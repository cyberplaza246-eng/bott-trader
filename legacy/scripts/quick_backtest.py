#!/usr/bin/env python3
"""
Quick vectorised backtest — computes indicators ONCE then scans for signals.
Much faster than the per-candle approach on 17k+ bars.

Usage:  python scripts/quick_backtest.py
"""
import os, sys, math
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import bot_logger
from src.instruments import REGISTRY

# ── ATR Strategy constants ──────────────────────────────
SL_ATR_MULT      = 0.8
TP_BASE_RATIO    = 1.3
TP_EXPANDING     = 1.3
TP_CONTRACTING   = 1.3
ENTRY_THRESHOLD  = 0.70
SPREAD_SIM       = {sym: spec.spread_default for sym, spec in REGISTRY.items()}
COOLDOWN_BARS    = 10
MAX_HOLD_BARS    = 15
VOL_SPIKE_MULT   = 1.2
STRUCTURE_LOOKBACK = 5
ATR_PERIOD       = 14
RSI_PERIOD       = 14
EMA_SHORT        = 20
EMA_LONG         = 50
EMA_TREND        = 200
ADX_PERIOD       = 14
VOL_PERIOD       = 20

# Session windows per pair (UTC hours)
SESSION_WINDOWS = {
    'EUR/USD': (7, 17),
    'GBP/USD': (7, 17),
    'USD/JPY': (0, 17),
    'MES': (13, 20),      # US regular trading hours
    'MNQ': (13, 20),
}


def compute_indicators(df):
    """Compute all indicators once (vectorised)."""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float)

    # EMAs
    df['ema20']  = c.ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema50']  = c.ewm(span=EMA_LONG,  adjust=False).mean()
    df['ema200'] = c.ewm(span=EMA_TREND, adjust=False).mean()

    # ATR
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / (loss + 1e-10)
    df['rsi'] = 100 - 100 / (1 + rs)

    # ADX (simplified +DI/-DI via directional movement)
    up_move = h - h.shift(1)
    dn_move = l.shift(1) - l
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_s = tr.rolling(ADX_PERIOD).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(ADX_PERIOD).mean() / (atr_s + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(ADX_PERIOD).mean() / (atr_s + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(ADX_PERIOD).mean()

    # Volume SMA
    df['vol_sma'] = v.rolling(VOL_PERIOD).mean()

    # ATR streak (rolling count of consecutive rises)
    atr_diff = df['atr'].diff()
    rise = (atr_diff > 0).astype(int)
    # count consecutive rises (simple proxy: rolling 4‑bar sum ≥ 3 → expanding)
    df['atr_rise4'] = rise.rolling(4).sum()

    # Body ratio
    df['body'] = (c - o).abs()
    df['range'] = h - l

    # Structure break (5-bar high/low)
    df['high5'] = h.rolling(STRUCTURE_LOOKBACK).max().shift(1)
    df['low5']  = l.rolling(STRUCTURE_LOOKBACK).min().shift(1)

    return df


def detect_regime(atr_rise4):
    if atr_rise4 >= 3:
        return 'expanding'
    elif atr_rise4 <= 1:
        return 'contracting'
    return 'neutral'


def score_entry(row, direction):
    """Score a potential entry 0..1 based on ATR strategy rules."""
    score = 0.0

    # EMA20 pullback: price near EMA20 (within 0.5×ATR)
    ema20_dist = abs(row['close'] - row['ema20'])
    if ema20_dist <= 0.5 * row['atr']:
        score += 0.20

    # RSI 40-60
    if 40 <= row['rsi'] <= 60:
        score += 0.15

    # Structure break
    if direction == 'BUY' and row['close'] > row['high5']:
        score += 0.25
    elif direction == 'SELL' and row['close'] < row['low5']:
        score += 0.25

    # Engulfing / momentum candle (body > 60% of range)
    if row['range'] > 0 and row['body'] / row['range'] > 0.60:
        score += 0.20

    # Volume spike
    if row['vol_sma'] > 0 and row['volume'] > row['vol_sma'] * VOL_SPIKE_MULT:
        score += 0.20

    return score


def run_backtest(filepath, pair, initial_balance=1000):
    """Vectorised backtest on 5m data."""
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    bot_logger.info(f"Loaded {len(df)} candles for {pair}")

    df = compute_indicators(df)
    spread = SPREAD_SIM.get(pair, 0.00015)
    _spec = REGISTRY.get(pair)
    pip_size = _spec.tick_size if _spec else (0.01 if 'JPY' in pair else 0.0001)

    balance = initial_balance
    trades = []
    last_exit_idx = -COOLDOWN_BARS

    start = max(200, ATR_PERIOD + EMA_TREND + 5)  # warm-up

    for i in range(start, len(df)):
        row = df.iloc[i]

        # Skip if indicators NaN
        if pd.isna(row['atr']) or pd.isna(row['ema200']) or pd.isna(row['rsi']):
            continue

        # Cooldown
        if (i - last_exit_idx) < COOLDOWN_BARS:
            continue

        # ATR floor: must exceed minimum for pair
        min_atr = spread * 5  # ATR > 5× spread
        if row['atr'] < min_atr:
            continue

        # Spread check: spread < 20% of ATR
        if spread > 0.20 * row['atr']:
            continue

        # Skip contracting regime (backtest shows heavy losses)
        regime = detect_regime(row['atr_rise4'])
        if regime == 'contracting':
            continue

        # Session filter — per-pair windows
        session_start, session_end = SESSION_WINDOWS.get(pair, (7, 17))
        if 'datetime' in df.columns:
            hr = row['datetime'].hour
            if session_start <= session_end:
                if not (session_start <= hr <= session_end):
                    continue
            else:  # Wraps midnight (e.g. 22-6)
                if not (hr >= session_start or hr <= session_end):
                    continue
            # Prefer peak liquidity hours for scoring bonus
            peak_hours = (10 <= hr <= 12) or (13 <= hr <= 15)
        else:
            peak_hours = True

        # Determine bias from EMA alignment
        ema20, ema50, ema200 = row['ema20'], row['ema50'], row['ema200']
        if ema20 > ema50 > ema200:
            direction = 'BUY'
        elif ema20 < ema50 < ema200:
            direction = 'SELL'
        else:
            continue  # No clear bias

        # ADX filter: need trending market (22 = grid-search optimal)
        if row['adx'] < 22:
            continue

        # EMA200 confirmation: price must be on right side of EMA200
        if direction == 'BUY' and row['close'] < row['ema200']:
            continue
        if direction == 'SELL' and row['close'] > row['ema200']:
            continue

        # Score entry
        score = score_entry(row, direction)
        if score < ENTRY_THRESHOLD:
            continue

        # ── ENTRY ──
        entry_price = row['close']
        atr_val = row['atr']
        # regime already computed above for the contracting skip

        sl_distance = SL_ATR_MULT * atr_val
        tp_ratio = {'expanding': TP_EXPANDING, 'contracting': TP_CONTRACTING}.get(regime, TP_BASE_RATIO)
        tp_distance = sl_distance * tp_ratio

        if direction == 'BUY':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # ── SIM FORWARD ──
        exit_reason = None
        exit_price = entry_price
        exit_idx = i

        for j in range(i + 1, min(i + MAX_HOLD_BARS + 1, len(df))):
            bar = df.iloc[j]

            if direction == 'BUY':
                if bar['high'] >= tp_price:
                    exit_reason, exit_price, exit_idx = 'TP', tp_price, j
                    break
                if bar['low'] <= sl_price:
                    exit_reason, exit_price, exit_idx = 'SL', sl_price, j
                    break
            else:
                if bar['low'] <= tp_price:
                    exit_reason, exit_price, exit_idx = 'TP', tp_price, j
                    break
                if bar['high'] >= sl_price:
                    exit_reason, exit_price, exit_idx = 'SL', sl_price, j
                    break

        if exit_reason is None:
            # Time exit at last bar
            exit_idx = min(i + MAX_HOLD_BARS, len(df) - 1)
            exit_price = df['close'].iloc[exit_idx]
            exit_reason = 'TIME'

        # P&L — standard lot (100k units). For JPY pairs: divide by rate to get USD P&L
        if direction == 'BUY':
            raw_pnl = (exit_price - entry_price - spread) * 100_000
        else:
            raw_pnl = (entry_price - exit_price - spread) * 100_000

        # JPY pairs: raw P&L is in JPY, convert to USD
        if 'JPY' in pair:
            pnl = raw_pnl / entry_price  # approximate USD conversion
        else:
            pnl = raw_pnl

        pips = abs(exit_price - entry_price) / pip_size
        r_multiple = ((exit_price - entry_price) / sl_distance) if direction == 'BUY' else ((entry_price - exit_price) / sl_distance)

        trades.append({
            'idx': i,
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'sl': sl_price,
            'tp': tp_price,
            'reason': exit_reason,
            'pnl': pnl,
            'pips': pips,
            'r': r_multiple,
            'hold': exit_idx - i,
            'atr': atr_val,
            'score': score,
            'regime': regime,
            'time': row.get('datetime', ''),
        })
        balance += pnl
        last_exit_idx = exit_idx

    return trades, balance


def print_results(pair, trades, balance, initial):
    """Print formatted results."""
    if not trades:
        bot_logger.info(f"\n{pair}: No trades generated")
        return

    wins  = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    tp_trades = [t for t in trades if t['reason'] == 'TP']
    sl_trades = [t for t in trades if t['reason'] == 'SL']
    time_exits = [t for t in trades if t['reason'] == 'TIME']

    total_profit = sum(t['pnl'] for t in wins)
    total_loss   = sum(t['pnl'] for t in losses)
    pf = total_profit / abs(total_loss) if total_loss else float('inf')

    avg_win  = np.mean([t['pnl'] for t in wins])   if wins   else 0
    avg_loss = np.mean([t['pnl'] for t in losses])  if losses else 0
    avg_r    = np.mean([t['r'] for t in trades])
    avg_hold = np.mean([t['hold'] for t in trades])

    # Max drawdown
    equity = [initial]
    for t in trades:
        equity.append(equity[-1] + t['pnl'])
    peak = initial
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)

    # Regime breakdown
    regimes = {}
    for t in trades:
        r = t['regime']
        if r not in regimes:
            regimes[r] = {'count': 0, 'pnl': 0, 'wins': 0}
        regimes[r]['count'] += 1
        regimes[r]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            regimes[r]['wins'] += 1

    print(f"\n{'='*60}")
    print(f"  {pair} — ATR-Centric Backtest Results")
    print(f"{'='*60}")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Win rate:         {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.1f}%")
    print(f"  TP hits:          {len(tp_trades)}  SL hits: {len(sl_trades)}  Time exits: {len(time_exits)}")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Total P&L:        ${sum(t['pnl'] for t in trades):.2f}")
    print(f"  Avg win:          ${avg_win:.2f}   Avg loss: ${avg_loss:.2f}")
    print(f"  Avg R-multiple:   {avg_r:.2f}R")
    print(f"  Avg hold time:    {avg_hold:.1f} bars")
    print(f"  Max drawdown:     {max_dd*100:.1f}%")
    print(f"  Final balance:    ${balance:.2f}  ({(balance-initial)/initial*100:+.1f}%)")

    print(f"\n  Regime breakdown:")
    for r, d in sorted(regimes.items()):
        wr = d['wins']/d['count']*100 if d['count'] else 0
        print(f"    {r:14s}  {d['count']:3d} trades  WR={wr:.0f}%  P&L=${d['pnl']:.2f}")

    print(f"\n  Sample trades (first 8):")
    for t in trades[:8]:
        tag = "WIN " if t['pnl'] > 0 else "LOSS"
        print(f"    {tag} {t['direction']:4s} {t['time']}  "
              f"entry={t['entry']:.5f} exit={t['exit']:.5f}  "
              f"{t['pips']:.1f}p  {t['r']:+.2f}R  ${t['pnl']:+.1f}  "
              f"ATR={t['atr']:.5f}  [{t['reason']}]  score={t['score']:.2f}")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    initial = 1000

    pairs = {
        'EUR/USD': os.path.join(data_dir, 'EUR_USD_5m.csv'),
        'GBP/USD': os.path.join(data_dir, 'GBP_USD_5m.csv'),
        'USD/JPY': os.path.join(data_dir, 'USD_JPY_1h.csv'),  # 1h data (no 5m available)
    }

    grand_trades = []
    grand_balance = initial

    for pair, fp in pairs.items():
        if not os.path.exists(fp):
            print(f"  {pair}: data file not found ({fp})")
            continue
        trades, balance = run_backtest(fp, pair, grand_balance)
        print_results(pair, trades, balance, grand_balance)
        grand_trades.extend(trades)
        grand_balance = balance

    if grand_trades:
        wins = sum(1 for t in grand_trades if t['pnl'] > 0)
        print(f"\n{'='*60}")
        print(f"  COMBINED SUMMARY")
        print(f"{'='*60}")
        print(f"  Total trades:  {len(grand_trades)}")
        print(f"  Win rate:      {wins/len(grand_trades)*100:.1f}%")
        print(f"  Final balance: ${grand_balance:.2f} ({(grand_balance-initial)/initial*100:+.1f}%)")


if __name__ == '__main__':
    main()
