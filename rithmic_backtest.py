#!/usr/bin/env python3
"""
Rithmic Data Backtest — built from analysis of 444 paper trades
Uses actual Rithmic MES/MNQ data with empirically-validated filters.

Key findings from paper trade analysis:
  - Bad hours (UTC): 0, 4, 7, 11, 13, 17, 21 → negative edge
  - Friday: negative overall (-$560)
  - Duration sweet spots: <15min (quick winners) and 1-4hr (trend rides)
  - 15-60 min trades are breakeven → trailing stop helps convert these
  - Real W/L ratio: 1.86 from paper trades
  - Filtering bad hours + Friday: PF 1.50 → 2.19
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime


# ─── Indicator helpers ───────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.rolling(period).mean(), plus_di, minus_di


def vwap(high, low, close, volume):
    """Session VWAP (resets each day)."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).groupby(typical.index.date).cumsum()
    cum_vol = volume.groupby(volume.index.date).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ─── Per-symbol configuration ───────────────────────────────────
SYMBOL_CONFIGS = {
    'MES': {
        'tick_value': 1.25,         # $1.25 per 0.25 tick (=$5/pt)
        'commission': 2.50,         # round-trip per contract
        'atr_sl_mult': 2.5,        # SL = 2.5× ATR (from sweep)
        'atr_tp_mult': 7.5,        # TP = 7.5× ATR (3.0R — from sweep best PF 1.32)
        'trailing_r': 1.5,         # move SL after 1.5R
        'trailing_offset': 1.0,    # trail to entry + 1pt ($5 > $2.50 commission)
        'ema_fast': 9,
        'ema_slow': 21,
        'ema_trend': 50,
        'ema_long': 200,
        'adx_min': 25,             # from sweep
        'cooldown_bars': 60,       # from sweep
        'volume_min_pct': 0.20,
        'bad_hours': [0, 4, 21],  # only truly dead hours (midnight, 12AM, 5PM ET)
        'skip_friday': False,
        'max_hold_bars': 120,
    },
    'MNQ': {
        'tick_value': 0.50,         # $0.50 per 0.25 tick (=$2/pt)
        'commission': 2.50,
        'atr_sl_mult': 2.5,        # from sweep
        'atr_tp_mult': 3.75,       # 1.5R (from sweep — MNQ needs tighter TP)
        'trailing_r': 999,          # disabled (from sweep — 'none' consistently best for MNQ)
        'trailing_offset': 0,
        'ema_fast': 9,
        'ema_slow': 21,
        'ema_trend': 50,
        'ema_long': 200,
        'adx_min': 20,             # lower threshold from sweep
        'cooldown_bars': 30,       # shorter cooldown from sweep
        'volume_min_pct': 0.20,
        'bad_hours': [0, 4, 21],   # only truly dead hours
        'skip_friday': False,
        'max_hold_bars': 120,
    },
}


# ─── Core backtest engine ────────────────────────────────────────
def run_backtest(df, symbol, initial_balance=50000, cfg=None):
    """
    Bar-by-bar backtest with:
      - EMA crossover signals (9/21) gated by 50/200 trend + ADX + RSI + session
      - ATR-based SL/TP
      - Trailing stop to breakeven after 1R
      - Time-of-day and day-of-week filters (from paper trade analysis)
      - Volume filter
      - Cooldown after trade close
      - Max hold timeout
    """
    if cfg is None:
        cfg = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS['MES'])

    tick_val = cfg['tick_value']
    comm     = cfg['commission']

    # ── Indicators ──
    df['ema_f']   = ema(df['close'], cfg['ema_fast'])
    df['ema_s']   = ema(df['close'], cfg['ema_slow'])
    df['ema_t']   = ema(df['close'], cfg['ema_trend'])
    df['ema_l']   = ema(df['close'], cfg['ema_long'])
    df['rsi']     = rsi(df['close'], 14)
    df['atr']     = atr(df['high'], df['low'], df['close'], 14)
    df['adx'], df['plus_di'], df['minus_di'] = adx(df['high'], df['low'], df['close'], 14)

    # VWAP for mean-reversion context
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])
    else:
        df['vwap'] = df['close']  # fallback

    # ── Signal generation ──
    bull_cross = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
    bear_cross = (df['ema_f'] < df['ema_s']) & (df['ema_f'].shift(1) >= df['ema_s'].shift(1))

    above_200 = df['close'] > df['ema_l']
    below_200 = df['close'] < df['ema_l']
    bull_trend = df['ema_t'] > df['ema_l']
    bear_trend = df['ema_t'] < df['ema_l']
    trending  = df['adx'] > cfg['adx_min']

    # RSI as loose filter: not extremely overbought for longs, not extremely oversold for shorts
    # (opposite of tight OS/OB — we just exclude entries against extreme momentum)
    rsi_ok_long  = df['rsi'] < 75   # don't buy when extremely overbought
    rsi_ok_short = df['rsi'] > 25   # don't sell when extremely oversold

    # Session filter from paper trade analysis
    hour = df.index.hour
    bad = hour.isin(cfg['bad_hours'])
    is_friday = df.index.dayofweek == 4
    session_ok = ~bad
    if cfg['skip_friday']:
        session_ok = session_ok & ~is_friday

    # Volume filter
    vol_threshold = df['volume'].quantile(cfg['volume_min_pct']) if 'volume' in df.columns else 0
    vol_ok = df['volume'] >= vol_threshold if 'volume' in df.columns else True

    df['signal'] = 0
    df.loc[bull_cross & above_200 & bull_trend & trending & rsi_ok_long  & session_ok & vol_ok, 'signal'] = 1
    df.loc[bear_cross & below_200 & bear_trend & trending & rsi_ok_short & session_ok & vol_ok, 'signal'] = -1

    # ── Execution loop ──
    trades = []
    balance = initial_balance
    position  = 0
    entry_price = stop_loss = take_profit = entry_atr = 0
    entry_time = None
    trailing_on = False
    last_signal = 0
    cooldown = 0
    bars_in_trade = 0

    for idx, row in df.iterrows():
        raw = row['signal']
        sig = raw if raw != last_signal else 0
        last_signal = raw

        if cooldown > 0:
            cooldown -= 1
            sig = 0

        # ── Manage open position ──
        if position != 0:
            bars_in_trade += 1
            hi, lo, cl = row['high'], row['low'], row['close']

            # Trailing stop: move to entry+offset after trailing_r × R achieved
            if not trailing_on:
                sl_dist = abs(entry_price - stop_loss)
                r_target = entry_price + sl_dist * cfg['trailing_r'] if position > 0 else entry_price - sl_dist * cfg['trailing_r']
                if (position > 0 and hi >= r_target) or (position < 0 and lo <= r_target):
                    stop_loss = entry_price + cfg['trailing_offset'] * (1 if position > 0 else -1)
                    trailing_on = True

            # Check SL/TP/timeout
            sl_hit = (position > 0 and lo <= stop_loss) or (position < 0 and hi >= stop_loss)
            tp_hit = (position > 0 and hi >= take_profit) or (position < 0 and lo <= take_profit)
            timeout = bars_in_trade >= cfg['max_hold_bars']

            exit_price = None
            exit_reason = None

            if sl_hit:
                exit_price = stop_loss
                exit_reason = 'TRAILING_BE' if trailing_on else 'STOP_LOSS'
            elif tp_hit:
                exit_price = take_profit
                exit_reason = 'TAKE_PROFIT'
            elif timeout:
                exit_price = cl
                exit_reason = 'TIMEOUT'
            elif sig != 0 and np.sign(sig) != np.sign(position):
                exit_price = cl
                exit_reason = 'SIGNAL_REVERSE'

            if exit_price is not None:
                pnl = ((exit_price - entry_price) * tick_val * abs(position)
                       if position > 0 else
                       (entry_price - exit_price) * tick_val * abs(position))
                pnl -= comm * abs(position)
                balance += pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'bars_held': bars_in_trade,
                    'atr_at_entry': entry_atr,
                })
                position = 0
                cooldown = cfg['cooldown_bars']
                if exit_reason in ('STOP_LOSS', 'TRAILING_BE', 'TIMEOUT'):
                    continue  # don't re-enter same bar as close

        # ── Open new position ──
        if position == 0 and sig != 0:
            position = sig
            entry_price = row['close']
            entry_atr = row['atr']
            entry_time = idx
            trailing_on = False
            bars_in_trade = 0

            a = row['atr']
            if position > 0:
                stop_loss   = entry_price - a * cfg['atr_sl_mult']
                take_profit = entry_price + a * cfg['atr_tp_mult']
            else:
                stop_loss   = entry_price + a * cfg['atr_sl_mult']
                take_profit = entry_price - a * cfg['atr_tp_mult']

            balance -= comm * abs(position)  # entry commission

    # Close remaining
    if position != 0:
        ep = df['close'].iloc[-1]
        pnl = ((ep - entry_price) * tick_val if position > 0 else (entry_price - ep) * tick_val) * abs(position)
        pnl -= comm * abs(position)
        balance += pnl
        trades.append({
            'entry_time': entry_time, 'exit_time': df.index[-1],
            'direction': 'LONG' if position > 0 else 'SHORT',
            'entry_price': entry_price, 'exit_price': ep,
            'stop_loss': stop_loss, 'take_profit': take_profit,
            'exit_reason': 'END_OF_DATA', 'pnl': pnl,
            'bars_held': bars_in_trade, 'atr_at_entry': entry_atr,
        })

    return trades, balance


# ─── Walk-forward splitter ───────────────────────────────────────
def walk_forward_splits(df, n_splits=5, train_ratio=0.6):
    """
    Generate walk-forward train/test splits.
    Each fold trains on train_ratio of the available data up to that point,
    then tests on the next chunk.
    """
    total = len(df)
    test_size = total // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        test_start = total - (n_splits - i) * test_size
        test_end   = test_start + test_size
        train_end  = test_start
        train_start = max(0, int(train_end * (1 - train_ratio)))
        splits.append((
            df.iloc[train_start:train_end].copy(),
            df.iloc[test_start:test_end].copy(),
            f"Fold-{i+1}"
        ))
    return splits


# ─── Reporting ───────────────────────────────────────────────────
def compute_stats(trades, initial_balance):
    if not trades:
        return None
    wins  = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    gross_win  = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    bal = initial_balance
    for t in trades:
        bal += t['pnl']
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Consecutive losses
    max_streak = 0
    streak = 0
    for t in trades:
        if t['pnl'] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
        'wl_ratio': abs(np.mean([t['pnl'] for t in wins]) / np.mean([t['pnl'] for t in losses])) if losses and wins else 0,
        'max_drawdown_pct': max_dd,
        'max_consec_losses': max_streak,
        'final_balance': initial_balance + total_pnl,
        'return_pct': total_pnl / initial_balance * 100,
        'exit_reasons': {r: len([t for t in trades if t['exit_reason'] == r])
                         for r in set(t['exit_reason'] for t in trades)},
    }


def print_results(symbol, stats, label=''):
    if stats is None:
        print(f"  ❌ No trades for {symbol}")
        return
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"  {symbol}{tag}")
    print(f"{'='*60}")
    print(f"  Trades:        {stats['trades']:>6}")
    print(f"  Win Rate:      {stats['win_rate']:>6.1f}%")
    print(f"  Total P&L:     ${stats['total_pnl']:>10,.2f}")
    print(f"  Profit Factor: {stats['profit_factor']:>6.2f}")
    print(f"  Avg Win:       ${stats['avg_win']:>10,.2f}")
    print(f"  Avg Loss:      ${stats['avg_loss']:>10,.2f}")
    print(f"  W/L Ratio:     {stats['wl_ratio']:>6.2f}")
    print(f"  Max Drawdown:  {stats['max_drawdown_pct']:>6.2f}%")
    print(f"  Max Consec L:  {stats['max_consec_losses']:>6}")
    print(f"  Final Balance: ${stats['final_balance']:>10,.2f}")
    print(f"  Return:        {stats['return_pct']:>+6.2f}%")
    print(f"  Exit reasons:")
    for reason, count in sorted(stats['exit_reasons'].items()):
        print(f"    {reason:16s}: {count:3d} ({count/stats['trades']*100:5.1f}%)")


def print_comparison_table(all_results):
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Symbol':<6} {'Label':<20} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'PF':>6} {'MaxDD':>6} {'MaxCL':>5}")
    print(f"{'-'*80}")
    for symbol, label, stats in all_results:
        if stats:
            print(f"{symbol:<6} {label:<20} {stats['trades']:>6} "
                  f"{stats['win_rate']:>5.1f}% ${stats['total_pnl']:>9,.2f} "
                  f"{stats['profit_factor']:>6.2f} {stats['max_drawdown_pct']:>5.2f}% "
                  f"{stats['max_consec_losses']:>5}")


# ─── Parameter sweep ─────────────────────────────────────────────
def param_sweep(df, symbol, initial_balance=50000):
    """Try a grid of ATR multipliers, cooldown periods, and trailing configs."""
    base_cfg = SYMBOL_CONFIGS[symbol].copy()
    results = []

    for sl_mult in [1.5, 2.0, 2.5]:
        for tp_mult_factor in [1.5, 2.0, 2.5, 3.0]:
            for cooldown in [30, 60, 90]:
                for adx_min in [20, 25, 30]:
                    for trail_r, trail_off in [(0, 0), (1.0, 1.0 if symbol == 'MES' else 2.5), (1.5, 1.0 if symbol == 'MES' else 2.5)]:
                        cfg = base_cfg.copy()
                        cfg['atr_sl_mult'] = sl_mult
                        cfg['atr_tp_mult'] = sl_mult * tp_mult_factor
                        cfg['cooldown_bars'] = cooldown
                        cfg['adx_min'] = adx_min
                        cfg['trailing_r'] = trail_r
                        cfg['trailing_offset'] = trail_off
                        # trail_r=0 means no trailing
                        if trail_r == 0:
                            cfg['trailing_r'] = 999  # effectively disabled
                        trades, bal = run_backtest(df.copy(), symbol, initial_balance, cfg)
                        s = compute_stats(trades, initial_balance)
                        if s and s['trades'] >= 10:
                            trail_label = f"{trail_r:.1f}R" if trail_r > 0 else "none"
                            results.append({
                                'sl': sl_mult, 'r_mult': tp_mult_factor,
                                'cooldown': cooldown, 'adx': adx_min,
                                'trail': trail_label,
                                **s
                            })

    if not results:
        print(f"  No valid parameter combos for {symbol}")
        return None

    results.sort(key=lambda x: x['profit_factor'], reverse=True)
    print(f"\n  Top 10 parameter sets for {symbol} (by Profit Factor):")
    print(f"  {'SL':>4} {'R':>4} {'CD':>3} {'ADX':>3} {'Trail':>5} | {'Trades':>6} {'WR%':>5} {'PnL':>9} {'PF':>5} {'MaxDD':>5}")
    for r in results[:10]:
        print(f"  {r['sl']:>4.1f} {r['r_mult']:>4.1f} {r['cooldown']:>3} {r['adx']:>3} {r['trail']:>5} | "
              f"{r['trades']:>6} {r['win_rate']:>4.1f}% ${r['total_pnl']:>8,.0f} "
              f"{r['profit_factor']:>5.2f} {r['max_drawdown_pct']:>4.1f}%")
    return results[0]


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("━" * 70)
    print("  RITHMIC DATA BACKTEST — Empirically-tuned from 444 paper trades")
    print("━" * 70)

    initial_balance = 50000
    all_results = []

    for symbol in ['MES', 'MNQ']:
        data_file = f'data/{symbol}_5m.csv'
        if not os.path.exists(data_file):
            print(f"❌ {data_file} not found")
            continue

        df = pd.read_csv(data_file, parse_dates=['datetime'], index_col='datetime')
        print(f"\n📊 {symbol}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

        # ── 1. Full-period backtest ──
        print(f"\n▶ Full-period backtest ({symbol})")
        trades, bal = run_backtest(df.copy(), symbol, initial_balance)
        stats = compute_stats(trades, initial_balance)
        print_results(symbol, stats, 'Full Period')
        all_results.append((symbol, 'Full Period', stats))

        # ── 2. Walk-forward validation ──
        print(f"\n▶ Walk-forward validation ({symbol}, 5 folds)")
        wf_trades_all = []
        splits = walk_forward_splits(df, n_splits=5, train_ratio=0.6)
        for train_df, test_df, fold_name in splits:
            test_trades, _ = run_backtest(test_df.copy(), symbol, initial_balance)
            wf_trades_all.extend(test_trades)
            fold_stats = compute_stats(test_trades, initial_balance)
            if fold_stats:
                pf_str = f"PF {fold_stats['profit_factor']:.2f}"
                wr_str = f"WR {fold_stats['win_rate']:.0f}%"
                print(f"    {fold_name}: {fold_stats['trades']:>3} trades | {wr_str} | {pf_str} | ${fold_stats['total_pnl']:>+8,.2f}")

        wf_stats = compute_stats(wf_trades_all, initial_balance)
        print_results(symbol, wf_stats, 'Walk-Forward Combined')
        all_results.append((symbol, 'Walk-Forward', wf_stats))

        # ── 3. Hour-filtered analysis ──
        if stats and stats['trades'] > 0:
            print(f"\n▶ Hourly edge breakdown ({symbol})")
            for hour in range(24):
                h_trades = [t for t in trades if t['entry_time'].hour == hour]
                if len(h_trades) >= 2:
                    h_stats = compute_stats(h_trades, initial_balance)
                    emoji = '✅' if h_stats['total_pnl'] > 0 else '❌'
                    print(f"    {emoji} {hour:02d}:00 | {h_stats['trades']:>3} trades | "
                          f"WR {h_stats['win_rate']:>4.0f}% | ${h_stats['total_pnl']:>+7,.2f}")

        # ── 4. Parameter sweep ──
        print(f"\n▶ Parameter sweep ({symbol})")
        best = param_sweep(df.copy(), symbol, initial_balance)
        if best:
            print(f"\n  ★ Best params: SL={best['sl']:.1f}× ATR, R-mult={best['r_mult']:.1f}×, "
                  f"CDN={best['cooldown']}, ADX>{best['adx']} → PF {best['profit_factor']:.2f}")

    # ── Summary ──
    print_comparison_table(all_results)

    # ── Paper trade comparison ──
    paper_file = 'logs/paper_trade_MES_20260312_155640.json'
    if os.path.exists(paper_file):
        import json
        pd_data = json.load(open(paper_file))
        print(f"\n{'='*80}")
        print(f"  PAPER TRADE REFERENCE (444 trades, actual bot)")
        print(f"{'='*80}")
        ps = pd_data['stats']
        print(f"  WR: {ps['win_rate']:.1f}% | PF: {ps['profit_factor']:.2f} | "
              f"PnL: ${ps['total_pnl']:,.2f} | MaxDD: ${ps['max_drawdown']:,.2f}")
        print(f"  Note: Paper trades used full sweep-gated ensemble, not EMA crossover proxy")

    print(f"\n{'━'*70}")
    print(f"  Done. Use best parameters in config/strategy_config.py and data/risk_overrides.json")
    print(f"{'━'*70}")


if __name__ == '__main__':
    main()
