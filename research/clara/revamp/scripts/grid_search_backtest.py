#!/usr/bin/env python3
"""
Parameter grid search over the quick backtest to find best ATR config.
Tests combinations of SL mult, TP ratio, entry threshold, and ADX filter.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ────────────────────────────────────────────────
SPREAD_SIM       = {'EUR/USD': 0.00012, 'GBP/USD': 0.00016}
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


def compute_indicators(df):
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float)

    df['ema20']  = c.ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema50']  = c.ewm(span=EMA_LONG,  adjust=False).mean()
    df['ema200'] = c.ewm(span=EMA_TREND, adjust=False).mean()

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / (loss + 1e-10)
    df['rsi'] = 100 - 100 / (1 + rs)

    up_move = h - h.shift(1)
    dn_move = l.shift(1) - l
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_s = tr.rolling(ADX_PERIOD).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(ADX_PERIOD).mean() / (atr_s + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(ADX_PERIOD).mean() / (atr_s + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(ADX_PERIOD).mean()

    df['vol_sma'] = v.rolling(VOL_PERIOD).mean()

    atr_diff = df['atr'].diff()
    rise = (atr_diff > 0).astype(int)
    df['atr_rise4'] = rise.rolling(4).sum()

    # ATR rolling average (for max ATR cap)
    df['atr_avg20'] = df['atr'].rolling(20).mean()

    df['body'] = (c - o).abs()
    df['range'] = h - l
    df['high5'] = h.rolling(STRUCTURE_LOOKBACK).max().shift(1)
    df['low5']  = l.rolling(STRUCTURE_LOOKBACK).min().shift(1)

    return df


def score_entry(row, direction):
    score = 0.0
    ema20_dist = abs(row['close'] - row['ema20'])
    if ema20_dist <= 0.5 * row['atr']:
        score += 0.20
    if 40 <= row['rsi'] <= 60:
        score += 0.15
    if direction == 'BUY' and row['close'] > row['high5']:
        score += 0.25
    elif direction == 'SELL' and row['close'] < row['low5']:
        score += 0.25
    if row['range'] > 0 and row['body'] / row['range'] > 0.60:
        score += 0.20
    if row['vol_sma'] > 0 and row['volume'] > row['vol_sma'] * VOL_SPIKE_MULT:
        score += 0.20
    return score


def run_config(df_dict, sl_mult, tp_ratio, threshold, adx_min, skip_contracting,
               max_atr_cap):
    """Run backtest with given params across all pairs."""
    all_trades = []
    balance = 1000

    for pair, df in df_dict.items():
        spread = SPREAD_SIM.get(pair, 0.00015)
        pip_size = 0.01 if 'JPY' in pair else 0.0001
        last_exit_idx = -COOLDOWN_BARS
        start = max(220, ATR_PERIOD + EMA_TREND + 5)

        for i in range(start, len(df)):
            row = df.iloc[i]
            if pd.isna(row['atr']) or pd.isna(row['ema200']) or pd.isna(row['rsi']):
                continue
            if (i - last_exit_idx) < COOLDOWN_BARS:
                continue
            if row['atr'] < spread * 5:
                continue
            if spread > 0.20 * row['atr']:
                continue

            # ATR regime
            ar4 = row['atr_rise4']
            if ar4 >= 3:
                regime = 'expanding'
            elif ar4 <= 1:
                regime = 'contracting'
            else:
                regime = 'neutral'

            if skip_contracting and regime == 'contracting':
                continue

            # Max ATR cap: skip if ATR > cap × rolling average
            if max_atr_cap > 0 and not pd.isna(row['atr_avg20']):
                if row['atr'] > max_atr_cap * row['atr_avg20']:
                    continue

            # Session
            if 'datetime' in df.columns:
                hr = row['datetime'].hour
                if not (7 <= hr <= 12 or 13 <= hr <= 17):
                    continue

            ema20, ema50, ema200 = row['ema20'], row['ema50'], row['ema200']
            if ema20 > ema50 > ema200:
                direction = 'BUY'
            elif ema20 < ema50 < ema200:
                direction = 'SELL'
            else:
                continue

            if row['adx'] < adx_min:
                continue

            score = score_entry(row, direction)
            if score < threshold:
                continue

            entry_price = row['close']
            sl_distance = sl_mult * row['atr']
            tp_distance = sl_distance * tp_ratio

            if direction == 'BUY':
                sl_p = entry_price - sl_distance
                tp_p = entry_price + tp_distance
            else:
                sl_p = entry_price + sl_distance
                tp_p = entry_price - tp_distance

            exit_reason = None
            exit_price = entry_price
            exit_idx = i

            for j in range(i + 1, min(i + MAX_HOLD_BARS + 1, len(df))):
                bar = df.iloc[j]
                if direction == 'BUY':
                    if bar['high'] >= tp_p:
                        exit_reason, exit_price, exit_idx = 'TP', tp_p, j; break
                    if bar['low'] <= sl_p:
                        exit_reason, exit_price, exit_idx = 'SL', sl_p, j; break
                else:
                    if bar['low'] <= tp_p:
                        exit_reason, exit_price, exit_idx = 'TP', tp_p, j; break
                    if bar['high'] >= sl_p:
                        exit_reason, exit_price, exit_idx = 'SL', sl_p, j; break

            if exit_reason is None:
                exit_idx = min(i + MAX_HOLD_BARS, len(df) - 1)
                exit_price = df['close'].iloc[exit_idx]
                exit_reason = 'TIME'

            if direction == 'BUY':
                pnl = (exit_price - entry_price - spread) * 100_000
            else:
                pnl = (entry_price - exit_price - spread) * 100_000

            all_trades.append(pnl)
            balance += pnl
            last_exit_idx = exit_idx

    n = len(all_trades)
    if n == 0:
        return {'n': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'balance': 1000}

    wins = sum(1 for p in all_trades if p > 0)
    total_w = sum(p for p in all_trades if p > 0)
    total_l = sum(p for p in all_trades if p <= 0)
    pf = total_w / abs(total_l) if total_l else 99

    return {
        'n': n,
        'wr': wins / n * 100,
        'pf': pf,
        'pnl': sum(all_trades),
        'balance': balance,
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    df_dict = {}
    for pair, fname in [('EUR/USD', 'EUR_USD_5m.csv'), ('GBP/USD', 'GBP_USD_5m.csv')]:
        fp = os.path.join(data_dir, fname)
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
            df = compute_indicators(df)
            df_dict[pair] = df

    if not df_dict:
        print("No data found"); return

    print(f"{'SL':>4} {'TP':>4} {'Thr':>5} {'ADX':>4} {'Skip':>5} {'Cap':>4} | "
          f"{'#':>3} {'WR%':>5} {'PF':>5} {'P&L':>8} {'Bal':>8}")
    print("-" * 75)

    best = None

    for sl_mult in [0.8, 1.0, 1.2]:
        for tp_ratio in [1.0, 1.2, 1.3, 1.4, 1.6]:
            for threshold in [0.60, 0.70, 0.80]:
                for adx_min in [15, 18, 22]:
                    for skip_c in [True, False]:
                        for atr_cap in [0, 1.5, 2.0]:
                            r = run_config(df_dict, sl_mult, tp_ratio, threshold,
                                           adx_min, skip_c, atr_cap)
                            if r['n'] < 5:
                                continue

                            tag = ""
                            if best is None or r['pnl'] > best['pnl']:
                                best = {**r, 'sl': sl_mult, 'tp': tp_ratio,
                                        'thr': threshold, 'adx': adx_min,
                                        'skip': skip_c, 'cap': atr_cap}
                                tag = " <-- BEST"

                            if r['pf'] >= 0.9 or r['pnl'] > -50:
                                print(f"{sl_mult:4.1f} {tp_ratio:4.1f} {threshold:5.2f} "
                                      f"{adx_min:4d} {str(skip_c):>5} {atr_cap:4.1f} | "
                                      f"{r['n']:3d} {r['wr']:5.1f} {r['pf']:5.2f} "
                                      f"${r['pnl']:7.1f} ${r['balance']:7.1f}{tag}")

    if best:
        print(f"\n{'='*75}")
        print(f"BEST CONFIG: SL={best['sl']}×ATR  TP={best['tp']}R  "
              f"Threshold={best['thr']}  ADX>={best['adx']}  "
              f"SkipContracting={best['skip']}  ATR_Cap={best['cap']}")
        print(f"  {best['n']} trades | WR={best['wr']:.1f}% | PF={best['pf']:.2f} | "
              f"P&L=${best['pnl']:.1f} | Balance=${best['balance']:.1f}")


if __name__ == '__main__':
    main()
