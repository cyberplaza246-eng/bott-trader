#!/usr/bin/env python3
"""
Compare rithmic backtest results: BEFORE vs AFTER applying MTF backtest improvements.

Improvements being tested:
  1. MNQ ADX threshold: 20 → 22
  2. Tighter RSI filter: <75/>25 → 40-55 long / 45-60 short
  3. TP ratio adjustment: test 1.8R for both symbols
  4. Volume filter tightened
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rithmic_backtest import (
    run_backtest, compute_stats, print_results, SYMBOL_CONFIGS, ema, rsi, atr, adx
)


def run_backtest_improved(df, symbol, initial_balance=50000, cfg=None):
    """Modified backtest with tighter RSI and volume filters from MTF backtest findings."""
    if cfg is None:
        cfg = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS['MES']).copy()

    tick_val = cfg['tick_value']
    comm = cfg['commission']

    # Indicators
    df['ema_f'] = ema(df['close'], cfg['ema_fast'])
    df['ema_s'] = ema(df['close'], cfg['ema_slow'])
    df['ema_t'] = ema(df['close'], cfg['ema_trend'])
    df['ema_l'] = ema(df['close'], cfg['ema_long'])
    df['rsi'] = rsi(df['close'], 14)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['adx'], df['plus_di'], df['minus_di'] = adx(df['high'], df['low'], df['close'], 14)

    # Volume
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
    else:
        df['vol_ratio'] = 1.0

    # Signal generation with TIGHTER filters
    bull_cross = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
    bear_cross = (df['ema_f'] < df['ema_s']) & (df['ema_f'].shift(1) >= df['ema_s'].shift(1))

    above_200 = df['close'] > df['ema_l']
    below_200 = df['close'] < df['ema_l']
    bull_trend = df['ema_t'] > df['ema_l']
    bear_trend = df['ema_t'] < df['ema_l']
    trending = df['adx'] > cfg['adx_min']

    # IMPROVEMENT 1: Tighter RSI — pre-expansion zone from MTF backtest
    rsi_ok_long = (df['rsi'] >= 40) & (df['rsi'] <= 55)   # Was: < 75
    rsi_ok_short = (df['rsi'] >= 45) & (df['rsi'] <= 60)  # Was: > 25

    # Session filter
    hour = df.index.hour
    bad = hour.isin(cfg['bad_hours'])
    is_friday = df.index.dayofweek == 4
    session_ok = ~bad
    if cfg['skip_friday']:
        session_ok = session_ok & ~is_friday

    # IMPROVEMENT 2: Volume must be >= 1.0x average (was quantile-based)
    vol_ok = df['vol_ratio'] >= 1.0

    df['signal'] = 0
    df.loc[bull_cross & above_200 & bull_trend & trending & rsi_ok_long & session_ok & vol_ok, 'signal'] = 1
    df.loc[bear_cross & below_200 & bear_trend & trending & rsi_ok_short & session_ok & vol_ok, 'signal'] = -1

    # Execution loop (same as original)
    trades = []
    balance = initial_balance
    position = 0
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

        if position != 0:
            bars_in_trade += 1
            hi, lo, cl = row['high'], row['low'], row['close']

            if not trailing_on:
                sl_dist = abs(entry_price - stop_loss)
                r_target = entry_price + sl_dist * cfg['trailing_r'] if position > 0 else entry_price - sl_dist * cfg['trailing_r']
                if (position > 0 and hi >= r_target) or (position < 0 and lo <= r_target):
                    stop_loss = entry_price + cfg['trailing_offset'] * (1 if position > 0 else -1)
                    trailing_on = True

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
                    'entry_time': entry_time, 'exit_time': idx,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'stop_loss': stop_loss, 'take_profit': take_profit,
                    'exit_reason': exit_reason, 'pnl': pnl,
                    'bars_held': bars_in_trade, 'atr_at_entry': entry_atr,
                })
                position = 0
                cooldown = cfg['cooldown_bars']
                if exit_reason in ('STOP_LOSS', 'TRAILING_BE', 'TIMEOUT'):
                    continue

        if position == 0 and sig != 0:
            position = sig
            entry_price = row['close']
            entry_atr = row['atr']
            entry_time = idx
            trailing_on = False
            bars_in_trade = 0

            a = row['atr']
            if position > 0:
                stop_loss = entry_price - a * cfg['atr_sl_mult']
                take_profit = entry_price + a * cfg['atr_tp_mult']
            else:
                stop_loss = entry_price + a * cfg['atr_sl_mult']
                take_profit = entry_price - a * cfg['atr_tp_mult']

            balance -= comm * abs(position)

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


def main():
    print("━" * 70)
    print("  BEFORE vs AFTER — Applying MTF Backtest Improvements")
    print("━" * 70)
    print()
    print("  Improvements applied:")
    print("    1. MNQ ADX: 20 → 22 (stronger trends only)")
    print("    2. RSI: <75/>25 → 40-55 long / 45-60 short (pre-expansion zone)")
    print("    3. Volume: quantile-based → >= 1.0x 20-period average")
    print("    4. TP ratio: test 1.8R for both symbols")
    print()

    initial_balance = 50000

    for symbol in ['MES', 'MNQ']:
        data_file = f'data/{symbol}_5m.csv'
        if not os.path.exists(data_file):
            print(f"❌ {data_file} not found")
            continue

        df = pd.read_csv(data_file, parse_dates=['datetime'], index_col='datetime')
        print(f"\n{'='*70}")
        print(f"  {symbol}: {len(df)} bars")
        print(f"{'='*70}")

        # ── BEFORE (original params) ──
        print(f"\n  ▶ BEFORE (original parameters)")
        cfg_before = SYMBOL_CONFIGS[symbol].copy()
        trades_before, bal_before = run_backtest(df.copy(), symbol, initial_balance, cfg_before)
        stats_before = compute_stats(trades_before, initial_balance)

        # ── AFTER (improved params) ──
        print(f"\n  ▶ AFTER (improved parameters)")
        cfg_after = SYMBOL_CONFIGS[symbol].copy()
        # IMPROVEMENT: MNQ ADX 20 → 22
        if symbol == 'MNQ':
            cfg_after['adx_min'] = 22
        # IMPROVEMENT: TP ratio → 1.8R
        cfg_after['atr_tp_mult'] = cfg_after['atr_sl_mult'] * 1.8
        trades_after, bal_after = run_backtest_improved(df.copy(), symbol, initial_balance, cfg_after)
        stats_after = compute_stats(trades_after, initial_balance)

        # ── Comparison ──
        print(f"\n  {'─'*66}")
        print(f"  {'METRIC':<20} {'BEFORE':>15} {'AFTER':>15} {'CHANGE':>15}")
        print(f"  {'─'*66}")
        if stats_before and stats_after:
            metrics = [
                ('Trades', stats_before['trades'], stats_after['trades'], ''),
                ('Win Rate', f"{stats_before['win_rate']:.1f}%", f"{stats_after['win_rate']:.1f}%", 
                 f"{stats_after['win_rate'] - stats_before['win_rate']:+.1f}%"),
                ('Profit Factor', f"{stats_before['profit_factor']:.2f}", f"{stats_after['profit_factor']:.2f}",
                 f"{stats_after['profit_factor'] - stats_before['profit_factor']:+.2f}"),
                ('Total P&L', f"${stats_before['total_pnl']:,.2f}", f"${stats_after['total_pnl']:,.2f}",
                 f"${stats_after['total_pnl'] - stats_before['total_pnl']:+,.2f}"),
                ('Avg Win', f"${stats_before['avg_win']:,.2f}", f"${stats_after['avg_win']:,.2f}", ''),
                ('Avg Loss', f"${stats_before['avg_loss']:,.2f}", f"${stats_after['avg_loss']:,.2f}", ''),
                ('W/L Ratio', f"{stats_before['wl_ratio']:.2f}", f"{stats_after['wl_ratio']:.2f}",
                 f"{stats_after['wl_ratio'] - stats_before['wl_ratio']:+.2f}"),
                ('Max Drawdown', f"{stats_before['max_drawdown_pct']:.2f}%", f"{stats_after['max_drawdown_pct']:.2f}%",
                 f"{stats_after['max_drawdown_pct'] - stats_before['max_drawdown_pct']:+.2f}%"),
                ('Max Consec L', stats_before['max_consec_losses'], stats_after['max_consec_losses'], ''),
                ('Return', f"{stats_before['return_pct']:+.2f}%", f"{stats_after['return_pct']:+.2f}%",
                 f"{stats_after['return_pct'] - stats_before['return_pct']:+.2f}%"),
            ]
            for name, before, after, change in metrics:
                print(f"  {name:<20} {str(before):>15} {str(after):>15} {str(change):>15}")

            # Validation
            pf_ok = stats_after['profit_factor'] >= 1.3
            dd_ok = stats_after['max_drawdown_pct'] < 10
            wr_ok = 45 <= stats_after['win_rate'] <= 55
            improved = stats_after['profit_factor'] > stats_before['profit_factor']

            print(f"\n  ✅ VALIDATION:")
            print(f"    PF ≥ 1.3:         {'✅' if pf_ok else '❌'} ({stats_after['profit_factor']:.2f})")
            print(f"    DD < 10%:         {'✅' if dd_ok else '❌'} ({stats_after['max_drawdown_pct']:.2f}%)")
            print(f"    WR 45-55%:        {'✅' if wr_ok else '⚠️'} ({stats_after['win_rate']:.1f}%)")
            print(f"    PF improved:      {'✅' if improved else '❌'} ({stats_before['profit_factor']:.2f} → {stats_after['profit_factor']:.2f})")

        elif stats_before:
            print(f"  BEFORE: {stats_before['trades']} trades, PF {stats_before['profit_factor']:.2f}")
            print(f"  AFTER: No trades (filters too tight)")
        else:
            print(f"  No baseline trades")

    print(f"\n{'━'*70}")
    print(f"  Done.")
    print(f"{'━'*70}")


if __name__ == '__main__':
    main()
