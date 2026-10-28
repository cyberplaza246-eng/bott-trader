#!/usr/bin/env python
"""
Run the optimized clean scalper strategy on MES and MNQ.

Usage:
    python scripts/run_clean_scalper.py           # Backtest both
    python scripts/run_clean_scalper.py MES       # Backtest MES only
    python scripts/run_clean_scalper.py --live    # Live mode (TODO)
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.clean_scalper import run_backtest, CleanScalper


def main():
    pairs = ['MES', 'MNQ']
    live_mode = '--live' in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] not in ('--live', '-h', '--help'):
        pairs = [sys.argv[1]]
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return
    
    print("=" * 70)
    print("Clean Scalper Strategy - Optimized Settings")
    print("=" * 70)
    print()
    print("Strategy Rules:")
    print("  1. EMA Trend: Price vs 200 EMA, 9 EMA vs 21 EMA")
    print("  2. RSI: Cross above 40 (long) / below 60 (short)")
    print("  3. MACD: Bullish/bearish crossovers")
    print("  4. Volume: > 20-period average")
    print("  5. ATR: > 20-period average (active market)")
    print("  6. Sweep: Bonus for swing high/low rejections")
    print()
    print(f"Min confirmations: {CleanScalper.MIN_CONFIRMATIONS}")
    print()
    
    if live_mode:
        print("LIVE MODE: Not implemented yet")
        print("Use this backtest to validate before going live.")
        return
    
    # Run backtests
    total_pnl = 0
    total_trades = 0
    total_winners = 0
    
    for pair in pairs:
        try:
            df_5m = pd.read_csv(f'data/{pair}_5m.csv', parse_dates=['datetime'])
            df_1m = pd.read_csv(f'data/{pair}_1m.csv', parse_dates=['datetime'])
        except FileNotFoundError:
            print(f"Data not found for {pair}, skipping...")
            continue
        
        config = CleanScalper.INSTRUMENT_CONFIG.get(pair, {})
        sl = config.get('sl_ticks', 12)
        tp = config.get('tp_ticks', 30)
        
        print(f"{pair}: SL={sl} ticks, TP={tp} ticks ({tp/sl:.1f}x R:R)")
        
        results = run_backtest(df_5m, df_1m, pair)
        
        print(f"  Trades: {results['total_trades']} | Win Rate: {results['win_rate']:.1f}%")
        print(f"  Net P&L: ${results['net_pnl']:.2f} | Max DD: {results['max_drawdown_pct']:.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print()
        
        total_pnl += results['net_pnl']
        total_trades += results['total_trades']
        total_winners += results['winners']
    
    if len(pairs) > 1:
        combined_wr = total_winners / total_trades * 100 if total_trades > 0 else 0
        print("-" * 70)
        print(f"COMBINED: {total_trades} trades | {combined_wr:.1f}% WR | ${total_pnl:.2f}")
        print(f"Return on $50K: {total_pnl/50000*100:.2f}%")


if __name__ == '__main__':
    main()
