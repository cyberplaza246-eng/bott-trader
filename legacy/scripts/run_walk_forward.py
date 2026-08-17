"""
Walk-Forward Backtest Runner

Runs walk-forward analysis on historical data to validate the strategy
without overfitting.

Usage:
  python -m scripts.run_walk_forward
  python -m scripts.run_walk_forward --pair EUR/USD --splits 5 --mode rolling
  python -m scripts.run_walk_forward --slippage 0.5
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backtest.walk_forward import WalkForwardEngine
from config.strategy_config import (
    WALKFORWARD_SPLITS,
    WALKFORWARD_TRAIN_PCT,
    WALKFORWARD_MODE,
    BACKTEST_SLIPPAGE_PIPS,
    PAIRS,
)


def load_data(pair: str, timeframe: str = '5m') -> pd.DataFrame:
    """Load historical CSV data for a pair."""
    pair_clean = pair.replace('/', '_')
    tf_map = {'5m': '5m', '1m': '1m', '1h': '1h'}
    tf = tf_map.get(timeframe, '5m')
    path = f'data/{pair_clean}_{tf}.csv'

    if not os.path.exists(path):
        print(f"❌ Data file not found: {path}")
        return None

    df = pd.read_csv(path)
    if 'datetime' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"📊 Loaded {len(df)} candles from {path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Backtest')
    parser.add_argument('--pair', type=str, default=None,
                        help='Currency pair (default: all configured pairs)')
    parser.add_argument('--splits', type=int, default=WALKFORWARD_SPLITS,
                        help=f'Number of walk-forward splits (default: {WALKFORWARD_SPLITS})')
    parser.add_argument('--train-pct', type=float, default=WALKFORWARD_TRAIN_PCT,
                        help=f'Training fraction (default: {WALKFORWARD_TRAIN_PCT})')
    parser.add_argument('--mode', type=str, default=WALKFORWARD_MODE,
                        choices=['rolling', 'anchored'],
                        help=f'Walk-forward mode (default: {WALKFORWARD_MODE})')
    parser.add_argument('--slippage', type=float, default=BACKTEST_SLIPPAGE_PIPS,
                        help=f'Slippage per trade in pips (default: {BACKTEST_SLIPPAGE_PIPS})')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance (default: 10000)')
    parser.add_argument('--threshold', type=float, default=0.30,
                        help='Confidence threshold (default: 0.30)')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Primary timeframe (default: 5m)')
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else PAIRS

    print("=" * 70)
    print("🔄 WALK-FORWARD ANALYSIS")
    print(f"   Mode: {args.mode} | Splits: {args.splits} | Train: {args.train_pct:.0%}")
    print(f"   Slippage: {args.slippage} pips | Balance: ${args.balance:,.0f}")
    print(f"   Pairs: {', '.join(pairs)}")
    print("=" * 70)

    all_results = {}

    for pair in pairs:
        print(f"\n{'─' * 50}")
        print(f"📈 Running walk-forward for {pair}")
        print(f"{'─' * 50}")

        df = load_data(pair, args.timeframe)
        if df is None or len(df) < 500:
            print(f"⚠️ Skipping {pair}: insufficient data")
            continue

        # Also try loading 1M data for confluence
        df_1m = load_data(pair, '1m')

        wf = WalkForwardEngine(initial_balance=args.balance)
        result = wf.run_walk_forward(
            df, pair,
            train_pct=args.train_pct,
            n_splits=args.splits,
            mode=args.mode,
            confidence_threshold=args.threshold,
            timeframe_key=args.timeframe,
            df_1m=df_1m,
            slippage_pips=args.slippage,
        )

        all_results[pair] = result

        # Print results
        print(f"\n📊 {pair} Walk-Forward Results (Out-of-Sample):")
        print(f"   Total OOS Trades:  {result['total_trades']}")
        print(f"   Win Rate:          {result['win_rate']:.1f}%")
        print(f"   Profit Factor:     {result['profit_factor']:.2f}")
        print(f"   Sharpe Ratio:      {result['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:      {result['max_drawdown']:.2f}%")
        print(f"   Total Return:      {result['return_percent']:.2f}%")
        print(f"   Final Balance:     ${result['final_balance']:.2f}")

        if result.get('overfit_ratio', 0) > 0:
            print(f"   Avg Train WR:      {result['avg_train_win_rate']:.1f}%")
            print(f"   Avg Test WR:       {result['avg_test_win_rate']:.1f}%")
            overfit = result['overfit_ratio']
            flag = "⚠️ POSSIBLE OVERFIT" if overfit > 1.3 else "✅ HEALTHY"
            print(f"   Overfit Ratio:     {overfit:.2f} {flag}")

        if 'fold_results' in result:
            print(f"\n   Per-Fold Breakdown:")
            for fold in result['fold_results']:
                print(
                    f"     Fold {fold['fold']}: "
                    f"train={fold['train_trades']}t/{fold['train_win_rate']:.0f}%WR "
                    f"→ test={fold['test_trades']}t/{fold['test_win_rate']:.0f}%WR "
                    f"PF={fold['test_profit_factor']:.2f} "
                    f"DD={fold['test_max_drawdown']:.1f}%"
                )

    # Summary
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("📋 SUMMARY")
        print(f"{'=' * 70}")
        for pair, r in all_results.items():
            flag = "✅" if r['win_rate'] > 50 and r['profit_factor'] > 1.0 else "❌"
            print(
                f"  {flag} {pair}: WR={r['win_rate']:.1f}% "
                f"PF={r['profit_factor']:.2f} "
                f"DD={r['max_drawdown']:.1f}% "
                f"Return={r['return_percent']:.2f}%"
            )

    print("\n✅ Walk-forward analysis complete")


if __name__ == '__main__':
    main()
