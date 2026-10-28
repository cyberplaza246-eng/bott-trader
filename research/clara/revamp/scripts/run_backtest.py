#!/usr/bin/env python
"""
Backtesting script — 9-Model Scalping Ensemble
Usage:
    python scripts/run_backtest.py --pair EUR/USD
    python scripts/run_backtest.py --pair GBP/USD --balance 500 --confidence 0.45
    python scripts/run_backtest.py --all           # run both EUR/USD and GBP/USD
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.backtest_engine import BacktestEngine
from src.broker.mt5_connector import MT5Connector
from config.strategy_config import PAIRS, SCALPING_PAIRS
from src.utils.logger import bot_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run 9-model scalping backtest')
    parser.add_argument('--pair', default=None, help='Symbol (default: first in PAIRS, e.g. MES or EUR/USD)')
    parser.add_argument('--all', action='store_true', help='Run all configured pairs')
    parser.add_argument('--start', default=None, help='Start date (YYYY-MM-DD), auto-detected from CSV if omitted')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD), auto-detected from CSV if omitted')
    parser.add_argument('--balance', type=float, default=50000, help='Initial balance (default: 50000)')
    parser.add_argument('--confidence', type=float, default=0.45, help='Min confidence (default: 0.45)')
    parser.add_argument('--agreement', type=int, default=2, help='Min models agreement (default: 2)')
    parser.add_argument('--tf', default='5m', choices=['1m', '5m'], help='Scalping config key (default: 5m)')
    parser.add_argument('--no-confluence', action='store_true', help='Disable 1M confluence detection')
    parser.add_argument('--candles', type=int, default=0, help='Limit primary candles (0=all)')
    parser.add_argument('--spread', type=float, default=None,
                        help='Override spread in pips (default: per-pair config, e.g. EUR/USD=1.5)')
    parser.add_argument('--commission', type=float, default=7.0,
                        help='Round-trip commission per standard lot in USD (default: 7.0)')
    parser.add_argument('--slippage', type=float, default=0.3,
                        help='Slippage per trade in pips (default: 0.3)')

    return parser.parse_args()


def fetch_historical_data(pair, start_date=None, end_date=None, interval='5m', num_candles=5000):
    """
    Load historical data from local CSV files or MT5.
    Auto-detects date range from the CSV when start/end are None.
    Falls back to synthetic data generation when no source is available.
    """
    csv_filename = f"data/{pair.replace('/', '_')}_{interval}.csv"

    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            df['datetime'] = pd.to_datetime(df['datetime'])

            if start_date:
                df = df[df['datetime'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.to_datetime(end_date)]

            if len(df) > 0:
                bot_logger.info(f"✅ Loaded {len(df)} candles from {csv_filename} "
                                f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
                return df
        except Exception as e:
            bot_logger.warning(f"Failed to load CSV: {e}")

    # Try generating fresh data
    try:
        from src.data.historical_downloader import HistoricalDownloader
        dl = HistoricalDownloader()
        df = dl.download(pair, days=60, interval=interval)
        if df is not None and len(df) > 100:
            if start_date:
                df = df[df['datetime'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.to_datetime(end_date)]
            bot_logger.info(f"✅ Downloaded {len(df)} candles for {pair} ({interval})")
            return df
    except Exception as e:
        bot_logger.warning(f"Download failed: {e}")

    # Fallback to MT5 relay
    try:
        connector = MT5Connector()
        tf_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240}
        tf_min = tf_map.get(interval, 5)
        df = connector.get_candles(pair, timeframe_minutes=tf_min, num_candles=num_candles)

        if df is None:
            bot_logger.error(f"Failed to fetch data for {pair}")
            return None

        df['datetime'] = pd.to_datetime(df['datetime'])
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]

        bot_logger.info(f"Loaded {len(df)} candles for {pair}")
        return df

    except Exception as e:
        bot_logger.error(f"Error fetching historical data: {str(e)}")
        return None


def run_backtest(pair, start_date, end_date, initial_balance,
                 confidence_threshold, min_agreement, timeframe_key='5m',
                 use_confluence=True, candles_limit=0,
                 spread_pips=None, commission_per_lot=7.0, slippage_pips=0.3):
    """Run a single-pair backtest and print results."""
    bot_logger.info("=" * 70)
    bot_logger.info(f"🔬 9-Model Scalping Backtest")
    bot_logger.info(f"  Pair: {pair}")
    bot_logger.info(f"  Primary TF: {timeframe_key}")
    bot_logger.info(f"  Confluence: {'1M enabled' if use_confluence else 'off'}")
    bot_logger.info(f"  Initial Balance: ${initial_balance:,.2f}")
    bot_logger.info(f"  Confidence Threshold: {confidence_threshold:.0%}")
    bot_logger.info(f"  Min Models Agreement: {min_agreement}/9")
    bot_logger.info(f"  Spread: {spread_pips if spread_pips is not None else 'per-pair config'} pips")
    bot_logger.info(f"  Commission: ${commission_per_lot:.1f}/lot round-trip")
    bot_logger.info(f"  Slippage: {slippage_pips:.1f} pips")
    bot_logger.info("=" * 70)

    # Load primary timeframe data
    historical_data = fetch_historical_data(pair, start_date, end_date, interval=timeframe_key)

    # Trim to --candles limit (keep tail so indicators warm up)
    if candles_limit and candles_limit > 0 and historical_data is not None and len(historical_data) > candles_limit:
        historical_data = historical_data.tail(candles_limit).reset_index(drop=True)
        bot_logger.info(f"Trimmed to last {candles_limit} candles for speed")

    if historical_data is None or len(historical_data) < 250:
        bot_logger.error(f"Insufficient {timeframe_key} data for backtest (need ≥ 250 candles)")
        return None

    # Load 1M data for confluence (only when primary is 5m)
    df_1m = None
    if use_confluence and timeframe_key == '5m':
        df_1m = fetch_historical_data(pair, start_date, end_date, interval='1m')
        if df_1m is not None and len(df_1m) < 500:
            bot_logger.warning(f"Insufficient 1M data for confluence ({len(df_1m)} candles), "
                               f"running without confluence")
            df_1m = None

    engine = BacktestEngine(
        initial_balance=initial_balance,
        slippage_pips=slippage_pips,
        commission_per_lot=commission_per_lot,
        spread_pips=spread_pips,
    )

    results = engine.run_backtest(
        historical_data,
        pair,
        confidence_threshold=confidence_threshold,
        min_agreement=min_agreement,
        timeframe_key=timeframe_key,
        df_1m=df_1m,
    )

    _print_results(results)
    return results


def _print_results(results):
    """Pretty-print backtest results."""
    print("\n" + "=" * 70)
    print("📊 9-MODEL SCALPING BACKTEST RESULTS")
    print("=" * 70)
    print(f"Pair: {results['pair']}")
    print(f"Total Trades: {results['total_trades']}")

    if results['total_trades'] > 0:
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Avg Pips/Trade: {results['avg_pips']:.1f}")
        print(f"Avg Win Pips: {results['avg_win_pips']:.1f}")
        print(f"Avg Loss Pips: {results['avg_loss_pips']:.1f}")
        print(f"Timeout Exits: {results['timeout_exits']}")
        print()
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Profit: ${results['total_profit']:,.2f}")
        print(f"Return: {results['return_percent']:.2f}%")

        # Cost breakdown
        if results.get('total_costs', 0) > 0:
            print()
            print(f"--- Cost Breakdown ---")
            print(f"Spread Cost: ${results.get('total_spread_cost', 0):,.2f}")
            print(f"Commission Cost: ${results.get('total_commission_cost', 0):,.2f}")
            print(f"Slippage Cost: ${results.get('total_slippage_cost', 0):,.2f}")
            print(f"Total Costs: ${results.get('total_costs', 0):,.2f}")
            gross = results['total_profit'] + results.get('total_costs', 0)
            print(f"Gross Profit (before costs): ${gross:,.2f}")
            print(f"Spread: {results.get('spread_pips', 0):.1f} pips | "
                  f"Commission: ${results.get('commission_per_lot', 0):.1f}/lot")
    else:
        print(f"Initial Balance: ${results.get('initial_balance', 50):,.2f}")
        print(f"Final Balance: ${results.get('final_balance', 50):,.2f}")

    print("=" * 70)

    # Assessment
    print("\n📈 Assessment:")
    if results['total_trades'] == 0:
        print("❌ No trades executed — lower confidence threshold or add more data")
    elif results['total_trades'] < 10:
        print("⚠️  Very few trades — results not statistically significant")
    elif results['win_rate'] < 40:
        print("❌ Win rate below 40% — strategy needs tuning")
    elif results['profit_factor'] >= 1.5 and results['win_rate'] >= 50:
        print("✅ Excellent — profitable strategy, ready for paper trading!")
    elif results['profit_factor'] >= 1.2 and results['win_rate'] >= 45:
        print("✅ Good — strategy is profitable, monitor on paper")
    elif results['profit_factor'] >= 1.0:
        print("⚠️  Break-even — review parameters before going live")
    else:
        print("❌ Negative expectancy — do not trade live until improved")

    print()


def main():
    """Main entry point"""
    args = parse_args()

    pair = args.pair or PAIRS[0]
    pairs = PAIRS if args.all else [pair]
    use_confluence = not args.no_confluence
    all_results = {}

    for pair in pairs:
        result = run_backtest(
            pair=pair,
            start_date=args.start,
            end_date=args.end,
            initial_balance=args.balance,
            confidence_threshold=args.confidence,
            min_agreement=args.agreement,
            timeframe_key=args.tf,
            use_confluence=use_confluence,
            candles_limit=args.candles,
            spread_pips=args.spread,
            commission_per_lot=args.commission,
            slippage_pips=args.slippage,
        )
        if result:
            all_results[pair] = result

    # Summary if multiple pairs
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("📊 COMBINED SUMMARY")
        print("=" * 70)
        total_profit = sum(r['total_profit'] for r in all_results.values())
        total_trades = sum(r['total_trades'] for r in all_results.values())
        total_wins = sum(r['winning_trades'] for r in all_results.values())
        total_costs = sum(r.get('total_costs', 0) for r in all_results.values())
        combined_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        print(f"Total Trades: {total_trades}")
        print(f"Combined Win Rate: {combined_wr:.1f}%")
        print(f"Combined Profit: ${total_profit:,.2f}")
        print(f"Combined Costs: ${total_costs:,.2f}")
        for pair, r in all_results.items():
            print(f"  {pair}: {r['return_percent']:+.2f}% "
                  f"({r['total_trades']} trades, WR {r['win_rate']:.1f}%, "
                  f"costs ${r.get('total_costs', 0):,.2f})")
        print("=" * 70)


if __name__ == '__main__':
    main()
