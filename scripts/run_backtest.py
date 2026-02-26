#!/usr/bin/env python
"""
Backtesting script
Usage: python scripts/run_backtest.py --pair EUR/USD --start 2022-01-01 --end 2024-12-31
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
from src.utils.logger import bot_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run backtest on historical forex data')
    parser.add_argument('--pair', default='EUR/USD', help='Currency pair (default: EUR/USD)')
    parser.add_argument('--start', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--confidence', type=float, default=0.75, help='Min confidence (default: 0.75)')
    parser.add_argument('--agreement', type=int, default=3, help='Min models agreement (default: 3)')
    
    return parser.parse_args()


def fetch_historical_data(pair, start_date, end_date, num_candles=5000):
    """
    Load historical data from local CSV files or MT5.
    Tries CSV first (faster), falls back to MT5 relay if needed.
    """
    # Try loading from CSV first
    csv_filename = f"data/{pair.replace('/', '_')}_1h.csv"
    
    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            df['datetime'] = pd.to_datetime(df['datetime'])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
            
            if len(df) > 0:
                bot_logger.info(f"✅ Loaded {len(df)} candles from {csv_filename}")
                return df
        except Exception as e:
            bot_logger.warning(f"Failed to load CSV: {e}")
    
    # Fallback to MT5 relay
    try:
        connector = MT5Connector()
        
        # Fetch data - MT5 can fetch up to 5000 candles per request
        df = connector.get_candles(pair, timeframe_minutes=60, num_candles=num_candles)
        
        if df is None:
            bot_logger.error(f"Failed to fetch data for {pair}")
            return None
        
        # Filter by date range
        df['datetime'] = pd.to_datetime(df['datetime'])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
        
        bot_logger.info(f"Loaded {len(df)} candles for {pair} from {start_date} to {end_date}")
        
        return df
    
    except Exception as e:
        bot_logger.error(f"Error fetching historical data: {str(e)}")
        return None


def run_backtest(pair, start_date, end_date, initial_balance, confidence_threshold, min_agreement):
    """Run backtest"""
    
    bot_logger.info("=" * 70)
    bot_logger.info(f"🔬 Starting Backtest")
    bot_logger.info(f"  Pair: {pair}")
    bot_logger.info(f"  Period: {start_date} to {end_date}")
    bot_logger.info(f"  Initial Balance: ${initial_balance:,.2f}")
    bot_logger.info(f"  Confidence Threshold: {confidence_threshold:.0%}")
    bot_logger.info(f"  Min Models Agreement: {min_agreement}/4")
    bot_logger.info("=" * 70)
    
    # Fetch data
    bot_logger.info("Fetching historical data...")
    historical_data = fetch_historical_data(pair, start_date, end_date)
    
    if historical_data is None or len(historical_data) < 100:
        bot_logger.error("Insufficient data for backtest")
        return None
    
    # Run backtest
    bot_logger.info("Running backtest engine...")
    engine = BacktestEngine(initial_balance=initial_balance)
    
    results = engine.run_backtest(
        historical_data,
        pair,
        confidence_threshold=confidence_threshold,
        min_agreement=min_agreement
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
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
        print(f"\nInitial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Profit: ${results['total_profit']:,.2f}")
        print(f"Return: {results['return_percent']:.2f}%")
    else:
        print(f"Initial Balance: ${results.get('initial_balance', 50):,.2f}")
        print(f"Final Balance: ${results.get('final_balance', 50):,.2f}")
    
    print("=" * 70)
    
    # Assessment
    print("\n📈 Assessment:")
    if results['total_trades'] == 0:
        print("❌ No trades executed - adjust confidence threshold or add more data")
    elif results['win_rate'] < 40:
        print("❌ Win rate too low - strategy needs improvement")
    elif results['win_rate'] >= 50 and results['sharpe_ratio'] >= 1.0:
        print("✅ Good results - ready for paper trading!")
    elif results['win_rate'] >= 55 and results['sharpe_ratio'] >= 1.5:
        print("✅ Excellent results - ready for live trading!")
    else:
        print("⚠️  Mixed results - review strategy parameters")
    
    return results


def main():
    """Main entry point"""
    args = parse_args()
    
    run_backtest(
        pair=args.pair,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        confidence_threshold=args.confidence,
        min_agreement=args.agreement
    )


if __name__ == '__main__':
    main()
