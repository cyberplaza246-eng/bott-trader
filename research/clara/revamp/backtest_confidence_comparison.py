#!/usr/bin/env python
"""
Backtest comparison: Run the same backtest with 3 different confidence thresholds.
Compares: 40%, 45%, 50%

Usage: python backtest_confidence_comparison.py
"""
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.backtest_engine import BacktestEngine
from src.data.historical_downloader import HistoricalDownloader

def run_backtest_with_threshold(df_5m, df_1m, threshold, symbol='MES'):
    """Run backtest with a specific confidence threshold."""
    os.environ['ENSEMBLE_CONFIDENCE_THRESHOLD'] = str(threshold)
    
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {symbol} | Confidence Threshold: {threshold*100:.0f}%")
    print(f"{'='*70}")
    
    engine = BacktestEngine(initial_balance=50000, slippage_pips=0.5)
    
    try:
        result = engine.run_backtest(
            df_5m, 
            symbol, 
            timeframe_key='5m',
            df_1m=df_1m
        )
        
        if result:
            print(f"\n✅ Backtest Complete")
            print(f"   Total Trades:      {result.get('total_trades', 0)}")
            print(f"   Winning Trades:    {result.get('winning_trades', 0)}")
            print(f"   Losing Trades:     {result.get('losing_trades', 0)}")
            print(f"   Win Rate:          {result.get('win_rate', 0)*100:.1f}%")
            print(f"   Total P&L:         ${result.get('total_pnl', 0):,.2f}")
            print(f"   Final Balance:     ${result.get('final_balance', 0):,.2f}")
            print(f"   Max Drawdown:      ${result.get('max_drawdown', 0):,.2f}")
            print(f"   Profit Factor:     {result.get('profit_factor', 0):.2f}x")
            print(f"   Avg Win:           ${result.get('avg_win', 0):,.2f}")
            print(f"   Avg Loss:          ${result.get('avg_loss', 0):,.2f}")
            return result
        else:
            print("❌ Backtest failed - no results")
            return None
    except Exception as e:
        print(f"❌ Backtest error: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("  CONFIDENCE THRESHOLD COMPARISON BACKTEST")
    print("="*70)
    
    # Load historical data
    print("\n📊 Loading historical data...")
    
    data_dir = 'data'
    
    # Try to load from existing CSVs
    csv_5m = os.path.join(data_dir, 'MES_5m.csv')
    csv_1m = os.path.join(data_dir, 'MES_1m.csv')
    
    if os.path.exists(csv_5m) and os.path.exists(csv_1m):
        print(f"   Loading {csv_5m}")
        df_5m = pd.read_csv(csv_5m)
        df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
        
        print(f"   Loading {csv_1m}")
        df_1m = pd.read_csv(csv_1m)
        df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
        
        print(f"   ✅ 5M: {len(df_5m)} candles | 1M: {len(df_1m)} candles")
    else:
        print(f"   ⚠️  CSV files not found, downloading...")
        dl = HistoricalDownloader()
        
        # Download last 30 days of 5-min data
        df_5m = dl.download('MES', 5, lookback_days=30)
        df_1m = dl.download('MES', 1, lookback_days=30)
        
        if df_5m is None or df_1m is None:
            print("   ❌ Failed to download data")
            return
        
        print(f"   ✅ 5M: {len(df_5m)} candles | 1M: {len(df_1m)} candles")
    
    # Ensure minimum data
    if len(df_5m) < 100 or len(df_1m) < 100:
        print(f"   ⚠️  Insufficient data (need 100+ candles)")
        return
    
    # Run backtests with different thresholds
    thresholds = [0.40, 0.45, 0.50]
    results = {}
    
    for threshold in thresholds:
        result = run_backtest_with_threshold(df_5m, df_1m, threshold, symbol='MES')
        results[threshold] = result
    
    # Comparison summary
    print(f"\n\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Threshold':<12} {'Trades':<10} {'Win%':<8} {'P&L':<15} {'Profit Factor':<15}")
    print("-" * 70)
    
    best_threshold = None
    best_pnl = float('-inf')
    
    for threshold in thresholds:
        result = results.get(threshold)
        if result:
            trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0) * 100
            pnl = result.get('total_pnl', 0)
            profit_factor = result.get('profit_factor', 0)
            
            print(f"{threshold*100:.0f}%{'':<6} {trades:<10} {win_rate:<8.1f} ${pnl:>12,.2f}  {profit_factor:>12.2f}x")
            
            if pnl > best_pnl:
                best_pnl = pnl
                best_threshold = threshold
    
    print("-" * 70)
    if best_threshold:
        print(f"\n✅ BEST THRESHOLD: {best_threshold*100:.0f}% (P&L: ${best_pnl:,.2f})")
    
    print("\n💡 INTERPRETATION:")
    print("  • More trades ≠ Better - look at Win Rate & Profit Factor")
    print("  • Profit Factor > 2.0x is excellent")
    print("  • Profit Factor 1.5-2.0x is good")
    print("  • Win Rate > 55% is strong in trending markets")

if __name__ == '__main__':
    main()
