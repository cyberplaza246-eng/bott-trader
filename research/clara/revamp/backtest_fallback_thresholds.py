#!/usr/bin/env python
"""
Test fallback confidence thresholds: 45% vs 50% vs 55%
Compares which threshold produces better results on historical data.

Usage: python backtest_fallback_thresholds.py
"""
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.backtest_engine import BacktestEngine

def run_backtest_with_fallback(df_5m, df_1m, fallback_threshold, symbol='MES'):
    """Run backtest with specific fallback threshold."""
    
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {symbol} | Fallback Threshold: {fallback_threshold*100:.0f}%")
    print(f"{'='*70}")
    
    engine = BacktestEngine(initial_balance=50000, slippage_pips=0.5)
    
    try:
        # Monkey-patch the threshold into engine
        engine.fallback_threshold = fallback_threshold
        
        result = engine.run_backtest(
            df_5m, 
            symbol, 
            timeframe_key='5m',
            df_1m=df_1m,
            confidence_threshold=fallback_threshold  # Pass the threshold
        )
        
        if result:
            print(f"\n✅ Backtest Complete")
            trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            pnl = result.get('total_pnl', 0)
            profit_factor = result.get('profit_factor', 0)
            drawdown = result.get('max_drawdown', 0)
            
            print(f"   Total Trades:      {trades}")
            print(f"   Winning Trades:    {result.get('winning_trades', 0)}")
            print(f"   Losing Trades:     {result.get('losing_trades', 0)}")
            print(f"   Win Rate:          {win_rate*100:.1f}%")
            print(f"   Total P&L:         ${pnl:,.2f}")
            print(f"   Final Balance:     ${result.get('final_balance', 0):,.2f}")
            print(f"   Max Drawdown:      ${drawdown:,.2f}")
            print(f"   Profit Factor:     {profit_factor:.2f}x")
            print(f"   Avg Win:           ${result.get('avg_win', 0):,.2f}")
            print(f"   Avg Loss:          ${result.get('avg_loss', 0):,.2f}")
            
            return {
                'threshold': fallback_threshold,
                'trades': trades,
                'win_rate': win_rate,
                'pnl': pnl,
                'profit_factor': profit_factor,
                'drawdown': drawdown,
                'final_balance': result.get('final_balance', 0)
            }
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
    print("  FALLBACK THRESHOLD COMPARISON")
    print("  Testing: 45% vs 50% vs 55%")
    print("="*70)
    
    # Load historical data
    print("\n📊 Loading historical data...")
    
    data_dir = 'data'
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
        print(f"   ⚠️  CSV files not found")
        return
    
    if len(df_5m) < 100 or len(df_1m) < 100:
        print(f"   ⚠️  Insufficient data (need 100+ candles)")
        return
    
    # Test thresholds
    thresholds = [0.45, 0.50, 0.55]
    results = {}
    
    for threshold in thresholds:
        result = run_backtest_with_fallback(df_5m, df_1m, threshold, symbol='MES')
        results[threshold] = result
    
    # Comparison summary
    print(f"\n\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Threshold':<12} {'Trades':<10} {'Win%':<8} {'P&L':<12} {'PF':<8} {'Drawdown':<12}")
    print("-" * 70)
    
    best_threshold = None
    best_pnl = float('-inf')
    
    for threshold in thresholds:
        result = results.get(threshold)
        if result:
            trades = result.get('trades', 0)
            win_rate = result.get('win_rate', 0) * 100
            pnl = result.get('pnl', 0)
            profit_factor = result.get('profit_factor', 0)
            drawdown = result.get('drawdown', 0)
            
            print(f"{threshold*100:.0f}%{'':<6} {trades:<10} {win_rate:<8.1f} ${pnl:>10,.0f}  {profit_factor:>6.2f}x  ${drawdown:>10,.0f}")
            
            if pnl > best_pnl:
                best_pnl = pnl
                best_threshold = threshold
    
    print("-" * 70)
    if best_threshold:
        print(f"\n✅ BEST THRESHOLD: {best_threshold*100:.0f}% (P&L: ${best_pnl:,.2f})")
        print(f"\n💡 Recommendation:")
        if best_threshold == 0.45:
            print("   45% is more aggressive - more trades, higher risk/reward")
        elif best_threshold == 0.50:
            print("   50% is balanced - good trade-off between frequency and quality")
        elif best_threshold == 0.55:
            print("   55% is conservative - fewer trades, higher win rate")
    
    print("\n📊 How to Use Results:")
    print("   • Profit Factor > 2.0x = Excellent")
    print("   • Win Rate > 55% = Strong in trending markets")
    print("   • Lower P&L but close PF might indicate better consistency")

if __name__ == '__main__':
    main()
