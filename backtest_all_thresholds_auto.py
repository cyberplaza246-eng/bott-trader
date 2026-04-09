#!/usr/bin/env python
"""
Automated Threshold Comparison: 45% vs 50% vs 55%
Tests all three confidence thresholds and generates comparison report.

Usage: python backtest_all_thresholds_auto.py
"""
import os
import sys
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.backtest_engine import BacktestEngine

def run_backtest_with_threshold(df_5m, df_1m, threshold_value, symbol='MES'):
    """Run backtest by temporarily modifying the threshold."""
    
    print(f"\n{'='*80}")
    print(f"  BACKTEST: {symbol} | Threshold Floor: {threshold_value*100:.0f}%")
    print(f"{'='*80}")
    
    try:
        # Create engine
        engine = BacktestEngine(initial_balance=50000, slippage_pips=0.5)
        
        result = engine.run_backtest(
            df_5m, 
            symbol, 
            timeframe_key='5m',
            df_1m=df_1m,
            confidence_threshold=threshold_value,
        )
        
        if result:
            print(f"\n✅ Backtest Complete")
            
            trades = result.get('total_trades', 0)
            wins = result.get('winning_trades', 0)
            losses = result.get('losing_trades', 0)
            win_rate = result.get('win_rate', 0)
            pnl = result.get('total_pnl', 0)
            profit_factor = result.get('profit_factor', 0)
            drawdown = result.get('max_drawdown', 0)
            final_balance = result.get('final_balance', 0)
            avg_win = result.get('avg_win', 0)
            avg_loss = result.get('avg_loss', 0)
            
            print(f"   Total Trades:      {trades}")
            print(f"   Winning Trades:    {wins}")
            print(f"   Losing Trades:     {losses}")
            print(f"   Win Rate:          {win_rate*100:.1f}%")
            print(f"   Total P&L:         ${pnl:,.2f}")
            print(f"   Final Balance:     ${final_balance:,.2f}")
            print(f"   Max Drawdown:      ${drawdown:,.2f}")
            print(f"   Profit Factor:     {profit_factor:.2f}x")
            print(f"   Avg Win/Loss:      ${avg_win:,.2f} / ${avg_loss:,.2f}")
            
            return {
                'threshold': threshold_value,
                'threshold_pct': threshold_value * 100,
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'pnl': pnl,
                'profit_factor': profit_factor,
                'drawdown': drawdown,
                'final_balance': final_balance,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
            }
        else:
            print("❌ Backtest failed - no results")
            return None
            
    except Exception as e:
        print(f"❌ Backtest error: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("  CONFIDENCE THRESHOLD AUTOMATED COMPARISON")
    print("  Testing: 45% vs 50% vs 55%")
    print("="*80)
    
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
        print(f"   ❌ CSV files not found at {csv_5m} or {csv_1m}")
        return
    
    if len(df_5m) < 100 or len(df_1m) < 100:
        print(f"   ⚠️  Insufficient data (have {len(df_5m)} 5M candles, need 100+)")
        return
    
    # Test thresholds
    thresholds = [0.45, 0.50, 0.55]
    results = {}
    
    for threshold in thresholds:
        result = run_backtest_with_threshold(df_5m, df_1m, threshold, symbol='MES')
        results[threshold] = result
    
    # Generate comprehensive comparison
    print(f"\n\n{'='*80}")
    print("  DETAILED COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Table 1: Performance metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"\n{'Threshold':<12} {'Trades':<8} {'W/L':<12} {'Win%':<8} {'P&L':<15} {'PF':<8} {'Drawdown':<12}")
    print("-" * 80)
    
    best_pnl = float('-inf')
    best_pnl_threshold = None
    best_wr = 0
    best_wr_threshold = None
    best_pf = 0
    best_pf_threshold = None
    
    for threshold in thresholds:
        result = results.get(threshold)
        if result:
            threshold_str = f"{result['threshold_pct']:.0f}%"
            trades = result['trades']
            wins = result['wins']
            losses = result['losses']
            wr_pct = result['win_rate'] * 100
            pnl = result['pnl']
            pf = result['profit_factor']
            dd = result['drawdown']
            
            print(f"{threshold_str:<12} {trades:<8} {wins}/{losses:<10} {wr_pct:<8.1f} ${pnl:>12,.0f}  {pf:>6.2f}x  ${dd:>10,.0f}")
            
            # Track bests
            if pnl > best_pnl:
                best_pnl = pnl
                best_pnl_threshold = threshold
            if wr_pct > best_wr:
                best_wr = wr_pct
                best_wr_threshold = threshold
            if pf > best_pf:
                best_pf = pf
                best_pf_threshold = threshold
    
    print("-" * 80)
    
    # Table 2: Risk metrics
    print(f"\n💰 RISK & EFFICIENCY METRICS:")
    print(f"\n{'Threshold':<12} {'Avg Win':<15} {'Avg Loss':<15} {'Win/Loss Ratio':<15} {'Final Balance':<15}")
    print("-" * 80)
    
    for threshold in thresholds:
        result = results.get(threshold)
        if result:
            threshold_str = f"{result['threshold_pct']:.0f}%"
            avg_w = result['avg_win']
            avg_l = result['avg_loss']
            ratio = abs(avg_w / avg_l) if avg_l != 0 else 0
            balance = result['final_balance']
            
            print(f"{threshold_str:<12} ${avg_w:>12,.0f}  ${avg_l:>12,.0f}  {ratio:>13.2f}x  ${balance:>12,.0f}")
    
    print("-" * 80)
    
    # Recommendations
    print(f"\n\n{'='*80}")
    print("  🎯 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n✅ BEST OVERALL P&L: {best_pnl_threshold*100:.0f}% (${best_pnl:,.2f})")
    if best_pnl_threshold == 0.45:
        print("   → Most aggressive | Trade more frequently | Higher variance")
    elif best_pnl_threshold == 0.50:
        print("   → Balanced approach | Good trade frequency | Moderate consistency")
    elif best_pnl_threshold == 0.55:
        print("   → Most conservative | Fewer but higher quality trades | Steady profits")
    
    print(f"\n✅ BEST WIN RATE: {best_wr_threshold*100:.0f}% ({best_wr:.1f}%)")
    print("   → More consistent win percentage")
    
    print(f"\n✅ BEST PROFIT FACTOR: {best_pf_threshold*100:.0f}% ({best_pf:.2f}x)")
    print("   → Better ratio of wins to losses")
    
    print(f"\n\n📋 THRESHOLD INTERPRETATION:")
    print(f"   45% = Entry at lower confidence → More trades, higher risk")
    print(f"   50% = Balanced threshold → Medium trades, medium risk")
    print(f"   55% = Entry at higher confidence → Fewer trades, lower risk")
    
    print(f"\n💡 TYPICAL TARGETS:")
    print(f"   • Profit Factor > 2.0x = Excellent")
    print(f"   • Profit Factor 1.5-2.0x = Good")
    print(f"   • Win Rate > 55% = Strong")
    print(f"   • Drawdown < 10% = Acceptable")
    
    # Save results to JSON
    output_file = f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Results saved to: {output_file}")

if __name__ == '__main__':
    main()
