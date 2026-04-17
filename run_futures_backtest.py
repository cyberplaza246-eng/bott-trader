#!/usr/bin/env python3
"""
Simple futures backtest script
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backtest.backtest_engine import BacktestEngine

def main():
    print("🚀 Running Futures Backtest on MES")
    print("="*50)

    # Load data
    print("📊 Loading MES futures data...")

    try:
        df_5m = pd.read_csv('data/MES_5m.csv')
        df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
        df_5m = df_5m.set_index('datetime')

        df_1m = pd.read_csv('data/MES_1m.csv')
        df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
        df_1m = df_1m.set_index('datetime')

        print(f"✅ Loaded {len(df_5m)} 5M candles and {len(df_1m)} 1M candles")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Run backtest
    print("\n🤖 Running backtest...")
    try:
        engine = BacktestEngine(initial_balance=50000, slippage_pips=0.5)
        result = engine.run_backtest(df_5m, 'MES', timeframe_key='5m', df_1m=df_1m)

        if result:
            print("✅ Backtest completed successfully!")
            print("\n📊 RESULTS:")
            print("-" * 40)
            print(f"Total Trades: {result.get('total_trades', 0)}")
            print(f"Winning Trades: {result.get('winning_trades', 0)}")
            print(f"Losing Trades: {result.get('losing_trades', 0)}")
            print(f"Win Rate: {result.get('win_rate', 0)*100:.1f}%")
            print(f"Total P&L: ${result.get('total_pnl', 0):,.2f}")
            print(f"Profit Factor: {result.get('profit_factor', 0):.2f}x")
            print(f"Max Drawdown: ${result.get('max_drawdown', 0):,.2f}")
            print(f"Final Balance: ${result.get('final_balance', 0):,.2f}")

            # Additional metrics
            if result.get('total_trades', 0) > 0:
                avg_win = result.get('avg_win', 0)
                avg_loss = result.get('avg_loss', 0)
                print(f"Average Win: ${avg_win:,.2f}")
                print(f"Average Loss: ${avg_loss:,.2f}")
                if avg_loss != 0:
                    print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x")

        else:
            print("❌ Backtest returned no results")

    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()