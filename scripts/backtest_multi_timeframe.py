#!/usr/bin/env python3
"""
Multi-Timeframe Scalping Backtest Engine
Tests 1-minute and 5-minute scalping strategy on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.multi_timeframe_scalper import MultiTimeframeScalpingAnalyzer, MultiTimeframeScalpingTrader
from src.utils.logger import TradeLogger


class MultiTimeframeScalpingBacktester:
    """Backtest multi-timeframe scalping strategy on historical data"""
    
    def __init__(self, pair_1m=None, pair_5m=None, start_balance=10000):
        """
        Initialize backtest engine
        
        Args:
            pair_1m: CSV file with 1M data (optional)
            pair_5m: CSV file with 5M data (optional)
            start_balance: Starting account balance
        """
        self.pair_1m = pair_1m or 'data/GBP_USD_1m.csv'
        self.pair_5m = pair_5m or 'data/GBP_USD_5m.csv'
        self.start_balance = start_balance
        self.current_balance = start_balance
        self.logger = TradeLogger()
        
        # Analysis engine
        self.analyzer = MultiTimeframeScalpingAnalyzer()
        self.trader = MultiTimeframeScalpingTrader(self.analyzer)
        
        # Results tracking
        self.trades = []
        self.signals = {'1m': [], '5m': [], 'confluence': []}
        self.stats = {}
        
    def load_data(self, pair='GBP/USD'):
        """Load historical OHLC data from CSV files"""
        print(f"Loading historical data for {pair}...")
        
        # Try to load 1M data
        try:
            df_1m = pd.read_csv(self.pair_1m)
            df_1m['time'] = pd.to_datetime(df_1m['time'])
            df_1m = df_1m.sort_values('time')
            print(f"✓ 1M data loaded: {len(df_1m)} candles")
        except Exception as e:
            print(f"✗ Failed to load 1M data: {e}")
            df_1m = None
        
        # Try to load 5M data
        try:
            df_5m = pd.read_csv(self.pair_5m)
            df_5m['time'] = pd.to_datetime(df_5m['time'])
            df_5m = df_5m.sort_values('time')
            print(f"✓ 5M data loaded: {len(df_5m)} candles")
        except Exception as e:
            print(f"✗ Failed to load 5M data: {e}")
            df_5m = None
        
        return df_1m, df_5m
    
    def resample_5m_from_1m(self, df_1m):
        """Create 5M candles from 1M data"""
        if df_1m is None:
            return None
        
        df = df_1m.copy()
        df.set_index('time', inplace=True)
        
        # Resample to 5-minute candles
        df_5m = pd.DataFrame({
            'open': df['open'].resample('5min').first(),
            'high': df['high'].resample('5min').max(),
            'low': df['low'].resample('5min').min(),
            'close': df['close'].resample('5min').last(),
            'volume': df['volume'].resample('5min').sum(),
        }).dropna()
        
        df_5m.reset_index(inplace=True)
        return df_5m
    
    def backtest_1m(self, df_1m, pair='GBP/USD'):
        """Backtest 1-minute signals"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING 1-MINUTE SIGNALS - {pair}")
        print(f"{'='*60}")
        
        if df_1m is None or len(df_1m) < 100:
            print("✗ Insufficient 1M data")
            return None
        
        trades_1m = []
        balance = self.start_balance
        
        # Process candles
        for i in range(100, len(df_1m)):
            candles = df_1m.iloc[max(0, i-100):i+1]
            
            # Analyze 1M signal
            signal = self.analyzer.get_signal_1m(candles, pair)
            
            if signal and signal.get('signal') in ['BUY', 'SELL']:
                self.signals['1m'].append({
                    'time': df_1m.iloc[i]['time'],
                    'signal': signal['signal'],
                    'confidence': signal.get('confidence', 0),
                })
                
                # Simulate trade
                entry_price = df_1m.iloc[i]['close']
                sl_pips = np.random.randint(5, 9) if pair == 'GBP/USD' else np.random.randint(4, 7)
                tp_pips = sl_pips * 1.2
                
                trade = {
                    'type': '1M',
                    'pair': pair,
                    'time': df_1m.iloc[i]['time'],
                    'signal': signal['signal'],
                    'entry': entry_price,
                    'sl': entry_price - (sl_pips/10000 if 'USD' not in pair else sl_pips*0.0001),
                    'tp': entry_price + (tp_pips/10000 if 'USD' not in pair else tp_pips*0.0001),
                    'confidence': signal.get('confidence', 0),
                    'status': 'pending',
                    'exit_price': None,
                    'profit': 0,
                }
                
                # Check if hit SL or TP in next few candles
                for j in range(i+1, min(i+10, len(df_1m))):
                    high = df_1m.iloc[j]['high']
                    low = df_1m.iloc[j]['low']
                    
                    if signal['signal'] == 'BUY':
                        if low <= trade['sl']:
                            trade['status'] = 'stopped_loss'
                            trade['exit_price'] = trade['sl']
                            trade['profit'] = -sl_pips * 0.0001
                            break
                        elif high >= trade['tp']:
                            trade['status'] = 'take_profit'
                            trade['exit_price'] = trade['tp']
                            trade['profit'] = tp_pips * 0.0001
                            break
                    else:
                        if high >= trade['sl']:
                            trade['status'] = 'stopped_loss'
                            trade['exit_price'] = trade['sl']
                            trade['profit'] = -sl_pips * 0.0001
                            break
                        elif low <= trade['tp']:
                            trade['status'] = 'take_profit'
                            trade['exit_price'] = trade['tp']
                            trade['profit'] = tp_pips * 0.0001
                            break
                
                if trade['status'] == 'pending':
                    trade['status'] = 'timeout'
                    trade['exit_price'] = df_1m.iloc[min(i+9, len(df_1m)-1)]['close']
                
                trades_1m.append(trade)
                balance += trade['profit'] * 100000  # Position size
        
        # Calculate 1M stats
        if trades_1m:
            wins = [t for t in trades_1m if t['profit'] > 0]
            losses = [t for t in trades_1m if t['profit'] < 0]
            
            print(f"\n1-Minute Statistics:")
            print(f"  Total Trades: {len(trades_1m)}")
            print(f"  Wins: {len(wins)} ({len(wins)/len(trades_1m)*100:.1f}%)")
            print(f"  Losses: {len(losses)} ({len(losses)/len(trades_1m)*100:.1f}%)")
            print(f"  Total Profit: ${sum(t['profit'] for t in trades_1m) * 100000:.2f}")
            print(f"  Final Balance: ${balance:.2f}")
        
        return trades_1m
    
    def backtest_5m(self, df_5m, pair='GBP/USD'):
        """Backtest 5-minute signals"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING 5-MINUTE SIGNALS - {pair}")
        print(f"{'='*60}")
        
        if df_5m is None or len(df_5m) < 100:
            print("✗ Insufficient 5M data")
            return None
        
        trades_5m = []
        balance = self.start_balance
        
        # Process candles
        for i in range(100, len(df_5m)):
            candles = df_5m.iloc[max(0, i-100):i+1]
            
            # Analyze 5M signal
            signal = self.analyzer.get_signal_5m(candles, pair)
            
            if signal and signal.get('signal') in ['BUY', 'SELL']:
                self.signals['5m'].append({
                    'time': df_5m.iloc[i]['time'],
                    'signal': signal['signal'],
                    'confidence': signal.get('confidence', 0),
                })
                
                # Simulate trade
                entry_price = df_5m.iloc[i]['close']
                sl_pips = np.random.randint(8, 13) if pair == 'GBP/USD' else np.random.randint(6, 11)
                tp_pips = sl_pips * 1.5
                
                trade = {
                    'type': '5M',
                    'pair': pair,
                    'time': df_5m.iloc[i]['time'],
                    'signal': signal['signal'],
                    'entry': entry_price,
                    'sl': entry_price - (sl_pips/10000 if 'USD' not in pair else sl_pips*0.0001),
                    'tp': entry_price + (tp_pips/10000 if 'USD' not in pair else tp_pips*0.0001),
                    'confidence': signal.get('confidence', 0),
                    'status': 'pending',
                    'exit_price': None,
                    'profit': 0,
                }
                
                # Check if hit SL or TP in next few candles (less strict for 5M)
                for j in range(i+1, min(i+20, len(df_5m))):
                    high = df_5m.iloc[j]['high']
                    low = df_5m.iloc[j]['low']
                    
                    if signal['signal'] == 'BUY':
                        if low <= trade['sl']:
                            trade['status'] = 'stopped_loss'
                            trade['exit_price'] = trade['sl']
                            trade['profit'] = -sl_pips * 0.0001
                            break
                        elif high >= trade['tp']:
                            trade['status'] = 'take_profit'
                            trade['exit_price'] = trade['tp']
                            trade['profit'] = tp_pips * 0.0001
                            break
                    else:
                        if high >= trade['sl']:
                            trade['status'] = 'stopped_loss'
                            trade['exit_price'] = trade['sl']
                            trade['profit'] = -sl_pips * 0.0001
                            break
                        elif low <= trade['tp']:
                            trade['status'] = 'take_profit'
                            trade['exit_price'] = trade['tp']
                            trade['profit'] = tp_pips * 0.0001
                            break
                
                if trade['status'] == 'pending':
                    trade['status'] = 'timeout'
                    trade['exit_price'] = df_5m.iloc[min(i+19, len(df_5m)-1)]['close']
                
                trades_5m.append(trade)
                balance += trade['profit'] * 100000  # Position size
        
        # Calculate 5M stats
        if trades_5m:
            wins = [t for t in trades_5m if t['profit'] > 0]
            losses = [t for t in trades_5m if t['profit'] < 0]
            
            print(f"\n5-Minute Statistics:")
            print(f"  Total Trades: {len(trades_5m)}")
            print(f"  Wins: {len(wins)} ({len(wins)/len(trades_5m)*100:.1f}%)")
            print(f"  Losses: {len(losses)} ({len(losses)/len(trades_5m)*100:.1f}%)")
            print(f"  Total Profit: ${sum(t['profit'] for t in trades_5m) * 100000:.2f}")
            print(f"  Final Balance: ${balance:.2f}")
        
        return trades_5m
    
    def backtest_confluence(self, trades_1m, trades_5m):
        """Analyze confluence between 1M and 5M signals"""
        if not trades_1m or not trades_5m:
            return None
        
        print(f"\n{'='*60}")
        print(f"CONFLUENCE ANALYSIS")
        print(f"{'='*60}")
        
        confluence_trades = []
        
        # Find trades within same timeframe window
        for t1 in trades_1m:
            for t5 in trades_5m:
                # Check if signals are within 5 minutes of each other
                time_diff = abs((t1['time'] - t5['time']).total_seconds())
                
                if time_diff < 300:  # Within 5 minutes
                    if t1['signal'] == t5['signal']:
                        confluence = {
                            'type': 'confluence',
                            '1m_signal': t1['signal'],
                            '5m_signal': t5['signal'],
                            'time': t1['time'],
                            'confidence_1m': t1.get('confidence', 0),
                            'confidence_5m': t5.get('confidence', 0),
                            'combined_confidence': (t1.get('confidence', 0) + t5.get('confidence', 0)) / 2,
                            'profit': (t1['profit'] + t5['profit']) / 2,  # Average profit
                        }
                        confluence_trades.append(confluence)
                        self.signals['confluence'].append(confluence)
        
        if confluence_trades:
            wins = [t for t in confluence_trades if t['profit'] > 0]
            print(f"\nConfluence Statistics:")
            print(f"  Total Confluence Signals: {len(confluence_trades)}")
            print(f"  Combined Wins: {len(wins)} ({len(wins)/len(confluence_trades)*100:.1f}%)")
            print(f"  Avg Confidence: {np.mean([t['combined_confidence'] for t in confluence_trades]):.2f}")
        else:
            print(f"\nNo confluence signals found")
        
        return confluence_trades
    
    def print_summary(self):
        """Print complete backtest summary"""
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"Start Balance: ${self.start_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total Return: {((self.current_balance - self.start_balance) / self.start_balance * 100):.2f}%")
        print(f"\nTotal 1M Signals: {len(self.signals['1m'])}")
        print(f"Total 5M Signals: {len(self.signals['5m'])}")
        print(f"Confluence Signals: {len(self.signals['confluence'])}")
        
        # Export results to file
        results = {
            'start_balance': self.start_balance,
            'final_balance': self.current_balance,
            'total_signals_1m': len(self.signals['1m']),
            'total_signals_5m': len(self.signals['5m']),
            'confluence_signals': len(self.signals['confluence']),
            'signals': self.signals,
        }
        
        with open('backtest_results_multi_tf.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to backtest_results_multi_tf.json")


def main():
    """Run multi-timeframe backtest"""
    print("Multi-Timeframe Scalping Backtest")
    print("=" * 60)
    
    # Initialize backtest engine
    bt = MultiTimeframeScalpingBacktester()
    
    # Load data for both pairs
    for pair in ['GBP/USD', 'EUR/USD']:
        print(f"\n{'='*60}")
        print(f"TESTING PAIR: {pair}")
        print(f"{'='*60}\n")
        
        # Load historical data
        df_1m, df_5m = bt.load_data(pair)
        
        # If 5M data missing, resample from 1M
        if df_5m is None and df_1m is not None:
            print("Resampling 5M data from 1M...")
            df_5m = bt.resample_5m_from_1m(df_1m)
        
        # Run backtests
        trades_1m = bt.backtest_1m(df_1m, pair) if df_1m is not None else None
        trades_5m = bt.backtest_5m(df_5m, pair) if df_5m is not None else None
        
        # Check confluence
        if trades_1m and trades_5m:
            bt.backtest_confluence(trades_1m, trades_5m)
    
    # Print final summary
    bt.print_summary()


if __name__ == '__main__':
    main()
