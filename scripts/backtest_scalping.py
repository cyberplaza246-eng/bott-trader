#!/usr/bin/env python3
"""
Scalping Strategy Backtest — Validate performance on historical data

Tests the 5-minute scalping strategy on historical GBPUSD/EURUSD data.
Generates performance metrics and trade-by-trade analysis.

Usage:
    python scripts/backtest_scalping.py
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.utils.logger import bot_logger


class ScalpingBacktester:
    """Backtest scalping strategy on historical data"""
    
    def __init__(self, initial_balance=1000):
        """Initialize backtester.
        
        Args:
            initial_balance: Starting capital in USD
        """
        self.analyzer = ScalpingAnalyzer()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
        bot_logger.info("🧪 Scalping Backtest Engine Initialized")
    
    def load_data(self, filepath):
        """Load OHLCV data from CSV.
        
        Args:
            filepath: Path to CSV file (expected format: datetime,open,high,low,close,volume)
            
        Returns:
            DataFrame or None
        """
        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['datetime'] if 'datetime' in pd.read_csv(filepath, nrows=1).columns else False
            )
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                bot_logger.error(f"CSV missing required columns. Found: {df.columns.tolist()}")
                return None
            
            # Ensure datetime index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime').reset_index(drop=True)
            else:
                df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
            
            bot_logger.info(f"Loaded {len(df)} candles from {filepath}")
            return df
        
        except Exception as e:
            bot_logger.error(f"Error loading data: {e}")
            return None
    
    def backtest_pair(self, df, pair='EUR/USD', verbose=True):
        """Run backtest on single pair.
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair
            verbose: Print detailed output
            
        Returns:
            dict: Backtest results
        """
        if df is None or len(df) < 200:
            bot_logger.error(f"Insufficient data for {pair}: {len(df)} candles")
            return {}
        
        trades = []
        entry_signal = None
        entry_idx = None
        entry_price = None
        entry_signal_info = None
        
        if verbose:
            bot_logger.info(f"\n{'='*70}")
            bot_logger.info(f"Backtesting {pair} — {len(df)} candles")
            bot_logger.info(f"{'='*70}")
        
        # Process each candle
        for idx in range(200, len(df)):  # Start at 200 to have enough history
            current_df = df.iloc[:idx+1].reset_index(drop=True)
            
            # Generate signal
            signal = self.analyzer.get_signal(current_df, pair)
            
            # ── Entry Logic ──
            if entry_signal is None and signal['signal'] in ['BUY', 'SELL']:
                # Check cooldown (avoid overtrading)
                if trades and (idx - trades[-1]['exit_idx']) < 10:  # 10 candles = 50 minutes
                    continue
                
                entry_signal = signal['signal']
                entry_idx = idx
                entry_price = df['close'].iloc[idx]
                entry_signal_info = signal
            
            # ── Exit Logic ──
            if entry_signal is not None:
                current_price = df['close'].iloc[idx]
                sl_price = entry_signal_info['stop_loss']
                tp_price = entry_signal_info['take_profit']
                
                exit_reason = None
                exit_price = current_price
                profit_loss = 0
                
                if entry_signal == 'BUY':
                    # Check TP hit
                    if df['high'].iloc[idx] >= tp_price:
                        exit_reason = 'TP_HIT'
                        exit_price = tp_price
                        profit_loss = (tp_price - entry_price) * 100000  # Standard lot
                    # Check SL hit
                    elif df['low'].iloc[idx] <= sl_price:
                        exit_reason = 'SL_HIT'
                        exit_price = sl_price
                        profit_loss = (sl_price - entry_price) * 100000
                    # Check time-based exit (max 15 candles = 75 minutes)
                    elif (idx - entry_idx) >= 15:
                        exit_reason = 'TIME_EXIT'
                        exit_price = current_price
                        profit_loss = (current_price - entry_price) * 100000
                
                elif entry_signal == 'SELL':
                    # Check TP hit
                    if df['low'].iloc[idx] <= tp_price:
                        exit_reason = 'TP_HIT'
                        exit_price = tp_price
                        profit_loss = (entry_price - tp_price) * 100000
                    # Check SL hit
                    elif df['high'].iloc[idx] >= sl_price:
                        exit_reason = 'SL_HIT'
                        exit_price = sl_price
                        profit_loss = (entry_price - sl_price) * 100000
                    # Check time-based exit
                    elif (idx - entry_idx) >= 15:
                        exit_reason = 'TIME_EXIT'
                        exit_price = current_price
                        profit_loss = (entry_price - current_price) * 100000
                
                if exit_reason:
                    # Record trade
                    trade = {
                        'pair': pair,
                        'direction': entry_signal,
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_time': df['datetime'].iloc[entry_idx] if 'datetime' in df else f"Candle {entry_idx}",
                        'exit_time': df['datetime'].iloc[idx] if 'datetime' in df else f"Candle {idx}",
                        'hold_candles': idx - entry_idx,
                        'exit_reason': exit_reason,
                        'profit_loss': profit_loss,
                        'confidence': entry_signal_info['confidence'],
                        'setup_type': entry_signal_info.get('setup', 'unknown'),
                    }
                    
                    # Calculate pips
                    if pair in ['EUR/USD', 'GBP/USD']:
                        trade['pips'] = abs(exit_price - entry_price) * 10000
                    else:
                        trade['pips'] = abs(exit_price - entry_price) * 100
                    
                    trades.append(trade)
                    self.balance += profit_loss
                    
                    if verbose and idx % 50 == 0:  # Print updates
                        status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                        bot_logger.info(
                            f"{status}: {entry_signal} @ {entry_price:.5f} > "
                            f"{exit_price:.5f} | "
                            f"{trade['pips']:.1f}p | "
                            f"${profit_loss:.2f} | "
                            f"{idx}/{len(df)}"
                        )
                    
                    # Reset for next trade
                    entry_signal = None
                    entry_idx = None
                    entry_price = None
                    entry_signal_info = None
        
        # Calculate statistics
        stats = self._calculate_stats(trades)
        
        if verbose:
            self._print_stats(pair, stats, trades)
        
        return {
            'pair': pair,
            'trades': trades,
            'stats': stats,
            'datapoints': len(df),
        }
    
    def _calculate_stats(self, trades):
        """Calculate performance statistics.
        
        Args:
            trades: List of trade dicts
            
        Returns:
            dict: Performance stats
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit_per_trade': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
            }
        
        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(profits),
            'losing_trades': len(losses),
            'win_rate': len(profits) / len(trades) if trades else 0,
            'total_pnl': total_profit + total_loss,
            'avg_profit_per_trade': (total_profit + total_loss) / len(trades) if trades else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(losses) if losses else 0,
            'profit_factor': total_profit / abs(total_loss) if total_loss != 0 else 0,
            'avg_hold_candles': np.mean([t['hold_candles'] for t in trades]) if trades else 0,
            'avg_pips': np.mean([t['pips'] for t in trades]) if trades else 0,
        }
    
    def _print_stats(self, pair, stats, trades):
        """Print formatted statistics."""
        bot_logger.info(f"\n{'─'*70}")
        bot_logger.info(f"Results for {pair}")
        bot_logger.info(f"{'─'*70}")
        bot_logger.info(f"Total trades:        {stats['total_trades']}")
        bot_logger.info(f"Winning trades:      {stats['winning_trades']} ({stats['win_rate']*100:.1f}%)")
        bot_logger.info(f"Losing trades:       {stats['losing_trades']}")
        bot_logger.info(f"Total P&L:           ${stats['total_pnl']:.2f}")
        bot_logger.info(f"Avg P&L per trade:   ${stats['avg_profit_per_trade']:.2f}")
        bot_logger.info(f"Max profit:          ${stats['max_profit']:.2f}")
        bot_logger.info(f"Max loss:            ${stats['max_loss']:.2f}")
        bot_logger.info(f"Profit factor:       {stats['profit_factor']:.2f}")
        bot_logger.info(f"Avg hold time:       {stats['avg_hold_candles']:.1f} candles (≈{stats['avg_hold_candles']*5:.0f} min)")
        bot_logger.info(f"Avg pips per trade:  {stats['avg_pips']:.1f}")
        
        bot_logger.info(f"\nFinal Balance: ${self.balance:.2f} "
                       f"(Gain: ${self.balance - self.initial_balance:.2f}, "
                       f"{((self.balance - self.initial_balance)/self.initial_balance)*100:.1f}%)")
        
        # Sample trades
        if trades:
            bot_logger.info(f"\nFirst 5 trades:")
            for i, t in enumerate(trades[:5]):
                status = "✅" if t['profit_loss'] > 0 else "❌"
                bot_logger.info(
                    f"  {status} {t['direction']:4s} @ {t['entry_price']:.5f} > {t['exit_price']:.5f} | "
                    f"{t['pips']:6.1f}p | ${t['profit_loss']:7.2f} | {t['exit_reason']:10s}"
                )


def main():
    """Run backtest"""
    # Look for historical data files
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Try to find data files (use 5m for scalping, with 1m available)
    data_files = {
        'EUR/USD': os.path.join(data_dir, 'EUR_USD_5m.csv'),
        'GBP/USD': os.path.join(data_dir, 'GBP_USD_5m.csv'),
    }
    
    backtester = ScalpingBacktester(initial_balance=1000)
    
    results = {}
    for pair, filepath in data_files.items():
        if os.path.exists(filepath):
            df = backtester.load_data(filepath)
            if df is not None:
                result = backtester.backtest_pair(df, pair, verbose=True)
                if result:
                    results[pair] = result
        else:
            bot_logger.warning(f"Data file not found: {filepath}")
    
    # Summary
    if results:
        bot_logger.info(f"\n{'='*70}")
        bot_logger.info("BACKTEST SUMMARY")
        bot_logger.info(f"{'='*70}")
        
        total_trades = sum(r['stats']['total_trades'] for r in results.values())
        total_profit = sum(r['stats']['total_pnl'] for r in results.values())
        
        bot_logger.info(f"Total trades across pairs: {total_trades}")
        bot_logger.info(f"Total P&L: ${total_profit:.2f}")
        bot_logger.info(f"Final Balance: ${backtester.balance:.2f}")
    else:
        bot_logger.error("No valid data files found. Ensure CSV files are in data/ directory.")


if __name__ == '__main__':
    main()
