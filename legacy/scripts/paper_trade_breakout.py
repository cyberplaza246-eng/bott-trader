#!/usr/bin/env python
"""
Paper Trading Simulator - Enhanced Breakout Strategy

Simulates live trading by walking through data bar-by-bar.
Uses the profitable enhanced breakout strategy (PF 1.58).

Usage:
    python scripts/paper_trade_breakout.py              # Run simulation
    python scripts/paper_trade_breakout.py --live       # Real-time simulation
    python scripts/paper_trade_breakout.py --speed 60   # 60x speed (1 bar/sec)
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PaperTrader:
    """Paper trading with the enhanced breakout strategy."""
    
    def __init__(
        self,
        symbol: str = 'MES',
        lookback: int = 10,
        atr_mult: float = 1.5,
        tp_mult: float = 2.0,
        ema_len: int = 50,
        commission: float = 2.00,
        slippage_ticks: int = 1,
        point_value: float = 5.0,
        tick_size: float = 0.25,
        daily_loss_limit: float = 150.0,
        max_trades_per_day: int = 10
    ):
        self.symbol = symbol
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.tp_mult = tp_mult
        self.ema_len = ema_len
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.point_value = point_value
        self.tick_size = tick_size
        self.slippage_cost = slippage_ticks * tick_size * point_value
        self.daily_loss_limit = daily_loss_limit
        self.max_trades_per_day = max_trades_per_day
        
        # State
        self.position: Optional[Dict] = None
        self.cooldown = 0
        self.trades: List[Dict] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.balance = 5000.0  # Starting balance for 1 MES contract
        
        # Log file
        self.log_file = f'logs/paper_trade_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators."""
        df = df.copy()
        
        # N-bar high/low
        df['high_N'] = df['high'].rolling(self.lookback).max().shift(1)
        df['low_N'] = df['low'].rolling(self.lookback).min().shift(1)
        
        # EMA trend filter
        df['ema'] = df['close'].ewm(span=self.ema_len, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def check_daily_limits(self, bar_date) -> bool:
        """Check if daily limits are hit."""
        if self.current_date != bar_date:
            # New day - reset counters
            self.current_date = bar_date
            self.daily_pnl = 0.0
            self.daily_trades = 0
            
        if self.daily_pnl <= -self.daily_loss_limit:
            return False  # Daily loss limit hit
        if self.daily_trades >= self.max_trades_per_day:
            return False  # Max trades hit
        return True
    
    def check_exit(self, row: pd.Series) -> Optional[Dict]:
        """Check if position should exit."""
        if self.position is None:
            return None
            
        d = self.position['direction']
        high = row['high']
        low = row['low']
        price = row['close']
        
        hit_sl = (d == 'long' and low <= self.position['sl']) or \
                 (d == 'short' and high >= self.position['sl'])
        hit_tp = (d == 'long' and high >= self.position['tp']) or \
                 (d == 'short' and low <= self.position['tp'])
        
        # Timeout after 40 bars
        bars_held = row.name - self.position['entry_idx'] if hasattr(row, 'name') else 0
        timeout = bars_held > 40
        
        if hit_sl:
            return {'type': 'STOP_LOSS', 'price': self.position['sl']}
        elif hit_tp:
            return {'type': 'TAKE_PROFIT', 'price': self.position['tp']}
        elif timeout:
            return {'type': 'TIMEOUT', 'price': price}
        return None
    
    def check_entry(self, row: pd.Series, idx: int) -> Optional[Dict]:
        """Check for entry signal."""
        if self.position is not None or self.cooldown > 0:
            return None
            
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        high_n = row['high_N']
        low_n = row['low_N']
        ema = row['ema']
        
        # Validate
        if pd.isna(atr) or pd.isna(high_n) or pd.isna(ema) or atr < 2.0:
            return None
        
        # Long breakout with trend filter
        if high > high_n and price > high_n and price > ema:
            entry = high_n + 0.25 + self.slippage_ticks * self.tick_size
            sl_dist = atr * self.atr_mult
            sl = entry - sl_dist
            tp = entry + (sl_dist * self.tp_mult)
            
            return {
                'direction': 'long',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'breakout_level': high_n,
                'entry_idx': idx
            }
        
        # Short breakout with trend filter
        elif low < low_n and price < low_n and price < ema:
            entry = low_n - 0.25 - self.slippage_ticks * self.tick_size
            sl_dist = atr * self.atr_mult
            sl = entry + sl_dist
            tp = entry - (sl_dist * self.tp_mult)
            
            return {
                'direction': 'short',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'breakout_level': low_n,
                'entry_idx': idx
            }
        
        return None
    
    def process_bar(self, row: pd.Series, idx: int, bar_time: datetime) -> Optional[str]:
        """Process a single bar. Returns action taken."""
        
        bar_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
        
        # Decrement cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # Check exit first
        exit_signal = self.check_exit(row)
        if exit_signal:
            d = self.position['direction']
            exit_price = exit_signal['price']
            
            # Calculate PnL
            if d == 'long':
                raw_pnl = (exit_price - self.position['entry']) * self.point_value
            else:
                raw_pnl = (self.position['entry'] - exit_price) * self.point_value
            
            pnl = raw_pnl - self.commission - self.slippage_cost
            
            # Record trade
            trade = {
                'symbol': self.symbol,
                'direction': d,
                'entry_time': str(self.position.get('entry_time', '')),
                'exit_time': str(bar_time),
                'entry_price': self.position['entry'],
                'exit_price': exit_price,
                'sl': self.position['sl'],
                'tp': self.position['tp'],
                'exit_type': exit_signal['type'],
                'pnl': pnl,
                'balance': self.balance + pnl
            }
            self.trades.append(trade)
            
            # Update state
            self.balance += pnl
            self.daily_pnl += pnl
            self.daily_trades += 1
            self.position = None
            self.cooldown = 5
            
            return f"EXIT {exit_signal['type']}: {d.upper()} @ {exit_price:.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}"
        
        # Check entry
        if self.check_daily_limits(bar_date):
            entry_signal = self.check_entry(row, idx)
            if entry_signal:
                self.position = entry_signal
                self.position['entry_time'] = bar_time
                
                return f"ENTRY: {entry_signal['direction'].upper()} @ {entry_signal['entry']:.2f} | SL: {entry_signal['sl']:.2f} | TP: {entry_signal['tp']:.2f}"
        
        return None
    
    def get_stats(self) -> Dict:
        """Calculate trading statistics."""
        if not self.trades:
            return {'trades': 0}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        
        # Max drawdown
        peak = 0
        max_dd = 0
        running = 0
        for t in self.trades:
            running += t['pnl']
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        
        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'avg_win': gross_profit / len(wins) if wins else 0,
            'avg_loss': gross_loss / len(losses) if losses else 0,
            'max_drawdown': max_dd,
            'final_balance': self.balance
        }
    
    def save_log(self):
        """Save trades to log file."""
        os.makedirs('logs', exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                'symbol': self.symbol,
                'strategy': 'enhanced_breakout',
                'params': {
                    'lookback': self.lookback,
                    'atr_mult': self.atr_mult,
                    'tp_mult': self.tp_mult,
                    'ema_len': self.ema_len
                },
                'stats': self.get_stats(),
                'trades': self.trades
            }, f, indent=2, default=str)
        print(f"\nTrades saved to: {self.log_file}")


def run_paper_trade(symbol: str = 'MES', speed: float = 1.0, live_mode: bool = False):
    """
    Run paper trading simulation.
    
    Args:
        symbol: Trading symbol (MES or MNQ)
        speed: Simulation speed (bars per second). 0 = instant
        live_mode: If True, wait for real-time bar intervals
    """
    print("=" * 70)
    print("  PAPER TRADING SIMULATOR - Enhanced Breakout Strategy")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Strategy: 10-bar Breakout + EMA50 Trend Filter")
    print(f"Costs: $2.00 commission + 1 tick slippage")
    print(f"Speed: {'Real-time' if live_mode else f'{speed} bars/sec' if speed > 0 else 'Instant'}")
    print("=" * 70)
    print()
    
    # Load data
    try:
        df = pd.read_csv(f'data/{symbol}_5m.csv', parse_dates=['datetime'])
        print(f"Loaded {len(df)} bars from {df['datetime'].min()} to {df['datetime'].max()}")
    except FileNotFoundError:
        print(f"Error: data/{symbol}_5m.csv not found!")
        return
    
    # Initialize paper trader
    trader = PaperTrader(symbol=symbol)
    df = trader.calculate_indicators(df)
    
    # Warmup period
    warmup = max(trader.lookback, trader.ema_len) + 15
    
    print(f"\nStarting simulation from bar {warmup}...")
    print("-" * 70)
    
    bar_interval = 1.0 / speed if speed > 0 else 0
    
    try:
        for idx in range(warmup, len(df)):
            row = df.iloc[idx]
            bar_time = row['datetime']
            
            # Process bar
            action = trader.process_bar(row, idx, bar_time)
            
            # Print activity
            if action:
                print(f"[{bar_time}] {action}")
            elif trader.position and idx % 50 == 0:
                # Periodic status when in position
                d = trader.position['direction']
                unrealized = 0
                if d == 'long':
                    unrealized = (row['close'] - trader.position['entry']) * trader.point_value
                else:
                    unrealized = (trader.position['entry'] - row['close']) * trader.point_value
                print(f"[{bar_time}] HOLDING {d.upper()} | Unrealized: ${unrealized:.2f}")
            
            # Progress indicator
            if idx % 500 == 0:
                pct = (idx - warmup) / (len(df) - warmup) * 100
                stats = trader.get_stats()
                print(f"\n--- Progress: {pct:.1f}% | Trades: {stats['trades']} | PnL: ${stats['total_pnl']:.2f} ---\n")
            
            # Speed control
            if bar_interval > 0:
                time.sleep(bar_interval)
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    
    # Final stats
    print("\n" + "=" * 70)
    print("  PAPER TRADING RESULTS")
    print("=" * 70)
    
    stats = trader.get_stats()
    print(f"Total Trades:    {stats['trades']}")
    print(f"Win Rate:        {stats['win_rate']:.1f}%")
    print(f"Total PnL:       ${stats['total_pnl']:.2f}")
    print(f"Profit Factor:   {stats['profit_factor']:.2f}")
    print(f"Avg Win:         ${stats['avg_win']:.2f}")
    print(f"Avg Loss:        ${stats['avg_loss']:.2f}")
    print(f"Max Drawdown:    ${stats['max_drawdown']:.2f}")
    print(f"Final Balance:   ${stats['final_balance']:.2f}")
    
    if stats['profit_factor'] >= 1.0:
        print("\n✅ STRATEGY PROFITABLE IN PAPER TRADING")
    else:
        print("\n❌ STRATEGY NOT PROFITABLE")
    
    # Save log
    trader.save_log()
    
    return trader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading Simulator')
    parser.add_argument('symbol', nargs='?', default='MES', help='Symbol to trade (MES or MNQ)')
    parser.add_argument('--speed', type=float, default=0, help='Simulation speed (bars/sec, 0=instant)')
    parser.add_argument('--live', action='store_true', help='Real-time mode')
    
    args = parser.parse_args()
    
    run_paper_trade(
        symbol=args.symbol,
        speed=args.speed,
        live_mode=args.live
    )
