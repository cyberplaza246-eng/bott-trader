#!/usr/bin/env python
"""
LIVE TRADING - Enhanced Breakout Strategy

This script runs the profitable Enhanced Breakout strategy LIVE with real money.
Strategy: 10-bar breakout + EMA50 trend filter (PF 1.50 in paper trading)

RISK WARNING: This trades REAL MONEY. Use at your own risk.
Start with 1 micro contract (MES = $1.25/tick, MNQ = $0.50/tick).

Requirements:
1. MT5 relay running on Windows: python relay_server.py
2. Proper API token set
3. Funded trading account

Usage:
    python start_live_breakout.py              # MES default
    python start_live_breakout.py --symbol MNQ # Trade MNQ
    python start_live_breakout.py --paper      # Paper mode (no real orders)
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configuration
RELAY_URL = os.getenv('MT5_RELAY_URL', 'http://127.0.0.1:5555')
RELAY_TOKEN = os.getenv('MT5_RELAY_TOKEN', 'change-me-to-a-secret')
AUTH_HEADERS = {"Authorization": f"Bearer {RELAY_TOKEN}"}

# Strategy parameters (validated profitable)
PARAMS = {
    'lookback': 10,
    'atr_mult': 1.5,
    'tp_mult': 2.0,
    'ema_len': 50
}

# Risk settings
RISK_SETTINGS = {
    'contracts': 1,              # Start with 1 micro
    'daily_loss_limit': 150.0,   # Max daily loss $150
    'max_trades_per_day': 8,     # Max trades per day
    'cooldown_bars': 5           # Bars between trades
}


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    volume: float
    sl: float
    tp: float
    entry_time: datetime


class LiveBreakoutTrader:
    """Live trading with Enhanced Breakout Strategy."""
    
    def __init__(self, symbol: str = 'MES', paper_mode: bool = False):
        self.symbol = symbol
        self.paper_mode = paper_mode
        
        # Symbol specs
        if symbol == 'MES':
            self.point_value = 5.0
            self.tick_size = 0.25
        elif symbol == 'MNQ':
            self.point_value = 2.0
            self.tick_size = 0.25
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        # Strategy params
        self.lookback = PARAMS['lookback']
        self.atr_mult = PARAMS['atr_mult']
        self.tp_mult = PARAMS['tp_mult']
        self.ema_len = PARAMS['ema_len']
        
        # Risk management
        self.contracts = RISK_SETTINGS['contracts']
        self.daily_loss_limit = RISK_SETTINGS['daily_loss_limit']
        self.max_trades_per_day = RISK_SETTINGS['max_trades_per_day']
        self.cooldown_bars = RISK_SETTINGS['cooldown_bars']
        
        # State
        self.position: Optional[Position] = None
        self.cooldown = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.trades: List[Dict] = []
        
        # Paper mode state
        self.paper_balance = 5000.0
        
        # Log file
        mode = "paper" if paper_mode else "live"
        self.log_file = f'logs/{mode}_breakout_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def connect(self) -> bool:
        """Verify relay connection."""
        try:
            r = requests.get(f"{RELAY_URL}/ping", timeout=5)
            data = r.json()
            if not data.get('mt5_connected'):
                print(f"❌ MT5 not connected to relay")
                return False
            
            if not self.paper_mode:
                acct = requests.get(f"{RELAY_URL}/account", headers=AUTH_HEADERS, timeout=5).json()
                print(f"✅ MT5 Connected | Account: {acct.get('login')} | Balance: ${acct.get('balance'):,.2f}")
            else:
                print(f"✅ Relay running | PAPER MODE - No real trades will be placed")
            return True
        except Exception as e:
            print(f"❌ Relay error: {e}")
            return False
    
    def get_candles(self, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from MT5."""
        try:
            r = requests.get(
                f"{RELAY_URL}/candles",
                params={'symbol': self.symbol, 'timeframe': 'M5', 'count': count},
                headers=AUTH_HEADERS,
                timeout=10
            )
            data = r.json()
            if 'candles' not in data:
                print(f"❌ No candles returned: {data}")
                return None
            
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
            return df
        except Exception as e:
            print(f"❌ Error getting candles: {e}")
            return None
    
    def get_price(self) -> Optional[Dict]:
        """Get current bid/ask."""
        try:
            r = requests.get(
                f"{RELAY_URL}/price",
                params={'symbol': self.symbol},
                headers=AUTH_HEADERS,
                timeout=5
            )
            return r.json()
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        try:
            r = requests.get(f"{RELAY_URL}/positions", headers=AUTH_HEADERS, timeout=5)
            return r.json().get('positions', [])
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []
    
    def place_order(self, direction: str, entry: float, sl: float, tp: float) -> Optional[int]:
        """Place market order with SL/TP."""
        if self.paper_mode:
            ticket = int(time.time() * 1000) % 1000000
            print(f"📝 [PAPER] {direction.upper()} {self.symbol} @ {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            return ticket
        
        try:
            order_type = 'buy' if direction == 'long' else 'sell'
            r = requests.post(
                f"{RELAY_URL}/order",
                headers=AUTH_HEADERS,
                json={
                    'symbol': self.symbol,
                    'type': order_type,
                    'volume': float(self.contracts),
                    'price': entry,
                    'sl': sl,
                    'tp': tp,
                    'comment': f'breakout_{direction}'
                },
                timeout=10
            )
            data = r.json()
            if data.get('success'):
                print(f"✅ ORDER FILLED: {direction.upper()} {self.symbol} @ {data.get('price', entry):.2f}")
                return data.get('ticket', 0)
            else:
                print(f"❌ Order failed: {data}")
                return None
        except Exception as e:
            print(f"❌ Order error: {e}")
            return None
    
    def close_position(self, ticket: int, price: float) -> bool:
        """Close position by ticket."""
        if self.paper_mode:
            return True
        
        try:
            r = requests.post(
                f"{RELAY_URL}/close",
                headers=AUTH_HEADERS,
                json={'ticket': ticket},
                timeout=10
            )
            data = r.json()
            return data.get('success', False)
        except Exception as e:
            print(f"❌ Close error: {e}")
            return False
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy indicators."""
        df = df.copy()
        df['high_N'] = df['high'].rolling(self.lookback).max().shift(1)
        df['low_N'] = df['low'].rolling(self.lookback).min().shift(1)
        df['ema'] = df['close'].ewm(span=self.ema_len, adjust=False).mean()
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = 0
            print(f"\n📅 New trading day: {today}")
        
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        if self.daily_trades >= self.max_trades_per_day:
            return False
        return True
    
    def check_entry_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for entry signal on latest bar."""
        if self.position or self.cooldown > 0:
            return None
        
        row = df.iloc[-1]
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        high_n = row['high_N']
        low_n = row['low_N']
        ema = row['ema']
        
        if pd.isna(atr) or pd.isna(high_n) or pd.isna(ema) or atr < 2.0:
            return None
        
        # Long breakout
        if high > high_n and price > high_n and price > ema:
            entry = high_n + self.tick_size
            sl_dist = atr * self.atr_mult
            sl = entry - sl_dist
            tp = entry + (sl_dist * self.tp_mult)
            return {'direction': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr}
        
        # Short breakout
        elif low < low_n and price < low_n and price < ema:
            entry = low_n - self.tick_size
            sl_dist = atr * self.atr_mult
            sl = entry + sl_dist
            tp = entry - (sl_dist * self.tp_mult)
            return {'direction': 'short', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr}
        
        return None
    
    def check_exit(self, df: pd.DataFrame) -> Optional[str]:
        """Check if position should exit (for paper mode timeout only)."""
        # In live mode, MT5 handles SL/TP automatically
        # This is only for paper mode timeout
        if not self.paper_mode or not self.position:
            return None
        
        row = df.iloc[-1]
        d = self.position.direction
        
        # Check SL/TP hit
        if d == 'long':
            if row['low'] <= self.position.sl:
                return 'STOP_LOSS'
            if row['high'] >= self.position.tp:
                return 'TAKE_PROFIT'
        else:
            if row['high'] >= self.position.sl:
                return 'STOP_LOSS'
            if row['low'] <= self.position.tp:
                return 'TAKE_PROFIT'
        
        return None
    
    def process_exit(self, exit_type: str, exit_price: float):
        """Process position exit."""
        if not self.position:
            return
        
        d = self.position.direction
        if d == 'long':
            raw_pnl = (exit_price - self.position.entry_price) * self.point_value * self.contracts
        else:
            raw_pnl = (self.position.entry_price - exit_price) * self.point_value * self.contracts
        
        # Subtract costs
        commission = 2.00
        slippage = self.tick_size * self.point_value
        pnl = raw_pnl - commission - slippage
        
        # Record trade
        trade = {
            'symbol': self.symbol,
            'direction': d,
            'entry_time': str(self.position.entry_time),
            'exit_time': str(datetime.now()),
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl
        }
        self.trades.append(trade)
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if self.paper_mode:
            self.paper_balance += pnl
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} EXIT {exit_type}: {d.upper()} @ {exit_price:.2f} | PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}")
        
        self.position = None
        self.cooldown = self.cooldown_bars
    
    def sync_position(self):
        """Sync with MT5 positions (live mode only)."""
        if self.paper_mode:
            return
        
        positions = self.get_positions()
        my_positions = [p for p in positions if p.get('symbol') == self.symbol]
        
        if self.position and not my_positions:
            # Position was closed (hit SL/TP)
            # Get last price as exit price estimate
            price = self.get_price()
            if price:
                exit_price = price.get('bid', self.position.entry_price)
                self.process_exit('SL/TP', exit_price)
        
        elif my_positions and not self.position:
            # External position detected
            pos = my_positions[0]
            self.position = Position(
                ticket=pos['ticket'],
                symbol=pos['symbol'],
                direction='long' if pos['type'] == 0 else 'short',
                entry_price=pos['price_open'],
                volume=pos['volume'],
                sl=pos.get('sl', 0),
                tp=pos.get('tp', 0),
                entry_time=datetime.now()
            )
            print(f"📋 Synced existing position: {self.position.direction} @ {self.position.entry_price}")
    
    def save_log(self):
        """Save trade log."""
        os.makedirs('logs', exist_ok=True)
        
        stats = {'trades': 0}
        if self.trades:
            wins = [t for t in self.trades if t['pnl'] > 0]
            losses = [t for t in self.trades if t['pnl'] <= 0]
            gross_profit = sum(t['pnl'] for t in wins) if wins else 0
            gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
            
            stats = {
                'trades': len(self.trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(self.trades) * 100,
                'total_pnl': sum(t['pnl'] for t in self.trades),
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0
            }
        
        with open(self.log_file, 'w') as f:
            json.dump({
                'symbol': self.symbol,
                'paper_mode': self.paper_mode,
                'stats': stats,
                'trades': self.trades
            }, f, indent=2, default=str)
        
        print(f"📁 Log saved: {self.log_file}")
    
    def run(self):
        """Main trading loop."""
        print("\n" + "=" * 70)
        print("  LIVE BREAKOUT TRADING")
        print("=" * 70)
        print(f"Symbol:     {self.symbol}")
        print(f"Contracts:  {self.contracts}")
        print(f"Mode:       {'PAPER' if self.paper_mode else '⚠️  LIVE - REAL MONEY'}")
        print(f"Strategy:   10-bar Breakout + EMA50 Filter")
        print(f"Risk:       Max ${self.daily_loss_limit} daily loss, {self.max_trades_per_day} trades/day")
        print("=" * 70)
        
        if not self.paper_mode:
            print("\n⚠️  WARNING: LIVE MODE - Real money at risk!")
            print("Press Ctrl+C within 10 seconds to cancel...")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nCancelled.")
                return
        
        print("\n🚀 Starting trading loop...\n")
        
        last_bar_time = None
        
        try:
            while True:
                # Check daily limits
                if not self.check_daily_limits():
                    print(f"⏸️  Daily limits reached - waiting for next day...")
                    time.sleep(300)
                    continue
                
                # Sync position with broker
                self.sync_position()
                
                # Get candles
                df = self.get_candles(count=100)
                if df is None or len(df) < self.lookback + self.ema_len + 15:
                    print("⚠️  Insufficient data, retrying...")
                    time.sleep(30)
                    continue
                
                df = self.calculate_indicators(df)
                current_bar = df.iloc[-1]['datetime']
                
                # Only process on new bar
                if last_bar_time == current_bar:
                    time.sleep(5)
                    continue
                
                last_bar_time = current_bar
                
                # Decrement cooldown
                if self.cooldown > 0:
                    self.cooldown -= 1
                
                # Check for exit (paper mode)
                if self.paper_mode and self.position:
                    exit_type = self.check_exit(df)
                    if exit_type:
                        row = df.iloc[-1]
                        if exit_type == 'TAKE_PROFIT':
                            exit_price = self.position.tp
                        else:
                            exit_price = self.position.sl
                        self.process_exit(exit_type, exit_price)
                
                # Check for entry
                if not self.position:
                    signal = self.check_entry_signal(df)
                    if signal:
                        ticket = self.place_order(
                            signal['direction'],
                            signal['entry'],
                            signal['sl'],
                            signal['tp']
                        )
                        if ticket:
                            self.position = Position(
                                ticket=ticket,
                                symbol=self.symbol,
                                direction=signal['direction'],
                                entry_price=signal['entry'],
                                volume=self.contracts,
                                sl=signal['sl'],
                                tp=signal['tp'],
                                entry_time=datetime.now()
                            )
                            print(f"📊 ATR: {signal['atr']:.2f} | Risk: ${abs(signal['entry'] - signal['sl']) * self.point_value:.2f}")
                
                # Status update every few bars
                row = df.iloc[-1]
                print(f"[{current_bar}] {self.symbol} {row['close']:.2f} | Pos: {self.position.direction if self.position else 'None'} | Daily PnL: ${self.daily_pnl:.2f}")
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Trading stopped by user")
        finally:
            self.save_log()
            
            print("\n" + "=" * 70)
            print("  SESSION SUMMARY")
            print("=" * 70)
            print(f"Trades today:  {self.daily_trades}")
            print(f"Daily PnL:     ${self.daily_pnl:.2f}")
            if self.trades:
                wins = len([t for t in self.trades if t['pnl'] > 0])
                print(f"Win rate:      {wins}/{len(self.trades)} ({wins/len(self.trades)*100:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Breakout Trading')
    parser.add_argument('--symbol', default='MES', help='Symbol (MES or MNQ)')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (no real orders)')
    
    args = parser.parse_args()
    
    trader = LiveBreakoutTrader(symbol=args.symbol, paper_mode=args.paper)
    
    if not trader.connect():
        print("\n❌ Could not connect to MT5 relay")
        print("Make sure relay_server.py is running on your Windows machine")
        sys.exit(1)
    
    trader.run()


if __name__ == "__main__":
    main()
