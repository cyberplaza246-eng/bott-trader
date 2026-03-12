#!/usr/bin/env python
"""
LIVE TRADING - Enhanced Breakout Strategy via Rithmic (Tradesea/Lucid)

Strategy: 10-bar breakout + EMA50 trend filter (PF 1.50 validated)

Requirements:
1. Rithmic credentials in .env (already configured for LucidTrading)
2. Active Tradesea/Lucid account with trading permissions

Usage:
    python start_live_rithmic.py              # MES default
    python start_live_rithmic.py --symbol MNQ # Trade MNQ
    python start_live_rithmic.py --paper      # Paper mode (signals only)
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.broker.rithmic_connector import RithmicConnector
from src.utils.logger import bot_logger, trades_logger

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

# Symbol specs
SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25},
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25},
}


@dataclass
class Position:
    order_id: str
    symbol: str
    direction: str
    entry_price: float
    size: int
    sl: float
    tp: float
    entry_time: datetime


class LiveRithmicTrader:
    """Live trading with Enhanced Breakout Strategy via Rithmic."""
    
    def __init__(self, symbol: str = 'MES', paper_mode: bool = False, skip_confirm: bool = False):
        self.skip_confirm = skip_confirm
        self.symbol = symbol
        self.paper_mode = paper_mode
        
        # Symbol specs
        spec = SYMBOL_SPECS.get(symbol)
        if not spec:
            raise ValueError(f"Unsupported symbol: {symbol}")
        self.point_value = spec['point_value']
        self.tick_size = spec['tick_size']
        
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
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        
        # Log file
        mode = "paper" if paper_mode else "live"
        self.log_file = f'logs/{mode}_rithmic_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def connect(self) -> bool:
        """Initialize Rithmic connection."""
        try:
            self.broker = RithmicConnector()
            self.broker.initialize()
            
            if not self.broker.connected:
                print("❌ Rithmic not connected - check credentials in .env")
                print(f"   RITHMIC_USER_ID: {os.getenv('RITHMIC_USER_ID', 'NOT SET')}")
                print(f"   RITHMIC_SYSTEM: {os.getenv('RITHMIC_SYSTEM', 'NOT SET')}")
                return False
            
            # Get account info
            acct = self.broker.get_account_info()
            print(f"✅ Rithmic Connected!")
            print(f"   System: {acct.get('system', 'Unknown')}")
            print(f"   Balance: ${acct.get('balance', 0):,.2f}")
            print(f"   Equity: ${acct.get('equity', 0):,.2f}")
            
            if self.paper_mode:
                print(f"   Mode: PAPER - No real orders will be placed")
            else:
                print(f"   Mode: ⚠️  LIVE - Real money at risk!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles(self, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic."""
        try:
            df = self.broker.get_candles(self.symbol, timeframe_minutes=5, num_candles=count)
            if df is None or len(df) < 20:
                print(f"⚠️  Insufficient candle data: {len(df) if df is not None else 0}")
                return None
            return df
        except Exception as e:
            print(f"❌ Error getting candles: {e}")
            return None
    
    def get_price(self) -> Optional[Dict]:
        """Get current bid/ask."""
        try:
            return self.broker.get_latest_price(self.symbol)
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
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
            print(f"🛑 Daily loss limit hit: ${self.daily_pnl:.2f}")
            return False
        if self.daily_trades >= self.max_trades_per_day:
            print(f"🛑 Max trades hit: {self.daily_trades}/{self.max_trades_per_day}")
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
    
    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        direction = signal['direction']
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        
        if self.paper_mode:
            order_id = f"paper_{int(time.time())}"
            print(f"📝 [PAPER] {direction.upper()} {self.symbol} @ {entry:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            self.position = Position(
                order_id=order_id,
                symbol=self.symbol,
                direction=direction,
                entry_price=entry,
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc)
            )
            return True
        
        # Real order
        order_type = 'buy' if direction == 'long' else 'sell'
        result = self.broker.place_order(
            symbol=self.symbol,
            order_type=order_type,
            size=self.contracts,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp
        )
        
        if result:
            self.position = Position(
                order_id=result.get('ticket', str(int(time.time()))),
                symbol=self.symbol,
                direction=direction,
                entry_price=result.get('entry_price', entry),
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc)
            )
            print(f"✅ ORDER FILLED: {direction.upper()} {self.symbol}")
            print(f"   Entry: {self.position.entry_price:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            trades_logger.info(f"ENTRY {direction} {self.symbol} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f}")
            return True
        else:
            print(f"❌ Order failed!")
            return False
    
    def check_position_exit(self, df: pd.DataFrame) -> Optional[str]:
        """Check if position hit SL/TP (for paper mode mainly)."""
        if not self.position:
            return None
        
        row = df.iloc[-1]
        d = self.position.direction
        
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
            raw_pnl = (exit_price - self.position.entry_price) * self.point_value * self.position.size
        else:
            raw_pnl = (self.position.entry_price - exit_price) * self.point_value * self.position.size
        
        # Subtract costs ($2 commission + 1 tick slippage)
        commission = 2.00
        slippage = self.tick_size * self.point_value
        pnl = raw_pnl - commission - slippage
        
        # Record trade
        trade = {
            'symbol': self.symbol,
            'direction': d,
            'entry_time': str(self.position.entry_time),
            'exit_time': str(datetime.now(timezone.utc)),
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl
        }
        self.trades.append(trade)
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} EXIT {exit_type}: {d.upper()} @ {exit_price:.2f}")
        print(f"   PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}")
        trades_logger.info(f"EXIT {exit_type} {d} {self.symbol} @ {exit_price:.2f} PnL=${pnl:.2f}")
        
        self.position = None
        self.cooldown = self.cooldown_bars
    
    def sync_broker_position(self):
        """Sync with actual broker position (live mode)."""
        if self.paper_mode or not self.broker:
            return
        
        # This would check actual positions from the broker
        # and update self.position accordingly
        # For now, relies on bracket orders handling SL/TP
        pass
    
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
                'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
                'total_pnl': sum(t['pnl'] for t in self.trades),
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0
            }
        
        with open(self.log_file, 'w') as f:
            json.dump({
                'symbol': self.symbol,
                'paper_mode': self.paper_mode,
                'strategy': 'enhanced_breakout',
                'params': PARAMS,
                'stats': stats,
                'trades': self.trades
            }, f, indent=2, default=str)
        
        print(f"📁 Log saved: {self.log_file}")
    
    def run(self):
        """Main trading loop."""
        print("\n" + "=" * 70)
        print("  LIVE BREAKOUT TRADING - Rithmic/Tradesea")
        print("=" * 70)
        print(f"Symbol:     {self.symbol}")
        print(f"Contracts:  {self.contracts}")
        print(f"Mode:       {'PAPER' if self.paper_mode else '⚠️ LIVE'}")
        print(f"Strategy:   10-bar Breakout + EMA50 Filter")
        print(f"Risk:       Max ${self.daily_loss_limit} daily loss")
        print("=" * 70)
        
        if not self.paper_mode and not self.skip_confirm:
            print("\n⚠️  WARNING: LIVE MODE - Real money at risk!")
            print("Press Ctrl+C within 10 seconds to cancel...")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nCancelled.")
                return
        elif not self.paper_mode:
            print("\n⚠️  LIVE MODE - Starting immediately (--yes flag)")
        
        print("\n🚀 Starting trading loop...\n")
        
        last_bar_time = None
        
        try:
            while True:
                # Check daily limits
                if not self.check_daily_limits():
                    print(f"⏸️  Limits reached - waiting...")
                    time.sleep(300)
                    continue
                
                # Sync position with broker
                self.sync_broker_position()
                
                # Get candles
                df = self.get_candles(count=100)
                if df is None or len(df) < self.lookback + self.ema_len + 15:
                    print("⚠️  Insufficient data, retrying in 30s...")
                    time.sleep(30)
                    continue
                
                df = self.calculate_indicators(df)
                
                # Get current bar time
                if 'datetime' in df.columns:
                    current_bar = df.iloc[-1]['datetime']
                elif 'time' in df.columns:
                    current_bar = df.iloc[-1]['time']
                else:
                    current_bar = datetime.now()
                
                # Only process on new bar
                if last_bar_time == current_bar:
                    time.sleep(5)
                    continue
                
                last_bar_time = current_bar
                
                # Decrement cooldown
                if self.cooldown > 0:
                    self.cooldown -= 1
                
                # Check for exit (paper mode or monitor)
                if self.position:
                    exit_type = self.check_position_exit(df)
                    if exit_type:
                        if exit_type == 'TAKE_PROFIT':
                            exit_price = self.position.tp
                        else:
                            exit_price = self.position.sl
                        self.process_exit(exit_type, exit_price)
                
                # Check for entry
                if not self.position:
                    signal = self.check_entry_signal(df)
                    if signal:
                        if self.place_order(signal):
                            print(f"📊 ATR: {signal['atr']:.2f}")
                            risk = abs(signal['entry'] - signal['sl']) * self.point_value
                            print(f"   Risk: ${risk:.2f} per contract")
                
                # Status update
                row = df.iloc[-1]
                pos_str = f"{self.position.direction}" if self.position else "None"
                cooldown_str = f"(CD:{self.cooldown})" if self.cooldown > 0 else ""
                print(f"[{current_bar}] {self.symbol} {row['close']:.2f} | Pos: {pos_str} {cooldown_str} | Daily: ${self.daily_pnl:.2f}")
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Trading stopped by user")
        finally:
            self.save_log()
            if self.broker:
                self.broker.shutdown()
            
            print("\n" + "=" * 70)
            print("  SESSION SUMMARY")
            print("=" * 70)
            print(f"Trades:    {len(self.trades)}")
            print(f"Daily PnL: ${self.daily_pnl:.2f}")
            if self.trades:
                wins = len([t for t in self.trades if t['pnl'] > 0])
                print(f"Win rate:  {wins}/{len(self.trades)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Breakout Trading via Rithmic')
    parser.add_argument('--symbol', default='MES', help='Symbol (MES or MNQ)')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip 10-second confirmation')
    
    args = parser.parse_args()
    
    trader = LiveRithmicTrader(symbol=args.symbol, paper_mode=args.paper, skip_confirm=args.yes)
    
    if not trader.connect():
        print("\n❌ Could not connect to Rithmic")
        print("\nCheck your .env file has:")
        print("  RITHMIC_USER_ID=your_user")
        print("  RITHMIC_PASSWORD=your_password")
        print("  RITHMIC_SYSTEM=LucidTrading")
        sys.exit(1)
    
    trader.run()


if __name__ == "__main__":
    main()
