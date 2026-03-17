#!/usr/bin/env python
"""
LIVE TRADING - Full AI Ensemble via Rithmic (Tradesea/Lucid)

Strategy: Sweep-Gated Entry + ML Confirmations (PF 1.16+ validated)
- IntelligentTrader: ML classifiers (Gradient Boost, SVC, Neural Net)
- AdvancedStrategies: 8 strategies (FibMACD, StochRSI, HeikinAshi, etc)
- Dynamic SL/TP: Swing-based stops with trailing
- EMA/Technical confirmation layer

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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.broker.rithmic_connector import RithmicConnector
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger, trades_logger

# Strategy parameters (validated profitable)
PARAMS = {
    'lookback': 10,
    'atr_mult': 1.0,            # SL = 1.0 × ATR (breathing room)
    'tp_mult': 0.5,             # TP = 0.5 × ATR (tight for quick wins)
    'ema_len': 50,
    'tp_tighten': 1.0,          # Disabled - using direct ATR multipliers above
}

# Risk settings
RISK_SETTINGS = {
    'contracts': 1,              # Start with 1 micro
    'daily_loss_limit': 150.0,   # Max daily loss $150
    'max_trades_per_day': 8,     # Max trades per day
    'cooldown_bars': 1,          # Bars between trades (1 bar = ~5 min)
    'max_positions': 2,          # Max concurrent positions
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
    managed_by_broker: bool = False


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
        self.tp_tighten = PARAMS.get('tp_tighten', 1.0)  # 0.15 = keep 15% of TP distance (85% closer)
        self.ema_len = PARAMS['ema_len']
        
        # Risk management
        self.contracts = RISK_SETTINGS['contracts']
        self.daily_loss_limit = RISK_SETTINGS['daily_loss_limit']
        self.max_trades_per_day = RISK_SETTINGS['max_trades_per_day']
        self.cooldown_bars = RISK_SETTINGS['cooldown_bars']
        self.max_positions = RISK_SETTINGS.get('max_positions', 2)
        
        # State - support multiple positions
        self.positions: Dict[str, Position] = {}  # order_id -> Position
        self.cooldown = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.trades: List[Dict] = []
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        
        # Full AI ensemble system
        self.ensemble: Optional[EnsembleTrader] = None
        self.technical = TechnicalAnalyzer()
        
        # Log file
        mode = "paper" if paper_mode else "live"
        self.log_file = f'logs/{mode}_rithmic_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def connect(self) -> bool:
        """Initialize Rithmic connection (skipped in paper mode)."""
        try:
            # Paper mode: skip Rithmic, use Yahoo Finance for data
            if self.paper_mode:
                print("📝 PAPER MODE - Skipping Rithmic connection")
                print("   Using Yahoo Finance for market data")
                self.broker = None
                
                # Initialize full AI ensemble without broker
                print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
                self.ensemble = EnsembleTrader(broker=None)
                print("✅ AI Ensemble ready!")
                return True
            
            # Live mode: connect to Rithmic with session-limit handling
            import logging
            
            # Track ForcedLogout to prevent infinite reconnect loops
            self._forced_logout_count = 0
            _trader_ref = self
            
            class ForcedLogoutFilter(logging.Filter):
                def filter(self, record):
                    if 'ForcedLogout' in record.getMessage():
                        _trader_ref._forced_logout_count += 1
                        if _trader_ref._forced_logout_count >= 3:
                            print("\n⚠️  Rithmic ForcedLogout detected 3 times!")
                            print("   Switching to PAPER MODE with Yahoo Finance data.")
                            print("   Fix: Close all other Rithmic sessions, then restart.\n")
                            _trader_ref.paper_mode = True
                            if _trader_ref.broker:
                                try:
                                    _trader_ref.broker.shutdown()
                                except Exception:
                                    pass
                            _trader_ref.broker = None
                    return True
            
            # Install filter on rithmic loggers
            for logger_name in ['rithmic.plant.ticker', 'rithmic.plant.history', 'rithmic']:
                rith_logger = logging.getLogger(logger_name)
                rith_logger.addFilter(ForcedLogoutFilter())
            
            self.broker = RithmicConnector()
            self.broker.initialize()
            
            # Check if we got forced-logout during init
            if self.paper_mode:
                print("📝 Fell back to PAPER MODE due to Rithmic session limits")
                self.broker = None
                self.ensemble = EnsembleTrader(broker=None)
                print("✅ AI Ensemble ready (paper mode)!")
                return True
            
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
            print(f"   Mode: ⚠️  LIVE - Real money at risk!")
            
            # Initialize full AI ensemble
            print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
            self.ensemble = EnsembleTrader(broker=self.broker)
            print("✅ AI Ensemble ready!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles(self, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic or Yahoo Finance (paper mode)."""
        try:
            # Paper mode or no broker: use Yahoo Finance
            if self.broker is None:
                return self._get_yahoo_candles(count)
            
            df = self.broker.get_candles(self.symbol, timeframe_minutes=5, num_candles=count)
            min_bars = self.lookback + self.ema_len + 15  # 75 bars needed
            if df is None or len(df) < min_bars:
                # Fallback to Yahoo - Rithmic returned insufficient data (normal on Sunday open)
                return self._get_yahoo_candles(count)
            return df
        except Exception as e:
            print(f"❌ Error getting candles: {e}")
            return self._get_yahoo_candles(count)
    
    def _get_yahoo_candles(self, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from Yahoo Finance as fallback."""
        try:
            import yfinance as yf
            
            # Map futures symbols to Yahoo tickers
            ticker_map = {
                'MES': 'ES=F',  # E-mini S&P 500 futures
                'MNQ': 'NQ=F',  # E-mini Nasdaq futures
            }
            ticker = ticker_map.get(self.symbol, f'{self.symbol}=F')
            
            data = yf.download(ticker, period='5d', interval='5m', progress=False)
            if data.empty:
                return None
            
            df = data.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            
            if 'datetime' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'})
            
            return df.tail(count).reset_index(drop=True)
        except Exception as e:
            bot_logger.warning(f"Yahoo Finance fallback failed: {e}")
            return None
    
    def get_price(self) -> Optional[Dict]:
        """Get current bid/ask."""
        try:
            if self.broker is None:
                # Paper mode: get last close from Yahoo
                df = self._get_yahoo_candles(5)
                if df is not None and len(df) > 0:
                    price = float(df['close'].iloc[-1])
                    return {'bid': price - 0.25, 'ask': price + 0.25, 'last': price}
                return None
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
        """Check for entry signal using full AI Ensemble."""
        if len(self.positions) >= self.max_positions or self.cooldown > 0:
            return None
        
        # Use full EnsembleTrader for signal
        if self.ensemble is None:
            return self._check_simple_breakout(df)
        
        try:
            # Add indicators via technical analyzer
            df_enriched = self.technical.calculate_indicators(df)
            
            # Get ensemble signal (includes sweep gate, ML, advanced strategies)
            signal_result = self.ensemble.get_trading_signal(df_enriched, self.symbol)
            
            # Check if should trade
            if not self.ensemble.should_trade(signal_result):
                return None
            
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            if signal not in ('BUY', 'SELL'):
                return None
            
            row = df.iloc[-1]
            price = row['close']
            atr = df_enriched['atr'].iloc[-1] if 'atr' in df_enriched.columns else row.get('atr', 10.0)
            
            if pd.isna(atr) or atr < 2.0:
                atr = 10.0  # Fallback ATR for futures
            
            # Simple ATR-based SL/TP (skip dynamic for consistency)
            sl_dist = atr * self.atr_mult  # SL = 1 ATR
            tp_dist = atr * self.tp_mult   # TP = 0.5 ATR
            if signal == 'BUY':
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist
            
            bot_logger.info(f"📍 SL: {sl_dist:.2f} pts (1×ATR) | TP: {tp_dist:.2f} pts (0.5×ATR)")
            
            direction = 'long' if signal == 'BUY' else 'short'
            
            bot_logger.info(f"🎯 ENSEMBLE SIGNAL: {signal} @ {price:.2f} (conf={confidence:.1%})")
            bot_logger.info(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            return {'direction': direction, 'entry': price, 'sl': sl, 'tp': tp, 'atr': atr, 'confidence': confidence}
            
        except Exception as e:
            bot_logger.warning(f"Ensemble signal error: {e}, falling back to breakout")
            return self._check_simple_breakout(df)
    
    def _check_simple_breakout(self, df: pd.DataFrame) -> Optional[Dict]:
        """Fallback simple breakout strategy."""
        row = df.iloc[-1]
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr', 10.0)
        high_n = row.get('high_N', row['high'])
        low_n = row.get('low_N', row['low'])
        ema = row.get('ema', price)
        
        if pd.isna(atr) or atr < 2.0:
            return None
        
        # Long breakout
        if high > high_n and price > high_n and price > ema:
            entry = high_n + self.tick_size
            sl_dist = atr * self.atr_mult  # SL = 1 ATR
            tp_dist = atr * self.tp_mult   # TP = 0.5 ATR
            sl = entry - sl_dist
            tp = entry + tp_dist
            return {'direction': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr}
        
        # Short breakout
        elif low < low_n and price < low_n and price < ema:
            entry = low_n - self.tick_size
            sl_dist = atr * self.atr_mult  # SL = 1 ATR
            tp_dist = atr * self.tp_mult   # TP = 0.5 ATR
            sl = entry + sl_dist
            tp = entry - tp_dist
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
            
            pos = Position(
                order_id=order_id,
                symbol=self.symbol,
                direction=direction,
                entry_price=entry,
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc)
            )
            self.positions[order_id] = pos
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
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
            order_id = result.get('ticket', str(int(time.time())))
            pos = Position(
                order_id=order_id,
                symbol=self.symbol,
                direction=direction,
                entry_price=result.get('entry_price', entry),
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc)
            )
            self.positions[order_id] = pos
            print(f"✅ ORDER FILLED: {direction.upper()} {self.symbol}")
            print(f"   Entry: {pos.entry_price:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            trades_logger.info(f"ENTRY {direction} {self.symbol} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f}")
            return True
        else:
            print(f"❌ Order failed!")
            return False
    
    def check_position_exit(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Check if any positions hit SL/TP, with trailing stop and breakeven."""
        exits = []
        if not self.positions:
            return exits
        
        row = df.iloc[-1]
        current_price = row['close']
        
        for order_id, pos in list(self.positions.items()):
            if pos.managed_by_broker:
                continue

            d = pos.direction
            entry = pos.entry_price
            initial_risk = abs(entry - pos.sl)
            
            if d == 'long':
                profit_pts = current_price - entry
                
                # Move SL to breakeven when price moves 1x risk in our favor
                if profit_pts >= initial_risk and pos.sl < entry:
                    old_sl = pos.sl
                    pos.sl = entry + (self.tick_size * 2)  # Slightly above entry
                    print(f"🔒 BREAKEVEN: {self.symbol} LONG SL moved {old_sl:.2f} → {pos.sl:.2f}")
                
                # Trail SL: when price moves 1.5x risk, trail at 1x risk behind price
                if profit_pts >= initial_risk * 1.5:
                    trail_sl = current_price - initial_risk
                    if trail_sl > pos.sl:
                        old_sl = pos.sl
                        pos.sl = trail_sl
                        print(f"📈 TRAIL: {self.symbol} LONG SL moved {old_sl:.2f} → {pos.sl:.2f}")
                
                if row['low'] <= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif row['high'] >= pos.tp:
                    exits.append((order_id, 'TAKE_PROFIT'))
            else:
                profit_pts = entry - current_price
                
                # Move SL to breakeven when price moves 1x risk in our favor
                if profit_pts >= initial_risk and pos.sl > entry:
                    old_sl = pos.sl
                    pos.sl = entry - (self.tick_size * 2)  # Slightly below entry
                    print(f"🔒 BREAKEVEN: {self.symbol} SHORT SL moved {old_sl:.2f} → {pos.sl:.2f}")
                
                # Trail SL: when price moves 1.5x risk, trail at 1x risk behind price
                if profit_pts >= initial_risk * 1.5:
                    trail_sl = current_price + initial_risk
                    if trail_sl < pos.sl:
                        old_sl = pos.sl
                        pos.sl = trail_sl
                        print(f"📈 TRAIL: {self.symbol} SHORT SL moved {old_sl:.2f} → {pos.sl:.2f}")
                
                if row['high'] >= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif row['low'] <= pos.tp:
                    exits.append((order_id, 'TAKE_PROFIT'))
        
        return exits
    
    def process_exit(self, order_id: str, exit_type: str, exit_price: float):
        """Process position exit."""
        if order_id not in self.positions:
            return
        
        pos = self.positions[order_id]
        d = pos.direction
        if d == 'long':
            raw_pnl = (exit_price - pos.entry_price) * self.point_value * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * self.point_value * pos.size
        
        # Subtract costs ($2 commission + 1 tick slippage)
        commission = 2.00
        slippage = self.tick_size * self.point_value
        pnl = raw_pnl - commission - slippage
        
        # Record trade
        trade = {
            'symbol': self.symbol,
            'direction': d,
            'entry_time': str(pos.entry_time),
            'exit_time': str(datetime.now(timezone.utc)),
            'entry_price': pos.entry_price,
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
        
        del self.positions[order_id]
        print(f"   Remaining positions: {len(self.positions)}/{self.max_positions}")
        self.cooldown = self.cooldown_bars
    
    def sync_broker_position(self):
        """Sync with actual broker position (live mode)."""
        if self.paper_mode or not self.broker:
            return

        # Preserve locally tracked positions created during this process.
        # This sync is mainly for restarts, where live broker positions exist
        # but self.positions starts empty.
        if self.positions:
            return

        try:
            broker_positions = self.broker.get_open_positions(self.symbol)
        except Exception as e:
            bot_logger.warning(f"Could not sync broker positions: {e}")
            return

        if not broker_positions:
            return

        synced_positions: Dict[str, Position] = {}
        for broker_pos in broker_positions:
            raw_size = int(abs(broker_pos.get('size', 0) or 0))
            if raw_size <= 0:
                continue

            direction = 'long' if (broker_pos.get('size', 0) or 0) > 0 else 'short'
            avg_price = float(broker_pos.get('avg_price', 0.0) or 0.0)
            
            # Skip invalid positions with no price
            if avg_price <= 0:
                bot_logger.warning(f"⚠️ Skipping sync: invalid avg_price={avg_price}")
                continue

            # Calculate protective SL/TP using current ATR
            try:
                df = self.get_candles(count=30)
                if df is not None and len(df) >= 15:
                    df = self.calculate_indicators(df)
                    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else 10.0
                else:
                    atr = 10.0
            except Exception:
                atr = 10.0

            sl_dist = atr * self.atr_mult
            tp_dist = sl_dist * self.tp_mult * self.tp_tighten
            if direction == 'long':
                sl = avg_price - sl_dist
                tp = avg_price + tp_dist
            else:
                sl = avg_price + sl_dist
                tp = avg_price - tp_dist

            for index in range(raw_size):
                order_id = f"broker_sync_{self.symbol}_{index + 1}"
                synced_positions[order_id] = Position(
                    order_id=order_id,
                    symbol=self.symbol,
                    direction=direction,
                    entry_price=avg_price,
                    size=1,
                    sl=sl,
                    tp=tp,
                    entry_time=datetime.now(timezone.utc),
                    managed_by_broker=False,
                )

        if synced_positions:
            self.positions = synced_positions
            for oid, p in synced_positions.items():
                print(
                    f"🔄 Synced {p.direction.upper()} {self.symbol} @ {p.entry_price:.2f} "
                    f"SL={p.sl:.2f} TP={p.tp:.2f} (ATR-based)"
                )
            print(f"🔄 Total synced: {len(self.positions)} position(s) for {self.symbol}")
    
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
        print("  LIVE TRADING - Rithmic + Full AI Ensemble")
        print("=" * 70)
        print(f"Symbol:     {self.symbol}")
        print(f"Contracts:  {self.contracts}")
        print(f"Mode:       {'PAPER' if self.paper_mode else '⚠️ LIVE'}")
        print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
        print(f"Features:   Dynamic SL/TP, Trailing Stops, 8 Confirmation Models")
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
                
                # Check for exits on all positions (paper mode or monitor)
                if self.positions:
                    exits = self.check_position_exit(df)
                    for order_id, exit_type in exits:
                        pos = self.positions.get(order_id)
                        if pos:
                            if exit_type == 'TAKE_PROFIT':
                                exit_price = pos.tp
                            else:
                                exit_price = pos.sl
                            self.process_exit(order_id, exit_type, exit_price)
                
                # Check for entry (only if room for more positions)
                if len(self.positions) < self.max_positions:
                    signal = self.check_entry_signal(df)
                    if signal:
                        if self.place_order(signal):
                            print(f"📊 ATR: {signal['atr']:.2f}")
                            risk = abs(signal['entry'] - signal['sl']) * self.point_value
                            print(f"   Risk: ${risk:.2f} per contract")
                
                # Status update
                row = df.iloc[-1]
                pos_str = f"{len(self.positions)}/{self.max_positions}" if self.positions else "None"
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
