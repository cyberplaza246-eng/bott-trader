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
    python start_live_rithmic.py                       # MES default
    python start_live_rithmic.py --symbol MNQ          # Trade MNQ
    python start_live_rithmic.py --symbols MES MNQ     # Scan both, trade best setup
    python start_live_rithmic.py --paper               # Paper mode (signals only)
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any
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
    'atr_mult': 1.5,
    'tp_mult': 2.0,
    'ema_len': 50
}

TRAIL_SETTINGS = {
    # Start trailing once price has moved this fraction of TP distance.
    'activation_tp_fraction': float(os.getenv('TRAIL_ACTIVATION_TP_FRACTION', '0.35')),
    # Alternative activation based on initial risk (R-multiple).
    'activation_r': float(os.getenv('TRAIL_ACTIVATION_R', '0.30')),
    # Trail stop behind favorable move using ATR.
    'atr_mult': float(os.getenv('TRAIL_ATR_MULT', '0.6')),
    # Minimum SL movement required before submitting a broker update.
    'min_step_ticks': int(os.getenv('TRAIL_MIN_STEP_TICKS', '1')),
}

ENV_MAX_CONCURRENT_TRADES = max(1, int(os.getenv('MAX_CONCURRENT_TRADES', '3')))

# Risk settings
RISK_SETTINGS = {
    'contracts': 1,              # Start with 1 micro
    'daily_loss_limit': 150.0,   # Max daily loss $150
    'max_trades_per_day': 50,    # Max trades per day
    'cooldown_bars': 5,          # Bars between trades
    'max_positions': ENV_MAX_CONCURRENT_TRADES,  # Max concurrent positions
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
    trailing_enabled: bool = True


class LiveRithmicTrader:
    """Live trading with Enhanced Breakout Strategy via Rithmic."""
    
    def __init__(self, symbol: str = 'MES', symbols: Optional[List[str]] = None, paper_mode: bool = False, skip_confirm: bool = False, paper_orders: bool = False):
        self.skip_confirm = skip_confirm
        self.symbols = symbols or [symbol]
        self.symbol = self.symbols[0]
        self.paper_mode = paper_mode
        self.paper_orders = paper_orders  # Live Rithmic data + simulated orders
        
        for sym in self.symbols:
            if sym not in SYMBOL_SPECS:
                raise ValueError(f"Unsupported symbol: {sym}")
        
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
        self.max_positions = RISK_SETTINGS.get('max_positions', 2)
        
        # State - support multiple positions
        self.positions: Dict[str, Position] = {}  # order_id -> Position
        self.cooldown = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.trades: List[Dict] = []
        self._warmup_log_state: Dict[str, tuple] = {}
        self.symbol_priority = [s.strip() for s in os.getenv('SYMBOL_PRIORITY', 'MNQ,MES').split(',') if s.strip()]
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        
        # Full AI ensemble system
        self.ensemble: Optional[EnsembleTrader] = None
        self.technical = TechnicalAnalyzer()
        
        # Log file
        mode = "paper" if paper_mode else "live"
        symbols_tag = "_".join(self.symbols)
        self.log_file = f'logs/{mode}_rithmic_{symbols_tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def _spec(self, symbol: str) -> Dict[str, float]:
        return SYMBOL_SPECS[symbol]

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        return round(round(price / tick_size) * tick_size, 8)

    def update_trailing_stops(self, symbol: str, df: pd.DataFrame):
        """Move SL before TP is reached to protect unrealized gains."""
        if not self.positions or df is None or len(df) == 0:
            return

        row = df.iloc[-1]
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        atr = float(row.get('atr', np.nan))
        if not np.isfinite(atr) or atr <= 0:
            return

        tick = self._spec(symbol)['tick_size']
        min_move = max(1, TRAIL_SETTINGS['min_step_ticks']) * tick
        activation_frac = max(0.05, min(0.95, TRAIL_SETTINGS['activation_tp_fraction']))
        activation_r = max(0.05, TRAIL_SETTINGS.get('activation_r', 0.30))
        trail_dist = max(tick, atr * max(0.05, TRAIL_SETTINGS['atr_mult']))

        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue
            if not pos.trailing_enabled:
                continue

            tp_dist = abs(pos.tp - pos.entry_price)
            if tp_dist <= 0:
                continue

            if pos.direction == 'long':
                favorable_move = bar_high - pos.entry_price
                # Start trailing once either TP-fraction OR R-based threshold is reached.
                risk_dist = max(tick, pos.entry_price - pos.sl)
                activation_move = min(tp_dist * activation_frac, risk_dist * activation_r)
                if favorable_move < activation_move:
                    continue

                breakeven_sl = pos.entry_price + tick
                trail_sl = bar_high - trail_dist
                candidate_sl = self._round_to_tick(max(breakeven_sl, trail_sl), tick)

                # Keep stop on the valid side of market and only ratchet upward.
                if candidate_sl >= bar_low:
                    candidate_sl = self._round_to_tick(bar_low - tick, tick)
                if candidate_sl <= pos.sl or (candidate_sl - pos.sl) < min_move:
                    continue
            else:
                favorable_move = pos.entry_price - bar_low
                # Start trailing once either TP-fraction OR R-based threshold is reached.
                risk_dist = max(tick, pos.sl - pos.entry_price)
                activation_move = min(tp_dist * activation_frac, risk_dist * activation_r)
                if favorable_move < activation_move:
                    continue

                breakeven_sl = pos.entry_price - tick
                trail_sl = bar_low + trail_dist
                candidate_sl = self._round_to_tick(min(breakeven_sl, trail_sl), tick)

                # Keep stop on the valid side of market and only ratchet downward.
                if candidate_sl <= bar_high:
                    candidate_sl = self._round_to_tick(bar_high + tick, tick)
                if candidate_sl >= pos.sl or (pos.sl - candidate_sl) < min_move:
                    continue

            old_sl = pos.sl
            if self.paper_mode:
                pos.sl = candidate_sl
                print(f"   🔒 {symbol} trail: SL {old_sl:.2f} → {candidate_sl:.2f}")
                continue

            if self.broker and self.broker.modify_position(ticket=order_id, sl=candidate_sl, tp=pos.tp):
                pos.sl = candidate_sl
                print(f"   🔒 {symbol} trail: SL {old_sl:.2f} → {candidate_sl:.2f}")
                bot_logger.info(
                    f"Trailing stop updated {symbol} {order_id}: {old_sl:.2f} -> {candidate_sl:.2f}"
                )
                continue

            if self.broker and hasattr(self.broker, 'get_order_info'):
                order_info = self.broker.get_order_info(order_id)
                if order_info and not order_info.get('supports_stop_modify', True):
                    pos.trailing_enabled = False
                    bot_logger.error(
                        f"Trailing disabled for {symbol} {order_id}: native stop was not attached at entry"
                    )
                    print(f"   ⚠️ {symbol} trail disabled for {order_id}: no native stop from entry")
        
    def connect(self) -> bool:
        """Initialize Rithmic connection (skipped in paper mode)."""
        try:
            # Paper mode: skip Rithmic, use Yahoo Finance for data
            if self.paper_mode and not self.paper_orders:
                print("📝 PAPER MODE - Skipping Rithmic connection")
                print("   Using Yahoo Finance for market data")
                self.broker = None
                
                # Initialize full AI ensemble without broker
                print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
                self.ensemble = EnsembleTrader(broker=None)
                print("✅ AI Ensemble ready!")
                return True
            
            # Live mode: connect to Rithmic
            os.environ['RITHMIC_DISABLE_YAHOO_FALLBACK'] = 'true'
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
            print(f"   Mode: ⚠️  LIVE - Real money at risk!")
            
            # Initialize full AI ensemble
            print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
            self.ensemble = EnsembleTrader(broker=self.broker)
            print("✅ AI Ensemble ready!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic only in live mode."""
        try:
            # Paper mode or no broker: use Yahoo Finance
            if self.broker is None:
                return self._get_yahoo_candles(count)
            
            df = self.broker.get_candles(symbol, timeframe_minutes=5, num_candles=count)
            min_bars = self.lookback + self.ema_len + 15  # 75 bars needed
            if df is None or len(df) < min_bars:
                rith_bars = 0 if df is None else len(df)
                local_df = self._get_local_history_candles(symbol, count)
                merged = self._merge_candles(local_df, df, count)
                merged_bars = 0 if merged is None else len(merged)
                # Avoid repeating the same warm-up message every loop.
                state_key = (rith_bars, merged_bars)
                if self._warmup_log_state.get(symbol) != state_key:
                    self._warmup_log_state[symbol] = state_key
                    print(
                        f"⚠️  {symbol} Rithmic returned {rith_bars} bars, need {min_bars} - "
                        f"using local warm history ({merged_bars} bars total)"
                    )
                return merged
            return df
        except Exception as e:
            print(f"❌ Error getting candles for {symbol}: {e}")
            return None

    def _get_local_history_candles(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Load local 5m history CSV as indicator warm-start (no Yahoo usage)."""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            path = os.path.join(data_dir, f'{symbol}_5m.csv')
            if not os.path.exists(path):
                return None

            df = pd.read_csv(path)
            df.columns = [str(c).lower() for c in df.columns]
            if 'datetime' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'})
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
                df = df.dropna(subset=['datetime'])

            required = {'open', 'high', 'low', 'close'}
            if not required.issubset(set(df.columns)):
                return None

            keep_cols = [c for c in ['datetime', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            return df[keep_cols].tail(count).reset_index(drop=True)
        except Exception as e:
            bot_logger.warning(f"Local history warm-start failed for {symbol}: {e}")
            return None

    def _merge_candles(
        self,
        history_df: Optional[pd.DataFrame],
        live_df: Optional[pd.DataFrame],
        count: int,
    ) -> Optional[pd.DataFrame]:
        """Merge local history with live bars and keep latest rows."""
        if history_df is None and live_df is None:
            return None
        if history_df is None:
            return live_df.tail(count).reset_index(drop=True)
        if live_df is None:
            return history_df.tail(count).reset_index(drop=True)

        merged = pd.concat([history_df, live_df], ignore_index=True)
        if 'datetime' in merged.columns:
            merged['datetime'] = pd.to_datetime(merged['datetime'], utc=True, errors='coerce')
            merged = merged.dropna(subset=['datetime'])
            merged = merged.drop_duplicates(subset=['datetime'], keep='last')
            merged = merged.sort_values('datetime')
        else:
            merged = merged.drop_duplicates(keep='last')
        return merged.tail(count).reset_index(drop=True)
    
    def _get_yahoo_candles(self, count: int = 100, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch candles from Yahoo Finance as fallback."""
        try:
            import yfinance as yf
            
            # Map futures symbols to Yahoo tickers
            ticker_map = {
                'MES': 'ES=F',  # E-mini S&P 500 futures
                'MNQ': 'NQ=F',  # E-mini Nasdaq futures
            }
            target = symbol or self.symbol
            ticker = ticker_map.get(target, f'{target}=F')
            
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
    
    def get_price(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Get current bid/ask."""
        try:
            if self.broker is None:
                # Paper mode: get last close from Yahoo
                df = self._get_yahoo_candles(5, symbol=symbol)
                if df is not None and len(df) > 0:
                    price = float(df['close'].iloc[-1])
                    return {'bid': price - 0.25, 'ask': price + 0.25, 'last': price}
                return None
            return self.broker.get_latest_price(symbol or self.symbol)
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
    
    def check_entry_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Check for entry signal using full AI Ensemble."""
        if len(self.positions) >= self.max_positions or self.cooldown > 0:
            return None
        
        # Use full EnsembleTrader for signal
        if self.ensemble is None:
            return self._check_simple_breakout(symbol, df)
        
        try:
            # Add indicators via technical analyzer
            df_enriched = self.technical.calculate_indicators(df)
            
            # Get ensemble signal (includes sweep gate, ML, advanced strategies)
            signal_result = self.ensemble.get_trading_signal(df_enriched, symbol)
            
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
            
            # Use dynamic SL/TP if available
            if hasattr(self.ensemble, 'get_dynamic_sl_tp'):
                direction = 'BUY' if signal == 'BUY' else 'SELL'
                sltp = self.ensemble.get_dynamic_sl_tp(
                    df_enriched,
                    direction,
                    price,
                    symbol=symbol,
                    sr_levels=signal_result.get('sr_levels'),
                    sweep_wick=signal_result.get('sweep_wick'),
                    timeframe='5m',
                )
                sl = sltp['sl_price']
                tp = sltp['tp_price']
                sltp_method = sltp.get('method', 'unknown') if isinstance(sltp, dict) else 'unknown'
            else:
                sl_dist = atr * self.atr_mult
                if signal == 'BUY':
                    sl = price - sl_dist
                    tp = price + (sl_dist * self.tp_mult)
                else:
                    sl = price + sl_dist
                    tp = price - (sl_dist * self.tp_mult)
                sltp_method = 'atr_fallback'
            
            direction = 'long' if signal == 'BUY' else 'short'
            
            bot_logger.info(f"🎯 ENSEMBLE SIGNAL: {symbol} {signal} @ {price:.2f} (conf={confidence:.1%})")
            bot_logger.info(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            bot_logger.info(f"   SL/TP method: {sltp_method}")
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry': price,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'confidence': confidence,
            }
            
        except Exception as e:
            bot_logger.warning(f"Ensemble signal error: {e}, falling back to breakout")
            return self._check_simple_breakout(symbol, df)
    
    def _check_simple_breakout(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Fallback simple breakout strategy."""
        row = df.iloc[-1]
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr', 10.0)
        high_n = row.get('high_N', row['high'])
        low_n = row.get('low_N', row['low'])
        ema = row.get('ema', price)
        
        tick_size = self._spec(symbol)['tick_size']
        if pd.isna(atr) or atr < 2.0:
            return None
        
        # Long breakout
        if high > high_n and price > high_n and price > ema:
            entry = high_n + tick_size
            sl_dist = atr * self.atr_mult
            sl = entry - sl_dist
            tp = entry + (sl_dist * self.tp_mult)
            return {'symbol': symbol, 'direction': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr, 'confidence': 0.5}
        
        # Short breakout
        elif low < low_n and price < low_n and price < ema:
            entry = low_n - tick_size
            sl_dist = atr * self.atr_mult
            sl = entry + sl_dist
            tp = entry - (sl_dist * self.tp_mult)
            return {'symbol': symbol, 'direction': 'short', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr, 'confidence': 0.5}
        
        return None
    
    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        symbol = signal['symbol']
        tick_size = self._spec(symbol)['tick_size']
        direction = signal['direction']
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        
        if self.paper_mode or self.paper_orders:
            order_id = f"paper_{int(time.time())}"
            tag = "PAPER-ORDERS" if self.paper_orders else "PAPER"
            side = "LONG" if direction == 'long' else "SHORT"
            print(f"📝 [{tag}] {side} {symbol} @ {entry:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            pos = Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                trailing_enabled=True,
            )
            self.positions[order_id] = pos
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            return True
        
        # Real order
        order_type = 'buy' if direction == 'long' else 'sell'
        result = self.broker.place_order(
            symbol=symbol,
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
                symbol=symbol,
                direction=direction,
                entry_price=result.get('entry_price', entry),
                size=self.contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                trailing_enabled=bool(result.get('supports_stop_modify', True)),
            )
            self.positions[order_id] = pos
            print(f"✅ ORDER FILLED: {direction.upper()} {symbol}")
            print(f"   Entry: {pos.entry_price:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            if not pos.trailing_enabled:
                print(f"   ⚠️ Native stop missing on entry - trailing disabled for {order_id}")
                bot_logger.error(
                    f"Order {order_id} for {symbol} filled without native stop support; trailing is disabled"
                )
            trades_logger.info(f"ENTRY {direction} {symbol} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f}")
            return True
        else:
            print(f"❌ Order failed!")
            return False
    
    def check_position_exit(self, symbol: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Check if any positions hit SL/TP (for paper mode mainly). Returns list of (order_id, exit_type)."""
        exits = []
        if not self.positions:
            return exits
        
        row = df.iloc[-1]
        
        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue
            d = pos.direction
            
            if d == 'long':
                if row['low'] <= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif row['high'] >= pos.tp:
                    exits.append((order_id, 'TAKE_PROFIT'))
            else:
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
        spec = self._spec(pos.symbol)
        d = pos.direction
        if d == 'long':
            raw_pnl = (exit_price - pos.entry_price) * spec['point_value'] * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * spec['point_value'] * pos.size
        
        # Subtract costs ($2 commission + 1 tick slippage)
        commission = 2.00
        slippage = spec['tick_size'] * spec['point_value']
        pnl = raw_pnl - commission - slippage
        
        # Record trade
        trade = {
            'symbol': pos.symbol,
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
        trades_logger.info(f"EXIT {exit_type} {d} {pos.symbol} @ {exit_price:.2f} PnL=${pnl:.2f}")
        
        del self.positions[order_id]
        print(f"   Remaining positions: {len(self.positions)}/{self.max_positions}")
        self.cooldown = self.cooldown_bars
    
    def sync_broker_position(self):
        """Sync with actual broker position (live mode).
        
        Queries Rithmic for the actual open position count and removes
        any positions from self.positions that have been closed by SL/TP.
        """
        if self.paper_mode or self.paper_orders or not self.broker:
            return
        
        try:
            # Get actual positions from broker
            broker_positions = []
            for sym in self.symbols:
                broker_positions.extend(self.broker.get_open_positions(symbol=sym))
            broker_order_ids = {str(p.get('ticket')) for p in broker_positions if p.get('ticket')}
            
            # Find positions we think are open but broker says are closed
            our_order_ids = set(self.positions.keys())
            closed_ids = our_order_ids - broker_order_ids
            
            # Remove closed positions and log the exit
            for order_id in closed_ids:
                pos = self.positions.get(order_id)
                if pos:
                    # Position was closed by broker (likely SL or TP hit)
                    # Since we don't know the exact exit price, estimate from SL/TP
                    # Rithmic bracket orders typically close at the trigger price
                    bot_logger.info(f"🔄 Position sync: {order_id} closed by broker (SL/TP hit)")
                    
                    # Log minimal exit info
                    d = pos.direction.upper()
                    self.trades.append({
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else '',
                        'exit_time': datetime.now(timezone.utc).isoformat(),
                        'symbol': pos.symbol,
                        'direction': d,
                        'entry': pos.entry_price,
                        'exit': 0,  # Unknown - broker handled it
                        'pnl': 0,   # Unknown - will update from account later
                        'exit_type': 'BROKER_SYNC',
                        'sl': pos.sl,
                        'tp': pos.tp,
                    })
                    
                    del self.positions[order_id]
                    print(f"🔄 Position {order_id} removed (closed by broker)")
            
            # Also check if broker has more positions than we track (rare, but possible)
            if len(broker_positions) > len(self.positions):
                extra_count = len(broker_positions) - len(self.positions)
                bot_logger.warning(f"⚠️ Broker has {extra_count} positions not tracked by bot")
            
            # Log sync if any changes
            if closed_ids:
                print(f"📊 Position sync: {len(self.positions)}/{self.max_positions} active")
                
        except Exception as e:
            bot_logger.warning(f"Position sync error: {e}")
    
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
                'symbols': self.symbols,
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
        print(f"Symbols:    {', '.join(self.symbols)}")
        print(f"Contracts:  {self.contracts}")
        if self.paper_orders:
            mode_str = '📝 PAPER-ORDERS (live data, simulated fills)'
        elif self.paper_mode:
            mode_str = 'PAPER'
        else:
            mode_str = '⚠️ LIVE'
        print(f"Mode:       {mode_str}")
        print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
        print(f"Features:   Dynamic SL/TP, Trailing Stops, 8 Confirmation Models")
        print(f"Risk:       Max ${self.daily_loss_limit} daily loss")
        print("=" * 70)
        
        if self.paper_orders:
            print("\n📝 PAPER-ORDERS MODE - Connected to live Rithmic data, orders are simulated.")
        elif not self.paper_mode and not self.skip_confirm:
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
        
        last_bar_time: Dict[str, Any] = {}
        
        try:
            while True:
                # Check daily limits
                if not self.check_daily_limits():
                    print(f"⏸️  Limits reached - waiting...")
                    time.sleep(300)
                    continue
                
                # Sync position with broker
                self.sync_broker_position()
                
                any_new_bar = False
                cycle_candidates: List[Dict[str, Any]] = []

                for symbol in self.symbols:
                    df = self.get_candles(symbol=symbol, count=100)
                    if df is None or len(df) < self.lookback + self.ema_len + 15:
                        print(f"⚠️  {symbol} insufficient data, waiting...")
                        continue

                    df = self.calculate_indicators(df)

                    if 'datetime' in df.columns:
                        current_bar = df.iloc[-1]['datetime']
                    elif 'time' in df.columns:
                        current_bar = df.iloc[-1]['time']
                    else:
                        current_bar = datetime.now()

                    if last_bar_time.get(symbol) == current_bar:
                        continue
                    last_bar_time[symbol] = current_bar
                    any_new_bar = True

                    if self.positions:
                        exits = self.check_position_exit(symbol, df)
                        for order_id, exit_type in exits:
                            pos = self.positions.get(order_id)
                            if pos:
                                exit_price = pos.tp if exit_type == 'TAKE_PROFIT' else pos.sl
                                self.process_exit(order_id, exit_type, exit_price)

                        # Trail remaining open positions after processing exits.
                        self.update_trailing_stops(symbol, df)

                    if len(self.positions) < self.max_positions and self.cooldown == 0:
                        signal = self.check_entry_signal(symbol, df)
                        if signal:
                            cycle_candidates.append(signal)

                    row = df.iloc[-1]
                    pos_syms = [p.symbol for p in self.positions.values()]
                    pos_str = f"{len(self.positions)}/{self.max_positions} {pos_syms}" if self.positions else "None"
                    cooldown_str = f"(CD:{self.cooldown})" if self.cooldown > 0 else ""
                    print(f"[{current_bar}] {symbol} {row['close']:.2f} | Pos: {pos_str} {cooldown_str} | Daily: ${self.daily_pnl:.2f}")

                if not any_new_bar:
                    time.sleep(5)
                    continue

                if self.cooldown > 0:
                    self.cooldown -= 1

                if len(self.positions) < self.max_positions and cycle_candidates:
                    priority_rank = {sym: idx for idx, sym in enumerate(self.symbol_priority)}
                    best = max(
                        cycle_candidates,
                        key=lambda s: (
                            float(s.get('confidence', 0.0)),
                            -priority_rank.get(s.get('symbol', ''), 999),
                        ),
                    )
                    if len(cycle_candidates) > 1:
                        print(
                            f"🏆 Best setup this cycle: {best['symbol']} "
                            f"({best.get('confidence', 0.0):.1%} confidence, priority={self.symbol_priority})"
                        )
                    if self.place_order(best):
                        spec = self._spec(best['symbol'])
                        print(f"📊 {best['symbol']} ATR: {best['atr']:.2f}")
                        risk = abs(best['entry'] - best['sl']) * spec['point_value']
                        print(f"   Risk: ${risk:.2f} per contract")

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
    parser.add_argument('--symbol', default='MES', choices=['MES', 'MNQ'], help='Single symbol mode')
    parser.add_argument('--symbols', nargs='+', choices=['MES', 'MNQ'], help='Multi-symbol mode, e.g. --symbols MES MNQ')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (Yahoo data + simulated orders)')
    parser.add_argument('--paper-orders', action='store_true', dest='paper_orders', help='Live Rithmic data + simulated orders')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip 10-second confirmation')
    
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else [args.symbol]
    trader = LiveRithmicTrader(symbol=symbols[0], symbols=symbols, paper_mode=args.paper, skip_confirm=args.yes, paper_orders=args.paper_orders)
    
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
