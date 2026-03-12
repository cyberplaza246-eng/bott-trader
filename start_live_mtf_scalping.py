#!/usr/bin/env python
"""
LIVE TRADING - Multi-Timeframe Scalping Strategy via Rithmic

Strategy: 5M Trend Filter + 1M Entry Signals
Symbols: MES, MNQ, NQ, MGC (Micro Gold)
- Max 2 positions at a time across all symbols
- MES: TP 80 ticks below resistance
- Others: Standard 1.5:1 R:R

Requirements:
1. Rithmic credentials in .env
2. Active Tradesea/Lucid account with trading permissions

Usage:
    python start_live_mtf_scalping.py              # Trade all symbols
    python start_live_mtf_scalping.py --symbols MES MNQ  # Specific symbols
    python start_live_mtf_scalping.py --paper      # Paper mode
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

import pytz
import smtplib
from email.mime.text import MIMEText

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Email config (must be after load_dotenv)
EMAIL_NOTIFY = True  # Enabled
EMAIL_TO = 'paraflix246@gmail.com'
EMAIL_SUBJECT_PREFIX = '[Ai-bot]'
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASS = os.getenv('EMAIL_PASS', '')

def send_email(subject, body):
    """Send email notification via Gmail SMTP."""
    if not EMAIL_NOTIFY or not EMAIL_USER or not EMAIL_PASS:
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = f"{EMAIL_SUBJECT_PREFIX} {subject}"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_TO
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print(f"[Email] Failed to send: {e}")

from src.broker.rithmic_connector import RithmicConnector
from src.utils.logger import bot_logger, trades_logger

# ── Trading Hours Logic ─────────────────────────────────────────────
def is_market_open_et(now=None):
    """Return True if CME Globex is open (ET), else False."""
    # CME Globex: Sunday 6pm ET to Friday 5pm ET, daily break 5-6pm ET
    if now is None:
        now = datetime.now(pytz.timezone('US/Eastern'))
    else:
        now = now.astimezone(pytz.timezone('US/Eastern'))
    wd = now.weekday()  # 0=Mon, 6=Sun
    hour, minute = now.hour, now.minute
    # Daily break: 17:00-18:00 ET
    if hour == 17:
        return False
    # Weekend close: Fri 17:00 ET to Sun 18:00 ET
    if wd == 4 and hour >= 17:  # Friday after 5pm
        return False
    if wd == 5:  # Saturday
        return False
    if wd == 6 and hour < 18:  # Sunday before 6pm
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Parameters (Validated via backtest)
# ══════════════════════════════════════════════════════════════════════════════

# 5M Trend Filter
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
ADX_THRESHOLD = 18
ADX_PERIOD = 14

# 1M Entry Conditions
ENTRY_EMA_FAST = 9
ENTRY_EMA_SLOW = 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 35, 60
RSI_SHORT_MIN, RSI_SHORT_MAX = 40, 65
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_RATIO_THRESHOLD = 0.5   # Lowered for more trades
BB_PERIOD, BB_STD = 20, 2
BB_EXTREME_LOW, BB_EXTREME_HIGH = 0.05, 0.95

# Risk Settings
ATR_PERIOD = 14
ATR_MULT = 1.2             # SL = 1.2 × ATR
TP_MULT = 1.5              # TP = 1.5 × SL
TP_BUFFER_ATR_MULT = 0.0   # TP capped exactly at resistance/support (no buffer)
RESISTANCE_LOOKBACK = 20   # 5M bars for swing high/low

# Pullback settings
MAX_PULLBACK_ATR = 1.5     # Max distance from EMA for pullback (matches backtest)

# Risk Management
DAILY_LOSS_LIMIT = 350.0
MAX_TRADES_PER_DAY = 6
MAX_POSITIONS = 2          # Max 2 positions at a time
CONTRACTS = 1

# Symbol specs - includes all tradeable futures
SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25},   # Micro E-mini S&P
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25},   # Micro E-mini Nasdaq
    'NQ': {'point_value': 20.0, 'tick_size': 0.25},   # E-mini Nasdaq (full)
    'MGC': {'point_value': 10.0, 'tick_size': 0.10},  # Micro Gold
}

# Default symbols to trade
DEFAULT_SYMBOLS = ['MES', 'MNQ', 'NQ', 'MGC']


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
    initial_sl: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Indicator Calculations
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, DI+, DI-"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(series: pd.Series, period: int = 20, std_dev: int = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    pctb = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, pctb


# ══════════════════════════════════════════════════════════════════════════════
#  Live Trader Class
# ══════════════════════════════════════════════════════════════════════════════

class LiveMTFScalper:
    """Live trading with Multi-Timeframe Scalping Strategy via Rithmic."""
    
    def __init__(self, symbols: List[str] = None, paper_mode: bool = False, skip_confirm: bool = False):
        self.skip_confirm = skip_confirm
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.paper_mode = paper_mode
        
        # Validate symbols
        for sym in self.symbols:
            if sym not in SYMBOL_SPECS:
                raise ValueError(f"Unsupported symbol: {sym}")
        
        # State - track multiple positions
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.trades: List[Dict] = []
        
        # 5M context cache per symbol
        self.context_cache: Dict[str, Dict] = {}
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        
        # Log file
        mode = "paper" if paper_mode else "live"
        self.log_file = f'logs/{mode}_mtf_multi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def connect(self) -> bool:
        """Initialize Rithmic connection."""
        try:
            self.broker = RithmicConnector()
            self.broker.initialize()
            
            if not self.broker.connected:
                print("❌ Rithmic not connected - check credentials in .env")
                return False
            
            acct = self.broker.get_account_info()
            print(f"✅ Rithmic Connected!")
            print(f"   System: {acct.get('system', 'Unknown')}")
            print(f"   Balance: ${acct.get('balance', 0):,.2f}")
            
            if self.paper_mode:
                print(f"   Mode: PAPER - No real orders")
            else:
                print(f"   Mode: ⚠️  LIVE - Real money!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles(self, symbol: str, timeframe_minutes: int, count: int = 200) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic for a specific symbol."""
        try:
            df = self.broker.get_candles(symbol, timeframe_minutes=timeframe_minutes, num_candles=count)
            if df is None or len(df) < 50:
                return None
            return df
        except Exception as e:
            print(f"❌ Error getting {symbol} candles: {e}")
            return None
    
    def add_1m_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 1M entry indicators."""
        df = df.copy()
        
        # EMAs
        df['ema_9'] = calculate_ema(df['close'], ENTRY_EMA_FAST)
        df['ema_21'] = calculate_ema(df['close'], ENTRY_EMA_SLOW)
        
        # ATR
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        
        # MACD
        _, _, df['macd_hist'] = calculate_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df['macd_hist_prev'] = df['macd_hist'].shift(1)
        
        # Volume ratio
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        # Bollinger %B
        _, _, df['bb_pctb'] = calculate_bollinger(df['close'], BB_PERIOD, BB_STD)
        
        return df
    
    def add_5m_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 5M trend indicators."""
        df = df.copy()
        
        # Trend EMAs
        df['ema_50'] = calculate_ema(df['close'], TREND_EMA_FAST)
        df['ema_200'] = calculate_ema(df['close'], TREND_EMA_SLOW)
        
        # ADX
        df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df, ADX_PERIOD)
        
        # ATR
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        
        return df
    
    def get_5m_context(self, df_5m: pd.DataFrame) -> Dict:
        """Get current 5M trend context."""
        if df_5m is None or len(df_5m) < RESISTANCE_LOOKBACK:
            return {'trend': None, 'adx': 0, 'di_plus': 0, 'di_minus': 0, 'resistance': 0, 'support': 0}
        
        recent = df_5m.tail(RESISTANCE_LOOKBACK)
        row = recent.iloc[-1]
        
        # Trend direction
        if row['ema_50'] > row['ema_200']:
            trend = 'bullish'
        elif row['ema_50'] < row['ema_200']:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Swing high/low for resistance/support
        resistance = recent['high'].max()
        support = recent['low'].min()
        
        return {
            'trend': trend,
            'adx': row['adx'],
            'di_plus': row['di_plus'],
            'di_minus': row['di_minus'],
            'atr': row['atr'],
            'resistance': resistance,
            'support': support,
        }
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = 0
            print(f"\n📅 New trading day: {today}")
        
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            print(f"🛑 Daily loss limit hit: ${self.daily_pnl:.2f}")
            return False
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            print(f"🛑 Max trades hit: {self.daily_trades}/{MAX_TRADES_PER_DAY}")
            return False
        return True
    
    def check_long_entry(self, row_1m: pd.Series, ctx_5m: Dict, verbose: bool = False) -> bool:
        """Check if long entry conditions are met."""
        # 5M Trend Filter
        if ctx_5m['trend'] != 'bullish':
            return False
        if ctx_5m['adx'] < ADX_THRESHOLD:
            return False
        if ctx_5m['di_plus'] <= ctx_5m['di_minus']:
            return False
        
        # 1M Entry Conditions
        price = row_1m['close']
        ema_9 = row_1m['ema_9']
        ema_21 = row_1m['ema_21']
        rsi = row_1m['rsi']
        macd_hist = row_1m['macd_hist']
        macd_hist_prev = row_1m['macd_hist_prev']
        volume_ratio = row_1m['volume_ratio']
        bb_pctb = row_1m['bb_pctb']
        atr = row_1m['atr']
        
        # Price above EMA9 (with small tolerance)
        ema9_tolerance = atr * 0.1
        if price < (ema_9 - ema9_tolerance):
            if verbose: print(f"      ❌ Price {price:.2f} < EMA9 {ema_9:.2f} - {ema9_tolerance:.2f}")
            return False
        
        # Pullback to EMA zone (within 1.5 ATR of EMA21)
        if abs(price - ema_21) > (atr * MAX_PULLBACK_ATR):
            if verbose: print(f"      ❌ Pullback: {abs(price-ema_21):.2f} (need within {atr*MAX_PULLBACK_ATR:.2f})")
            return False
        
        # RSI filter
        if pd.isna(rsi) or not (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX):
            if verbose: print(f"      ❌ RSI: {rsi:.1f} (need {RSI_LONG_MIN}-{RSI_LONG_MAX})")
            return False
        
        # MACD rising (matching backtest - just needs to be rising, not positive)
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            return False
        if macd_hist <= macd_hist_prev:
            if verbose: print(f"      ❌ MACD: hist={macd_hist:.4f} not rising (prev={macd_hist_prev:.4f})")
            return False
        
        # Volume confirmation
        if pd.isna(volume_ratio) or volume_ratio < VOLUME_RATIO_THRESHOLD:
            if verbose: print(f"      ❌ Volume: {volume_ratio:.2f}x (need ≥{VOLUME_RATIO_THRESHOLD})")
            return False
        
        # Bollinger filter
        if pd.isna(bb_pctb) or bb_pctb <= BB_EXTREME_LOW or bb_pctb >= BB_EXTREME_HIGH:
            if verbose: print(f"      ❌ BB%B: {bb_pctb:.2f} (need {BB_EXTREME_LOW}-{BB_EXTREME_HIGH})")
            return False
        
        return True
    
    def check_short_entry(self, row_1m: pd.Series, ctx_5m: Dict, verbose: bool = False) -> bool:
        """Check if short entry conditions are met."""
        # 5M Trend Filter
        if ctx_5m['trend'] != 'bearish':
            return False
        if ctx_5m['adx'] < ADX_THRESHOLD:
            return False
        if ctx_5m['di_minus'] <= ctx_5m['di_plus']:
            return False
        
        # 1M Entry Conditions
        price = row_1m['close']
        ema_9 = row_1m['ema_9']
        ema_21 = row_1m['ema_21']
        rsi = row_1m['rsi']
        macd_hist = row_1m['macd_hist']
        macd_hist_prev = row_1m['macd_hist_prev']
        volume_ratio = row_1m['volume_ratio']
        bb_pctb = row_1m['bb_pctb']
        atr = row_1m['atr']
        
        # Price below EMA9 (with small tolerance)
        ema9_tolerance = atr * 0.1
        if price > (ema_9 + ema9_tolerance):
            if verbose: print(f"      ❌ Price {price:.2f} > EMA9 {ema_9:.2f} + {ema9_tolerance:.2f}")
            return False
        
        # Pullback to EMA zone (within 1.5 ATR of EMA21)
        if abs(price - ema_21) > (atr * MAX_PULLBACK_ATR):
            if verbose: print(f"      ❌ Pullback: {abs(price-ema_21):.2f} (need within {atr*MAX_PULLBACK_ATR:.2f})")
            return False
        
        # RSI filter
        if pd.isna(rsi) or not (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX):
            if verbose: print(f"      ❌ RSI: {rsi:.1f} (need {RSI_SHORT_MIN}-{RSI_SHORT_MAX})")
            return False
        
        # MACD falling (matching backtest - just needs to be falling, not negative)
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            return False
        if macd_hist >= macd_hist_prev:
            if verbose: print(f"      ❌ MACD: hist={macd_hist:.4f} not falling (prev={macd_hist_prev:.4f})")
            return False
        
        # Volume confirmation
        if pd.isna(volume_ratio) or volume_ratio < VOLUME_RATIO_THRESHOLD:
            if verbose: print(f"      ❌ Volume: {volume_ratio:.2f}x (need ≥{VOLUME_RATIO_THRESHOLD})")
            return False
        
        # Bollinger filter
        if pd.isna(bb_pctb) or bb_pctb <= BB_EXTREME_LOW or bb_pctb >= BB_EXTREME_HIGH:
            if verbose: print(f"      ❌ BB%B: {bb_pctb:.2f} (need {BB_EXTREME_LOW}-{BB_EXTREME_HIGH})")
            return False
        
        return True
    
    def check_entry_signal(self, symbol: str, df_1m: pd.DataFrame, ctx_5m: Dict) -> Optional[Dict]:
        """Check for entry signal on latest 1M bar for a symbol."""
        # Skip if already have position in this symbol
        if symbol in self.positions:
            return None
        
        # Check max positions limit
        if len(self.positions) >= MAX_POSITIONS:
            return None
        
        row = df_1m.iloc[-1]
        atr = row['atr']
        
        if pd.isna(atr) or atr <= 0:
            return None
        
        sl_distance = atr * ATR_MULT
        tp_distance = sl_distance * TP_MULT
        entry_price = row['close']
        
        # Enable verbose for symbols with valid 5M trend
        verbose = ctx_5m['adx'] >= ADX_THRESHOLD
        
        if self.check_long_entry(row, ctx_5m, verbose=verbose):
            # Always cap TP at resistance - ATR buffer (applies to ALL symbols)
            tp_rr = entry_price + tp_distance
            tp_buffer = atr * TP_BUFFER_ATR_MULT
            tp_resistance = ctx_5m['resistance'] - tp_buffer
            # Use whichever is closer to entry (safer)
            tp_final = min(tp_rr, tp_resistance) if tp_resistance > entry_price else tp_rr
            
            return {
                'symbol': symbol,
                'direction': 'long',
                'entry': entry_price,
                'sl': entry_price - sl_distance,
                'tp': tp_final,
                'atr': atr
            }
        
        elif self.check_short_entry(row, ctx_5m, verbose=verbose):
            # Always cap TP at support + ATR buffer (applies to ALL symbols)
            tp_rr = entry_price - tp_distance
            tp_buffer = atr * TP_BUFFER_ATR_MULT
            tp_support = ctx_5m['support'] + tp_buffer
            # Use whichever is closer to entry (safer)
            tp_final = max(tp_rr, tp_support) if tp_support < entry_price else tp_rr
            
            return {
                'symbol': symbol,
                'direction': 'short',
                'entry': entry_price,
                'sl': entry_price + sl_distance,
                'tp': tp_final,
                'atr': atr
            }
        
        return None
    
    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        symbol = signal['symbol']
        direction = signal['direction']
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        
        spec = SYMBOL_SPECS[symbol]
        point_value = spec['point_value']
        
        if self.paper_mode:
            order_id = f"paper_{symbol}_{int(time.time())}"
            print(f"📝 [PAPER] {direction.upper()} {symbol} @ {entry:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            self.positions[symbol] = Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=CONTRACTS,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                initial_sl=sl
            )
            self.daily_trades += 1
            return True
        
        # Live order
        try:
            side = 'BUY' if direction == 'long' else 'SELL'
            result = self.broker.place_order(
                symbol=symbol,
                order_type=side,
                size=CONTRACTS,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp
            )
            
            if result and result.get('ticket'):
                print(f"✅ {direction.upper()} {symbol} @ {entry:.2f}")
                print(f"   Order ID: {result['ticket']}")
                print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
                
                # Send email notification
                send_email(
                    f"Trade: {direction.upper()} {symbol}",
                    f"Trade Placed!\n\nSymbol: {symbol}\nDirection: {direction.upper()}\nEntry: {entry:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}\nOrder ID: {result['ticket']}\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                
                self.positions[symbol] = Position(
                    order_id=result['ticket'],
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    size=CONTRACTS,
                    sl=sl,
                    tp=tp,
                    entry_time=datetime.now(timezone.utc),
                    initial_sl=sl
                )
                self.daily_trades += 1
                return True
            else:
                print(f"❌ Order failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Order error: {e}")
            return False
    
    def test_order(self):
        """Test broker connectivity and order capability."""
        print("\n" + "="*60)
        print("🧪 ORDER SYSTEM TEST")
        print("="*60)
        
        test_symbol = self.symbols[0] if self.symbols else 'MES'
        
        # Step 1: Connect
        print(f"\n[1/3] 📡 Connecting to broker...")
        try:
            if not self.connect():
                print(f"❌ Failed to connect to broker")
                return False
            print(f"✅ Connected!")
            time.sleep(1)
            
            # Step 2: Get quote
            print(f"\n[2/3] 📊 Getting {test_symbol} quote...")
            quote = self.broker.get_latest_price(test_symbol)
            
            if not quote:
                print(f"❌ Failed to get quote for {test_symbol}")
                return False
            
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)
            last = quote.get('last', 0)
            current_price = last if last > 0 else (bid + ask) / 2 if bid and ask else 0
            
            if current_price == 0:
                print(f"❌ No valid price for {test_symbol}")
                return False
                
            print(f"✅ Quote received: Bid={bid:.2f} Ask={ask:.2f} Last={last:.2f}")
            
            # Step 3: Test order (with immediate close)
            print(f"\n[3/3] 📝 Testing order placement...")
            print(f"   This will place a REAL order and immediately close it!")
            
            confirm = input("   Continue? (type 'yes'): ")
            if confirm.lower() != 'yes':
                print("   Test skipped.")
                return True
            
            # Place order at market with tight brackets
            spec = SYMBOL_SPECS.get(test_symbol, SYMBOL_SPECS['MES'])
            sl = current_price - (5 * spec['tick_size'])  # 5 ticks SL
            tp = current_price + (3 * spec['tick_size'])  # 3 ticks TP
            
            result = self.broker.place_order(
                symbol=test_symbol,
                order_type='BUY',
                size=1,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp
            )
            
            if result and result.get('ticket'):
                order_id = result['ticket']
                print(f"✅ Order placed! Ticket: {order_id}")
                
                # Immediately close
                time.sleep(0.3)
                print(f"🗑️ Closing position...")
                close_result = self.broker.close_position(symbol=test_symbol)
                
                if close_result:
                    print(f"✅ Position closed!")
                    print("\n" + "="*60)
                    print("✅ ORDER SYSTEM VERIFIED - WORKING!")
                    print("="*60 + "\n")
                    return True
                else:
                    print(f"⚠️ Close failed - CHECK BROKER MANUALLY!")
                    print(f"   You may have an open position on {test_symbol}")
                    return True  # Order worked
            else:
                print(f"❌ Order failed: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.broker:
                self.broker.shutdown()
    
    def check_exit(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position for symbol should be closed."""
        if symbol not in self.positions:
            return None
        
        # Validate price is reasonable (not 0 or negative)
        if current_price <= 0:
            return None
        
        position = self.positions[symbol]
        
        # Sanity check: price should be within reasonable range of entry
        # (prevents false triggers from bad data)
        max_deviation = abs(position.entry_price) * 0.1  # 10% max deviation
        if abs(current_price - position.entry_price) > max_deviation:
            print(f"   ⚠️ Ignoring suspicious price {current_price:.2f} (entry was {position.entry_price:.2f})")
            return None
        
        if position.direction == 'long':
            if current_price <= position.sl:
                return 'SL'
            if current_price >= position.tp:
                return 'TP'
        else:  # short
            if current_price >= position.sl:
                return 'SL'
            if current_price <= position.tp:
                return 'TP'
        
        return None
    
    def close_position(self, symbol: str, reason: str, exit_price: float):
        """Close position for a symbol."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        spec = SYMBOL_SPECS[symbol]
        point_value = spec['point_value']
        
        # Calculate P&L with real fill price
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * point_value * position.size
        else:
            pnl = (position.entry_price - exit_price) * point_value * position.size
        self.daily_pnl += pnl
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} Closed {symbol} {position.direction.upper()} @ {exit_price:.2f} ({reason})")
        print(f"   P&L: ${pnl:+.2f} | Daily: ${self.daily_pnl:+.2f}")
        
        # Send email notification for closed trade
        send_email(
            f"Closed: {symbol} ({reason})",
            f"Trade Closed!\n\nSymbol: {symbol}\nDirection: {position.direction.upper()}\nEntry: {position.entry_price:.2f}\nExit: {exit_price:.2f}\nReason: {reason}\nP&L: ${pnl:+.2f}\nDaily P&L: ${self.daily_pnl:+.2f}\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        # Log trade
        trade = {
            'symbol': symbol,
            'direction': position.direction,
            'entry': position.entry_price,
            'exit': exit_price,
            'sl': position.sl,
            'tp': position.tp,
            'pnl': pnl,
            'reason': reason,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)
        # Save to log
        with open(self.log_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
        del self.positions[symbol]
        return
    def run(self, duration_minutes: int = 480):
        """Main trading loop - scans all symbols."""
        print(f"\n{'='*70}")
        print(f"  🔪 MULTI-TIMEFRAME SCALPING BOT - MULTI-SYMBOL")
        print(f"{'='*70}")
        print(f"Strategy: 5M Trend + 1M Entry")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Max Positions: {MAX_POSITIONS}")
        print(f"Daily Loss Limit: ${DAILY_LOSS_LIMIT}")
        print(f"Max Trades/Day: {MAX_TRADES_PER_DAY}")
        print(f"{'='*70}\n")
        
        if not self.connect():
            return
        
        if not self.paper_mode and not self.skip_confirm:
            confirm = input("\n⚠️  LIVE TRADING MODE - Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        print(f"\n🚀 Starting trading loop...")
        print(f"   Will run until {end_time.strftime('%H:%M:%S')}\n")
        
        loop_count = 0
        while datetime.now() < end_time:
            try:
                loop_count += 1

                # ── Trading Hours Check ──
                if not is_market_open_et():
                    now_et = datetime.now(pytz.timezone('US/Eastern'))
                    print(f"⏸️  Market closed (CME Globex hours) - {now_et.strftime('%a %H:%M ET')}")
                    # Sleep until next open: if daily break, sleep 65min; if weekend, sleep 2h
                    if now_et.weekday() == 4 and now_et.hour >= 17:
                        # Friday after 5pm, sleep until Sunday 6pm
                        days = (6 - now_et.weekday()) % 7
                        next_open = now_et.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=days)
                        sleep_sec = (next_open - now_et).total_seconds()
                        sleep_sec = max(sleep_sec, 3600)
                    elif now_et.hour == 17:
                        # Daily break, sleep 65min
                        sleep_sec = 65 * 60
                    elif now_et.weekday() == 5:
                        # Saturday, sleep until Sunday 6pm
                        next_open = now_et + timedelta(days=1)
                        next_open = next_open.replace(hour=18, minute=0, second=0, microsecond=0)
                        sleep_sec = (next_open - now_et).total_seconds()
                        sleep_sec = max(sleep_sec, 3600)
                    elif now_et.weekday() == 6 and now_et.hour < 18:
                        # Sunday before 6pm
                        next_open = now_et.replace(hour=18, minute=0, second=0, microsecond=0)
                        sleep_sec = (next_open - now_et).total_seconds()
                        sleep_sec = max(sleep_sec, 3600)
                    else:
                        sleep_sec = 3600
                    time.sleep(sleep_sec)
                    continue
                
                # Check daily limits
                if not self.check_daily_limits():
                    print("⏸️  Daily limits reached - waiting for next day")
                    time.sleep(300)
                    continue
                
                # Process each symbol
                for symbol in self.symbols:
                    # Longer delay between symbols to prevent Rithmic lock timeout
                    time.sleep(3)
                    
                    # Fetch data for this symbol
                    df_1m = self.get_candles(symbol, timeframe_minutes=1, count=100)
                    time.sleep(2)  # Longer pause between requests
                    df_5m = self.get_candles(symbol, timeframe_minutes=5, count=RESISTANCE_LOOKBACK + 50)
                    
                    if df_1m is None or df_5m is None:
                        continue
                    
                    # Add indicators
                    df_1m = self.add_1m_indicators(df_1m)
                    df_5m = self.add_5m_indicators(df_5m)
                    
                    # Get 5M context
                    ctx_5m = self.get_5m_context(df_5m)
                    
                    # Show 5M trend status
                    trend = ctx_5m.get('trend', 'none')
                    adx = ctx_5m.get('adx', 0)
                    print(f"   {symbol}: 5M trend={trend} ADX={adx:.1f}")
                    
                    # Get current price
                    price_data = self.broker.get_latest_price(symbol)
                    time.sleep(1)  # Delay after price request
                    if not price_data:
                        continue
                    
                    bid = price_data.get('bid', 0)
                    ask = price_data.get('ask', 0)
                    last = price_data.get('last', 0)
                    
                    # Use last price if available, otherwise mid
                    if last > 0:
                        current_price = last
                    elif bid > 0 and ask > 0:
                        current_price = (bid + ask) / 2
                    else:
                        print(f"   ⚠️ No valid price for {symbol}")
                        continue
                    
                    # Check for exit on existing position
                    if symbol in self.positions:
                        exit_reason = self.check_exit(symbol, current_price)
                        if exit_reason:
                            # Close via broker and use current_price as exit
                            self.broker.close_position(symbol=symbol)
                            time.sleep(2)  # Delay after close
                            self.close_position(symbol, exit_reason, current_price)
                    
                    # Check for entry (only if under max positions)
                    if symbol not in self.positions and len(self.positions) < MAX_POSITIONS:
                        signal = self.check_entry_signal(symbol, df_1m, ctx_5m)
                        if signal:
                            print(f"🎯 Signal: {signal['direction'].upper()} {symbol} @ {signal['entry']:.2f}")
                            print(f"   SL: {signal['sl']:.2f} | TP: {signal['tp']:.2f}")
                            self.place_order(signal)
                            time.sleep(2)  # Delay after order
                
                # Status update every 10 loops
                if loop_count % 10 == 0:
                    pos_list = list(self.positions.keys())
                    pos_str = f"Positions: {pos_list}" if pos_list else "No positions"
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | {pos_str} | Daily P&L: ${self.daily_pnl:+.2f}")
                
                # Show scan status every loop
                print(f"🔍 Scan #{loop_count} | Trades today: {self.daily_trades}/{MAX_TRADES_PER_DAY}")
                
                # Wait for next bar
                time.sleep(60)  # 1-minute bars
                
            except KeyboardInterrupt:
                print("\n\n⛔ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Symbols traded: {', '.join(self.symbols)}")
        print(f"Total Trades: {len(self.trades)}")
        total_pnl = sum(t['pnl'] for t in self.trades)
        wins = len([t for t in self.trades if t['pnl'] > 0])
        losses = len([t for t in self.trades if t['pnl'] <= 0])
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Log saved to: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Timeframe Scalping Bot - Multi-Symbol')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        choices=['MES', 'MNQ', 'NQ', 'MGC'],
                        help='Symbols to trade (default: all)')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    parser.add_argument('--duration', type=int, default=480, help='Duration in minutes')
    parser.add_argument('--test', action='store_true', help='Place test order and cancel immediately')
    args = parser.parse_args()
    
    trader = LiveMTFScalper(
        symbols=args.symbols,
        paper_mode=args.paper,
        skip_confirm=args.yes
    )
    
    if args.test:
        trader.test_order()
    else:
        trader.run(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
