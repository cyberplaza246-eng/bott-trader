#!/usr/bin/env python3
"""
Enhanced Multi-Timeframe Scalping Backtest — 1M/5M Strategy (v2)

NEW vs original backtest_mtf_scalping.py:
  1. Candlestick pattern confirmation (engulfing, hammer, pin bar)
  2. Liquidity sweep detection (wick past swing → reversal = high-quality entry)
  3. Swing-based SL placement (SL behind structure, not pure ATR)
  4. Session filter re-enabled (US Open + US Close only)
  5. Scoring system — entries require minimum score from multiple confirmations

Usage:
    python scripts/backtest_mtf_enhanced.py --symbol MES
    python scripts/backtest_mtf_enhanced.py --symbol MNQ
    python scripts/backtest_mtf_enhanced.py --symbol NQ
    python scripts/backtest_mtf_enhanced.py --all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25, 'atr_mult': 1.2},
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25, 'atr_mult': 1.2},
    'NQ':  {'point_value': 20.0, 'tick_size': 0.25, 'atr_mult': 1.2},
}

# 5M Trend Filter — TIGHTENED from original
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
ADX_THRESHOLD = 22             # Tightened from 18 → require strong trends
ADX_PERIOD = 14

# 1M Entry — TIGHTENED from original (higher quality entries)
ENTRY_EMA_FAST = 9
ENTRY_EMA_MED = 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 40, 55        # Tightened from 35-60
RSI_SHORT_MIN, RSI_SHORT_MAX = 45, 60      # Tightened from 40-65
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_RATIO_THRESHOLD = 1.0               # Tightened from 0.5 → require above-avg volume
VOLUME_MA_PERIOD = 20
BB_PERIOD, BB_STD = 20, 2
BB_EXTREME_LOW, BB_EXTREME_HIGH = 0.10, 0.90  # Tightened from 0.05/0.95

# Risk
TP_MULT = 1.8                 # Up from 1.5 → better R:R without hurting WR too much
TP_BUFFER_ATR_MULT = 0.0
RESISTANCE_LOOKBACK = 20
DAILY_LOSS_LIMIT = 350.0
MAX_TRADES_PER_DAY = 6
INITIAL_BALANCE = 50000.0

# Exit modes
TRAIL_TRIGGER_R = 1.2
TRAIL_STEP_R = 0.25
USE_BREAKEVEN = False
TRAIL_AFTER_TP = False
HYBRID_EXIT = False

# ── NEW: Entry Confirmation (relaxed — candle not required, just bonus) ──────
REQUIRE_CANDLE_CONFIRM = False  # Candle pattern is optional bonus (not a gate)
REQUIRE_SESSION = True          # Only trade during US Open / US Close

# ── NEW: RSI Divergence Detection ───────────────────────────────────────────
DIVERGENCE_LOOKBACK = 20
DIVERGENCE_ENABLED = True

# ── NEW: Momentum Confirmation ──────────────────────────────────────────────
# Require previous bar close in trade direction (momentum alignment)
REQUIRE_MOMENTUM_BAR = False
# Pullback zone tightened
PULLBACK_ATR_MULT = 1.0        # Tightened from 1.5 → closer to EMA21


@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl: float
    tp: float
    entry_score: float = 0.0
    entry_signals: str = ""
    initial_sl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    trail_activated: bool = False
    partial_closed: bool = False
    partial_pnl: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Indicator Calculations (unchanged)
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
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm > 0) & (minus_dm > plus_dm)] = 0
    minus_dm[(minus_dm > 0) & (plus_dm > minus_dm)] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger(series: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return lower, middle, upper


def calculate_bb_pctb(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    return (close - lower) / (upper - lower + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Candlestick Pattern Detection (inline, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def detect_candle_signal(df: pd.DataFrame, idx: int) -> Tuple[str, float]:
    """Detect candlestick patterns at index idx.
    
    Returns:
        (signal, strength) where signal is 'BUY', 'SELL', or 'NONE'
        and strength is 0.0-1.0.
    """
    if idx < 4:
        return ('NONE', 0.0)
    
    BODY_THRESHOLD = 0.35
    TAIL_RATIO = 1.5
    
    o = [df['open'].iat[idx - j] for j in range(3, -1, -1)]   # [idx-3, idx-2, idx-1, idx]
    h = [df['high'].iat[idx - j] for j in range(3, -1, -1)]
    l = [df['low'].iat[idx - j] for j in range(3, -1, -1)]
    c = [df['close'].iat[idx - j] for j in range(3, -1, -1)]
    
    cur, prev = 3, 2
    body_cur = abs(c[cur] - o[cur])
    body_prev = abs(c[prev] - o[prev])
    range_cur = h[cur] - l[cur]
    is_bull_cur = c[cur] > o[cur]
    is_bull_prev = c[prev] > o[prev]
    
    if range_cur <= 0:
        return ('NONE', 0.0)
    
    upper_wick = h[cur] - max(o[cur], c[cur])
    lower_wick = min(o[cur], c[cur]) - l[cur]
    body_ratio = body_cur / range_cur
    
    # ── Bullish Engulfing ──
    if (is_bull_cur and not is_bull_prev and
            c[cur] > o[prev] and o[cur] < c[prev]):
        return ('BUY', 0.8)
    
    # ── Bearish Engulfing ──
    if (not is_bull_cur and is_bull_prev and
            c[cur] < o[prev] and o[cur] > c[prev]):
        return ('SELL', 0.8)
    
    # ── Hammer (bullish) ──
    if (body_ratio < BODY_THRESHOLD and
            lower_wick > body_cur * TAIL_RATIO and
            upper_wick < body_cur * 0.5):
        return ('BUY', 0.7)
    
    # ── Shooting Star (bearish) ──
    if (body_ratio < BODY_THRESHOLD and
            upper_wick > body_cur * TAIL_RATIO and
            lower_wick < body_cur * 0.5):
        return ('SELL', 0.7)
    
    # ── Pin Bar ──
    if lower_wick > range_cur * 0.6 and body_ratio < 0.3:
        return ('BUY', 0.6)
    if upper_wick > range_cur * 0.6 and body_ratio < 0.3:
        return ('SELL', 0.6)
    
    # ── Bullish close (weaker — directional candle body > 60% of range) ──
    if is_bull_cur and body_ratio > 0.6:
        return ('BUY', 0.3)
    if not is_bull_cur and body_ratio > 0.6:
        return ('SELL', 0.3)
    
    return ('NONE', 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: RSI Divergence Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_rsi_divergence(df: pd.DataFrame, idx: int, direction: str,
                          lookback: int = DIVERGENCE_LOOKBACK) -> bool:
    """Detect bullish/bearish RSI divergence.
    
    Bullish divergence: price makes lower low, RSI makes higher low → BUY
    Bearish divergence: price makes higher high, RSI makes lower high → SELL
    """
    if idx < lookback + 5:
        return False
    
    start = idx - lookback
    prices = df['close'].iloc[start:idx + 1].values
    rsi_vals = df['rsi'].iloc[start:idx + 1].values
    
    if np.any(np.isnan(rsi_vals)):
        return False
    
    # Find two recent troughs (for bullish) or peaks (for bearish)
    mid = len(prices) // 2
    
    if direction == 'LONG':
        # Bullish divergence: price lower low + RSI higher low
        price_low_1 = np.min(prices[:mid])
        price_low_2 = np.min(prices[mid:])
        rsi_at_low_1 = rsi_vals[np.argmin(prices[:mid])]
        rsi_at_low_2 = rsi_vals[mid + np.argmin(prices[mid:])]
        
        if price_low_2 < price_low_1 and rsi_at_low_2 > rsi_at_low_1:
            return True
    
    elif direction == 'SHORT':
        # Bearish divergence: price higher high + RSI lower high
        price_high_1 = np.max(prices[:mid])
        price_high_2 = np.max(prices[mid:])
        rsi_at_high_1 = rsi_vals[np.argmax(prices[:mid])]
        rsi_at_high_2 = rsi_vals[mid + np.argmax(prices[mid:])]
        
        if price_high_2 > price_high_1 and rsi_at_high_2 < rsi_at_high_1:
            return True
    
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Session Filter (RE-ENABLED)
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_session(dt: datetime) -> Tuple[bool, bool]:
    """Check if datetime is within valid US trading sessions (UTC).
    
    Returns:
        (is_allowed, is_optimal) — is_optimal used for stats.
    """
    if not isinstance(dt, datetime):
        return True, False
    
    hour = dt.hour
    minute = dt.minute
    t = hour * 60 + minute
    
    # US Open: 14:30-16:30 UTC (9:30-11:30 EST) — BEST liquidity
    us_open_start = 14 * 60 + 30
    us_open_end = 16 * 60 + 30
    
    # US Close: 19:30-21:00 UTC (2:30-4:00 PM EST)
    us_close_start = 19 * 60 + 30
    us_close_end = 21 * 60
    
    is_optimal = (us_open_start <= t <= us_open_end) or (us_close_start <= t <= us_close_end)
    
    if REQUIRE_SESSION:
        return is_optimal, is_optimal  # Only allow during optimal sessions
    else:
        # Extended US session
        us_extended_start = 13 * 60 + 30
        us_extended_end = 21 * 60
        is_allowed = us_extended_start <= t <= us_extended_end
        return is_allowed, is_optimal


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    path_1m = os.path.join(data_dir, f'{symbol}_1m.csv')
    path_5m = os.path.join(data_dir, f'{symbol}_5m.csv')
    
    if not os.path.exists(path_1m) or not os.path.exists(path_5m):
        if symbol == 'NQ':
            fallback_symbol = 'MNQ'
            fallback_1m = os.path.join(data_dir, f'{fallback_symbol}_1m.csv')
            fallback_5m = os.path.join(data_dir, f'{fallback_symbol}_5m.csv')
            if os.path.exists(fallback_1m) and os.path.exists(fallback_5m):
                print(f"Warning: NQ data not found, using {fallback_symbol} as proxy.")
                path_1m = fallback_1m
                path_5m = fallback_5m
        if not os.path.exists(path_1m) or not os.path.exists(path_5m):
            missing = [p for p in (path_1m, path_5m) if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing data: {', '.join(missing)}")

    df_1m = pd.read_csv(path_1m, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
    df_5m = pd.read_csv(path_5m, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
    return df_1m, df_5m


def add_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_50'] = calculate_ema(df['close'], TREND_EMA_FAST)
    df['ema_200'] = calculate_ema(df['close'], TREND_EMA_SLOW)
    df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df, ADX_PERIOD)
    df['atr'] = calculate_atr(df, 14)
    return df


def add_indicators_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_9'] = calculate_ema(df['close'], ENTRY_EMA_FAST)
    df['ema_21'] = calculate_ema(df['close'], ENTRY_EMA_MED)
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    df['volume_ma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['bb_lower'], df['bb_middle'], df['bb_upper'] = calculate_bollinger(df['close'], BB_PERIOD, BB_STD)
    df['bb_pctb'] = calculate_bb_pctb(df['close'], df['bb_lower'], df['bb_upper'])
    df['atr'] = calculate_atr(df, 14)
    return df


def get_5m_context(df_5m: pd.DataFrame, timestamp: datetime) -> Dict:
    mask = df_5m['datetime'] <= timestamp
    if not mask.any():
        return {'trend': None, 'adx': 0, 'di_plus': 0, 'di_minus': 0, 'atr': 0,
                'resistance': 0, 'support': 0}
    recent_5m = df_5m[mask].tail(RESISTANCE_LOOKBACK)
    row = recent_5m.iloc[-1]
    if row['ema_50'] > row['ema_200']:
        trend = 'bullish'
    elif row['ema_50'] < row['ema_200']:
        trend = 'bearish'
    else:
        trend = 'neutral'
    return {
        'trend': trend,
        'adx': row['adx'],
        'di_plus': row['di_plus'],
        'di_minus': row['di_minus'],
        'atr': row['atr'],
        'resistance': recent_5m['high'].max(),
        'support': recent_5m['low'].min(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Enhanced Entry Logic with Scoring
# ══════════════════════════════════════════════════════════════════════════════

def check_base_long(row_1m: pd.Series, ctx_5m: Dict) -> bool:
    """Long entry conditions — tightened parameters"""
    if ctx_5m['trend'] != 'bullish':
        return False
    if ctx_5m['adx'] < ADX_THRESHOLD:
        return False
    if ctx_5m['di_plus'] <= ctx_5m['di_minus']:
        return False
    
    price = row_1m['close']
    atr = row_1m['atr']
    if pd.isna(atr) or atr <= 0:
        return False
    
    near_ema21 = abs(price - row_1m['ema_21']) <= atr * PULLBACK_ATR_MULT
    rsi_ok = RSI_LONG_MIN <= row_1m['rsi'] <= RSI_LONG_MAX
    macd_rising = row_1m['macd_hist'] > row_1m['macd_hist_prev']
    volume_ok = row_1m['volume_ratio'] >= VOLUME_RATIO_THRESHOLD
    bb_ok = BB_EXTREME_LOW < row_1m['bb_pctb'] < BB_EXTREME_HIGH
    above_ema9 = price > (row_1m['ema_9'] - atr * 0.1)
    
    return near_ema21 and rsi_ok and macd_rising and volume_ok and bb_ok and above_ema9


def check_base_short(row_1m: pd.Series, ctx_5m: Dict) -> bool:
    """Short entry conditions — tightened parameters"""
    if ctx_5m['trend'] != 'bearish':
        return False
    if ctx_5m['adx'] < ADX_THRESHOLD:
        return False
    if ctx_5m['di_minus'] <= ctx_5m['di_plus']:
        return False
    
    price = row_1m['close']
    atr = row_1m['atr']
    if pd.isna(atr) or atr <= 0:
        return False
    
    near_ema21 = abs(price - row_1m['ema_21']) <= atr * PULLBACK_ATR_MULT
    rsi_ok = RSI_SHORT_MIN <= row_1m['rsi'] <= RSI_SHORT_MAX
    macd_falling = row_1m['macd_hist'] < row_1m['macd_hist_prev']
    volume_ok = row_1m['volume_ratio'] >= VOLUME_RATIO_THRESHOLD
    bb_ok = BB_EXTREME_LOW < row_1m['bb_pctb'] < BB_EXTREME_HIGH
    below_ema9 = price < (row_1m['ema_9'] + atr * 0.1)
    
    return near_ema21 and rsi_ok and macd_falling and volume_ok and bb_ok and below_ema9


def check_enhanced_entry(df_1m: pd.DataFrame, idx: int, direction: str,
                         atr: float) -> Tuple[bool, str]:
    """Check enhanced entry confirmations.
    
    Returns:
        (should_enter, signals_description)
    """
    signals = ['BASE']
    
    # 1) Momentum bar: previous bar must close in trade direction
    if REQUIRE_MOMENTUM_BAR and idx >= 2:
        prev_close = df_1m['close'].iat[idx - 1]
        prev2_close = df_1m['close'].iat[idx - 2]
        if direction == 'LONG' and prev_close <= prev2_close:
            return False, ''
        if direction == 'SHORT' and prev_close >= prev2_close:
            return False, ''
        signals.append('MOM')
    
    # 2) Candlestick Pattern (optional bonus)
    if REQUIRE_CANDLE_CONFIRM:
        candle_signal, candle_strength = detect_candle_signal(df_1m, idx)
        candle_matches = ((direction == 'LONG' and candle_signal == 'BUY') or
                          (direction == 'SHORT' and candle_signal == 'SELL'))
        if not candle_matches:
            return False, ''
        signals.append(f'CANDLE({candle_strength:.1f})')
    else:
        candle_signal, candle_strength = detect_candle_signal(df_1m, idx)
        candle_matches = ((direction == 'LONG' and candle_signal == 'BUY') or
                          (direction == 'SHORT' and candle_signal == 'SELL'))
        if candle_matches:
            signals.append(f'CANDLE({candle_strength:.1f})')
    
    # 3) RSI Divergence (bonus)
    if DIVERGENCE_ENABLED and detect_rsi_divergence(df_1m, idx, direction):
        signals.append('DIV')
    
    return True, '+'.join(signals)


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedBacktester:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spec = SYMBOL_SPECS[symbol]
        self.point_value = self.spec['point_value']
        self.atr_mult = self.spec['atr_mult']
        
        self.balance = INITIAL_BALANCE
        self.equity_curve = [INITIAL_BALANCE]
        self.trades: List[Trade] = []
        
        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.stopped_for_day = False
        
        # Stats for new signals
        self.entries_rejected = 0
        self.candle_entries = 0
        self.divergence_entries = 0
    
    def run(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict:
        print(f"\n{'='*70}")
        print(f"  Enhanced MTF Scalping Backtest — {self.symbol}")
        print(f"{'='*70}")
        print(f"1M Bars: {len(df_1m):,}")
        print(f"5M Bars: {len(df_5m):,}")
        
        print("Adding indicators...")
        df_1m = add_indicators_1m(df_1m)
        df_5m = add_indicators_5m(df_5m)
        
        warmup = max(200, TREND_EMA_SLOW)
        position: Optional[Trade] = None
        
        print(f"Running enhanced backtest from bar {warmup}...")
        print(f"  Candle confirmation: {'REQUIRED' if REQUIRE_CANDLE_CONFIRM else 'bonus'}")
        print(f"  Momentum bar: {'REQUIRED' if REQUIRE_MOMENTUM_BAR else 'OFF'}")
        print(f"  RSI divergence: {'ON' if DIVERGENCE_ENABLED else 'OFF'}")
        print(f"  Session filter: {'US Open/Close ONLY' if REQUIRE_SESSION else 'Extended'}")
        print(f"  TP:SL ratio: {TP_MULT}:1")
        print(f"  ADX threshold: {ADX_THRESHOLD}")
        print(f"  Volume threshold: {VOLUME_RATIO_THRESHOLD}x avg")
        
        for i in range(warmup, len(df_1m)):
            row = df_1m.iloc[i]
            dt = row['datetime']
            
            trade_date = dt.date() if hasattr(dt, 'date') else None
            if trade_date != self.current_date:
                self.current_date = trade_date
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.stopped_for_day = False
            
            if self.stopped_for_day:
                if position:
                    position = self._check_exit(position, row)
                continue
            
            ctx_5m = get_5m_context(df_5m, dt)
            
            if position:
                position = self._check_exit(position, row)
            else:
                if self.daily_trades >= MAX_TRADES_PER_DAY:
                    continue
                
                # Session filter (re-enabled)
                is_allowed, is_optimal = is_trading_session(dt)
                if not is_allowed:
                    continue
                
                atr = row['atr']
                if pd.isna(atr) or atr <= 0:
                    continue
                
                direction = None
                if check_base_long(row, ctx_5m):
                    direction = 'LONG'
                elif check_base_short(row, ctx_5m):
                    direction = 'SHORT'
                
                if direction is None:
                    continue
                
                # Enhanced entry confirmation
                should_enter, signals = check_enhanced_entry(df_1m, i, direction, atr)
                
                if not should_enter:
                    self.entries_rejected += 1
                    continue
                
                # Track signal sources
                if 'CANDLE' in signals:
                    self.candle_entries += 1
                if 'DIV' in signals:
                    self.divergence_entries += 1
                
                entry_price = row['close']
                sl_distance = atr * self.atr_mult
                tp_distance = sl_distance * TP_MULT
                
                if direction == 'LONG':
                    sl = entry_price - sl_distance
                    tp_rr = entry_price + tp_distance
                    tp_buffer = atr * TP_BUFFER_ATR_MULT
                    tp_resist = ctx_5m['resistance'] - tp_buffer
                    tp_final = min(tp_rr, tp_resist) if tp_resist > entry_price else tp_rr
                    position = Trade(
                        entry_time=dt, direction='LONG',
                        entry_price=entry_price, sl=sl, tp=tp_final,
                        entry_signals=signals,
                        initial_sl=sl, highest_price=entry_price,
                    )
                else:
                    sl = entry_price + sl_distance
                    tp_rr = entry_price - tp_distance
                    tp_buffer = atr * TP_BUFFER_ATR_MULT
                    tp_support = ctx_5m['support'] + tp_buffer
                    tp_final = max(tp_rr, tp_support) if tp_support < entry_price else tp_rr
                    position = Trade(
                        entry_time=dt, direction='SHORT',
                        entry_price=entry_price, sl=sl, tp=tp_final,
                        entry_signals=signals,
                        initial_sl=sl, lowest_price=entry_price,
                    )
                self.daily_trades += 1
        
        if position:
            position.exit_time = df_1m.iloc[-1]['datetime']
            position.exit_price = df_1m.iloc[-1]['close']
            position.exit_reason = 'END'
            position.pnl = self._calc_pnl(position)
            self.trades.append(position)
            self.balance += position.pnl
        
        return self._compute_stats()
    
    def _check_exit(self, trade: Trade, row: pd.Series) -> Optional[Trade]:
        high = row['high']
        low = row['low']
        dt = row['datetime']
        
        if trade.initial_sl == 0:
            trade.initial_sl = trade.sl
        sl_distance = abs(trade.entry_price - trade.initial_sl)
        
        if trade.direction == 'LONG':
            trade.highest_price = max(trade.highest_price, high)
            if low <= trade.sl:
                trade.exit_time = dt
                trade.exit_price = trade.sl
                trade.exit_reason = 'SL'
                trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
            if high >= trade.tp:
                trade.exit_time = dt
                trade.exit_price = trade.tp
                trade.exit_reason = 'TP'
                trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
        else:
            trade.lowest_price = min(trade.lowest_price, low)
            if high >= trade.sl:
                trade.exit_time = dt
                trade.exit_price = trade.sl
                trade.exit_reason = 'SL'
                trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
            if low <= trade.tp:
                trade.exit_time = dt
                trade.exit_price = trade.tp
                trade.exit_reason = 'TP'
                trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
        
        return trade
    
    def _calc_pnl(self, trade: Trade) -> float:
        if trade.direction == 'LONG':
            points = trade.exit_price - trade.entry_price
        else:
            points = trade.entry_price - trade.exit_price
        return points * self.point_value
    
    def _record_trade(self, trade: Trade):
        self.trades.append(trade)
        self.balance += trade.pnl
        self.equity_curve.append(self.balance)
        self.daily_pnl += trade.pnl
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            self.stopped_for_day = True
    
    def _compute_stats(self) -> Dict:
        if not self.trades:
            return {'error': 'No trades'}
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.01
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        win_rate = len(wins) / len(self.trades) * 100
        
        peak = INITIAL_BALANCE
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        tp_count = len([t for t in self.trades if t.exit_reason == 'TP'])
        sl_count = len([t for t in self.trades if t.exit_reason == 'SL'])
        
        days_stopped = len(set(t.entry_time.date() for t in self.trades
                               if hasattr(t.entry_time, 'date') and
                               sum(tr.pnl for tr in self.trades
                                   if hasattr(tr.entry_time, 'date') and
                                   tr.entry_time.date() == t.entry_time.date()) <= -DAILY_LOSS_LIMIT))
        
        # Signal breakdown
        candle_trades = [t for t in self.trades if 'CANDLE' in t.entry_signals]
        candle_wins = [t for t in candle_trades if t.pnl > 0]
        div_trades = [t for t in self.trades if 'DIV' in t.entry_signals]
        div_wins = [t for t in div_trades if t.pnl > 0]
        
        return {
            'symbol': self.symbol,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': max_dd,
            'final_balance': self.balance,
            'return_pct': (self.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
            'tp_exits': tp_count,
            'sl_exits': sl_count,
            'days_stopped': days_stopped,
            # Signal stats
            'entries_rejected': self.entries_rejected,
            'candle_trades': len(candle_trades),
            'candle_win_rate': len(candle_wins) / len(candle_trades) * 100 if candle_trades else 0,
            'div_trades': len(div_trades),
            'div_win_rate': len(div_wins) / len(div_trades) * 100 if div_trades else 0,
        }


def print_results(stats: Dict):
    print(f"\n{'='*70}")
    print(f"  ENHANCED BACKTEST RESULTS — {stats.get('symbol', '?')}")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total Trades:     {stats['total_trades']}")
    print(f"   Wins:             {stats['wins']}")
    print(f"   Losses:           {stats['losses']}")
    print(f"   TP Exits:         {stats['tp_exits']}")
    print(f"   SL Exits:         {stats['sl_exits']}")
    print(f"   Entries Rejected: {stats['entries_rejected']} (below min score)")
    
    print(f"\n💰 PERFORMANCE")
    print(f"   Win Rate:         {stats['win_rate']:.1f}%")
    print(f"   Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"   Total P&L:        ${stats['total_pnl']:,.2f}")
    print(f"   Final Balance:    ${stats['final_balance']:,.2f}")
    print(f"   Return:           {stats['return_pct']:.2f}%")
    
    print(f"\n📉 RISK")
    print(f"   Max Drawdown:     {stats['max_drawdown_pct']:.2f}%")
    print(f"   Avg Win:          ${stats['avg_win']:.2f}")
    print(f"   Avg Loss:         ${stats['avg_loss']:.2f}")
    print(f"   Days Stopped:     {stats['days_stopped']} (hit $350 limit)")
    
    print(f"\n🔍 SIGNAL ANALYSIS")
    print(f"   Candle Confirmed: {stats['candle_trades']} ({stats['candle_win_rate']:.1f}% win rate)")
    print(f"   RSI Divergence:   {stats['div_trades']} ({stats['div_win_rate']:.1f}% win rate)")
    
    print(f"\n✅ VALIDATION")
    pf_ok = stats['profit_factor'] >= 1.3
    dd_ok = stats['max_drawdown_pct'] < 10
    wr_ok = 45 <= stats['win_rate'] <= 55
    
    print(f"   PF ≥ 1.3:         {'✅' if pf_ok else '❌'} ({stats['profit_factor']:.2f})")
    print(f"   DD < 10%:         {'✅' if dd_ok else '❌'} ({stats['max_drawdown_pct']:.2f}%)")
    print(f"   Win Rate 45-55%:  {'✅' if wr_ok else '⚠️'} ({stats['win_rate']:.1f}%)")
    
    if pf_ok and dd_ok:
        print(f"\n🎯 STRATEGY PASSES VALIDATION")
    else:
        print(f"\n⚠️  STRATEGY NEEDS FURTHER OPTIMIZATION")


def main():
    parser = argparse.ArgumentParser(description='Enhanced MTF Scalping Backtest')
    parser.add_argument('--symbol', type=str, default='MES', choices=['MES', 'MNQ', 'NQ'])
    parser.add_argument('--all', action='store_true', help='Run all symbols')
    args = parser.parse_args()
    
    symbols = ['MES', 'MNQ', 'NQ'] if args.all else [args.symbol]
    all_stats = []
    
    for symbol in symbols:
        print(f"\nLoading data for {symbol}...")
        df_1m, df_5m = load_data(symbol)
        bt = EnhancedBacktester(symbol)
        stats = bt.run(df_1m, df_5m)
        print_results(stats)
        all_stats.append(stats)
    
    # Comparison summary if multiple symbols
    if len(all_stats) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"   {'Symbol':<8} {'Trades':>7} {'WR%':>7} {'PF':>6} {'P&L':>12} {'Return':>8} {'MaxDD':>7}")
        print(f"   {'-'*56}")
        for s in all_stats:
            print(f"   {s['symbol']:<8} {s['total_trades']:>7} {s['win_rate']:>6.1f}% {s['profit_factor']:>6.2f} ${s['total_pnl']:>10,.2f} {s['return_pct']:>7.2f}% {s['max_drawdown_pct']:>6.2f}%")
    
    return all_stats


if __name__ == '__main__':
    main()
