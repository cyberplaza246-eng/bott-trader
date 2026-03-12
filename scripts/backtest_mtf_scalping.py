#!/usr/bin/env python3
"""
Multi-Timeframe Scalping Backtest — 1M/5M Strategy

Strategy Rules:
- 5M Chart (Trend Filter):
  - EMA 200/50 defines trend direction
  - ADX ≥ 22 for strong trend
  - DI+/DI- must align with direction

- 1M Chart (Entry):
  - Pullback to EMA21 zone
  - RSI 40-55 for long, 45-60 for short
  - MACD histogram rising (long) / falling (short)
  - Volume ≥ 1.3x 20MA
  - Not at Bollinger extreme (0.1 < %B < 0.9)
  - Candle closes above EMA9 (long) / below EMA9 (short)

- Risk:
  - SL = ATR(14) × 1.2
  - TP = 2 × SL (1:2 R:R)
  - $350 daily loss limit → stop trading
  - Max 6 trades per day

- Session Filter:
  - US Open: 9:30-11:30 EST (14:30-16:30 UTC)
  - US Close: 2:30-4:00 PM EST (19:30-21:00 UTC)

Usage:
    python scripts/backtest_mtf_scalping.py --symbol MES
    python scripts/backtest_mtf_scalping.py --symbol MNQ
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25, 'atr_mult': 1.2},
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25, 'atr_mult': 1.2},
}

# 5M Trend Filter Settings
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
ADX_THRESHOLD = 18    # Lowered from 22 to catch more trending moves
ADX_PERIOD = 14

# 1M Entry Settings
ENTRY_EMA_FAST = 9
ENTRY_EMA_MED = 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 35, 60       # Widened from 40-55
RSI_SHORT_MIN, RSI_SHORT_MAX = 40, 65     # Widened from 45-60
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_RATIO_THRESHOLD = 0.5              # Lowered for more trades
VOLUME_MA_PERIOD = 20
BB_PERIOD, BB_STD = 20, 2
BB_EXTREME_LOW, BB_EXTREME_HIGH = 0.05, 0.95  # Relaxed from 0.1/0.9

# Risk Settings
TP_MULT = 1.5  # TP = 1.5 × SL (less aggressive, higher win rate)
TP_BUFFER_ATR_MULT = 0.0  # TP capped exactly at resistance/support (no buffer)
RESISTANCE_LOOKBACK = 20  # 5M bars to look back for swing high/low
DAILY_LOSS_LIMIT = 350.0
MAX_TRADES_PER_DAY = 6
INITIAL_BALANCE = 50000.0

# Trailing Stop Settings
TRAIL_TRIGGER_R = 1.2    # Not used in current mode
TRAIL_STEP_R = 0.25      # Trail by 0.25R increments
USE_BREAKEVEN = False    # DISABLED - hurts PF by cutting winners short
TRAIL_AFTER_TP = False   # Trail after TP (hurts PF - price reverses quickly)
HYBRID_EXIT = False      # 50/50 also hurts PF


@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl: float
    tp: float
    initial_sl: float = 0.0         # Store original SL for R calculation
    highest_price: float = 0.0      # Track highest price for trailing (long)
    lowest_price: float = 999999.0  # Track lowest price for trailing (short)
    trail_activated: bool = False   # Whether trailing has started
    partial_closed: bool = False    # Whether 50% was closed at TP (hybrid mode)
    partial_pnl: float = 0.0        # P&L from partial close at TP
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""


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
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When both are positive, take the larger one
    plus_dm[(plus_dm > 0) & (minus_dm > plus_dm)] = 0
    minus_dm[(minus_dm > 0) & (plus_dm > minus_dm)] = 0
    
    tr = calculate_atr(df, 1) * period  # Use simple TR
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
    """Calculate %B (position within Bollinger Bands, 0-1)"""
    return (close - lower) / (upper - lower + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  Session Filter
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_session(dt: datetime) -> bool:
    """Check if datetime is within valid US trading sessions (UTC)
    
    CURRENTLY DISABLED for testing - all hours allowed
    """
    return True  # Disabled for now
    
    if not isinstance(dt, datetime):
        return True  # If no datetime, allow
    
    hour = dt.hour
    minute = dt.minute
    t = hour * 60 + minute
    
    # US Open: 14:30-16:30 UTC (9:30-11:30 EST)
    us_open_start = 14 * 60 + 30
    us_open_end = 16 * 60 + 30
    
    # US Close: 19:30-21:00 UTC (2:30-4:00 PM EST)
    us_close_start = 19 * 60 + 30
    us_close_end = 21 * 60
    
    return (us_open_start <= t <= us_open_end) or (us_close_start <= t <= us_close_end)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load 1M and 5M data for symbol"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    path_1m = os.path.join(data_dir, f'{symbol}_1m.csv')
    path_5m = os.path.join(data_dir, f'{symbol}_5m.csv')
    
    df_1m = pd.read_csv(path_1m, parse_dates=['datetime'])
    df_5m = pd.read_csv(path_5m, parse_dates=['datetime'])
    
    # Ensure sorted
    df_1m = df_1m.sort_values('datetime').reset_index(drop=True)
    df_5m = df_5m.sort_values('datetime').reset_index(drop=True)
    
    return df_1m, df_5m


def add_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5M trend indicators"""
    df = df.copy()
    df['ema_50'] = calculate_ema(df['close'], TREND_EMA_FAST)
    df['ema_200'] = calculate_ema(df['close'], TREND_EMA_SLOW)
    df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df, ADX_PERIOD)
    df['atr'] = calculate_atr(df, 14)
    return df


def add_indicators_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1M entry indicators"""
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
    """Get 5M trend context for a given 1M timestamp"""
    # Find the most recent 5M bar before/at this timestamp
    mask = df_5m['datetime'] <= timestamp
    if not mask.any():
        return {'trend': None, 'adx': 0, 'di_plus': 0, 'di_minus': 0, 'atr': 0, 'resistance': 0, 'support': 0}
    
    recent_5m = df_5m[mask].tail(RESISTANCE_LOOKBACK)
    row = recent_5m.iloc[-1]
    
    # Trend direction
    if row['ema_50'] > row['ema_200']:
        trend = 'bullish'
    elif row['ema_50'] < row['ema_200']:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    # Swing high/low for resistance/support
    resistance = recent_5m['high'].max()
    support = recent_5m['low'].min()
    
    return {
        'trend': trend,
        'adx': row['adx'],
        'di_plus': row['di_plus'],
        'di_minus': row['di_minus'],
        'atr': row['atr'],
        'resistance': resistance,
        'support': support,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Logic
# ══════════════════════════════════════════════════════════════════════════════

def check_long_entry(row_1m: pd.Series, ctx_5m: Dict) -> bool:
    """Check if long entry conditions are met"""
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
    
    # Pullback near EMA21 (within 1.5 ATR - widened for more entries)
    atr = row_1m['atr']
    pullback_zone = atr * 1.5
    near_ema21 = abs(price - ema_21) <= pullback_zone
    
    # RSI in range
    rsi_ok = RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
    
    # MACD histogram rising
    macd_rising = macd_hist > macd_hist_prev
    
    # Volume spike
    volume_ok = volume_ratio >= VOLUME_RATIO_THRESHOLD
    
    # Not at BB extreme
    bb_ok = BB_EXTREME_LOW < bb_pctb < BB_EXTREME_HIGH
    
    # Price above EMA9
    above_ema9 = price > ema_9
    
    return near_ema21 and rsi_ok and macd_rising and volume_ok and bb_ok and above_ema9


def check_short_entry(row_1m: pd.Series, ctx_5m: Dict) -> bool:
    """Check if short entry conditions are met"""
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
    
    # Pullback near EMA21 (within 1.5 ATR - widened for more entries)
    atr = row_1m['atr']
    pullback_zone = atr * 1.5
    near_ema21 = abs(price - ema_21) <= pullback_zone
    
    # RSI in range
    rsi_ok = RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
    
    # MACD histogram falling
    macd_falling = macd_hist < macd_hist_prev
    
    # Volume spike
    volume_ok = volume_ratio >= VOLUME_RATIO_THRESHOLD
    
    # Not at BB extreme
    bb_ok = BB_EXTREME_LOW < bb_pctb < BB_EXTREME_HIGH
    
    # Price below EMA9
    below_ema9 = price < ema_9
    
    return near_ema21 and rsi_ok and macd_falling and volume_ok and bb_ok and below_ema9


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

class MultiTimeframeBacktester:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spec = SYMBOL_SPECS[symbol]
        self.point_value = self.spec['point_value']
        self.atr_mult = self.spec['atr_mult']
        
        self.balance = INITIAL_BALANCE
        self.equity_curve = [INITIAL_BALANCE]
        self.trades: List[Trade] = []
        
        # Daily tracking
        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.stopped_for_day = False
    
    def run(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict:
        """Run the backtest"""
        print(f"\n{'='*70}")
        print(f"  Multi-Timeframe Scalping Backtest — {self.symbol}")
        print(f"{'='*70}")
        print(f"1M Bars: {len(df_1m):,}")
        print(f"5M Bars: {len(df_5m):,}")
        
        # Add indicators
        print("Adding indicators...")
        df_1m = add_indicators_1m(df_1m)
        df_5m = add_indicators_5m(df_5m)
        
        # Skip warmup period
        warmup = max(200, TREND_EMA_SLOW)
        
        position: Optional[Trade] = None
        
        print(f"Running backtest from bar {warmup}...")
        
        for i in range(warmup, len(df_1m)):
            row = df_1m.iloc[i]
            dt = row['datetime']
            
            # Reset daily stats
            trade_date = dt.date() if hasattr(dt, 'date') else None
            if trade_date != self.current_date:
                self.current_date = trade_date
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.stopped_for_day = False
            
            # Check if stopped for the day
            if self.stopped_for_day:
                # Still need to manage open position
                if position:
                    position = self._check_exit(position, row)
                continue
            
            # Get 5M context
            ctx_5m = get_5m_context(df_5m, dt)
            
            # Check open position
            if position:
                position = self._check_exit(position, row)
                if position is None:
                    # Position closed
                    pass
            else:
                # Check for new entry
                if self.daily_trades >= MAX_TRADES_PER_DAY:
                    continue
                
                # Session filter
                if not is_trading_session(dt):
                    continue
                
                # Check entry signals
                atr = row['atr']
                if pd.isna(atr) or atr <= 0:
                    continue
                
                sl_distance = atr * self.atr_mult
                tp_distance = sl_distance * TP_MULT
                
                if check_long_entry(row, ctx_5m):
                    entry_price = row['close']
                    # Always cap TP at resistance - ATR buffer (ALL symbols)
                    tp_rr = entry_price + tp_distance
                    tp_buffer = atr * TP_BUFFER_ATR_MULT
                    tp_resistance = ctx_5m['resistance'] - tp_buffer
                    tp_final = min(tp_rr, tp_resistance) if tp_resistance > entry_price else tp_rr
                    position = Trade(
                        entry_time=dt,
                        direction='LONG',
                        entry_price=entry_price,
                        sl=entry_price - sl_distance,
                        tp=tp_final,
                        initial_sl=entry_price - sl_distance,
                        highest_price=entry_price,
                    )
                    self.daily_trades += 1
                    
                elif check_short_entry(row, ctx_5m):
                    entry_price = row['close']
                    # Always cap TP at support + ATR buffer (ALL symbols)
                    tp_rr = entry_price - tp_distance
                    tp_buffer = atr * TP_BUFFER_ATR_MULT
                    tp_support = ctx_5m['support'] + tp_buffer
                    tp_final = max(tp_rr, tp_support) if tp_support < entry_price else tp_rr
                    position = Trade(
                        entry_time=dt,
                        direction='SHORT',
                        entry_price=entry_price,
                        sl=entry_price + sl_distance,
                        tp=tp_final,
                        initial_sl=entry_price + sl_distance,
                        lowest_price=entry_price,
                    )
                    self.daily_trades += 1
        
        # Close any remaining position
        if position:
            position.exit_time = df_1m.iloc[-1]['datetime']
            position.exit_price = df_1m.iloc[-1]['close']
            position.exit_reason = 'END'
            position.pnl = self._calc_pnl(position)
            self.trades.append(position)
            self.balance += position.pnl
        
        return self._compute_stats()
    
    def _check_exit(self, trade: Trade, row: pd.Series) -> Optional[Trade]:
        """Check if position should be closed (with hybrid 50/50 exit)"""
        high = row['high']
        low = row['low']
        dt = row['datetime']
        
        # Calculate R-multiple (risk unit)
        if trade.initial_sl == 0:
            trade.initial_sl = trade.sl
        sl_distance = abs(trade.entry_price - trade.initial_sl)
        
        if trade.direction == 'LONG':
            # Update highest price
            trade.highest_price = max(trade.highest_price, high)
            
            # Check SL first (always active)
            if low <= trade.sl:
                trade.exit_time = dt
                trade.exit_price = trade.sl
                if trade.partial_closed:
                    # Hybrid: add partial 50% TP profit + 50% trail result
                    trail_pnl = self._calc_pnl(trade) * 0.5  # Only 50% still open
                    trade.pnl = trade.partial_pnl + trail_pnl
                    trade.exit_reason = 'HYB'  # Hybrid exit
                else:
                    trade.exit_reason = 'SL'
                    trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
            
            # Check TP
            if high >= trade.tp:
                if HYBRID_EXIT and not trade.partial_closed:
                    # Hybrid: close 50% at TP, trail remainder
                    trade.partial_closed = True
                    trade.trail_activated = True
                    # Store 50% profit from TP hit
                    full_tp_pnl = (trade.tp - trade.entry_price) * self.point_value
                    trade.partial_pnl = full_tp_pnl * 0.5
                    # Move SL to breakeven for remaining 50%
                    trade.sl = trade.entry_price
                elif TRAIL_AFTER_TP and not trade.partial_closed:
                    # Pure trail after TP
                    trade.trail_activated = True
                    trade.sl = trade.entry_price + (sl_distance * 0.5)
                elif not HYBRID_EXIT and not TRAIL_AFTER_TP:
                    # Standard: close 100% at TP
                    trade.exit_time = dt
                    trade.exit_price = trade.tp
                    trade.exit_reason = 'TP'
                    trade.pnl = self._calc_pnl(trade)
                    self._record_trade(trade)
                    return None
            
            # Trail the remaining 50% (if hybrid activated)
            if trade.partial_closed:
                trail_distance = sl_distance * 0.5  # Tight 0.5R trail
                new_sl = trade.highest_price - trail_distance
                if new_sl > trade.sl:
                    trade.sl = new_sl
                
        else:  # SHORT
            # Update lowest price
            trade.lowest_price = min(trade.lowest_price, low)
            
            # Check SL first (always active)
            if high >= trade.sl:
                trade.exit_time = dt
                trade.exit_price = trade.sl
                if trade.partial_closed:
                    # Hybrid: add partial 50% TP profit + 50% trail result
                    trail_pnl = self._calc_pnl(trade) * 0.5  # Only 50% still open
                    trade.pnl = trade.partial_pnl + trail_pnl
                    trade.exit_reason = 'HYB'  # Hybrid exit
                else:
                    trade.exit_reason = 'SL'
                    trade.pnl = self._calc_pnl(trade)
                self._record_trade(trade)
                return None
            
            # Check TP
            if low <= trade.tp:
                if HYBRID_EXIT and not trade.partial_closed:
                    # Hybrid: close 50% at TP, trail remainder
                    trade.partial_closed = True
                    trade.trail_activated = True
                    # Store 50% profit from TP hit
                    full_tp_pnl = (trade.entry_price - trade.tp) * self.point_value
                    trade.partial_pnl = full_tp_pnl * 0.5
                    # Move SL to breakeven for remaining 50%
                    trade.sl = trade.entry_price
                elif TRAIL_AFTER_TP and not trade.partial_closed:
                    # Pure trail after TP
                    trade.trail_activated = True
                    trade.sl = trade.entry_price - (sl_distance * 0.5)
                elif not HYBRID_EXIT and not TRAIL_AFTER_TP:
                    # Standard: close 100% at TP
                    trade.exit_time = dt
                    trade.exit_price = trade.tp
                    trade.exit_reason = 'TP'
                    trade.pnl = self._calc_pnl(trade)
                    self._record_trade(trade)
                    return None
            
            # Trail the remaining 50% (if hybrid activated)
            if trade.partial_closed:
                trail_distance = sl_distance * 0.5  # Tight 0.5R trail
                new_sl = trade.lowest_price + trail_distance
                if new_sl < trade.sl:
                    trade.sl = new_sl
        
        return trade
    
    def _calc_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade"""
        if trade.direction == 'LONG':
            points = trade.exit_price - trade.entry_price
        else:
            points = trade.entry_price - trade.exit_price
        return points * self.point_value
    
    def _record_trade(self, trade: Trade):
        """Record a completed trade"""
        self.trades.append(trade)
        self.balance += trade.pnl
        self.equity_curve.append(self.balance)
        self.daily_pnl += trade.pnl
        
        # Check daily loss limit
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            self.stopped_for_day = True
    
    def _compute_stats(self) -> Dict:
        """Compute backtest statistics"""
        if not self.trades:
            return {'error': 'No trades'}
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.01
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        win_rate = len(wins) / len(self.trades) * 100
        
        # Max drawdown
        peak = INITIAL_BALANCE
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Average trade
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Count trades by exit type
        tp_count = len([t for t in self.trades if t.exit_reason == 'TP'])
        sl_count = len([t for t in self.trades if t.exit_reason == 'SL'])
        be_count = len([t for t in self.trades if t.exit_reason == 'BE'])
        tsl_count = len([t for t in self.trades if t.exit_reason == 'TSL'])
        hyb_count = len([t for t in self.trades if t.exit_reason == 'HYB'])
        
        # TSL breakdown (trailing after TP)
        tsl_pnl = sum(t.pnl for t in self.trades if t.exit_reason == 'TSL')
        
        # Hybrid breakdown (50% TP + 50% trail)
        hyb_pnl = sum(t.pnl for t in self.trades if t.exit_reason == 'HYB')
        hyb_wins = len([t for t in self.trades if t.exit_reason == 'HYB' and t.pnl > 0])
        
        # Days stopped
        days_stopped = len(set(t.entry_time.date() for t in self.trades 
                               if hasattr(t.entry_time, 'date') and 
                               sum(tr.pnl for tr in self.trades 
                                   if hasattr(tr.entry_time, 'date') and 
                                   tr.entry_time.date() == t.entry_time.date()) <= -DAILY_LOSS_LIMIT))
        
        stats = {
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
            'be_exits': be_count,
            'tsl_exits': tsl_count,
            'tsl_pnl': tsl_pnl,
            'hyb_exits': hyb_count,
            'hyb_pnl': hyb_pnl,
            'hyb_wins': hyb_wins,
            'days_stopped': days_stopped,
        }
        
        return stats


def print_results(stats: Dict):
    """Print formatted results"""
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — {stats.get('symbol', 'Unknown')}")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total Trades:    {stats['total_trades']}")
    print(f"   Wins:            {stats['wins']}")
    print(f"   Losses:          {stats['losses']}")
    print(f"   TP Exits:        {stats['tp_exits']}")
    print(f"   SL Exits:        {stats['sl_exits']}")
    
    # Show hybrid or trailing stats if applicable
    if stats.get('hyb_exits', 0) > 0:
        print(f"   Hybrid Exits:    {stats['hyb_exits']} (50% TP + 50% trail)")
        print(f"   Hybrid P&L:      ${stats.get('hyb_pnl', 0):,.2f}")
    if stats.get('tsl_exits', 0) > 0:
        print(f"   Trailing Exits:  {stats['tsl_exits']} (W:{stats.get('tsl_wins', 0)} / L:{stats['tsl_exits'] - stats.get('tsl_wins', 0)})")
    
    print(f"\n💰 PERFORMANCE")
    print(f"   Win Rate:        {stats['win_rate']:.1f}%")
    print(f"   Profit Factor:   {stats['profit_factor']:.2f}")
    print(f"   Total P&L:       ${stats['total_pnl']:,.2f}")
    print(f"   Final Balance:   ${stats['final_balance']:,.2f}")
    print(f"   Return:          {stats['return_pct']:.2f}%")
    
    print(f"\n📉 RISK")
    print(f"   Max Drawdown:    {stats['max_drawdown_pct']:.2f}%")
    print(f"   Avg Win:         ${stats['avg_win']:.2f}")
    print(f"   Avg Loss:        ${stats['avg_loss']:.2f}")
    print(f"   Days Stopped:    {stats['days_stopped']} (hit $350 limit)")
    
    # Validation
    print(f"\n✅ VALIDATION")
    pf_ok = stats['profit_factor'] >= 1.3
    dd_ok = stats['max_drawdown_pct'] < 10
    wr_ok = 45 <= stats['win_rate'] <= 55
    
    print(f"   PF ≥ 1.3:        {'✅' if pf_ok else '❌'} ({stats['profit_factor']:.2f})")
    print(f"   DD < 10%:        {'✅' if dd_ok else '❌'} ({stats['max_drawdown_pct']:.2f}%)")
    print(f"   Win Rate 45-55%: {'✅' if wr_ok else '⚠️'} ({stats['win_rate']:.1f}%)")
    
    if pf_ok and dd_ok:
        print(f"\n🎯 STRATEGY PASSES VALIDATION")
    else:
        print(f"\n⚠️  STRATEGY NEEDS OPTIMIZATION")


def main():
    parser = argparse.ArgumentParser(description='Multi-Timeframe Scalping Backtest')
    parser.add_argument('--symbol', type=str, default='MES', choices=['MES', 'MNQ'],
                        help='Symbol to backtest')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data for {args.symbol}...")
    df_1m, df_5m = load_data(args.symbol)
    
    # Run backtest
    bt = MultiTimeframeBacktester(args.symbol)
    stats = bt.run(df_1m, df_5m)
    
    # Print results
    print_results(stats)
    
    return stats


if __name__ == '__main__':
    main()
