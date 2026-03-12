"""
Enhanced Scalping Strategy - Improved trend-following with multi-timeframe confirmation.

Key improvements over base strategy:
1. Strong trend filter (EMA200 + EMA50 + RSI alignment)
2. Trading hours filter (avoid lunch lull, trade optimal sessions)
3. Entry scoring system for higher quality trades
4. Regime-aware position sizing
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal:
    """Enhanced signal with dual TP levels."""
    direction: str  # 'long', 'short', 'none'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit1: float  # Partial exit (50%)
    take_profit2: float  # Runner to full target
    timestamp: pd.Timestamp
    entry_score: float = 0.0
    regime: str = 'unknown'
    trend_alignment: bool = False


class EnhancedScalpingStrategy:
    """
    Multi-timeframe scalping with improved filters.
    
    Uses 5M for trend direction, 1M for entry timing.
    """
    
    def __init__(
        self,
        min_score: float = 0.65,
        use_liquidity_sweep: bool = True,
        use_trading_hours: bool = True,
        atr_sl_mult: float = 1.5,
        tp1_ratio: float = 1.5,  # First TP at 1.5R
        tp2_ratio: float = 2.5,  # Runner to 2.5R
    ):
        self.min_score = min_score
        self.use_liquidity_sweep = use_liquidity_sweep
        self.use_trading_hours = use_trading_hours
        self.atr_sl_mult = atr_sl_mult
        self.tp1_ratio = tp1_ratio
        self.tp2_ratio = tp2_ratio
        
    def check_trend_alignment(self, data_5m: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """
        Check if trend is aligned for the trade direction.
        
        Returns (is_aligned, alignment_score)
        """
        if len(data_5m) < 210:
            return False, 0.0
            
        latest = data_5m.iloc[-1]
        price = latest['close']
        
        # Get indicators
        ema200 = latest.get('ema_200', latest.get('ema200'))
        ema50 = latest.get('ema_50', latest.get('ema50'))
        rsi = latest.get('rsi', 50)
        
        if pd.isna(ema200) or pd.isna(ema50):
            return False, 0.0
        
        score = 0.0
        
        if direction == 'long':
            # Price above EMA200
            if price > ema200:
                score += 0.30
            # EMA50 above EMA200 (bullish structure)
            if ema50 > ema200:
                score += 0.30
            # RSI bullish bias
            if rsi > 50:
                score += 0.20
            if rsi > 55:
                score += 0.10
            # Price above EMA50 (immediate trend)
            if price > ema50:
                score += 0.10
                
            aligned = (price > ema200 and ema50 > ema200 and rsi > 50)
            
        else:  # short
            # Price below EMA200
            if price < ema200:
                score += 0.30
            # EMA50 below EMA200 (bearish structure)
            if ema50 < ema200:
                score += 0.30
            # RSI bearish bias
            if rsi < 50:
                score += 0.20
            if rsi < 45:
                score += 0.10
            # Price below EMA50 (immediate trend)
            if price < ema50:
                score += 0.10
                
            aligned = (price < ema200 and ema50 < ema200 and rsi < 50)
        
        return aligned, score
    
    def is_trading_hours(self, dt: datetime = None) -> Tuple[bool, str]:
        """
        Check if current time is within optimal trading hours (EST).
        
        Returns (is_valid, session_name)
        """
        if not self.use_trading_hours:
            return True, 'any'
            
        if dt is None:
            now = datetime.now().time()
        else:
            now = dt.time() if hasattr(dt, 'time') else dt
        
        # Convert to EST (UTC-5) if needed - assume input is UTC
        # Optimal hours for MES/MNQ (EST):
        # - Morning session: 9:30-11:30
        # - Afternoon session: 14:30-16:00
        # - Avoid lunch: 12:00-14:00
        
        # In UTC terms (EST + 5):
        morning_start = time(14, 30)  # 9:30 EST
        morning_end = time(16, 30)    # 11:30 EST
        afternoon_start = time(19, 30) # 14:30 EST
        afternoon_end = time(21, 0)    # 16:00 EST
        lunch_start = time(17, 0)      # 12:00 EST
        lunch_end = time(19, 0)        # 14:00 EST
        
        # Early morning (pre-market)
        pre_market_start = time(13, 0)  # 8:00 EST
        pre_market_end = time(14, 30)   # 9:30 EST
        
        in_morning = morning_start <= now <= morning_end
        in_afternoon = afternoon_start <= now <= afternoon_end
        in_lunch = lunch_start <= now <= lunch_end
        in_premarket = pre_market_start <= now <= pre_market_end
        
        if in_lunch:
            return False, 'lunch_lull'
        if in_morning:
            return True, 'morning_session'
        if in_afternoon:
            return True, 'afternoon_session'
        if in_premarket:
            return True, 'pre_market'
            
        return False, 'off_hours'
    
    def detect_regime(self, data_5m: pd.DataFrame) -> str:
        """Detect current market regime."""
        if len(data_5m) < 50:
            return 'unknown'
            
        latest = data_5m.iloc[-1]
        adx = latest.get('adx', 25)
        
        # Count swing highs/lows in last 50 bars
        highs = data_5m['high'].tail(50)
        lows = data_5m['low'].tail(50)
        
        swing_count = 0
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swing_count += 1
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swing_count += 1
        
        if adx > 30:
            return 'trending'
        elif adx < 20 or swing_count > 8:
            return 'ranging'
        else:
            return 'transitional'
    
    def calculate_entry_score(self, data_1m: pd.DataFrame, data_5m: pd.DataFrame, 
                               direction: str) -> float:
        """
        Calculate entry quality score (0.0 to 1.0).
        
        Factors:
        - Trend alignment (5M)
        - RSI not extreme
        - Volume confirmation
        - ATR adequate
        - Recent momentum
        """
        score = 0.0
        
        if len(data_1m) < 20 or len(data_5m) < 50:
            return 0.0
        
        latest_1m = data_1m.iloc[-1]
        latest_5m = data_5m.iloc[-1]
        
        # 1. Trend alignment (40% weight)
        aligned, trend_score = self.check_trend_alignment(data_5m, direction)
        score += trend_score * 0.40
        
        # 2. RSI not overbought/oversold (20% weight)
        rsi_1m = latest_1m.get('rsi', 50)
        if direction == 'long':
            if 30 < rsi_1m < 65:  # Not overbought
                score += 0.20
            elif 40 < rsi_1m < 55:  # Ideal zone
                score += 0.15
        else:
            if 35 < rsi_1m < 70:  # Not oversold
                score += 0.20
            elif 45 < rsi_1m < 60:  # Ideal zone
                score += 0.15
        
        # 3. Volume confirmation (15% weight)
        if 'volume' in data_1m.columns:
            vol_avg = data_1m['volume'].tail(20).mean()
            vol_current = latest_1m['volume']
            if vol_current > vol_avg * 1.2:
                score += 0.15
            elif vol_current > vol_avg * 0.8:
                score += 0.10
        else:
            score += 0.10  # Neutral if no volume data
        
        # 4. ATR adequate (15% weight) - volatility present
        atr = latest_5m.get('atr', 0)
        atr_avg = data_5m['atr'].tail(20).mean() if 'atr' in data_5m.columns else atr
        if atr > atr_avg * 0.8:
            score += 0.15
        elif atr > atr_avg * 0.5:
            score += 0.10
        
        # 5. Recent momentum (10% weight)
        close_5m = data_5m['close'].tail(5)
        momentum = (close_5m.iloc[-1] - close_5m.iloc[0]) / close_5m.iloc[0] * 100
        if direction == 'long' and momentum > 0:
            score += 0.10
        elif direction == 'short' and momentum < 0:
            score += 0.10
        
        return min(1.0, score)
    
    def detect_liquidity_sweep(self, data: pd.DataFrame, direction: str, 
                                lookback: int = 20) -> Tuple[bool, Optional[float]]:
        """
        Detect if recent price action swept liquidity.
        
        Returns (sweep_detected, sweep_level)
        """
        if len(data) < lookback + 5:
            return False, None
        
        recent = data.tail(lookback + 5)
        
        # Find swing highs/lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent) - 2):
            if recent['high'].iloc[i] > recent['high'].iloc[i-1] and \
               recent['high'].iloc[i] > recent['high'].iloc[i-2] and \
               recent['high'].iloc[i] > recent['high'].iloc[i+1] and \
               recent['high'].iloc[i] > recent['high'].iloc[i+2]:
                swing_highs.append(recent['high'].iloc[i])
                
            if recent['low'].iloc[i] < recent['low'].iloc[i-1] and \
               recent['low'].iloc[i] < recent['low'].iloc[i-2] and \
               recent['low'].iloc[i] < recent['low'].iloc[i+1] and \
               recent['low'].iloc[i] < recent['low'].iloc[i+2]:
                swing_lows.append(recent['low'].iloc[i])
        
        last_bar = recent.iloc[-1]
        prev_bar = recent.iloc[-2]
        
        if direction == 'long':
            # Look for bearish sweep (took out lows, then reversed)
            if not swing_lows:
                return False, None
            recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
            # Price swept below and closed back above
            swept = prev_bar['low'] < recent_low and last_bar['close'] > recent_low
            return swept, recent_low if swept else None
            
        else:  # short
            # Look for bullish sweep (took out highs, then reversed)
            if not swing_highs:
                return False, None
            recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            # Price swept above and closed back below
            swept = prev_bar['high'] > recent_high and last_bar['close'] < recent_high
            return swept, recent_high if swept else None
    
    def calculate_levels(self, entry_price: float, direction: str, 
                         atr: float, regime: str) -> Tuple[float, float, float]:
        """
        Calculate SL and dual TP levels.
        
        Returns (stop_loss, tp1, tp2)
        """
        # Regime-adjusted SL
        sl_mult = self.atr_sl_mult
        if regime == 'ranging':
            sl_mult *= 0.9  # Tighter in ranging
        elif regime == 'trending':
            sl_mult *= 1.1  # Wider in trending
        
        sl_distance = atr * sl_mult
        
        if direction == 'long':
            stop_loss = entry_price - sl_distance
            tp1 = entry_price + (sl_distance * self.tp1_ratio)
            tp2 = entry_price + (sl_distance * self.tp2_ratio)
        else:
            stop_loss = entry_price + sl_distance
            tp1 = entry_price - (sl_distance * self.tp1_ratio)
            tp2 = entry_price - (sl_distance * self.tp2_ratio)
        
        return stop_loss, tp1, tp2
    
    def generate_signal(
        self,
        data_1m: pd.DataFrame,
        data_5m: pd.DataFrame,
        current_position: Optional[dict] = None,
        candle_dt: datetime = None
    ) -> EnhancedSignal:
        """
        Generate enhanced trading signal.
        
        Returns EnhancedSignal with dual TP levels.
        """
        # Default no-signal
        no_signal = EnhancedSignal(
            direction='none', confidence=0, entry_price=0,
            stop_loss=0, take_profit1=0, take_profit2=0,
            timestamp=pd.Timestamp.now()
        )
        
        if len(data_1m) < 50 or len(data_5m) < 210:
            return no_signal
        
        # Check trading hours
        if self.use_trading_hours:
            valid_hours, session = self.is_trading_hours(candle_dt)
            if not valid_hours:
                return no_signal
        else:
            session = 'any'
        
        # Get latest values
        latest_5m = data_5m.iloc[-1]
        price = data_1m['close'].iloc[-1]
        atr = latest_5m.get('atr', 0)
        
        if atr == 0 or pd.isna(atr):
            return no_signal
        
        # Determine trend bias (5M chart)
        ema200 = latest_5m.get('ema_200', latest_5m.get('ema200'))
        ema50 = latest_5m.get('ema_50', latest_5m.get('ema50'))
        rsi_5m = latest_5m.get('rsi', 50)
        
        if pd.isna(ema200) or pd.isna(ema50):
            return no_signal
        
        trend_bullish = (price > ema200 and ema50 > ema200 and rsi_5m > 50)
        trend_bearish = (price < ema200 and ema50 < ema200 and rsi_5m < 50)
        
        # Skip if no clear trend
        if not (trend_bullish or trend_bearish):
            return no_signal
        
        direction = 'long' if trend_bullish else 'short'
        
        # Detect market regime
        regime = self.detect_regime(data_5m)
        
        # Check liquidity sweep if enabled
        if self.use_liquidity_sweep:
            sweep_detected, sweep_level = self.detect_liquidity_sweep(data_5m, direction)
            if not sweep_detected:
                return no_signal
        
        # Calculate entry score
        entry_score = self.calculate_entry_score(data_1m, data_5m, direction)
        
        # Generate signal if score is high enough
        if entry_score >= self.min_score:
            # Check trend alignment
            aligned, _ = self.check_trend_alignment(data_5m, direction)
            
            stop, tp1, tp2 = self.calculate_levels(price, direction, atr, regime)
            
            # Final confidence = entry score with session bonus
            confidence = entry_score
            if session in ('morning_session', 'afternoon_session'):
                confidence = min(1.0, confidence + 0.05)
            
            return EnhancedSignal(
                direction=direction,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop,
                take_profit1=tp1,
                take_profit2=tp2,
                timestamp=pd.Timestamp.now(),
                entry_score=entry_score,
                regime=regime,
                trend_alignment=aligned
            )
        
        return no_signal


def integrate_with_backtest(engine, data_5m: pd.DataFrame, data_1m: pd.DataFrame,
                            idx: int, pair: str, candle_dt: datetime) -> dict:
    """
    Helper function to integrate enhanced strategy with backtest engine.
    
    Returns dict with signal info that can be used by backtest engine.
    """
    strategy = EnhancedScalpingStrategy(
        min_score=0.55,
        use_liquidity_sweep=True,
        use_trading_hours=True,
        atr_sl_mult=1.5,
        tp1_ratio=1.5,
        tp2_ratio=2.5
    )
    
    # Get relevant data slices
    lookback_1m = min(300, len(data_1m))
    lookback_5m = min(250, len(data_5m))
    
    if idx < 50:
        return {'signal': 'SKIP'}
    
    # Create subsets
    subset_5m = data_5m.iloc[max(0, idx - lookback_5m):idx + 1].copy()
    
    # For 1M, find matching bars
    if candle_dt is not None and 'datetime' in data_1m.columns:
        mask_1m = data_1m['datetime'] <= candle_dt
        subset_1m = data_1m.loc[mask_1m].tail(lookback_1m).copy()
    else:
        subset_1m = data_1m.tail(lookback_1m).copy()
    
    if len(subset_1m) < 50 or len(subset_5m) < 50:
        return {'signal': 'SKIP'}
    
    # Generate signal
    signal = strategy.generate_signal(
        data_1m=subset_1m,
        data_5m=subset_5m,
        candle_dt=candle_dt
    )
    
    if signal.direction == 'none':
        return {'signal': 'SKIP'}
    
    return {
        'signal': 'BUY' if signal.direction == 'long' else 'SELL',
        'confidence': signal.confidence,
        'entry_score': signal.entry_score,
        'stop_loss': signal.stop_loss,
        'take_profit1': signal.take_profit1,
        'take_profit2': signal.take_profit2,
        'regime': signal.regime,
        'trend_aligned': signal.trend_alignment
    }
