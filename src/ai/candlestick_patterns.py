"""
Candlestick Pattern Recognition

Detects high-probability candlestick patterns using pure Python.
No external dependencies required.
"""
import pandas as pd
import numpy as np
from src.utils.logger import bot_logger


class CandlestickPatternDetector:
    """
    Detect common candlestick reversal and continuation patterns:
      - Engulfing (bullish / bearish)
      - Hammer / Shooting Star
      - Doji
      - Morning / Evening Star
      - Three White Soldiers / Three Black Crows
      - Pin Bar
    """

    def __init__(self):
        self.body_threshold = 0.3   # Body < 30% of range → small body
        self.tail_ratio = 2.0       # Tail must be 2x body for hammer/pin

    def detect_patterns(self, df: pd.DataFrame) -> list:
        """
        Scan the last few candles for patterns.

        Args:
            df: DataFrame with columns open, high, low, close

        Returns:
            List of detected pattern dicts:
            [
                {
                    'pattern': 'BULLISH_ENGULFING',
                    'signal': 'BUY',
                    'strength': 0.0-1.0,
                    'candle_index': int,
                },
                ...
            ]
        """
        if len(df) < 5:
            return []

        patterns = []

        # We only need the last 5 candles
        recent = df.tail(5).reset_index(drop=True)

        # Helper columns
        body = (recent['close'] - recent['open']).abs()
        full_range = recent['high'] - recent['low']
        is_bullish = recent['close'] > recent['open']
        upper_wick = recent['high'] - recent[['open', 'close']].max(axis=1)
        lower_wick = recent[['open', 'close']].min(axis=1) - recent['low']

        # Last candle index in `recent`
        last = len(recent) - 1
        prev = last - 1
        prev2 = last - 2

        # ------- Engulfing -------
        if (is_bullish.iloc[last] and not is_bullish.iloc[prev] and
                recent['close'].iloc[last] > recent['open'].iloc[prev] and
                recent['open'].iloc[last] < recent['close'].iloc[prev]):
            patterns.append({
                'pattern': 'BULLISH_ENGULFING',
                'signal': 'BUY',
                'strength': 0.8,
                'candle_index': last,
            })

        if (not is_bullish.iloc[last] and is_bullish.iloc[prev] and
                recent['close'].iloc[last] < recent['open'].iloc[prev] and
                recent['open'].iloc[last] > recent['close'].iloc[prev]):
            patterns.append({
                'pattern': 'BEARISH_ENGULFING',
                'signal': 'SELL',
                'strength': 0.8,
                'candle_index': last,
            })

        # ------- Hammer / Shooting Star -------
        if full_range.iloc[last] > 0:
            body_ratio = body.iloc[last] / full_range.iloc[last]

            # Hammer (bullish): small body at top, long lower wick
            if (body_ratio < self.body_threshold and
                    lower_wick.iloc[last] > body.iloc[last] * self.tail_ratio and
                    upper_wick.iloc[last] < body.iloc[last] * 0.5):
                patterns.append({
                    'pattern': 'HAMMER',
                    'signal': 'BUY',
                    'strength': 0.7,
                    'candle_index': last,
                })

            # Shooting Star (bearish): small body at bottom, long upper wick
            if (body_ratio < self.body_threshold and
                    upper_wick.iloc[last] > body.iloc[last] * self.tail_ratio and
                    lower_wick.iloc[last] < body.iloc[last] * 0.5):
                patterns.append({
                    'pattern': 'SHOOTING_STAR',
                    'signal': 'SELL',
                    'strength': 0.7,
                    'candle_index': last,
                })

        # ------- Doji -------
        if full_range.iloc[last] > 0:
            if body.iloc[last] / full_range.iloc[last] < 0.1:
                patterns.append({
                    'pattern': 'DOJI',
                    'signal': 'HOLD',  # Indecision – context-dependent
                    'strength': 0.5,
                    'candle_index': last,
                })

        # ------- Morning Star (bullish 3-candle) -------
        if prev2 >= 0:
            if (not is_bullish.iloc[prev2] and  # 1st: bearish
                    body.iloc[prev] < body.iloc[prev2] * 0.5 and  # 2nd: small body (star)
                    is_bullish.iloc[last] and  # 3rd: bullish
                    recent['close'].iloc[last] > (recent['open'].iloc[prev2] + recent['close'].iloc[prev2]) / 2):
                patterns.append({
                    'pattern': 'MORNING_STAR',
                    'signal': 'BUY',
                    'strength': 0.85,
                    'candle_index': last,
                })

            # Evening Star (bearish 3-candle)
            if (is_bullish.iloc[prev2] and  # 1st: bullish
                    body.iloc[prev] < body.iloc[prev2] * 0.5 and  # 2nd: small body (star)
                    not is_bullish.iloc[last] and  # 3rd: bearish
                    recent['close'].iloc[last] < (recent['open'].iloc[prev2] + recent['close'].iloc[prev2]) / 2):
                patterns.append({
                    'pattern': 'EVENING_STAR',
                    'signal': 'SELL',
                    'strength': 0.85,
                    'candle_index': last,
                })

        # ------- Three White Soldiers / Three Black Crows -------
        if prev2 >= 0:
            if (is_bullish.iloc[prev2] and is_bullish.iloc[prev] and is_bullish.iloc[last] and
                    recent['close'].iloc[prev] > recent['close'].iloc[prev2] and
                    recent['close'].iloc[last] > recent['close'].iloc[prev]):
                patterns.append({
                    'pattern': 'THREE_WHITE_SOLDIERS',
                    'signal': 'BUY',
                    'strength': 0.75,
                    'candle_index': last,
                })

            if (not is_bullish.iloc[prev2] and not is_bullish.iloc[prev] and not is_bullish.iloc[last] and
                    recent['close'].iloc[prev] < recent['close'].iloc[prev2] and
                    recent['close'].iloc[last] < recent['close'].iloc[prev]):
                patterns.append({
                    'pattern': 'THREE_BLACK_CROWS',
                    'signal': 'SELL',
                    'strength': 0.75,
                    'candle_index': last,
                })

        # ------- Pin Bar -------
        if full_range.iloc[last] > 0:
            body_ratio_last = body.iloc[last] / full_range.iloc[last]
            # Bullish pin bar: very long lower wick relative to body
            if (body_ratio_last < 0.4 and
                    lower_wick.iloc[last] > full_range.iloc[last] * 0.6):
                patterns.append({
                    'pattern': 'BULLISH_PIN_BAR',
                    'signal': 'BUY',
                    'strength': 0.7,
                    'candle_index': last,
                })

            # Bearish pin bar: very long upper wick relative to body
            if (body_ratio_last < 0.4 and
                    upper_wick.iloc[last] > full_range.iloc[last] * 0.6):
                patterns.append({
                    'pattern': 'BEARISH_PIN_BAR',
                    'signal': 'SELL',
                    'strength': 0.7,
                    'candle_index': last,
                })

        return patterns

    def get_pattern_signal(self, df: pd.DataFrame) -> dict:
        """
        Aggregate all detected patterns into a single signal.

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'patterns': [...],
                'reason': str,
            }
        """
        patterns = self.detect_patterns(df)

        if not patterns:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'patterns': [],
                'reason': 'No candlestick patterns detected',
            }

        buy_strength = sum(p['strength'] for p in patterns if p['signal'] == 'BUY')
        sell_strength = sum(p['strength'] for p in patterns if p['signal'] == 'SELL')

        pattern_names = [p['pattern'] for p in patterns]

        if buy_strength > sell_strength and buy_strength > 0.5:
            signal = 'BUY'
            confidence = min(buy_strength, 1.0)
        elif sell_strength > buy_strength and sell_strength > 0.5:
            signal = 'SELL'
            confidence = min(sell_strength, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.0

        return {
            'signal': signal,
            'confidence': confidence,
            'patterns': patterns,
            'reason': f"Patterns: {', '.join(pattern_names)}",
        }
