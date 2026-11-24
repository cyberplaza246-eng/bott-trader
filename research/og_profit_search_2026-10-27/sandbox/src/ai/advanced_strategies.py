"""
Advanced Trading Strategies — Ported from Binance-Futures-Trading-Bot

High-performance strategies for futures trading:
  - Fibonacci MACD (fib retracement + MACD confirmation)
  - StochRSI MACD (stochastic RSI + MACD crossover)
  - Golden Cross (EMA100/50/20 + RSI filter)
  - Triple EMA Stochastic (EMA stack + stoch crossover)
  - Heikin-Ashi EMA (smoothed candles + trend filter)
  - Volume Breakout (price + volume confirmation)
  - Stochastic Bollinger Bands

Source: github.com/conor19w/Binance-Futures-Trading-Bot
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils.logger import bot_logger


class AdvancedStrategies:
    """Collection of advanced trading strategies."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def get_signal(
        self,
        df: pd.DataFrame,
        strategy: str = 'stoch_rsi_macd',
        pair: str = None
    ) -> Dict:
        """
        Get trading signal from specified strategy.
        
        Args:
            df: OHLCV DataFrame with indicators
            strategy: Strategy name
            pair: Trading pair
            
        Returns:
            Dict with signal, confidence, and details
        """
        # Ensure we have enough data
        if len(df) < 200:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
        # Calculate indicators if not present
        df = self._ensure_indicators(df)
        
        # Route to strategy
        strategy_map = {
            'fib_macd': self._fib_macd_signal,
            'stoch_rsi_macd': self._stoch_rsi_macd_signal,
            'golden_cross': self._golden_cross_signal,
            'triple_ema_stoch': self._triple_ema_stoch_signal,
            'heikin_ashi_ema': self._heikin_ashi_ema_signal,
            'breakout': self._breakout_signal,
            'stoch_bb': self._stoch_bb_signal,
            'wick_reversal': self._wick_reversal_signal,
        }
        
        if strategy not in strategy_map:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Unknown strategy: {strategy}'}
            
        return strategy_map[strategy](df)
        
    def get_combined_signal(
        self,
        df: pd.DataFrame,
        strategies: List[str] = None,
        min_agreement: int = 2
    ) -> Dict:
        """
        Get combined signal from multiple strategies.
        
        Args:
            df: OHLCV DataFrame
            strategies: List of strategy names to use
            min_agreement: Minimum number of agreeing strategies
            
        Returns:
            Combined signal with confidence
        """
        if strategies is None:
            strategies = ['stoch_rsi_macd', 'golden_cross', 'triple_ema_stoch']
            
        buy_votes = 0
        sell_votes = 0
        total_confidence = 0
        details = {}
        
        for strat in strategies:
            result = self.get_signal(df, strat)
            details[strat] = result
            
            if result['signal'] == 'BUY':
                buy_votes += 1
                total_confidence += result['confidence']
            elif result['signal'] == 'SELL':
                sell_votes += 1
                total_confidence += result['confidence']
                
        # Determine final signal
        if buy_votes >= min_agreement and buy_votes > sell_votes:
            signal = 'BUY'
            confidence = total_confidence / len(strategies)
        elif sell_votes >= min_agreement and sell_votes > buy_votes:
            signal = 'SELL'
            confidence = total_confidence / len(strategies)
        else:
            signal = 'HOLD'
            confidence = 0.0
            
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'strategies': details
        }
        
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators if not present."""
        df = df.copy()
        
        # EMAs
        for period in [3, 6, 8, 9, 14, 20, 50, 100, 200]:
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = df['close'].ewm(span=period, adjust=False).mean()
                
        # RSI
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
        # MACD
        if 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
        # Stochastic RSI
        if 'stoch_k' not in df.columns:
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
        # Bollinger Bands
        if 'bb_middle' not in df.columns:
            df['bb_middle'] = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * std
            df['bb_lower'] = df['bb_middle'] - 2 * std
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
        # ATR
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            
        # Volume
        if 'volume_sma' not in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            
        return df
        
    def _fib_macd_signal(self, df: pd.DataFrame) -> Dict:
        """
        Fibonacci MACD Strategy.
        
        Looks for pullbacks to Fibonacci levels with MACD confirmation.
        Entry on bullish/bearish engulfing + MACD crossover.
        """
        idx = len(df) - 1
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_ = df['open'].values
        macd = df['macd'].values
        macd_signal = df['macd_signal'].values
        ema200 = df['ema_200'].values
        
        # Find peaks and troughs in last 100 bars
        period = min(100, idx - 5)
        peaks = []
        troughs = []
        peak_locs = []
        trough_locs = []
        
        for i in range(idx - period, idx - 2):
            if i < 2:
                continue
            if high[i] > high[i-1] and high[i] > high[i+1] and high[i] > high[i-2] and high[i] > high[i+2]:
                peaks.append(high[i])
                peak_locs.append(i)
            if low[i] < low[i-1] and low[i] < low[i+1] and low[i] < low[i-2] and low[i] < low[i+2]:
                troughs.append(low[i])
                trough_locs.append(i)
                
        if not peaks or not troughs:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No swing points'}
            
        # Determine trend
        if close[idx] > ema200[idx]:
            trend = 'UP'
            # Find recent high and low for uptrend fib
            max_close = max(peaks) if peaks else high[idx]
            min_close = min(troughs) if troughs else low[idx]
        else:
            trend = 'DOWN'
            max_close = max(peaks) if peaks else high[idx]
            min_close = min(troughs) if troughs else low[idx]
            
        # Calculate Fibonacci levels
        diff = max_close - min_close
        fib_236 = max_close - 0.236 * diff
        fib_382 = max_close - 0.382 * diff
        fib_500 = max_close - 0.500 * diff
        fib_618 = max_close - 0.618 * diff
        
        signal = 'HOLD'
        confidence = 0.0
        
        # Check for BUY setup (uptrend pullback)
        if trend == 'UP':
            # Price in fib zone
            in_fib_zone = (fib_382 <= close[idx] <= fib_236) or (fib_500 <= close[idx] <= fib_382)
            
            # Bullish engulfing
            bullish_engulf = (close[idx-2] < open_[idx-2] and 
                            open_[idx-2] < close[idx-1] < close[idx])
            
            # MACD crossover
            macd_cross_up = (macd_signal[idx-1] < macd[idx-1] or macd_signal[idx-2] < macd[idx-2]) and macd_signal[idx] > macd[idx]
            
            if in_fib_zone and bullish_engulf and macd_cross_up:
                signal = 'BUY'
                confidence = 0.75
                
        # Check for SELL setup (downtrend pullback)
        elif trend == 'DOWN':
            # Inverse fib levels for downtrend
            fib_236_inv = min_close + 0.236 * diff
            fib_382_inv = min_close + 0.382 * diff
            
            in_fib_zone = (fib_236_inv <= close[idx] <= fib_382_inv)
            
            # Bearish engulfing
            bearish_engulf = (close[idx-2] > open_[idx-2] and 
                             open_[idx-2] > close[idx-1] > close[idx])
            
            # MACD crossover down
            macd_cross_down = (macd_signal[idx-1] > macd[idx-1] or macd_signal[idx-2] > macd[idx-2]) and macd_signal[idx] < macd[idx]
            
            if in_fib_zone and bearish_engulf and macd_cross_down:
                signal = 'SELL'
                confidence = 0.75
                
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'fib_macd',
            'trend': trend,
            'fib_levels': {'236': fib_236, '382': fib_382, '500': fib_500, '618': fib_618}
        }
        
    def _stoch_rsi_macd_signal(self, df: pd.DataFrame) -> Dict:
        """
        Stochastic RSI + MACD Strategy.
        
        Entry when:
        - Stoch K&D in oversold/overbought (20/80)
        - RSI confirms direction (>50 for buy, <50 for sell)
        - MACD crossover confirms
        """
        idx = len(df) - 1
        stoch_k = df['stoch_k'].values
        stoch_d = df['stoch_d'].values
        rsi = df['rsi'].values
        macd = df['macd'].values
        macd_signal = df['macd_signal'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # BUY conditions
        buy_stoch = (stoch_k[idx] < 20 and stoch_d[idx] < 20) or \
                   (stoch_k[idx-1] < 20 and stoch_d[idx-1] < 20 and stoch_k[idx] < 80 and stoch_d[idx] < 80)
        buy_rsi = rsi[idx] > 50
        buy_macd = (macd[idx] > macd_signal[idx] and macd[idx-1] < macd_signal[idx-1]) or \
                  (macd[idx] > macd_signal[idx] and macd[idx-2] < macd_signal[idx-2])
        
        # SELL conditions
        sell_stoch = (stoch_k[idx] > 80 and stoch_d[idx] > 80) or \
                    (stoch_k[idx-1] > 80 and stoch_d[idx-1] > 80 and stoch_k[idx] > 20 and stoch_d[idx] > 20)
        sell_rsi = rsi[idx] < 50
        sell_macd = (macd[idx] < macd_signal[idx] and macd[idx-1] > macd_signal[idx-1]) or \
                   (macd[idx] < macd_signal[idx] and macd[idx-2] > macd_signal[idx-2])
        
        if buy_stoch and buy_rsi and buy_macd:
            signal = 'BUY'
            confidence = 0.70
        elif sell_stoch and sell_rsi and sell_macd:
            signal = 'SELL'
            confidence = 0.70
            
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'stoch_rsi_macd',
            'stoch_k': stoch_k[idx],
            'stoch_d': stoch_d[idx],
            'rsi': rsi[idx]
        }
        
    def _golden_cross_signal(self, df: pd.DataFrame) -> Dict:
        """
        Golden Cross Strategy.
        
        Entry when EMA20 crosses EMA50 with:
        - Price above/below EMA100 (trend filter)
        - RSI above/below 50 (momentum filter)
        """
        idx = len(df) - 1
        close = df['close'].values
        ema20 = df['ema_20'].values
        ema50 = df['ema_50'].values
        ema100 = df['ema_100'].values
        rsi = df['rsi'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # BUY: Price > EMA100, RSI > 50, EMA20 crosses above EMA50
        if close[idx] > ema100[idx] and rsi[idx] > 50:
            cross_up = (ema20[idx-1] < ema50[idx-1] and ema20[idx] > ema50[idx]) or \
                      (ema20[idx-2] < ema50[idx-2] and ema20[idx] > ema50[idx]) or \
                      (ema20[idx-3] < ema50[idx-3] and ema20[idx] > ema50[idx])
            if cross_up:
                signal = 'BUY'
                confidence = 0.65
                
        # SELL: Price < EMA100, RSI < 50, EMA20 crosses below EMA50
        elif close[idx] < ema100[idx] and rsi[idx] < 50:
            cross_down = (ema20[idx-1] > ema50[idx-1] and ema20[idx] < ema50[idx]) or \
                        (ema20[idx-2] > ema50[idx-2] and ema20[idx] < ema50[idx]) or \
                        (ema20[idx-3] > ema50[idx-3] and ema20[idx] < ema50[idx])
            if cross_down:
                signal = 'SELL'
                confidence = 0.65
                
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'golden_cross',
            'ema20': ema20[idx],
            'ema50': ema50[idx],
            'ema100': ema100[idx]
        }
        
    def _triple_ema_stoch_signal(self, df: pd.DataFrame) -> Dict:
        """
        Triple EMA + Stochastic Strategy.
        
        Entry when:
        - EMA8 > EMA14 > EMA50 (trend aligned)
        - Stochastic K crosses D
        """
        idx = len(df) - 1
        close = df['close'].values
        ema8 = df['ema_8'].values
        ema14 = df['ema_14'].values
        ema50 = df['ema_50'].values
        stoch_k = df['stoch_k'].values
        stoch_d = df['stoch_d'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # BUY: EMAs stacked bullish + stoch cross up
        ema_bullish = close[idx] > ema8[idx] > ema14[idx] > ema50[idx]
        stoch_cross_up = stoch_k[idx] > stoch_d[idx] and stoch_k[idx-1] < stoch_d[idx-1]
        
        # SELL: EMAs stacked bearish + stoch cross down
        ema_bearish = close[idx] < ema8[idx] < ema14[idx] < ema50[idx]
        stoch_cross_down = stoch_k[idx] < stoch_d[idx] and stoch_k[idx-1] > stoch_d[idx-1]
        
        if ema_bullish and stoch_cross_up:
            signal = 'BUY'
            confidence = 0.72
        elif ema_bearish and stoch_cross_down:
            signal = 'SELL'
            confidence = 0.72
            
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'triple_ema_stoch'
        }
        
    def _heikin_ashi_ema_signal(self, df: pd.DataFrame) -> Dict:
        """
        Heikin-Ashi + EMA Strategy.
        
        Uses smoothed Heikin-Ashi candles with EMA200 trend filter.
        Entry on HA candle color change + stochastic signal.
        """
        idx = len(df) - 1
        
        # Calculate Heikin-Ashi
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
        ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
        
        ema200 = df['ema_200'].values
        stoch_k = df['stoch_k'].values
        stoch_d = df['stoch_d'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # Bullish HA candle (close > open) above EMA200
        ha_bullish = ha_close.iloc[idx] > ha_open.iloc[idx]
        ha_bearish = ha_close.iloc[idx] < ha_open.iloc[idx]
        above_ema200 = ha_close.iloc[idx] > ema200[idx]
        below_ema200 = ha_close.iloc[idx] < ema200[idx]
        
        # Stochastic cross
        stoch_cross_up = stoch_k[idx] > stoch_d[idx] and stoch_k[idx-1] < stoch_d[idx-1]
        stoch_cross_down = stoch_k[idx] < stoch_d[idx] and stoch_k[idx-1] > stoch_d[idx-1]
        
        if ha_bullish and above_ema200 and stoch_cross_up:
            signal = 'BUY'
            confidence = 0.68
        elif ha_bearish and below_ema200 and stoch_cross_down:
            signal = 'SELL'
            confidence = 0.68
            
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'heikin_ashi_ema'
        }
        
    def _breakout_signal(self, df: pd.DataFrame) -> Dict:
        """
        Volume Breakout Strategy.
        
        Entry when price breaks recent high/low with volume confirmation.
        """
        idx = len(df) - 1
        period = 20
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Rolling max/min
        max_close = df['close'].rolling(period).max().values
        min_close = df['close'].rolling(period).min().values
        max_vol = df['volume'].rolling(period).max().values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # Breakout up with volume
        if close[idx] >= max_close[idx-1] and volume[idx] >= max_vol[idx-1] * 0.8:
            signal = 'BUY'
            confidence = 0.60
            
        # Breakdown with volume
        elif close[idx] <= min_close[idx-1] and volume[idx] >= max_vol[idx-1] * 0.8:
            signal = 'SELL'
            confidence = 0.60
            
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'breakout'
        }
        
    def _stoch_bb_signal(self, df: pd.DataFrame) -> Dict:
        """
        Stochastic + Bollinger Bands Strategy.
        
        Entry when stochastic signals with BB confirmation.
        """
        idx = len(df) - 1
        stoch_k = df['stoch_k'].values
        stoch_d = df['stoch_d'].values
        bb_pct = df['bb_pct'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # BUY: Stoch oversold + near lower BB
        if stoch_k[idx] < 20 and stoch_d[idx] < 20 and bb_pct[idx] < 0.2:
            if stoch_k[idx] > stoch_d[idx] and stoch_k[idx-1] < stoch_d[idx-1]:
                signal = 'BUY'
                confidence = 0.65
                
        # SELL: Stoch overbought + near upper BB
        elif stoch_k[idx] > 80 and stoch_d[idx] > 80 and bb_pct[idx] > 0.8:
            if stoch_k[idx] < stoch_d[idx] and stoch_k[idx-1] > stoch_d[idx-1]:
                signal = 'SELL'
                confidence = 0.65
                
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'stoch_bb',
            'bb_pct': bb_pct[idx]
        }
        
    def _wick_reversal_signal(self, df: pd.DataFrame) -> Dict:
        """
        Candle Wick Reversal Strategy.
        
        Detects rejection wicks after trend moves.
        - 3 green candles + red candle with huge upper wick = SELL
        - 3 red candles + green candle with huge lower wick = BUY
        """
        idx = len(df) - 1
        close = df['close'].values
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        
        signal = 'HOLD'
        confidence = 0.0
        
        # Check for bearish rejection (sell signal)
        # 3 green candles followed by red candle with huge upper wick
        if (close[idx-4] < close[idx-3] < close[idx-2] and  # 3 rising closes
            close[idx-1] < open_[idx-1] and  # Red candle
            close[idx] < close[idx-1]):  # Continuation
            
            body = abs(open_[idx-1] - close[idx-1])
            upper_wick = high[idx-1] - max(open_[idx-1], close[idx-1])
            lower_wick = min(open_[idx-1], close[idx-1]) - low[idx-1]
            
            if body > 0 and (upper_wick + lower_wick) > 10 * body:
                signal = 'SELL'
                confidence = 0.62
                
        # Check for bullish rejection (buy signal)
        # 3 red candles followed by green candle with huge lower wick
        elif (close[idx-4] > close[idx-3] > close[idx-2] and  # 3 falling closes
              close[idx-1] > open_[idx-1] and  # Green candle
              close[idx] > close[idx-1]):  # Continuation
            
            body = abs(open_[idx-1] - close[idx-1])
            upper_wick = high[idx-1] - max(open_[idx-1], close[idx-1])
            lower_wick = min(open_[idx-1], close[idx-1]) - low[idx-1]
            
            if body > 0 and (upper_wick + lower_wick) > 10 * body:
                signal = 'BUY'
                confidence = 0.62
                
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'wick_reversal'
        }


# Convenience function for quick access
def get_advanced_signal(df: pd.DataFrame, strategy: str = 'stoch_rsi_macd') -> Dict:
    """Quick access to advanced strategy signals."""
    strat = AdvancedStrategies()
    return strat.get_signal(df, strategy)


def get_combined_advanced_signal(
    df: pd.DataFrame,
    strategies: List[str] = None,
    min_agreement: int = 2
) -> Dict:
    """Quick access to combined strategy signals."""
    strat = AdvancedStrategies()
    return strat.get_combined_signal(df, strategies, min_agreement)
