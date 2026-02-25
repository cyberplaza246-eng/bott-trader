"""
Technical Analysis Indicators (using pandas_ta - pure Python, no C dependency)
"""
import pandas as pd
import numpy as np
import pandas_ta as pta
from src.utils.logger import bot_logger


class TechnicalAnalyzer:
    """Calculate technical indicators and generate signals"""
    
    def __init__(self):
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
    
    def calculate_indicators(self, df):
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # RSI
        df['rsi'] = pta.rsi(df['close'], length=self.rsi_period)
        
        # MACD
        macd_df = pta.macd(df['close'],
                           fast=self.macd_fast,
                           slow=self.macd_slow,
                           signal=self.macd_signal)
        if macd_df is not None:
            df['macd'] = macd_df.iloc[:, 0]           # MACD line
            df['macd_histogram'] = macd_df.iloc[:, 1]  # Histogram
            df['macd_signal'] = macd_df.iloc[:, 2]     # Signal line
        else:
            df['macd'] = 0.0
            df['macd_histogram'] = 0.0
            df['macd_signal'] = 0.0
        
        # Bollinger Bands
        bb_df = pta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
        if bb_df is not None:
            df['bb_lower'] = bb_df.iloc[:, 0]
            df['bb_middle'] = bb_df.iloc[:, 1]
            df['bb_upper'] = bb_df.iloc[:, 2]
        else:
            df['bb_lower'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_upper'] = df['close']
        
        # ATR (volatility)
        df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        # Stochastic
        stoch_df = pta.stoch(df['high'], df['low'], df['close'],
                             k=14, d=3, smooth_k=3)
        if stoch_df is not None:
            df['stoch_k'] = stoch_df.iloc[:, 0]
            df['stoch_d'] = stoch_df.iloc[:, 1]
        else:
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
        
        # Fill NaN values from indicator warmup period
        df = df.bfill().ffill()
        
        return df
    
    def get_signal(self, df):
        """
        Generate BUY/SELL signal based on technical analysis
        
        Returns:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'confidence': 0.0-1.0,
                'reason': 'explanation'
            }
        """
        if len(df) < self.bb_period:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        confidence = 0.0
        signals_count = 0
        reason_parts = []
        
        # RSI Analysis
        if latest['rsi'] < self.rsi_oversold:
            confidence += 0.25
            signals_count += 1
            reason_parts.append(f"RSI oversold ({latest['rsi']:.1f})")
        elif latest['rsi'] > self.rsi_overbought:
            confidence -= 0.25
            reason_parts.append(f"RSI overbought ({latest['rsi']:.1f})")
        
        # Bollinger Bands Analysis (mean reversion)
        bb_width = latest['bb_upper'] - latest['bb_lower']
        if latest['close'] < latest['bb_lower']:
            confidence += 0.25
            signals_count += 1
            reason_parts.append("Price below lower BB")
        elif latest['close'] > latest['bb_upper']:
            confidence -= 0.25
            reason_parts.append("Price above upper BB")
        
        # MACD Analysis
        if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
            confidence += 0.25
            signals_count += 1
            reason_parts.append("MACD bullish crossover")
        elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
            confidence -= 0.25
            reason_parts.append("MACD bearish crossover")
        
        # Stochastic Oscillator
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            confidence += 0.25
            signals_count += 1
            reason_parts.append("Stochastic oversold")
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            confidence -= 0.25
            reason_parts.append("Stochastic overbought")
        
        # Determine final signal (lowered thresholds for responsiveness)
        if confidence > 0.25:
            signal = 'BUY'
        elif confidence < -0.25:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        confidence = abs(confidence)  # Use absolute value for confidence score
        reason = " | ".join(reason_parts) if reason_parts else "No clear signals"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'reason': reason,
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        }
