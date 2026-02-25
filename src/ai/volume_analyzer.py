"""
Volume Analysis for Trading Signals
"""
import pandas as pd
import numpy as np
from src.utils.logger import bot_logger


class VolumeAnalyzer:
    """Analyze volume patterns for trading signals"""
    
    def __init__(self, volume_period=20):
        self.volume_period = volume_period
    
    def calculate_volume_profile(self, df):
        """
        Calculate volume profile and anomalies
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Enhanced DataFrame with volume metrics
        """
        df = df.copy()
        
        # Simple Moving Average of Volume
        df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
        
        # Volume standard deviation
        df['volume_std'] = df['volume'].rolling(window=self.volume_period).std()
        
        # Volume ratio (current / average)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-6)
        
        # Identify volume spikes (above 1.5x average)
        df['volume_spike'] = df['volume_ratio'] > 1.5
        
        return df
    
    def get_volume_signal(self, df):
        """
        Generate signal based on volume analysis
        
        Returns:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'confidence': 0.0-1.0,
                'reason': 'explanation'
            }
        """
        if len(df) < self.volume_period:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        df = self.calculate_volume_profile(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        confidence = 0.0
        reason_parts = []
        
        # Unusual volume spike
        if latest['volume_spike']:
            confidence += 0.3
            reason_parts.append(f"High volume spike ({latest['volume_ratio']:.2f}x average)")
        
        # Volume increasing on momentum
        if prev['volume'] < latest['volume']:
            # Check if price is moving
            price_change = (latest['close'] - prev['close']) / prev['close']
            if abs(price_change) > 0.001:  # 0.1% move
                confidence += 0.3
                reason_parts.append("Volume increasing with price move")
        
        # Recent high volume
        recent_volumes = df['volume'].tail(5).values
        if latest['volume'] >= np.percentile(recent_volumes, 75):
            confidence += 0.2
            reason_parts.append("Above average recent volume")
        
        signal = 'BUY' if confidence > 0.4 else 'HOLD'
        reason = " | ".join(reason_parts) if reason_parts else "Normal volume conditions"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'reason': reason,
            'current_volume': latest['volume'],
            'average_volume': latest['volume_sma'],
            'volume_ratio': latest['volume_ratio']
        }
