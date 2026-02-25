"""
Support and Resistance Level Detection

Identifies key price levels where price has historically reversed.
Used to:
  - Avoid buying into resistance / selling into support
  - Place smarter stop-loss / take-profit levels
  - Confirm breakout entries
"""
import numpy as np
import pandas as pd
from src.utils.logger import bot_logger


class SupportResistanceDetector:
    """
    Detect support/resistance using multiple methods:
      1. Swing highs/lows (pivot points)
      2. Volume-weighted price clusters
      3. Round number levels
    """

    def __init__(self, pivot_window: int = 10, cluster_tolerance: float = 0.001):
        """
        Args:
            pivot_window:      bars on each side to confirm a pivot
            cluster_tolerance: % distance to cluster nearby levels together
        """
        self.pivot_window = pivot_window
        self.cluster_tolerance = cluster_tolerance

    def detect_levels(self, df: pd.DataFrame, num_levels: int = 6) -> dict:
        """
        Detect support and resistance levels.

        Args:
            df:         DataFrame with OHLCV
            num_levels: max levels to return per side

        Returns:
            {
                'support_levels':    [float, ...],
                'resistance_levels': [float, ...],
                'current_price':     float,
                'nearest_support':   float,
                'nearest_resistance': float,
                'price_zone':        'AT_SUPPORT' | 'AT_RESISTANCE' | 'BETWEEN' | 'BREAKOUT',
                'zone_strength':     0.0-1.0,
            }
        """
        if len(df) < self.pivot_window * 2 + 1:
            current = df['close'].iloc[-1]
            return {
                'support_levels': [], 'resistance_levels': [],
                'current_price': current,
                'nearest_support': current * 0.99,
                'nearest_resistance': current * 1.01,
                'price_zone': 'BETWEEN', 'zone_strength': 0.0,
            }

        current_price = df['close'].iloc[-1]

        # 1. Find pivot highs and lows
        pivots = self._find_pivots(df)

        # 2. Find volume-weighted clusters
        vol_levels = self._volume_clusters(df)

        # 3. Round number levels
        round_levels = self._round_number_levels(current_price)

        # Combine and cluster all levels
        all_levels = pivots['highs'] + pivots['lows'] + vol_levels + round_levels
        clustered = self._cluster_levels(all_levels, current_price)

        # Separate into support and resistance
        support = sorted([l for l in clustered if l < current_price], reverse=True)
        resistance = sorted([l for l in clustered if l >= current_price])

        support = support[:num_levels]
        resistance = resistance[:num_levels]

        # Nearest levels
        nearest_support = support[0] if support else current_price * 0.995
        nearest_resistance = resistance[0] if resistance else current_price * 1.005

        # Determine price zone
        dist_to_support = abs(current_price - nearest_support) / current_price
        dist_to_resistance = abs(current_price - nearest_resistance) / current_price

        if dist_to_support < 0.001:
            zone = 'AT_SUPPORT'
            zone_strength = max(0, 1.0 - dist_to_support / 0.002)
        elif dist_to_resistance < 0.001:
            zone = 'AT_RESISTANCE'
            zone_strength = max(0, 1.0 - dist_to_resistance / 0.002)
        else:
            zone = 'BETWEEN'
            zone_strength = 0.3

        return {
            'support_levels': support,
            'resistance_levels': resistance,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'price_zone': zone,
            'zone_strength': zone_strength,
        }

    def get_sr_signal(self, df: pd.DataFrame) -> dict:
        """
        Generate a trading signal based on support/resistance context.

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'reason': str,
                'levels': { ... full level data ... },
            }
        """
        levels = self.detect_levels(df)
        zone = levels['price_zone']
        strength = levels['zone_strength']

        if zone == 'AT_SUPPORT':
            signal = 'BUY'
            confidence = min(strength * 0.8, 1.0)
            reason = f"Price at support {levels['nearest_support']:.5f}"
        elif zone == 'AT_RESISTANCE':
            signal = 'SELL'
            confidence = min(strength * 0.8, 1.0)
            reason = f"Price at resistance {levels['nearest_resistance']:.5f}"
        else:
            signal = 'HOLD'
            confidence = 0.0
            reason = f"Price between levels (S: {levels['nearest_support']:.5f}, R: {levels['nearest_resistance']:.5f})"

        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'levels': levels,
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _find_pivots(self, df: pd.DataFrame) -> dict:
        """Find pivot highs and pivot lows."""
        highs_col = df['high'].values
        lows_col = df['low'].values
        w = self.pivot_window

        pivot_highs = []
        pivot_lows = []

        for i in range(w, len(df) - w):
            # Pivot high: highest high in window
            if highs_col[i] == max(highs_col[i - w:i + w + 1]):
                pivot_highs.append(highs_col[i])

            # Pivot low: lowest low in window
            if lows_col[i] == min(lows_col[i - w:i + w + 1]):
                pivot_lows.append(lows_col[i])

        return {'highs': pivot_highs, 'lows': pivot_lows}

    def _volume_clusters(self, df: pd.DataFrame, num_bins: int = 50) -> list:
        """Find price levels with highest volume concentration."""
        price_min = df['low'].min()
        price_max = df['high'].max()

        if price_max - price_min < 1e-8:
            return []

        bins = np.linspace(price_min, price_max, num_bins + 1)
        volume_profile = np.zeros(num_bins)

        for _, row in df.iterrows():
            for j in range(num_bins):
                if bins[j] <= row['close'] <= bins[j + 1]:
                    volume_profile[j] += row['volume']
                    break

        # Top volume areas
        top_indices = volume_profile.argsort()[-5:]
        levels = [(bins[i] + bins[i + 1]) / 2 for i in top_indices if volume_profile[i] > 0]

        return levels

    def _round_number_levels(self, price: float) -> list:
        """Generate psychologically significant round-number levels."""
        if price > 10:  # JPY pairs
            step = 0.5
        else:
            step = 0.005  # EUR/USD etc.

        base = round(price / step) * step
        levels = [base + step * i for i in range(-3, 4)]
        return levels

    def _cluster_levels(self, levels: list, reference_price: float) -> list:
        """Cluster nearby levels together (take the average)."""
        if not levels:
            return []

        tolerance = reference_price * self.cluster_tolerance
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            if sorted_levels[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [sorted_levels[i]]

        clusters.append(np.mean(current_cluster))
        return clusters
