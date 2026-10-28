"""
Multi-Timeframe Confluence Analysis

Checks alignment across 5m, 1h, and 4h timeframes.
Trades are stronger when all timeframes agree on direction.
"""
import pandas as pd
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger


class MultiTimeframeAnalyzer:
    """
    Analyse multiple timeframes and score confluence.
    
    If 5m + 1h + 4h all show BUY → strong confluence (1.0)
    If only 2 agree → moderate (0.6)
    If they disagree → weak / no trade (0.0-0.3)
    """

    TIMEFRAMES = {
        'fast':   5,    # 5-minute
        'medium': 60,   # 1-hour
        'slow':   240,  # 4-hour
    }

    WEIGHTS = {'fast': 0.20, 'medium': 0.35, 'slow': 0.45}

    def __init__(self):
        self.technical = TechnicalAnalyzer()

    def analyze(self, broker, pair: str) -> dict:
        """
        Fetch data for each timeframe, run technical analysis, score confluence.

        Args:
            broker: MT5Connector (or simulation) instance with get_candles()
            pair:   currency pair string

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confluence_score': 0.0 – 1.0,
                'timeframe_signals': { 'fast': {...}, 'medium': {...}, 'slow': {...} },
                'reason': str,
            }
        """
        tf_signals = {}

        for label, tf_minutes in self.TIMEFRAMES.items():
            df = broker.get_candles(pair, timeframe_minutes=tf_minutes, num_candles=100)
            if df is None or len(df) < 30:
                tf_signals[label] = {
                    'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'
                }
                continue

            df_with_indicators = self.technical.calculate_indicators(df)
            signal = self.technical.get_signal(df_with_indicators)
            tf_signals[label] = signal

        # Score confluence
        buy_score = 0.0
        sell_score = 0.0

        for label, weight in self.WEIGHTS.items():
            sig = tf_signals[label]
            if sig['signal'] == 'BUY':
                buy_score += weight * sig['confidence']
            elif sig['signal'] == 'SELL':
                sell_score += weight * sig['confidence']

        # Determine direction
        if buy_score > sell_score and buy_score > 0.10:
            final_signal = 'BUY'
            confluence = min(buy_score / 0.5, 1.0)  # Normalise
        elif sell_score > buy_score and sell_score > 0.10:
            final_signal = 'SELL'
            confluence = min(sell_score / 0.5, 1.0)
        else:
            final_signal = 'HOLD'
            confluence = 0.0

        # Count agreements
        buy_count = sum(1 for s in tf_signals.values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in tf_signals.values() if s['signal'] == 'SELL')
        agreement = max(buy_count, sell_count)

        reason_parts = [
            f"{label.upper()}: {tf_signals[label]['signal']} ({tf_signals[label]['confidence']:.0%})"
            for label in self.TIMEFRAMES
        ]
        reason = " | ".join(reason_parts)

        return {
            'signal': final_signal,
            'confluence_score': confluence,
            'agreement': agreement,
            'timeframe_signals': tf_signals,
            'reason': reason,
        }
