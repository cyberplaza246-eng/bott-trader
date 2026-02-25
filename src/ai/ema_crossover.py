"""
EMA Crossover Model — Fast trend-following signal

Uses EMA20/EMA50 crossover + RSI + MACD confirmation.
Pure math, no ML — executes in <1ms.
"""
import pandas as pd
from src.utils.logger import bot_logger


class EMACrossoverAnalyzer:
    """Lightweight EMA crossover with RSI/MACD filters."""

    def get_signal(self, df):
        """
        Analyse the latest candles for an EMA crossover signal.

        Args:
            df: DataFrame with at least 'close' column (60+ rows ideal)

        Returns:
            dict with signal, confidence, reason
        """
        try:
            if df is None or len(df) < 55:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

            close = df['close']

            # EMAs
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            # RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            # Current & previous values
            cur_ema20 = ema20.iloc[-1]
            cur_ema50 = ema50.iloc[-1]
            prev_ema20 = ema20.iloc[-2]
            prev_ema50 = ema50.iloc[-2]
            cur_rsi = rsi.iloc[-1]
            cur_macd = macd_line.iloc[-1]
            cur_signal = signal_line.iloc[-1]

            if pd.isna(cur_rsi) or pd.isna(cur_ema50):
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Indicators warming up'}

            signal = 'HOLD'
            confidence = 0.0
            reasons = []

            # --- Bullish crossover ---
            if cur_ema20 > cur_ema50 and prev_ema20 <= prev_ema50:
                reasons.append('EMA20 crossed above EMA50')
                confidence = 0.6
                if cur_rsi < 70:
                    confidence += 0.1
                    reasons.append(f'RSI {cur_rsi:.0f} not overbought')
                if cur_macd > cur_signal:
                    confidence += 0.15
                    reasons.append('MACD confirms bullish')
                signal = 'BUY'

            # --- Bearish crossover ---
            elif cur_ema20 < cur_ema50 and prev_ema20 >= prev_ema50:
                reasons.append('EMA20 crossed below EMA50')
                confidence = 0.6
                if cur_rsi > 30:
                    confidence += 0.1
                    reasons.append(f'RSI {cur_rsi:.0f} not oversold')
                if cur_macd < cur_signal:
                    confidence += 0.15
                    reasons.append('MACD confirms bearish')
                signal = 'SELL'

            # --- Trend continuation (no crossover but clear trend) ---
            elif cur_ema20 > cur_ema50:
                spread = (cur_ema20 - cur_ema50) / cur_ema50
                if spread > 0.0005:
                    confidence = 0.45
                    reasons.append(f'Uptrend (spread {spread:.4f})')
                    if cur_rsi < 65:
                        confidence += 0.05
                    if cur_macd > cur_signal:
                        confidence += 0.10
                        reasons.append('MACD confirms')
                    signal = 'BUY'
            elif cur_ema20 < cur_ema50:
                spread = (cur_ema50 - cur_ema20) / cur_ema50
                if spread > 0.0005:
                    confidence = 0.45
                    reasons.append(f'Downtrend (spread {spread:.4f})')
                    if cur_rsi > 35:
                        confidence += 0.05
                    if cur_macd < cur_signal:
                        confidence += 0.10
                        reasons.append('MACD confirms')
                    signal = 'SELL'

            return {
                'signal': signal,
                'confidence': round(confidence, 2),
                'reason': ' | '.join(reasons) if reasons else 'No crossover signal',
            }

        except Exception as e:
            bot_logger.error(f"EMA crossover error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {e}'}
