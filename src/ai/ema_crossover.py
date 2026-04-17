"""
Volatility-Momentum Analyzer (formerly EMA Crossover)

ATR-centric ensemble model (weight ~0.12) that confirms volatility
expansion / contraction and EMA alignment strength.

Replaces old crossover logic with:
  - ATR expansion signal (3+ consecutive ATR rises → 0.55)
  - EMA alignment confirmation (20 > 50: +0.50 if strong spread)
  - ATR contraction warning (3+ ATR falls → confidence × 0.7)
  - EMA200 counter-trend penalty (× 0.6)
"""
import pandas as pd
from src.utils.logger import bot_logger


class EMACrossoverAnalyzer:
    """ATR / EMA alignment model — confirms momentum via volatility growth."""

    def get_signal(self, df):
        from src.utils.logger import bot_logger
        bot_logger.info("[EMA] get_signal called")
        """Analyse latest candles for EMA alignment + ATR momentum.

        Scoring:
          - EMA20 > EMA50 with decent spread  → base 0.50
          - EMA crossover (just happened)       → base 0.60
          - ATR trend rising 3+ candles         → +0.15
          - RSI not extreme (anti-exhaustion)   → +0.10
          - EMA200 with-trend                   → +0.10
          - ATR contraction penalty             → ×0.70
          - EMA200 counter-trend penalty        → ×0.60

        Args:
            df: DataFrame with at least 'close', 'high', 'low' (200+ rows)

        Returns:
            dict with signal, confidence, reason
        """
        try:
            if df is None or len(df) < 60:
                result = {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
                bot_logger.info(f"[EMA] Returning: {result}")
                return result

            close = df['close']
            high = df['high']
            low = df['low']
            has_ema200 = len(df) >= 200

            # -- EMAs -----------------------------------------------
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if has_ema200 else None

            # -- ATR(14) --------------------------------------------
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # -- RSI(14) --------------------------------------------
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # -- Current values ------------------------------------
            cur_ema20 = float(ema20.iloc[-1])
            cur_ema50 = float(ema50.iloc[-1])
            prev_ema20 = float(ema20.iloc[-2])
            prev_ema50 = float(ema50.iloc[-2])
            cur_ema200 = float(ema200.iloc[-1]) if has_ema200 else None
            cur_rsi = float(rsi.iloc[-1])
            cur_atr = float(atr.iloc[-1])
            cur_price = float(close.iloc[-1])

            if pd.isna(cur_rsi) or pd.isna(cur_atr):
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Indicators warming up'}
            if cur_ema200 is not None and pd.isna(cur_ema200):
                cur_ema200 = None

            # -- ATR trend: count consecutive rises/falls ----------
            atr_diff = atr.diff()
            atr_rise_count = 0
            atr_fall_count = 0
            for i in range(1, min(8, len(atr_diff))):
                val = float(atr_diff.iloc[-i])
                if val > 0:
                    atr_rise_count += 1
                else:
                    break
            for i in range(1, min(8, len(atr_diff))):
                val = float(atr_diff.iloc[-i])
                if val < 0:
                    atr_fall_count += 1
                else:
                    break

            atr_expanding = atr_rise_count >= 3
            atr_contracting = atr_fall_count >= 3

            # -- Signal generation -----------------------------------
            signal = 'HOLD'
            confidence = 0.0
            reasons = []

            above_200 = cur_ema200 is not None and cur_price > cur_ema200
            below_200 = cur_ema200 is not None and cur_price < cur_ema200

            # Crossover detection
            bullish_cross = cur_ema20 > cur_ema50 and prev_ema20 <= prev_ema50
            bearish_cross = cur_ema20 < cur_ema50 and prev_ema20 >= prev_ema50
            bullish_trend = cur_ema20 > cur_ema50
            bearish_trend = cur_ema20 < cur_ema50

            # --- BULLISH ---
            if bullish_cross or bullish_trend:
                signal = 'BUY'
                if bullish_cross:
                    confidence = 0.60
                    reasons.append('EMA20 crossed above EMA50')
                else:
                    ema_spread = (cur_ema20 - cur_ema50) / cur_ema50
                    if ema_spread > 0.00003:  # 0.003% relative spread
                        confidence = 0.50
                        reasons.append(f'Bullish EMA alignment (spread {ema_spread:.5f})')
                    else:
                        return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Weak EMA alignment'}

                # ATR expansion boost
                if atr_expanding:
                    confidence += 0.15
                    reasons.append(f'ATR expanding ({atr_rise_count} consecutive rises)')

                # (RSI filter removed — RSI handled by LiquiditySweep model only)

                # EMA200 alignment
                if cur_ema200 is None:
                    pass  # Skip EMA200 filter when insufficient data
                elif above_200:
                    confidence += 0.10
                    reasons.append('Price > EMA200 — with trend')
                else:
                    confidence *= 0.60
                    reasons.append('Price < EMA200 — counter-trend (×0.6)')

                # ATR contraction penalty
                if atr_contracting:
                    confidence *= 0.70
                    reasons.append(f'ATR contracting ({atr_fall_count} falls) — confidence penalised ×0.7')

            # --- BEARISH ---
            elif bearish_cross or bearish_trend:
                signal = 'SELL'
                if bearish_cross:
                    confidence = 0.60
                    reasons.append('EMA20 crossed below EMA50')
                else:
                    ema_spread = (cur_ema50 - cur_ema20) / cur_ema50
                    if ema_spread > 0.00003:  # 0.003% relative spread
                        confidence = 0.50
                        reasons.append(f'Bearish EMA alignment (spread {ema_spread:.5f})')
                    else:
                        return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Weak EMA alignment'}

                # ATR expansion boost
                if atr_expanding:
                    confidence += 0.15
                    reasons.append(f'ATR expanding ({atr_rise_count} consecutive rises)')

                # (RSI filter removed — RSI handled by LiquiditySweep model only)

                # EMA200 alignment
                if cur_ema200 is None:
                    pass  # Skip EMA200 filter when insufficient data
                elif below_200:
                    confidence += 0.10
                    reasons.append('Price < EMA200 — with trend')
                else:
                    confidence *= 0.60
                    reasons.append('Price > EMA200 — counter-trend (×0.6)')

                # ATR contraction penalty
                if atr_contracting:
                    confidence *= 0.70
                    reasons.append(f'ATR contracting ({atr_fall_count} falls) — confidence penalised ×0.7')

            confidence = round(min(confidence, 1.0), 2)

            result = {
                'signal': signal,
                'confidence': confidence,
                'reason': ' | '.join(reasons) if reasons else 'No alignment signal',
            }
            bot_logger.info(f"[EMA] Returning: {result}")
            return result

        except Exception as e:
            bot_logger.error(f"EMA/Volatility analyzer error: {e}")
            result = {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {e}'}
            bot_logger.info(f"[EMA] Returning: {result}")
            return result
