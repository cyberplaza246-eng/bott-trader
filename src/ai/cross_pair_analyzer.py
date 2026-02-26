"""
Cross-Pair Correlation Analyzer

Uses real-time price correlation between pairs to:
  - Confirm signals (if correlated pairs agree → stronger conviction)
  - Reject signals (if correlated pair diverges → warning)
  - Detect lead/lag relationships (EUR/USD sometimes leads GBP/USD)

Known forex correlations:
  EUR/USD ↔ GBP/USD   (strong positive, ~0.80-0.90)
  EUR/USD ↔ USD/JPY   (moderate negative, ~-0.40 to -0.60)
  GBP/USD ↔ USD/JPY   (moderate negative)
"""
import numpy as np
import pandas as pd
from src.utils.logger import bot_logger


# Static correlation map — direction and typical strength
# positive = pairs move together, negative = inverse
PAIR_CORRELATIONS = {
    ('EUR/USD', 'GBP/USD'): {'direction': 'positive', 'typical': 0.85},
    ('EUR/USD', 'USD/JPY'): {'direction': 'negative', 'typical': -0.50},
    ('GBP/USD', 'USD/JPY'): {'direction': 'negative', 'typical': -0.45},
}

# Reverse lookups
for (a, b), v in list(PAIR_CORRELATIONS.items()):
    PAIR_CORRELATIONS[(b, a)] = v


class CrossPairAnalyzer:
    """
    Analyze cross-pair correlations to confirm or reject trading signals.
    """

    # Minimum candles for reliable correlation calculation
    MIN_CANDLES = 50
    # Rolling correlation window
    CORRELATION_WINDOW = 30
    # Lead/lag detection window (candles)
    LEAD_LAG_WINDOW = 5

    def __init__(self):
        self._price_cache = {}  # {pair: pd.Series of recent closes}

    def update_prices(self, pair: str, df: pd.DataFrame):
        """
        Cache latest price data for a pair (called during each analysis cycle).

        Args:
            pair: e.g. 'EUR/USD'
            df:   DataFrame with 'close' column
        """
        if df is not None and 'close' in df.columns and len(df) >= self.MIN_CANDLES:
            self._price_cache[pair] = df['close'].values.copy()

    def get_correlation_signal(self, target_pair: str, target_signal: str) -> dict:
        """
        Check if correlated pairs confirm or reject the proposed signal.

        Args:
            target_pair:   pair we're about to trade (e.g. 'EUR/USD')
            target_signal: proposed direction ('BUY' or 'SELL')

        Returns:
            {
                'confirmation_score': -1.0 to +1.0 (positive = confirms, negative = rejects),
                'correlated_signals': {pair: 'AGREES'|'DISAGREES'|'NEUTRAL'},
                'lead_lag': {pair: 'LEADS'|'LAGS'|'SYNC'},
                'reason': str,
            }
        """
        if target_signal not in ('BUY', 'SELL'):
            return self._neutral_result("No directional signal")

        target_prices = self._price_cache.get(target_pair)
        if target_prices is None or len(target_prices) < self.MIN_CANDLES:
            return self._neutral_result("Insufficient target pair data")

        confirmation_scores = []
        correlated_signals = {}
        lead_lag_info = {}
        reason_parts = []

        for other_pair, other_prices in self._price_cache.items():
            if other_pair == target_pair:
                continue
            if len(other_prices) < self.MIN_CANDLES:
                continue

            corr_info = PAIR_CORRELATIONS.get((target_pair, other_pair))
            if corr_info is None:
                continue

            # Calculate rolling correlation
            min_len = min(len(target_prices), len(other_prices))
            t_prices = target_prices[-min_len:]
            o_prices = other_prices[-min_len:]

            # Returns-based correlation (more stable than price-level)
            t_returns = np.diff(t_prices) / (t_prices[:-1] + 1e-10)
            o_returns = np.diff(o_prices) / (o_prices[:-1] + 1e-10)

            if len(t_returns) < self.CORRELATION_WINDOW:
                continue

            # Recent rolling correlation
            recent_t = t_returns[-self.CORRELATION_WINDOW:]
            recent_o = o_returns[-self.CORRELATION_WINDOW:]
            live_corr = np.corrcoef(recent_t, recent_o)[0, 1]

            if np.isnan(live_corr):
                continue

            # Determine other pair's recent direction
            other_recent_return = np.sum(o_returns[-5:])  # Last 5 candles
            other_direction = 'BUY' if other_recent_return > 0 else 'SELL'

            # Does the other pair agree with our signal?
            expected_direction = corr_info['direction']
            if expected_direction == 'positive':
                agrees = (target_signal == other_direction)
            else:  # negative correlation
                agrees = (target_signal != other_direction)

            if agrees:
                correlated_signals[other_pair] = 'AGREES'
                # Stronger confirmation when live correlation matches expected
                score = abs(live_corr) * 0.5
                reason_parts.append(f"{other_pair} confirms ({live_corr:+.2f} corr)")
            else:
                correlated_signals[other_pair] = 'DISAGREES'
                score = -abs(live_corr) * 0.5
                reason_parts.append(f"⚠️ {other_pair} diverges ({live_corr:+.2f} corr)")

            confirmation_scores.append(score)

            # Lead/lag detection
            lead_lag = self._detect_lead_lag(t_returns, o_returns)
            lead_lag_info[other_pair] = lead_lag
            if lead_lag != 'SYNC':
                reason_parts.append(f"{other_pair} {lead_lag}")

        if not confirmation_scores:
            return self._neutral_result("No correlated pair data available")

        avg_score = np.mean(confirmation_scores)
        avg_score = max(-1.0, min(1.0, avg_score))

        return {
            'confirmation_score': avg_score,
            'correlated_signals': correlated_signals,
            'lead_lag': lead_lag_info,
            'reason': " | ".join(reason_parts) if reason_parts else "Cross-pair neutral",
        }

    def get_confidence_modifier(self, target_pair: str, target_signal: str) -> float:
        """
        Get a confidence multiplier based on cross-pair confirmation.

        Returns:
            0.85 - 1.15 multiplier for the signal's confidence.
            > 1.0 if correlated pairs confirm, < 1.0 if they diverge.
        """
        result = self.get_correlation_signal(target_pair, target_signal)
        score = result['confirmation_score']

        # Map score to multiplier: -1.0 → 0.85, 0.0 → 1.0, +1.0 → 1.15
        if score >= 0:
            modifier = 1.0 + score * 0.15
        else:
            modifier = 1.0 + score * 0.15  # score is negative, so this reduces

        return round(modifier, 3)

    def _detect_lead_lag(self, t_returns, o_returns) -> str:
        """
        Detect if one pair leads the other using cross-correlation.

        Returns 'LEADS' if other pair leads target, 'LAGS' if it lags, 'SYNC' if unclear.
        """
        try:
            n = min(len(t_returns), len(o_returns), 30)
            t = t_returns[-n:]
            o = o_returns[-n:]

            # Cross-correlation at different lags
            best_lag = 0
            best_corr = abs(np.corrcoef(t, o)[0, 1])

            for lag in range(1, self.LEAD_LAG_WINDOW + 1):
                if lag >= n:
                    break
                # Other pair shifted forward → other leads
                corr_lead = abs(np.corrcoef(t[lag:], o[:-lag])[0, 1])
                if corr_lead > best_corr + 0.05:
                    best_corr = corr_lead
                    best_lag = lag

                # Other pair shifted backward → other lags
                corr_lag = abs(np.corrcoef(t[:-lag], o[lag:])[0, 1])
                if corr_lag > best_corr + 0.05:
                    best_corr = corr_lag
                    best_lag = -lag

            if best_lag > 0:
                return 'LEADS'
            elif best_lag < 0:
                return 'LAGS'
            return 'SYNC'
        except Exception:
            return 'SYNC'

    @staticmethod
    def _neutral_result(reason: str) -> dict:
        return {
            'confirmation_score': 0.0,
            'correlated_signals': {},
            'lead_lag': {},
            'reason': reason,
        }
