"""
Regime-score argmax allocator — routes each bar's trade decision to whichever
strategy pool member fits that bar's highest regime score, instead of a
single hard threshold (compare to EnsembleStrategy's fixed ADX>=25/<20 gate).

SUPERSEDED: walk-forward testing showed this argmax-on-a-heuristic-score
selection rule performs badly (MNQ: -$41,495 OOS, t=-5.99; MES: -$59,928,
t=-8.07) — worse than several individual strategies. Kept for comparison,
but `src/regime/adaptive_engine.py` (expectancy-conditioned selection) is
the intended replacement; see that module and the project plan for why.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.regime.engine import compute_regime_scores
from src.strategies.base import StrategySignals

MIN_REGIME_CONFIDENCE = 0.34  # slightly above the 1/3 uniform baseline


class RegimeAllocatorStrategy:
    name = "regime_bot"

    def __init__(self, regime_strategy_map: Dict[str, List[object]] | None = None,
                 min_regime_confidence: float = MIN_REGIME_CONFIDENCE):
        if regime_strategy_map is None:
            from src.strategies.breakout import BreakoutStrategy
            from src.strategies.mean_reversion import MeanReversionStrategy
            from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
            from src.strategies.trend_following import TrendFollowingStrategy
            from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy

            regime_strategy_map = {
                "trend": [TrendFollowingStrategy(), VwapPullbackTrendStrategy()],
                "range": [MeanReversionStrategy()],
                "breakout": [OpeningRangeBreakoutStrategy(), BreakoutStrategy()],
            }
        self.regime_strategy_map = regime_strategy_map
        self.min_regime_confidence = min_regime_confidence

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        regime_scores = compute_regime_scores(df)

        sub_signals: Dict[str, List[StrategySignals]] = {
            regime: [strat.generate_signals(df) for strat in strategies]
            for regime, strategies in self.regime_strategy_map.items()
        }

        score_cols = {"trend": "trend_score", "range": "range_score", "breakout": "breakout_score"}
        probs_arr = regime_scores[[score_cols[r] for r in self.regime_strategy_map]].to_numpy()
        regime_names = list(self.regime_strategy_map)
        best_regime_idx = probs_arr.argmax(axis=1)
        best_regime_prob = probs_arr.max(axis=1)

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        for i in range(len(df)):
            if best_regime_prob[i] < self.min_regime_confidence:
                continue
            regime = regime_names[best_regime_idx[i]]
            for sig in sub_signals[regime]:
                signal_val = sig.entries.iloc[i]
                if signal_val != 0:
                    entries.iloc[i] = signal_val
                    stop_price.iloc[i] = sig.stop_price.iloc[i]
                    target_price.iloc[i] = sig.target_price.iloc[i]
                    break  # first strategy in that regime's list to fire wins

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
