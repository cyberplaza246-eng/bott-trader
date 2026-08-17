"""
Ensemble: ADX-based regime router rather than naive signal voting (trend and
mean-reversion signals routinely contradict each other, so voting would
cancel out most trades). Per bar:

  - ADX >= 25 (trending)   -> take trend-following signals
  - ADX < 20  (ranging)    -> take mean-reversion signals
  - ADX 20-25 (transition) -> neither trend nor mean-reversion trades
  - breakout signals are evaluated independently every bar and take
    priority over the regime-routed signal if both fire on the same bar
    (a real breakout should override a stale range-fade signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.breakout import BreakoutStrategy
from src.strategies.indicators import adx
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy

ADX_TREND_MIN = 25
ADX_RANGE_MAX = 20
ADX_PERIOD = 14


class EnsembleStrategy:
    name = "ensemble"

    def __init__(self) -> None:
        self._trend = TrendFollowingStrategy()
        self._mean_reversion = MeanReversionStrategy()
        self._breakout = BreakoutStrategy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        trend_sig = self._trend.generate_signals(df)
        mr_sig = self._mean_reversion.generate_signals(df)
        bo_sig = self._breakout.generate_signals(df)

        adx_ = adx(df["high"], df["low"], df["close"], ADX_PERIOD)
        trending = adx_ >= ADX_TREND_MIN
        ranging = adx_ < ADX_RANGE_MAX

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        entries[trending] = trend_sig.entries[trending]
        stop_price[trending] = trend_sig.stop_price[trending]
        target_price[trending] = trend_sig.target_price[trending]

        entries[ranging] = mr_sig.entries[ranging]
        stop_price[ranging] = mr_sig.stop_price[ranging]
        target_price[ranging] = mr_sig.target_price[ranging]

        # Breakout takes priority whenever it fires, regardless of regime.
        bo_fires = bo_sig.entries != 0
        entries[bo_fires] = bo_sig.entries[bo_fires]
        stop_price[bo_fires] = bo_sig.stop_price[bo_fires]
        target_price[bo_fires] = bo_sig.target_price[bo_fires]

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
