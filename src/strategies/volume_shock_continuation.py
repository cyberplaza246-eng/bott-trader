"""
Volume shock continuation — deliberately minimal: "does an unusually large
volume spike combined with a directional close predict continuation?"
Distinct mechanism from vol_expansion_momentum (which keys off ATR/range,
not volume specifically). No VWAP, no trend filter, no session windows.

Rule: volume >= `volume_shock_mult` x its own `volume_baseline_period`
-bar average, AND the bar closes beyond the prior bar's high/low (a
directional break, not just a big-volume doji) -> enter in that direction.
Stop at the shock bar's opposite extreme; target left to the engine's ATR
trailing stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals

VOLUME_BASELINE_PERIOD = 20
VOLUME_SHOCK_MULT = 3.0


class VolumeShockContinuationStrategy:
    name = "volume_shock_continuation"

    def __init__(self, volume_shock_mult: float = VOLUME_SHOCK_MULT, volume_baseline_period: int = VOLUME_BASELINE_PERIOD):
        self.volume_shock_mult = volume_shock_mult
        self.volume_baseline_period = volume_baseline_period

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

        avg_volume = volume.rolling(self.volume_baseline_period).mean()
        volume_shock = volume >= self.volume_shock_mult * avg_volume

        prior_high = high.shift(1)
        prior_low = low.shift(1)

        long_entry = volume_shock & (close > prior_high)
        short_entry = volume_shock & (close < prior_low)

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = low[long_entry]
        stop_price[short_entry] = high[short_entry]

        target_price = pd.Series(np.nan, index=df.index)  # trailed by engine

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
