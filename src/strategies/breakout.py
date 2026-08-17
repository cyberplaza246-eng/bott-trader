"""
Breakout: Donchian(20) channel break with a volume confirmation filter
(current volume > 1.5x its 20-bar average). Stop at the opposite channel
edge; target is trailed by the engine (ATR-based), same as trend-following.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import donchian_channel

DONCHIAN_PERIOD = 20
VOLUME_LOOKBACK = 20
VOLUME_MULT = 1.5


class BreakoutStrategy:
    name = "breakout"

    def __init__(self, donchian_period: int = DONCHIAN_PERIOD, volume_lookback: int = VOLUME_LOOKBACK,
                 volume_mult: float = VOLUME_MULT):
        self.donchian_period = donchian_period
        self.volume_lookback = volume_lookback
        self.volume_mult = volume_mult

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
        upper, lower = donchian_channel(high, low, self.donchian_period)
        # Compare against the *prior* channel so the breakout bar itself isn't
        # absorbed into the rolling max/min it's being tested against.
        prior_upper = upper.shift(1)
        prior_lower = lower.shift(1)
        avg_volume = volume.rolling(self.volume_lookback).mean()
        volume_confirmed = volume > self.volume_mult * avg_volume

        long_entry = (close > prior_upper) & volume_confirmed
        short_entry = (close < prior_lower) & volume_confirmed

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = prior_lower[long_entry]
        stop_price[short_entry] = prior_upper[short_entry]

        target_price = pd.Series(np.nan, index=df.index)  # trailed by engine

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
