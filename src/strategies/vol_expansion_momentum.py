"""
Momentum continuation after volatility expansion — deliberately minimal:
"does a volatility shock followed by a strongly directional bar predict
continuation in that direction?" No VWAP, no multi-day trend filter, no
session windows, no regime gating. One mechanism only.

Rule: ATR(14) jumps above `atr_expansion_mult` x its own `atr_baseline`
-period average on this bar, AND the bar's body is directional and large
relative to its own range (>= `min_body_range_pct` of high-low) -> enter in
the direction of that bar's close vs open. Stop beyond the shock bar's
opposite extreme; target left to the engine's ATR trailing stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import atr, sma

ATR_PERIOD = 14
ATR_BASELINE_PERIOD = 50
ATR_EXPANSION_MULT = 1.5
MIN_BODY_RANGE_PCT = 0.6


class VolExpansionMomentumStrategy:
    name = "vol_expansion_momentum"

    def __init__(self, atr_expansion_mult: float = ATR_EXPANSION_MULT,
                 min_body_range_pct: float = MIN_BODY_RANGE_PCT, atr_baseline_period: int = ATR_BASELINE_PERIOD):
        self.atr_expansion_mult = atr_expansion_mult
        self.min_body_range_pct = min_body_range_pct
        self.atr_baseline_period = atr_baseline_period

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]

        atr_ = atr(high, low, close, ATR_PERIOD)
        atr_baseline = sma(atr_, self.atr_baseline_period)
        vol_shock = atr_ > self.atr_expansion_mult * atr_baseline

        bar_range = (high - low).replace(0, np.nan)
        body = (close - open_).abs()
        directional = (body / bar_range) >= self.min_body_range_pct

        long_entry = vol_shock & directional & (close > open_)
        short_entry = vol_shock & directional & (close < open_)

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = low[long_entry]
        stop_price[short_entry] = high[short_entry]

        target_price = pd.Series(np.nan, index=df.index)  # trailed by engine

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
