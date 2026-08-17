"""
Trend following: EMA(20)/EMA(50) crossover confirmed by the EMA(200) trend
filter. Stop is ATR(14)-based; target is trailed by the engine rather than a
fixed take-profit (target_price left NaN => engine uses ATR trailing stop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import atr, ema

FAST, SLOW, TREND = 20, 50, 200
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0


class TrendFollowingStrategy:
    name = "trend"

    def __init__(self, fast: int = FAST, slow: int = SLOW, trend: int = TREND,
                 atr_period: int = ATR_PERIOD, atr_stop_mult: float = ATR_STOP_MULT):
        self.fast = fast
        self.slow = slow
        self.trend = trend
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        trend_ema = ema(close, self.trend)
        atr_ = atr(df["high"], df["low"], close, self.atr_period)

        cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        long_entry = cross_up & (close > trend_ema)
        short_entry = cross_down & (close < trend_ema)

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = close[long_entry] - self.atr_stop_mult * atr_[long_entry]
        stop_price[short_entry] = close[short_entry] + self.atr_stop_mult * atr_[short_entry]

        target_price = pd.Series(np.nan, index=df.index)  # trailed by engine

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
