"""
Mean reversion: fade closes outside Bollinger(20, 2sigma) confirmed by
RSI(14) overbought/oversold, gated by ADX(14) < 20 so it doesn't fight real
trends. Target is the middle band (mean); stop is the opposite band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import adx, bollinger_bands, rsi

BB_PERIOD, BB_STD = 20, 2.0
RSI_PERIOD = 14
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70
ADX_PERIOD = 14
ADX_RANGE_MAX = 20


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, bb_period: int = BB_PERIOD, bb_std: float = BB_STD,
                 rsi_period: int = RSI_PERIOD, rsi_oversold: float = RSI_OVERSOLD,
                 rsi_overbought: float = RSI_OVERBOUGHT, adx_period: int = ADX_PERIOD,
                 adx_range_max: float = ADX_RANGE_MAX):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_period = adx_period
        self.adx_range_max = adx_range_max

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        close = df["close"]
        upper, mid, lower = bollinger_bands(close, self.bb_period, self.bb_std)
        rsi_ = rsi(close, self.rsi_period)
        adx_ = adx(df["high"], df["low"], close, self.adx_period)

        ranging = adx_ < self.adx_range_max
        long_entry = (close <= lower) & (rsi_ < self.rsi_oversold) & ranging
        short_entry = (close >= upper) & (rsi_ > self.rsi_overbought) & ranging

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = lower[long_entry] - (mid[long_entry] - lower[long_entry])
        stop_price[short_entry] = upper[short_entry] + (upper[short_entry] - mid[short_entry])

        target_price = pd.Series(np.nan, index=df.index)
        target_price[long_entry] = mid[long_entry]
        target_price[short_entry] = mid[short_entry]

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
