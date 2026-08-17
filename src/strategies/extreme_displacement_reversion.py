"""
Mean reversion after statistically extreme displacement — deliberately
minimal: "does an unusually large N-bar price move, relative to its own
recent distribution, predict reversion?" No Bollinger bands, no RSI, no ADX
regime gate, no VWAP. One mechanism: a z-scored return.

Rule: compute the N-bar return, z-scored against its own rolling mean/std.
When |z| >= `z_threshold`, fade the move (enter opposite direction). Stop
beyond the extreme of the displacement window; target is reversion to the
`target_ema_period`-bar EMA (the "recent normal" price level).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import ema

DISPLACEMENT_BARS = 5
ZSCORE_WINDOW = 100
Z_THRESHOLD = 2.5
TARGET_EMA_PERIOD = 20


class ExtremeDisplacementReversionStrategy:
    name = "extreme_displacement_reversion"

    def __init__(self, displacement_bars: int = DISPLACEMENT_BARS, zscore_window: int = ZSCORE_WINDOW,
                 z_threshold: float = Z_THRESHOLD, target_ema_period: int = TARGET_EMA_PERIOD):
        self.displacement_bars = displacement_bars
        self.zscore_window = zscore_window
        self.z_threshold = z_threshold
        self.target_ema_period = target_ema_period

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        close, high, low = df["close"], df["high"], df["low"]

        n_bar_return = close.pct_change(self.displacement_bars)
        mean = n_bar_return.rolling(self.zscore_window).mean()
        std = n_bar_return.rolling(self.zscore_window).std()
        z = (n_bar_return - mean) / std.replace(0, np.nan)

        long_entry = z <= -self.z_threshold   # extreme down move -> fade long
        short_entry = z >= self.z_threshold    # extreme up move -> fade short

        window_low = low.rolling(self.displacement_bars).min()
        window_high = high.rolling(self.displacement_bars).max()
        target_ema = ema(close, self.target_ema_period)

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = window_low[long_entry]
        stop_price[short_entry] = window_high[short_entry]

        target_price = pd.Series(np.nan, index=df.index)
        target_price[long_entry] = target_ema[long_entry]
        target_price[short_entry] = target_ema[short_entry]

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
