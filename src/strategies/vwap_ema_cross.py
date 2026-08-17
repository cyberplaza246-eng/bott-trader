"""
VWAP + EMA(10/20) cross with confirmation delay — encodes a discretionary
process: wait for the fast/slow EMA to cross, then wait a few more candles
to confirm the cross is holding (and that price is on the correct side of
session VWAP) before entering, rather than reacting to the cross itself.

Entry rules, evaluated once per cross:
  1. EMA(10) crosses EMA(20) -> starts a "watch window" for that cross.
  2. If CONFIRM_BARS candles later the cross hasn't been invalidated by an
     opposite cross, AND close is on the trade side of session VWAP
     (above for long, below for short) -> enter.
  3. If VWAP never lines up by the confirmation bar, that cross is skipped
     (no late entries chasing a move well past the signal).
  4. Only one entry per cross event.

Stop is placed beyond the swing formed during the watch window (the low/high
since the cross), with a small ATR buffer. Target is left to the engine's
ATR trailing stop, same as trend-following, since this is a trend-following
entry style.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import atr, ema, session_vwap

FAST, SLOW = 10, 20
CONFIRM_BARS = 3
ATR_PERIOD = 14
ATR_STOP_BUFFER_MULT = 0.5


class VwapEmaCrossStrategy:
    name = "vwap_cross"

    def __init__(self, fast: int = FAST, slow: int = SLOW, confirm_bars: int = CONFIRM_BARS,
                 atr_period: int = ATR_PERIOD, atr_stop_buffer_mult: float = ATR_STOP_BUFFER_MULT):
        self.fast = fast
        self.slow = slow
        self.confirm_bars = confirm_bars
        self.atr_period = atr_period
        self.atr_stop_buffer_mult = atr_stop_buffer_mult

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        vwap = session_vwap(high, low, close, volume)
        atr_ = atr(high, low, close, self.atr_period)

        cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        cu = cross_up.to_numpy()
        cd = cross_down.to_numpy()
        cl = close.to_numpy()
        vw = vwap.to_numpy()
        at = atr_.to_numpy()
        lo = low.to_numpy()
        hi = high.to_numpy()

        cross_dir = 0
        cross_idx: int | None = None
        entered_for_cross = False

        for i in range(len(df)):
            if cu[i]:
                cross_dir, cross_idx, entered_for_cross = 1, i, False
            elif cd[i]:
                cross_dir, cross_idx, entered_for_cross = -1, i, False

            if cross_dir == 0 or entered_for_cross or cross_idx is None:
                continue

            bars_since = i - cross_idx
            if bars_since != self.confirm_bars:
                continue  # too early, or the confirmation bar has already passed

            if cross_dir == 1 and cl[i] > vw[i]:
                entries.iloc[i] = 1
                window_low = lo[cross_idx : i + 1].min()
                stop_price.iloc[i] = window_low - self.atr_stop_buffer_mult * at[i]
                entered_for_cross = True
            elif cross_dir == -1 and cl[i] < vw[i]:
                entries.iloc[i] = -1
                window_high = hi[cross_idx : i + 1].max()
                stop_price.iloc[i] = window_high + self.atr_stop_buffer_mult * at[i]
                entered_for_cross = True

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
