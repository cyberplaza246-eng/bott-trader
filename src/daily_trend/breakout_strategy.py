"""
Deliberately boring daily N-day breakout trend strategy. One mechanism,
no stacking, no filters:

    close > previous N-day high  -> long
    close < previous N-day low   -> short

"Previous" means the rolling window is shifted by one bar BEFORE taking the
max/min, so today's own high/low never leaks into the threshold it's being
tested against (the bug flagged in the earlier proposed version: comparing
close against a window that already includes today's own high is circular
and can never trigger correctly).

Initial stop is ATR(20)-based; there is no fixed target -- exits happen via
the engine's ATR trailing stop (same mechanism already used and tested
elsewhere in this project), which is the "maintain/exit according to a
predefined rule" from the spec, kept as simple as the entry rule.

Execution: the breakout is detected using day T's own close, but the
engine fills entries at the close of whichever bar carries the signal --
so the whole signal (entries/stop/target) is shifted forward one bar here,
meaning the fill happens on day T+1 (the next tradable session), never on
the same bar whose close produced the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import atr

ATR_PERIOD = 20
ATR_STOP_MULT = 2.0


class DailyBreakoutStrategy:
    def __init__(self, lookback: int):
        self.lookback = lookback
        self.name = f"daily_breakout_{lookback}"

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close = df["high"], df["low"], df["close"]

        previous_high = high.shift(1).rolling(self.lookback).max()
        previous_low = low.shift(1).rolling(self.lookback).min()

        long_entry = close > previous_high
        short_entry = close < previous_low

        atr_ = atr(high, low, close, ATR_PERIOD)

        entries = pd.Series(0, index=df.index)
        entries[long_entry] = 1
        entries[short_entry] = -1

        stop_price = pd.Series(np.nan, index=df.index)
        stop_price[long_entry] = close[long_entry] - ATR_STOP_MULT * atr_[long_entry]
        stop_price[short_entry] = close[short_entry] + ATR_STOP_MULT * atr_[short_entry]

        target_price = pd.Series(np.nan, index=df.index)  # trailed by engine

        # Execute next session, not on the bar whose close produced the signal.
        entries = entries.shift(1).fillna(0).astype(int)
        stop_price = stop_price.shift(1)
        target_price = target_price.shift(1)

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
