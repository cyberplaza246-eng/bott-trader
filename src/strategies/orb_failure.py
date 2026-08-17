"""
Opening Range Failure (fade the trapped breakout traders) — the mirror
image of OpeningRangeBreakoutStrategy. Instead of trading with a break of
the opening range, this watches for a break that *fails*: price closes
beyond the opening range, then within a few bars closes back inside it —
trapping the breakout traders who just entered, and fading back toward
VWAP/the opposite side of the range.

Rules:
  1. Opening range = high/low of the first `or_minutes` after RTH open
     (09:30 ET), per session.
  2. Watch for a breakout attempt (close beyond the range) within
     `watch_window_minutes` of the range forming.
  3. If within `failure_bars` bars the price closes back inside the range,
     that's a failed breakout -> enter fading back toward the range/VWAP.
  4. Stop beyond the extreme reached during the failed attempt; target is
     left to the engine's ATR trailing stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals

RTH_OPEN_HOUR, RTH_OPEN_MINUTE = 9, 30
OR_MINUTES = 15
WATCH_WINDOW_MINUTES = 120
FAILURE_BARS = 4
SESSION_TZ = "America/New_York"


class OrbFailureStrategy:
    name = "orb_failure"

    def __init__(self, or_minutes: int = OR_MINUTES, watch_window_minutes: int = WATCH_WINDOW_MINUTES,
                 failure_bars: int = FAILURE_BARS):
        self.or_minutes = or_minutes
        self.watch_window_minutes = watch_window_minutes
        self.failure_bars = failure_bars

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        local_index = df.index.tz_convert(SESSION_TZ)
        minutes_since_open = (local_index.hour * 60 + local_index.minute) - (RTH_OPEN_HOUR * 60 + RTH_OPEN_MINUTE)
        session_date = local_index.date

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        mins = np.asarray(minutes_since_open)
        dates = np.asarray(session_date)

        for d in pd.unique(dates):
            day_mask = dates == d
            or_idx = np.where(day_mask & (mins >= 0) & (mins < self.or_minutes))[0]
            if len(or_idx) == 0:
                continue

            or_high = high[or_idx].max()
            or_low = low[or_idx].min()

            watch_idx = np.where(
                day_mask & (mins >= self.or_minutes) & (mins < self.or_minutes + self.watch_window_minutes)
            )[0]

            breakout_dir = 0
            breakout_extreme = None
            bars_since_breakout = 0
            traded = False

            for i in watch_idx:
                if traded:
                    break

                if breakout_dir == 0:
                    if close[i] > or_high:
                        breakout_dir, breakout_extreme, bars_since_breakout = 1, high[i], 0
                    elif close[i] < or_low:
                        breakout_dir, breakout_extreme, bars_since_breakout = -1, low[i], 0
                    continue

                bars_since_breakout += 1
                if breakout_dir == 1:
                    breakout_extreme = max(breakout_extreme, high[i])
                else:
                    breakout_extreme = min(breakout_extreme, low[i])

                failed = (breakout_dir == 1 and close[i] < or_high) or (breakout_dir == -1 and close[i] > or_low)
                if failed:
                    entries.iloc[i] = -breakout_dir  # fade the failed breakout
                    stop_price.iloc[i] = breakout_extreme
                    traded = True
                elif bars_since_breakout >= self.failure_bars:
                    # Breakout held past the failure window -> stop watching,
                    # don't fade a breakout that's actually working.
                    breakout_dir = 0

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
