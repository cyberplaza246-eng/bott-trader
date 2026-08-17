"""
RTH Opening Range Breakout (ORB), anchored to the regular trading session
open (09:30 America/New_York, matching the NYSE cash-market open) — a
structurally different signal from the rolling-lookback BreakoutStrategy
already in this project.

Motivated by published research on index futures ORB and intraday
seasonality (see README references):
  - Raw/unfiltered ORB has weak-to-no edge on ES/NQ in several published
    backtests and a 2026 systematic falsification study on MNQ specifically.
  - Documented U-shaped intraday volume/volatility (high near the open,
    quiet at midday) motivates restricting entries to a window after the
    open and requiring volume confirmation on the breakout bar, rather than
    trading a rolling breakout at any time of day.

Rules:
  1. Opening range = high/low of the first `or_minutes` after RTH open
     (09:30 ET), computed per session (calendar day in America/New_York).
  2. Entries only allowed within `entry_window_minutes` of the RTH open
     (default 120 = 09:30-11:30 ET), and at most one entry per session.
  3. The breakout bar must close beyond the opening range AND have volume
     above `volume_mult` x the opening range's own average volume.
  4. Stop at the opposite side of the opening range; target left to the
     engine's ATR trailing stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals

RTH_OPEN_HOUR, RTH_OPEN_MINUTE = 9, 30  # America/New_York local time
OR_MINUTES = 15
ENTRY_WINDOW_MINUTES = 120
VOLUME_MULT = 1.3
SESSION_TZ = "America/New_York"


class OpeningRangeBreakoutStrategy:
    name = "orb"

    def __init__(self, or_minutes: int = OR_MINUTES, entry_window_minutes: int = ENTRY_WINDOW_MINUTES,
                 volume_mult: float = VOLUME_MULT):
        self.or_minutes = or_minutes
        self.entry_window_minutes = entry_window_minutes
        self.volume_mult = volume_mult

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
        volume = df["volume"].to_numpy()
        mins = np.asarray(minutes_since_open)
        dates = np.asarray(session_date)

        for d in pd.unique(dates):
            day_mask = dates == d

            or_idx = np.where(day_mask & (mins >= 0) & (mins < self.or_minutes))[0]
            if len(or_idx) == 0:
                continue  # no RTH bars this calendar day (weekend/holiday)

            or_high = high[or_idx].max()
            or_low = low[or_idx].min()
            or_avg_vol = volume[or_idx].mean()

            entry_idx = np.where(
                day_mask & (mins >= self.or_minutes) & (mins < self.or_minutes + self.entry_window_minutes)
            )[0]

            for i in entry_idx:
                volume_confirmed = volume[i] > self.volume_mult * or_avg_vol
                if close[i] > or_high and volume_confirmed:
                    entries.iloc[i] = 1
                    stop_price.iloc[i] = or_low
                    break
                if close[i] < or_low and volume_confirmed:
                    entries.iloc[i] = -1
                    stop_price.iloc[i] = or_high
                    break

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
