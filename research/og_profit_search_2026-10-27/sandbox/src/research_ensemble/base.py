"""Common strategy interface used by the backtest engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class StrategySignals:
    """Per-bar outputs a strategy hands to the backtest engine.

    entries: +1 (go long), -1 (go short), 0 (no new entry) per bar
    stop_price: initial stop-loss price for a new entry on that bar (NaN if entries==0)
    target_price: initial take-profit price for a new entry on that bar (NaN if entries==0 or trailing-only)
    """

    entries: pd.Series
    stop_price: pd.Series
    target_price: pd.Series


class Strategy(Protocol):
    name: str

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        ...
