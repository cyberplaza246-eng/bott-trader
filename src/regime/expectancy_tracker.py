"""
Regime-conditioned expectancy tracker — answers "what has this strategy
actually earned, net of costs, when this regime showed up before?" using
only trades that had already closed as of the query time. This is the
no-lookahead guarantee the whole adaptive engine depends on: a trade's
outcome isn't knowable until it exits, so it can only inform decisions made
strictly after its exit_time.

Each candidate strategy's trade population comes from its own independent
`run_backtest()` result (as if it traded every one of its own signals in
isolation) — not from trades it happened to "win" a selection contest for.
Every trade is tagged with the regime label (trend/range/breakout) that had
the highest score at that trade's *entry* bar.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.backtest.engine import BacktestResult

REGIME_COLUMNS = {"trend": "trend_score", "range": "range_score", "breakout": "breakout_score"}


def _regime_label_at(regime_scores: pd.DataFrame, ts: pd.Timestamp) -> str:
    row = regime_scores.loc[ts]
    return max(REGIME_COLUMNS, key=lambda label: row[REGIME_COLUMNS[label]])


@dataclass
class _SeriesEntry:
    exit_time: pd.Timestamp
    net_pnl: float


class RegimeExpectancyTracker:
    def __init__(self, strategy_results: Dict[str, BacktestResult], regime_scores: pd.DataFrame):
        self._series: Dict[Tuple[str, str], List[_SeriesEntry]] = {}

        for strategy_name, result in strategy_results.items():
            for trade in result.trades:
                if trade.entry_time not in regime_scores.index:
                    continue
                regime = _regime_label_at(regime_scores, trade.entry_time)
                key = (regime, strategy_name)
                self._series.setdefault(key, []).append(_SeriesEntry(trade.exit_time, trade.pnl))

        for key in self._series:
            self._series[key].sort(key=lambda e: e.exit_time)

        self._exit_times: Dict[Tuple[str, str], List[pd.Timestamp]] = {
            key: [e.exit_time for e in entries] for key, entries in self._series.items()
        }

    def expectancy_as_of(self, regime: str, strategy_name: str, as_of_time: pd.Timestamp) -> Tuple[float, int]:
        """Mean net P&L and sample size, using only trades with exit_time < as_of_time."""
        key = (regime, strategy_name)
        entries = self._series.get(key)
        if not entries:
            return 0.0, 0

        exit_times = self._exit_times[key]
        cutoff = bisect.bisect_left(exit_times, as_of_time)
        if cutoff == 0:
            return 0.0, 0

        pnls = [e.net_pnl for e in entries[:cutoff]]
        return sum(pnls) / len(pnls), len(pnls)
