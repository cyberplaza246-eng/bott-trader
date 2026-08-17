"""No-lookahead and correctness checks for RegimeExpectancyTracker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, Trade
from src.regime.expectancy_tracker import RegimeExpectancyTracker


def _trade(entry_time, exit_time, pnl):
    return Trade(
        symbol="MNQ", direction=1, entry_time=entry_time, entry_price=100.0,
        exit_time=exit_time, exit_price=101.0, contracts=1,
        gross_pnl=pnl, commission=0.0, pnl=pnl, exit_reason="target",
    )


def _regime_scores(index, trend_dominant_at):
    """All bars 'range' dominant except the given timestamps, which are 'trend' dominant."""
    trend = pd.Series(0.2, index=index)
    range_ = pd.Series(0.6, index=index)
    breakout = pd.Series(0.2, index=index)
    trend.loc[trend_dominant_at] = 0.8
    range_.loc[trend_dominant_at] = 0.1
    breakout.loc[trend_dominant_at] = 0.1
    return pd.DataFrame({"trend_score": trend, "range_score": range_, "breakout_score": breakout}, index=index)


def test_expectancy_ignores_trades_not_yet_closed():
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    entry_times = idx[[1, 3, 5, 7]]
    scores = _regime_scores(idx, trend_dominant_at=entry_times)

    trades = [
        _trade(idx[1], idx[2], 100.0),
        _trade(idx[3], idx[4], -50.0),
        _trade(idx[5], idx[6], 200.0),
    ]
    result = BacktestResult(symbol="MNQ", strategy="trend", timeframe="5m", trades=trades, equity_curve=pd.Series())
    tracker = RegimeExpectancyTracker({"trend": result}, scores)

    # As of idx[3] (before trade #2 exits at idx[4]), only trade #1 (exited idx[2]) counts.
    mean_pnl, n = tracker.expectancy_as_of("trend", "trend", idx[3])
    assert n == 1
    assert mean_pnl == 100.0

    # As of idx[6] (after trade #2's exit at idx[4], at trade #3's exit), trades 1&2 count
    # (trade #3 exits exactly at idx[6], so it is NOT < idx[6] and must not count yet).
    mean_pnl, n = tracker.expectancy_as_of("trend", "trend", idx[6])
    assert n == 2
    assert mean_pnl == (100.0 - 50.0) / 2

    # As of a time after all three exits, all three count.
    mean_pnl, n = tracker.expectancy_as_of("trend", "trend", idx[9])
    assert n == 3
    assert np.isclose(mean_pnl, (100.0 - 50.0 + 200.0) / 3)


def test_expectancy_zero_sample_before_any_trade_closes():
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    scores = _regime_scores(idx, trend_dominant_at=idx[[1]])
    trades = [_trade(idx[1], idx[4], 100.0)]
    result = BacktestResult(symbol="MNQ", strategy="trend", timeframe="5m", trades=trades, equity_curve=pd.Series())
    tracker = RegimeExpectancyTracker({"trend": result}, scores)

    mean_pnl, n = tracker.expectancy_as_of("trend", "trend", idx[2])
    assert n == 0
    assert mean_pnl == 0.0


def test_unknown_regime_strategy_pair_returns_zero_sample():
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    scores = _regime_scores(idx, trend_dominant_at=idx[[1]])
    result = BacktestResult(symbol="MNQ", strategy="trend", timeframe="5m", trades=[], equity_curve=pd.Series())
    tracker = RegimeExpectancyTracker({"trend": result}, scores)

    mean_pnl, n = tracker.expectancy_as_of("range", "mean_reversion", idx[4])
    assert n == 0
    assert mean_pnl == 0.0


def test_trades_tagged_by_regime_at_entry_not_exit():
    # A trade enters during a 'trend' bar but exits during a bar that would
    # be 'range' dominant if (incorrectly) looked up at exit time instead.
    idx = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")
    scores = _regime_scores(idx, trend_dominant_at=idx[[1]])  # only bar 1 is trend-dominant
    trades = [_trade(idx[1], idx[4], 77.0)]  # entry at trend bar, exit at range bar
    result = BacktestResult(symbol="MNQ", strategy="trend", timeframe="5m", trades=trades, equity_curve=pd.Series())
    tracker = RegimeExpectancyTracker({"trend": result}, scores)

    trend_mean, trend_n = tracker.expectancy_as_of("trend", "trend", idx[5])
    range_mean, range_n = tracker.expectancy_as_of("range", "trend", idx[5])
    assert trend_n == 1 and trend_mean == 77.0
    assert range_n == 0
