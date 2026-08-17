"""Integration invariants for run_adaptive_backtest on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.instruments import get_spec
from src.regime.adaptive_engine import MIN_SAMPLE, run_adaptive_backtest
from src.risk.position_sizing import contracts_for_risk
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.strategies.trend_following import TrendFollowingStrategy


def _make_choppy_trending_df(n_days=90, bars_per_day=288, seed=7):
    """Long enough, volatile enough synthetic data for trend/mean-reversion
    signals to fire repeatedly across many sessions -- needed so the
    expectancy tracker can actually accumulate >= MIN_SAMPLE trades within
    the test window."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n_days * bars_per_day, freq="5min", tz="UTC")
    price = 20000.0
    closes = []
    for day in range(n_days):
        drift = rng.choice([-1, 1]) * 8.0
        for _ in range(bars_per_day):
            price += rng.normal(drift / bars_per_day, 4.0)
            closes.append(price)
    closes = np.array(closes)
    high = closes + np.abs(rng.normal(3.0, 1.5, size=len(closes)))
    low = closes - np.abs(rng.normal(3.0, 1.5, size=len(closes)))
    volume = rng.integers(500, 3000, size=len(closes)).astype(float)
    return pd.DataFrame({"open": closes, "high": high, "low": low, "close": closes, "volume": volume}, index=idx)


def _default_pool():
    return {
        "trend": [TrendFollowingStrategy()],
        "range": [MeanReversionStrategy()],
        "breakout": [OpeningRangeBreakoutStrategy()],
    }


def test_no_executed_trade_below_min_sample():
    df = _make_choppy_trending_df()
    result, log = run_adaptive_backtest(df, "MNQ", "5m", _default_pool(), 50000, 0.5)

    executed = log[log["selected_strategy"] != "no_trade"]
    for _, row in executed.iterrows():
        strat_name = row["selected_strategy"]
        n_col = f"{strat_name}_ev_n"
        assert row[n_col] >= MIN_SAMPLE, f"trade executed with only {row[n_col]} prior samples"


def test_gross_minus_commission_equals_net_for_executed_trades():
    df = _make_choppy_trending_df()
    result, log = run_adaptive_backtest(df, "MNQ", "5m", _default_pool(), 50000, 0.5)

    closed = log.dropna(subset=["net_pnl"])
    if len(closed) == 0:
        return  # nothing closed in this synthetic window -- not a failure of the invariant
    reconciled = (closed["gross_pnl"] - closed["commission"] - closed["net_pnl"]).abs()
    assert (reconciled < 1e-6).all()


def test_research_log_schema_has_required_columns():
    df = _make_choppy_trending_df(n_days=20)
    result, log = run_adaptive_backtest(df, "MNQ", "5m", _default_pool(), 50000, 0.5)
    required = {
        "timestamp", "trend_score", "range_score", "breakout_score", "transition_score",
        "transition", "dominant_regime", "selected_strategy", "entry", "stop", "target",
        "position_size", "gross_pnl", "commission", "net_pnl",
    }
    assert required.issubset(set(log.columns))
    assert len(log) == len(df)


def test_transition_trade_uses_halved_risk_budget():
    df = _make_choppy_trending_df()
    result, log = run_adaptive_backtest(df, "MNQ", "5m", _default_pool(), 50000, 0.5)

    executed = log[log["selected_strategy"] != "no_trade"].reset_index(drop=True)
    if executed.empty:
        return

    spec = get_spec("MNQ")
    # result.equity_curve holds the exact equity the engine used for sizing
    # at each bar (already reflects any same-bar exit before the same-bar
    # entry decision), so look it up directly rather than reconstructing it.
    for _, row in executed.iterrows():
        equity_at_entry = result.equity_curve.loc[row["timestamp"]]
        full_size = contracts_for_risk(equity_at_entry, 0.5, row["entry"], row["stop"], spec)
        half_size = contracts_for_risk(equity_at_entry, 0.25, row["entry"], row["stop"], spec)
        if row["transition"]:
            assert row["position_size"] == half_size
        else:
            assert row["position_size"] == full_size
