"""Sanity checks for the regime score engine and allocator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.regime.engine import compute_regime_scores


def _make_session_df(n_days=10, bars_per_day=288, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n_days * bars_per_day, freq="5min", tz="UTC")
    price = 20000.0
    closes = []
    for _ in range(n_days * bars_per_day):
        price += rng.normal(0, 3.0)
        closes.append(price)
    closes = np.array(closes)
    high = closes + np.abs(rng.normal(2.0, 1.0, size=len(closes)))
    low = closes - np.abs(rng.normal(2.0, 1.0, size=len(closes)))
    volume = rng.integers(500, 2000, size=len(closes)).astype(float)
    return pd.DataFrame({"open": closes, "high": high, "low": low, "close": closes, "volume": volume}, index=idx)


def test_regime_scores_sum_to_one_and_are_bounded():
    df = _make_session_df()
    scores = compute_regime_scores(df)
    score_cols = scores[["trend_score", "range_score", "breakout_score"]]
    totals = score_cols.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-8)
    assert (score_cols.to_numpy() >= 0).all()
    assert (score_cols.to_numpy() <= 1).all()


def test_regime_scores_have_expected_columns():
    df = _make_session_df(n_days=3)
    scores = compute_regime_scores(df)
    assert set(scores.columns) == {"trend_score", "range_score", "breakout_score", "transition_score"}
    assert len(scores) == len(df)


def test_transition_score_is_nonnegative():
    df = _make_session_df(n_days=5)
    scores = compute_regime_scores(df)
    assert (scores["transition_score"] >= 0).all()
