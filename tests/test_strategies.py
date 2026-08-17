"""Each strategy should produce the expected signal shape on constructed patterns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies.breakout import BreakoutStrategy
from src.strategies.extreme_displacement_reversion import ExtremeDisplacementReversionStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.strategies.orb_failure import OrbFailureStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.vol_expansion_momentum import VolExpansionMomentumStrategy
from src.strategies.volume_shock_continuation import VolumeShockContinuationStrategy
from src.strategies.vwap_ema_cross import CONFIRM_BARS, VwapEmaCrossStrategy
from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy


def _make_df(closes, volumes=None, high_pad=1.0, low_pad=1.0):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="5min")
    closes = np.array(closes, dtype=float)
    if volumes is None:
        volumes = np.full(len(closes), 1000)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + high_pad,
            "low": closes - low_pad,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


def test_trend_following_flags_clean_uptrend():
    # Flat base then a steady, sustained rally long enough for EMA200 to warm up.
    closes = [20000] * 220 + [20000 + i * 3 for i in range(60)]
    df = _make_df(closes)
    signals = TrendFollowingStrategy().generate_signals(df)
    assert (signals.entries == 1).sum() >= 1
    long_entries = signals.entries[signals.entries == 1]
    for idx in long_entries.index:
        assert not np.isnan(signals.stop_price[idx])
        assert signals.stop_price[idx] < df.loc[idx, "close"]


def test_mean_reversion_fades_range_extremes():
    # Long flat warmup (ADX decays near 0) then a single-bar spike below the
    # lower Bollinger band with RSI deeply oversold, still in a low-ADX regime.
    closes = [20000] * 150 + [19950] + [20000] * 20
    df = _make_df(closes)
    signals = MeanReversionStrategy().generate_signals(df)
    assert (signals.entries != 0).sum() >= 1


def test_breakout_fires_on_range_break_with_volume():
    # Flat range for 25 bars, then a sharp breakout bar with a volume spike.
    closes = [20000 + (i % 3) for i in range(25)] + [20100]
    volumes = [1000] * 25 + [5000]
    df = _make_df(closes, volumes=volumes)
    signals = BreakoutStrategy().generate_signals(df)
    assert signals.entries.iloc[-1] == 1


def test_breakout_requires_volume_confirmation():
    # Same price breakout but volume stays flat -> should NOT fire.
    closes = [20000 + (i % 3) for i in range(25)] + [20100]
    volumes = [1000] * 26
    df = _make_df(closes, volumes=volumes)
    signals = BreakoutStrategy().generate_signals(df)
    assert signals.entries.iloc[-1] == 0


def test_vwap_cross_enters_after_confirmation_delay():
    # Flat base so EMA10/20 settle, then a sustained rally: EMA10 crosses
    # above EMA20, price runs well above VWAP, and stays there through the
    # confirmation window -> should enter long exactly CONFIRM_BARS after
    # the cross, not on the cross bar itself.
    closes = [20000] * 30 + [20000 + i * 4 for i in range(15)]
    df = _make_df(closes)
    signals = VwapEmaCrossStrategy().generate_signals(df)

    assert (signals.entries == 1).sum() == 1
    entry_idx = signals.entries[signals.entries == 1].index[0]
    entry_pos = df.index.get_loc(entry_idx)

    from src.strategies.indicators import ema
    fast_ema = ema(df["close"], 10)
    slow_ema = ema(df["close"], 20)
    cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    cross_pos = np.where(cross_up.to_numpy())[0][0]

    assert entry_pos - cross_pos == CONFIRM_BARS
    assert not np.isnan(signals.stop_price[entry_idx])
    assert signals.stop_price[entry_idx] < df.loc[entry_idx, "close"]


def test_vwap_cross_invalidated_by_opposite_cross_before_confirmation():
    # A sharp up-cross (bar 30) is immediately invalidated by a down-cross
    # (bar 31) before the 3-bar confirmation window for the *original* long
    # signal elapses -> no long entry at cross_idx + CONFIRM_BARS (bar 33),
    # regardless of what the newer down-cross's own watch window decides.
    closes = [20000] * 30 + [20030, 19960] + [19995] * 30
    df = _make_df(closes)
    signals = VwapEmaCrossStrategy().generate_signals(df)
    assert signals.entries.iloc[30 + CONFIRM_BARS] != 1


def _make_session_df(n_days=6, bars_per_day=288, seed=0, trend_per_day=0.0):
    """Continuous tz-aware 5m UTC bars spanning n_days calendar days, with
    mild randomness plus an optional per-day drift, for strategies that key
    off session/RTH boundaries (ORB, ORB-failure, VWAP-pullback-trend)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n_days * bars_per_day, freq="5min", tz="UTC")
    price = 20000.0
    closes = []
    for day in range(n_days):
        day_drift = trend_per_day * day
        for _ in range(bars_per_day):
            price += rng.normal(day_drift / bars_per_day, 3.0)
            closes.append(price)
    closes = np.array(closes)
    high = closes + np.abs(rng.normal(2.0, 1.0, size=len(closes)))
    low = closes - np.abs(rng.normal(2.0, 1.0, size=len(closes)))
    open_ = closes + rng.normal(0, 1.0, size=len(closes))
    volume = rng.integers(500, 2000, size=len(closes)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": closes, "volume": volume}, index=idx)


def test_orb_produces_valid_signal_shape():
    df = _make_session_df(n_days=10, trend_per_day=15.0, seed=1)
    signals = OpeningRangeBreakoutStrategy().generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])


def test_orb_failure_produces_valid_signal_shape():
    df = _make_session_df(n_days=10, trend_per_day=0.0, seed=2)
    signals = OrbFailureStrategy().generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])
        # Fading a failed breakout: stop sits beyond the failed extreme, on
        # the opposite side of price from a normal breakout continuation.


def test_vwap_pullback_trend_produces_valid_signal_shape_and_breakeven_attr():
    df = _make_session_df(n_days=15, trend_per_day=20.0, seed=3)
    strat = VwapPullbackTrendStrategy()
    signals = strat.generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])
        assert not np.isnan(signals.target_price[idx])  # fixed 2R target, not trailed
    assert strat.breakeven_r_mult == 1.0  # engine reads this via getattr


def test_vol_expansion_momentum_produces_valid_signal_shape():
    df = _make_session_df(n_days=10, trend_per_day=15.0, seed=4)
    signals = VolExpansionMomentumStrategy().generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])
        assert np.isnan(signals.target_price[idx])  # trailed by engine, not fixed


def test_extreme_displacement_reversion_produces_valid_signal_shape():
    df = _make_session_df(n_days=10, trend_per_day=0.0, seed=5)
    signals = ExtremeDisplacementReversionStrategy().generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])
        assert not np.isnan(signals.target_price[idx])  # reverts toward EMA, fixed target


def test_volume_shock_continuation_produces_valid_signal_shape():
    df = _make_session_df(n_days=10, trend_per_day=10.0, seed=6)
    signals = VolumeShockContinuationStrategy().generate_signals(df)
    assert set(signals.entries.unique()).issubset({-1, 0, 1})
    fired = signals.entries[signals.entries != 0]
    for idx in fired.index:
        assert not np.isnan(signals.stop_price[idx])
        assert np.isnan(signals.target_price[idx])  # trailed by engine, not fixed
