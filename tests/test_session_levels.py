"""Tests for session levels + level_15_5_1 entry gate."""
from datetime import datetime

import pandas as pd
import pytz
import pytest

from src.strategy.mnq_15m_ema_eod import (
    Ema15EodState,
    check_ema15_level_entry,
    ema15_level_mode_from_env,
    explain_ema15_level_skip,
)
from src.strategy.session_levels import (
    bias_15m_5m,
    compute_session_levels,
    level_interaction,
    one_min_level_confirm,
    structure_side_from_ohlc,
)

ET = pytz.timezone("US/Eastern")


def _synth_days(n_days=3, start_et="2026-08-18 09:30:00"):
    """Build multi-day RTH+overnight 1m bars for level math."""
    start = ET.localize(datetime.strptime(start_et, "%Y-%m-%d %H:%M:%S"))
    rows = []
    px = 20000.0
    for d in range(n_days):
        day = start.date().toordinal() + d
        day_dt = datetime.fromordinal(day)
        # overnight 18:00 prior → 09:29
        on_start = ET.localize(datetime(day_dt.year, day_dt.month, day_dt.day, 0, 0))
        # simpler: emit full calendar day minutes for weekdays only
        if day_dt.weekday() >= 5:
            continue
        for m in range(24 * 60):
            ts = ET.localize(datetime(day_dt.year, day_dt.month, day_dt.day, m // 60, m % 60))
            # skip weekend gap simulation — only weekdays
            px += 0.25 if m % 17 else -0.15
            hi, lo = px + 1.5, px - 1.5
            # mark prior day RTH extremes distinctly
            if d == 0 and 9 * 60 + 30 <= m < 16 * 60:
                hi = max(hi, 20100.0)
                lo = min(lo, 19900.0)
            # overnight bump before day 1 RTH
            if d == 1 and m < 9 * 60 + 30:
                hi = max(hi, 20080.0)
                lo = min(lo, 19940.0)
            rows.append({
                "datetime": ts.astimezone(pytz.UTC),
                "open": px,
                "high": hi,
                "low": lo,
                "close": px,
                "volume": 50 + (m % 20),
            })
    return pd.DataFrame(rows)


def test_ema15_level_mode_env(monkeypatch):
    monkeypatch.setenv("EMA15_LEVEL_MODE", "1")
    assert ema15_level_mode_from_env() is True
    monkeypatch.delenv("EMA15_LEVEL_MODE", raising=False)
    monkeypatch.setenv("SCALP_MODE", "level_15_5_1")
    assert ema15_level_mode_from_env() is True
    monkeypatch.setenv("SCALP_MODE", "ema15_eod")
    monkeypatch.setenv("EMA15_LEVEL_MODE", "0")
    assert ema15_level_mode_from_env() is False


def test_compute_session_levels_has_majors():
    df = _synth_days(3)
    # evaluate mid RTH on day 2
    now = ET.localize(datetime(2026, 8, 19, 10, 15)).astimezone(pytz.UTC)
    lv = compute_session_levels(df, now=now)
    assert lv["pdh"] is not None
    assert lv["pdl"] is not None
    assert lv["onh"] is not None
    assert lv["onl"] is not None
    assert lv["rth_open"] is not None
    assert lv["vwap"] is not None
    assert lv["ema10"] is not None
    assert lv["ema20"] is not None
    assert lv["day_high"] is not None
    assert lv["day_low"] is not None


def test_structure_hh_hl():
    idx = pd.date_range("2026-08-19 14:00", periods=8, freq="15min", tz="UTC")
    # rising highs and lows
    hi = [10, 11, 12, 13, 14, 15, 16, 17]
    lo = [1, 2, 3, 4, 5, 6, 7, 8]
    ohlc = pd.DataFrame({
        "open": hi, "high": hi, "low": lo, "close": hi, "volume": 1,
    }, index=idx)
    side, label = structure_side_from_ohlc(ohlc)
    assert side == 1
    assert label == "HH_HL"


def test_structure_mixed():
    idx = pd.date_range("2026-08-19 14:00", periods=6, freq="15min", tz="UTC")
    ohlc = pd.DataFrame({
        "open": [10, 12, 11, 13, 10, 14],
        "high": [11, 13, 12, 14, 11, 15],
        "low": [9, 8, 10, 7, 9, 8],
        "close": [10, 12, 11, 13, 10, 14],
        "volume": 1,
    }, index=idx)
    side, label = structure_side_from_ohlc(ohlc)
    assert side == 0
    assert label == "mixed"


def test_level_interaction_and_1m_confirm():
    # Build bars that break above ONH then retest
    onh = 20050.0
    idx = pd.date_range("2026-08-19 14:40", periods=10, freq="1min", tz="UTC")
    closes = [20040, 20045, 20052, 20055, 20053, 20051, 20050.5, 20051, 20052, 20053]
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "datetime": idx[i],
            "open": c - 0.5,
            "high": max(c, onh) + 0.5,
            "low": min(c, onh) - 0.5,
            "close": c,
            "volume": 10,
        })
    df = pd.DataFrame(rows)
    levels = {"onh": onh, "onl": 19900.0, "pdh": 20100.0, "pdl": 19800.0, "rth_open": 20000.0}
    inter = level_interaction(df, 1, levels, touch_pts=8.0)
    assert inter is not None
    assert inter["level_name"] == "onh"
    assert one_min_level_confirm(df, 1, inter) is True


def test_explain_waiting_for_level(monkeypatch):
    monkeypatch.setenv("EMA15_MIN_CONFIDENCE", "0")
    df = _synth_days(3)
    # Force price far from majors mid-range
    now = ET.localize(datetime(2026, 8, 19, 9, 40)).astimezone(pytz.UTC)
    # Inflate last close away from levels
    df = df.copy()
    df.loc[df.index[-1], "close"] = 20500.0
    df.loc[df.index[-1], "open"] = 20500.0
    df.loc[df.index[-1], "high"] = 20501.0
    df.loc[df.index[-1], "low"] = 20499.0
    why = explain_ema15_level_skip(df, Ema15EodState(), now=now)
    # May be mixed / disagree / waiting for level — any are valid; must not crash
    assert isinstance(why, str) and len(why) > 10


def test_check_level_entry_rejects_outside_window(monkeypatch):
    monkeypatch.setenv("EMA15_MIN_CONFIDENCE", "0")
    df = _synth_days(3)
    now = ET.localize(datetime(2026, 8, 19, 10, 5)).astimezone(pytz.UTC)  # gap between windows
    sig, _ = check_ema15_level_entry("MNQ", df, Ema15EodState(), now=now)
    assert sig is None


def test_bias_keys():
    df = _synth_days(3)
    b = bias_15m_5m(df)
    assert "bias_15m" in b
    assert "bias_5m" in b
    assert "structure_15m" in b
    assert "agree" in b
