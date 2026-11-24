from datetime import datetime

import pandas as pd
import pytz
import pytest

from src.strategy.mnq_15m_ema_eod import (
    ENTRY_WINDOWS,
    MAX_TRADES_DAY,
    Ema15EodState,
    allow_ema15_entry_confidence,
    check_ema15_eod_entry,
    ema15_entry_confidence_pct,
    ema15_min_confidence_from_env,
    explain_ema15_skip,
    high_confidence,
    load_1m_seed_csv,
    max_open_lots_from_env,
    max_trades_day_from_env,
    merge_1m_history,
    overlay_ticker_last_on_1m,
    parse_mnq_1m_datetime,
    quiet_1m_ok,
    release_ema15_window,
    sl_pts_from_atr,
    trend_15m_on_1m,
    window_requires_quality,
)


@pytest.fixture(autouse=True)
def _disable_min_confidence_for_legacy_tests(monkeypatch):
    """Existing recipe tests predate the % gate; opt out unless a test sets it."""
    monkeypatch.setenv("EMA15_MIN_CONFIDENCE", "0")


def _frame(n=120, start="2026-07-01 13:30:00"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = pd.Series(range(n), dtype=float) + 20000.0
    return pd.DataFrame({
        "datetime": idx,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 100,
        "atr": 4.0,
    })


def test_trend_up_on_rising_series():
    t = trend_15m_on_1m(_frame())
    assert int(t.iloc[-1]) == 1


def test_one_trade_per_window():
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET
    st = Ema15EodState()
    sig, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig is not None
    assert sig["direction"] == "long"
    assert abs(sig["entry"] - sig["sl"] - 30) < 0.01
    assert abs(sig["tp"] - sig["entry"] - 30) < 0.01
    sig2, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig2 is None


def test_second_window_can_fire():
    df = _frame()
    st = Ema15EodState()
    sig1, st = check_ema15_eod_entry(
        "MNQ", df, st,
        now=datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC),
        require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig1 is not None
    sig2, st = check_ema15_eod_entry(
        "MNQ", df, st,
        now=datetime(2026, 7, 1, 15, 5, tzinfo=pytz.UTC),  # 11:05 ET
        require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig2 is not None


def test_no_entry_before_935():
    df = _frame()
    now = datetime(2026, 7, 1, 13, 32, tzinfo=pytz.UTC)  # 9:32 ET
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig is None


def test_sl_clips_to_band():
    # L6 defaults: 1m ATR×1.5 clip 6–25
    assert sl_pts_from_atr(2.0) == 6.0
    assert sl_pts_from_atr(30.0) == 25.0
    assert sl_pts_from_atr(10.0) == 15.0
    # Legacy 15m band still available via explicit args
    assert sl_pts_from_atr(5.0, mult=2.0, sl_min=20, sl_max=60) == 20.0
    assert sl_pts_from_atr(40.0, mult=2.0, sl_min=20, sl_max=60) == 60.0
    assert sl_pts_from_atr(20.0, mult=2.0, sl_min=20, sl_max=60) == 40.0


def test_second_lot_blocked_without_confidence():
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, require_5m=False, sl_pts=30,
        open_count=1, open_direction="long", sep_min=99.0,
    )
    assert sig is None


def test_high_confidence_helper():
    assert high_confidence(1, 1, 0.5) is True
    assert high_confidence(1, -1, 0.9) is False
    assert high_confidence(1, 1, 0.2) is False


def test_ema15_entry_confidence_gate_helpers(monkeypatch):
    monkeypatch.delenv("EMA15_MIN_CONFIDENCE", raising=False)
    assert ema15_min_confidence_from_env() == 75
    soft = ema15_entry_confidence_pct(1, 1, 0.2, daily_ok=True, sep_min=0.45)
    assert soft["pct"] == 70
    assert allow_ema15_entry_confidence(soft["pct"], 75) is False
    strong = ema15_entry_confidence_pct(1, 1, 0.5, daily_ok=True, sep_min=0.45)
    assert strong["pct"] == 85
    assert allow_ema15_entry_confidence(strong["pct"], 75) is True
    assert allow_ema15_entry_confidence(75, 75) is True
    assert allow_ema15_entry_confidence(74, 75) is False


def test_ema15_min_confidence_blocks_below_allows_at_or_above(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET
    blocked, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, sl_pts=30,
        sep_min=99.0, min_confidence=75,
    )
    assert blocked is None

    monkeypatch.setattr(
        "src.strategy.mnq_15m_ema_eod.sep15_on_1m",
        lambda df_1m: pd.Series(1.0, index=df_1m.index),
    )
    allowed, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, sl_pts=30,
        sep_min=0.45, min_confidence=75,
    )
    # 50 base + 10 (5m confirm) + 15 (sep) = 75 with daily off (L6)
    assert allowed is not None
    assert allowed["entry_meta"]["entry_confidence_pct"] >= 75


def test_merge_keeps_latest_duplicate():
    seed = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-07-01 14:00:00+00:00"]),
        "close": [100.0],
    })
    live = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-07-01 14:00:00+00:00", "2026-07-01 14:01:00+00:00"]),
        "close": [101.0, 102.0],
    })
    out = merge_1m_history(seed, live)
    assert len(out) == 2
    assert float(out.loc[out["datetime"] == pd.Timestamp("2026-07-01 14:00:00+00:00"), "close"].iloc[0]) == 101.0


def test_daily_filter_skips_without_history():
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now, require_daily=True, require_60m=False, require_5m=False,
    )
    assert sig is None


def test_six_windows_and_afternoon_quality():
    assert MAX_TRADES_DAY == 20
    assert len(ENTRY_WINDOWS) == 6
    assert window_requires_quality(0) is False
    assert window_requires_quality(1) is False
    assert window_requires_quality(2) is False
    assert window_requires_quality(3) is True
    assert window_requires_quality(5) is True


def test_1015_window_can_fire():
    df = _frame()
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(),
        now=datetime(2026, 7, 1, 14, 20, tzinfo=pytz.UTC),  # 10:20 ET
        require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig is not None


def test_afternoon_window_blocked_without_confidence():
    df = _frame(n=200)
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(),
        now=datetime(2026, 7, 1, 16, 5, tzinfo=pytz.UTC),  # 12:05 ET
        require_daily=False, require_60m=False, sl_pts=30, sep_min=99.0,
    )
    assert sig is None


def test_next_window_info_uses_locked_slots_only():
    from src.strategy.mnq_15m_ema_eod import next_window_info

    inside = next_window_info(9 * 60 + 40)
    assert inside["in_window"] is True
    assert inside["window_name"] == "09:35"
    gap = next_window_info(10 * 60 + 5)
    assert gap["in_window"] is False
    assert gap["next_window_name"] == "10:15"
    after = next_window_info(15 * 60 + 10)
    assert after["next_window_name"] == "09:35"
    assert after["next_is_tomorrow"] is True


def test_parse_two_digit_year_timestamp():
    parsed = parse_mnq_1m_datetime(["26-05-27 13:11:00+00:00", "2026-05-27 13:12:00+00:00", "not-a-date"])
    assert parsed.iloc[0] == pd.Timestamp("2026-05-27 13:11:00", tz="UTC")
    assert parsed.iloc[1] == pd.Timestamp("2026-05-27 13:12:00", tz="UTC")
    assert pd.isna(parsed.iloc[2])


def test_load_1m_seed_csv_skips_bad_row(tmp_path):
    path = tmp_path / "MNQ_1m.csv"
    path.write_text(
        "datetime,open,high,low,close,volume\n"
        "26-05-27 13:11:00+00:00,30222.5,30227.75,30219.75,30225.25,1815.0\n"
        "2026-05-27 13:12:00+00:00,30224.75,30226.0,30206.0,30208.5,2437.0\n"
        "garbage,1,1,1,1,1\n",
        encoding="utf-8",
    )
    df = load_1m_seed_csv(str(path))
    assert df is not None
    assert len(df) == 2
    assert df.iloc[0]["datetime"] == pd.Timestamp("2026-05-27 13:11:00", tz="UTC")
    assert float(df.iloc[0]["close"]) == 30225.25


def _const_trend(val):
    def _fn(df_1m, **kwargs):
        return pd.Series(val, index=range(len(df_1m)))
    return _fn


def test_no_entry_when_60m_disagrees(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_60m_on_1m", _const_trend(-1))
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=True, require_5m=False, sl_pts=30,
    )
    assert sig is None


def test_entry_when_daily_15m_60m_agree(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_60m_on_1m", _const_trend(1))
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=True, require_5m=False, sl_pts=30,
    )
    assert sig is not None
    assert sig["direction"] == "long"
    assert sig["entry_meta"]["trend_60m"] == 1


def test_l6_requires_5m_agree(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_5m_on_1m", _const_trend(-1))
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, require_5m=True, sl_pts=12,
    )
    assert sig is None


def test_l6_entry_15m_5m_with_1m_atr_stop(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_5m_on_1m", _const_trend(1))
    monkeypatch.setattr(
        "src.strategy.mnq_15m_ema_eod.atr1_on_1m",
        lambda df_1m: pd.Series(10.0, index=df_1m.index),
    )
    monkeypatch.setattr(
        "src.strategy.mnq_15m_ema_eod.sep15_on_1m",
        lambda df_1m: pd.Series(1.0, index=df_1m.index),
    )
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, require_5m=True,
        sl_tf="1m", sl_atr_mult=1.5, sl_min=6, sl_max=25, min_confidence=75,
    )
    assert sig is not None
    assert sig["entry_meta"]["stack"] == "15m+5m"
    assert sig["entry_meta"]["sl_tf"] == "1m"
    assert abs(sig["entry_meta"]["sl_pts"] - 15.0) < 0.01
    assert abs(sig["tp"] - sig["entry"] - 15.0) < 0.01


def test_explain_skip_5m_disagree(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 15, 5, tzinfo=pytz.UTC)  # 11:05 ET
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_5m_on_1m", _const_trend(-1))
    msg = explain_ema15_skip(
        df, Ema15EodState(), now=now,
        require_daily=False, require_60m=False, require_5m=True,
    )
    assert "11:00" in msg
    assert "5m" in msg.lower()
    assert "must agree" in msg.lower()


def test_explain_skip_not_in_window_before_1100():
    df = _frame()
    now = datetime(2026, 8, 18, 14, 51, tzinfo=pytz.UTC)  # 10:51 ET
    msg = explain_ema15_skip(
        df, Ema15EodState(), now=now, require_daily=False, require_60m=False, require_5m=False,
    )
    assert "11:00" in msg
    assert "10:51" in msg
    assert "not in a window" in msg.lower()


def test_explain_skip_60m_disagree(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 15, 5, tzinfo=pytz.UTC)  # 11:05 ET
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_60m_on_1m", _const_trend(-1))
    msg = explain_ema15_skip(
        df, Ema15EodState(), now=now,
        require_daily=False, require_60m=True, require_5m=False,
    )
    assert "11:00" in msg
    assert "60m" in msg.lower()
    assert "short" in msg.lower()
    assert "long" in msg.lower()


def test_overlay_ticker_updates_current_minute():
    df = _frame(n=5, start="2026-08-18 14:56:00")
    now = datetime(2026, 8, 18, 15, 0, tzinfo=pytz.UTC)
    out = overlay_ticker_last_on_1m(df, 21000.0, now=now)
    assert len(out) == 5
    assert float(out.iloc[-1]["close"]) == 21000.0
    assert float(out.iloc[-1]["high"]) >= 21000.0


def test_overlay_ticker_appends_next_minute():
    df = _frame(n=5, start="2026-08-18 14:56:00")
    now = datetime(2026, 8, 18, 15, 1, tzinfo=pytz.UTC)
    out = overlay_ticker_last_on_1m(df, 21000.0, now=now)
    assert len(out) == 6
    assert float(out.iloc[-1]["close"]) == 21000.0


def test_overlay_ticker_stamps_clock_after_stale_gap():
    df = _frame(n=5, start="2026-08-18 11:37:00")  # 07:37 ET
    now = datetime(2026, 8, 18, 19, 38, tzinfo=pytz.UTC)  # 15:38 ET
    out = overlay_ticker_last_on_1m(df, 29748.0, now=now)
    assert len(out) == 6
    last = out.iloc[-1]
    assert float(last["close"]) == 29748.0
    assert float(last["open"]) == 29748.0
    assert abs(float(last["close"]) - float(last["open"])) < 0.01
    ts = pd.Timestamp(last["datetime"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    assert ts.floor("min") == pd.Timestamp(now).floor("min")


def test_explain_skip_after_1500_is_tomorrow_935():
    df = _frame()
    now = datetime(2026, 8, 18, 19, 38, tzinfo=pytz.UTC)  # 15:38 ET
    msg = explain_ema15_skip(
        df, Ema15EodState(), now=now, require_daily=False, require_60m=False,
    )
    assert "09:35" in msg
    assert "tomorrow" in msg.lower()
    assert "15:38" in msg
    assert "not in a window" in msg.lower()


def test_quiet_1m_rejects_stretched_bar():
    assert quiet_1m_ok(1, 100.0, 100.0) is True
    assert quiet_1m_ok(1, 100.0, 105.0) is False
    assert quiet_1m_ok(-1, 100.0, 94.0) is False
    assert quiet_1m_ok(-1, 100.0, 99.0) is True


def test_stretched_1m_does_not_burn_the_window():
    df = _frame()
    df.loc[df.index[-1], "open"] = float(df.iloc[-1]["close"]) - 8.0
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    st = Ema15EodState()
    sig, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig is None
    assert st.fired_windows == ()
    df.loc[df.index[-1], "open"] = float(df.iloc[-1]["close"])
    sig2, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig2 is not None
    assert sig2["entry_meta"]["entry_1m"] == "quiet_close"


def test_max_trades_day_from_env(monkeypatch):
    monkeypatch.delenv("EMA15_MAX_TRADES_DAY", raising=False)
    monkeypatch.delenv("MAX_TRADES_PER_DAY", raising=False)
    assert max_trades_day_from_env() == MAX_TRADES_DAY
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "4")
    assert max_trades_day_from_env() == 4
    monkeypatch.setenv("EMA15_MAX_TRADES_DAY", "3")
    assert max_trades_day_from_env() == 3


def test_max_open_lots_from_env(monkeypatch):
    monkeypatch.delenv("MNQ_MAX_POSITIONS", raising=False)
    monkeypatch.delenv("EMA15_MAX_OPEN_LOTS", raising=False)
    assert max_open_lots_from_env() == 2
    monkeypatch.setenv("MNQ_MAX_POSITIONS", "3")
    assert max_open_lots_from_env() == 3


def test_daily_cap_blocks_at_limit():
    df = _frame()
    st = Ema15EodState()
    st.day = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC).date()
    st.trades_today = 20
    st.fired_windows = tuple(range(len(ENTRY_WINDOWS)))
    sig, st = check_ema15_eod_entry(
        "MNQ", df, st,
        now=datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC),
        require_daily=False, require_60m=False, sl_pts=30,
        max_trades_day=20,
    )
    assert sig is None
    msg = explain_ema15_skip(
        df, st,
        now=datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC),
        require_daily=False, require_60m=False,
        max_trades_day=20,
    )
    assert "20/20" in msg


def test_release_window_allows_reentry_same_slot(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET, window 0
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_60m_on_1m", _const_trend(1))
    st = Ema15EodState()
    sig1, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig1 is not None
    assert 0 in st.fired_windows
    sig_blocked, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig_blocked is None
    st = release_ema15_window(st, 0)
    assert 0 not in st.fired_windows
    sig2, st = check_ema15_eod_entry(
        "MNQ", df, st, now=now, require_daily=False, require_60m=False, sl_pts=30,
    )
    assert sig2 is not None
    assert st.trades_today == 2


def test_all_six_window_starts_defined():
    labels = [f"{start // 60:02d}:{start % 60:02d}" for start, _end in ENTRY_WINDOWS]
    assert labels == ["09:35", "10:15", "11:00", "12:00", "13:30", "14:30"]
