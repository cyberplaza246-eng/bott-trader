from datetime import datetime

import pandas as pd
import pytz

from src.strategy.mnq_15m_ema_eod import (
    ENTRY_WINDOWS,
    MAX_TRADES_DAY,
    Ema15EodState,
    check_ema15_eod_entry,
    explain_ema15_skip,
    high_confidence,
    load_1m_seed_csv,
    merge_1m_history,
    overlay_ticker_last_on_1m,
    parse_mnq_1m_datetime,
    sl_pts_from_atr,
    trend_15m_on_1m,
    window_requires_quality,
)


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
    assert sl_pts_from_atr(5.0) == 20.0
    assert sl_pts_from_atr(40.0) == 60.0
    assert sl_pts_from_atr(20.0) == 40.0


def test_second_lot_blocked_without_confidence():
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now, require_daily=False, require_60m=False, sl_pts=30,
        open_count=1, open_direction="long", sep_min=99.0,
    )
    assert sig is None


def test_high_confidence_helper():
    assert high_confidence(1, 1, 0.5) is True
    assert high_confidence(1, -1, 0.9) is False
    assert high_confidence(1, 1, 0.2) is False


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
        "MNQ", df, Ema15EodState(), now=now, require_daily=True, require_60m=False,
    )
    assert sig is None


def test_six_windows_and_afternoon_quality():
    assert MAX_TRADES_DAY == 6
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
        "MNQ", df, Ema15EodState(), now=now, require_daily=False, sl_pts=30,
    )
    assert sig is None


def test_entry_when_daily_15m_60m_agree(monkeypatch):
    df = _frame()
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)  # 9:40 ET
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_15m_on_1m", _const_trend(1))
    monkeypatch.setattr("src.strategy.mnq_15m_ema_eod.trend_60m_on_1m", _const_trend(1))
    sig, _ = check_ema15_eod_entry(
        "MNQ", df, Ema15EodState(), now=now, require_daily=False, sl_pts=30,
    )
    assert sig is not None
    assert sig["direction"] == "long"
    assert sig["entry_meta"]["trend_60m"] == 1


def test_explain_skip_not_in_window_before_1100():
    df = _frame()
    now = datetime(2026, 8, 18, 14, 51, tzinfo=pytz.UTC)  # 10:51 ET
    msg = explain_ema15_skip(
        df, Ema15EodState(), now=now, require_daily=False, require_60m=False,
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
        df, Ema15EodState(), now=now, require_daily=False, require_60m=True,
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
