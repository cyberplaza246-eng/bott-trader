"""Overnight Globex edge search: session keys, flatten, grade. Not live."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytz

from scripts.backtest_overnight_edge import (
    Spec,
    WIN_ASIA,
    _summarize,
    globex_session_id,
    grade,
    in_window,
    simulate,
    split_is_oos,
)
from src.strategy.overnight_research import FLATTEN_MINUTE, GLOBEX_OPEN_MINUTE

ET = pytz.timezone("US/Eastern")


def test_session_keys_skip_rth_and_friday_close():
    mon = date(2026, 8, 17)
    assert globex_session_id(date(2026, 8, 16), GLOBEX_OPEN_MINUTE, 6) == "2026-08-17"  # Sun 18:00
    assert globex_session_id(mon, 60, 0) == "2026-08-17"  # Mon 01:00
    assert globex_session_id(mon, FLATTEN_MINUTE, 0) == "2026-08-17"  # Mon 09:25 still in session
    assert globex_session_id(mon, 9 * 60 + 30, 0) == ""  # RTH
    assert globex_session_id(mon, 12 * 60, 0) == ""
    assert globex_session_id(mon, GLOBEX_OPEN_MINUTE, 0) == "2026-08-18"  # Mon 18:00 -> Tue
    assert globex_session_id(date(2026, 8, 21), GLOBEX_OPEN_MINUTE, 4) == ""  # Fri 18:00 closed
    assert globex_session_id(date(2026, 8, 22), 60, 5) == ""  # Sat


def test_asia_window_and_grade_thresholds():
    assert in_window(20 * 60 + 5, WIN_ASIA) == 0
    assert in_window(3 * 60, WIN_ASIA) is None
    thin = {"trades": 10, "profit_factor": 2.0, "total_pnl": 400.0, "max_drawdown": 50.0}
    fat = {"trades": 40, "profit_factor": 1.3, "total_pnl": 800.0, "max_drawdown": 200.0}
    fail = {"trades": 40, "profit_factor": 0.9, "total_pnl": -100.0, "max_drawdown": 400.0}
    dd = {"trades": 40, "profit_factor": 1.4, "total_pnl": 200.0, "max_drawdown": 3000.0}
    assert grade(thin, thin) == "THIN"
    assert grade(fat, fat) == "PASS"
    assert grade(fat, fail) == "FAIL"
    assert grade(fat, dd) == "FAIL_DD"
    assert grade(thin, fat) == "THIN_IS"


def _bars(start_et: str, n: int, *, base=25000.0, drift=0.0, vol=10.0):
    start = ET.localize(datetime.strptime(start_et, "%Y-%m-%d %H:%M"))
    rows = []
    px = base
    for i in range(n):
        ts = start + pd.Timedelta(minutes=i)
        px += drift
        rows.append({
            "datetime": ts.astimezone(pytz.UTC),
            "open": px,
            "high": px + 1,
            "low": px - 1,
            "close": px,
            "volume": vol,
        })
    return pd.DataFrame(rows)


def _frame_from_df(df: pd.DataFrame) -> dict:
    from scripts.backtest_overnight_edge import build_session_arrays, overnight_vwap

    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    mins, dow, sess, can_enter, flatten = build_session_arrays(dt)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    n = len(df)
    return {
        "dt": dt,
        "hi": hi,
        "lo": lo,
        "cl": cl,
        "vol": vol,
        "t15": np.ones(n, dtype=int),
        "t60": np.ones(n, dtype=int),
        "td": np.ones(n, dtype=int),
        "atr": np.full(n, 20.0),
        "vwap": overnight_vwap(hi, lo, cl, vol, sess),
        "mins": mins,
        "dow": dow,
        "sess": sess,
        "can_enter": can_enter,
        "flatten": flatten,
    }


def test_bh_long_flattens_0925_not_rth():
    # 80 dummy bars then Sun 18:00 -> Mon 09:40 so flatten is inside the sample.
    pad = _bars("2026-08-16 16:00", 80, base=25000.0)
    night = _bars("2026-08-16 18:00", 16 * 60 + 40, base=25000.0, drift=50.0 / (16 * 60 + 40))
    df = pd.concat([pad, night], ignore_index=True)
    frame = _frame_from_df(df)
    spec = Spec("bh_long", "bh", use_atr_stop=False, tp_pts=1e9, side=1)
    trades = simulate(frame, spec)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.exit_reason == "FLAT"
    exit_et = pd.Timestamp(t.exit_time).tz_convert(ET)
    assert exit_et.hour == 9
    assert exit_et.minute == 25
    assert t.pnl > 0


def test_no_rth_entries_on_aligned_window():
    # Monday RTH only — Asia window is overnight; this sample has no Globex.
    rth = _bars("2026-08-17 09:30", 200, base=25000.0, drift=0.2)
    pad = _bars("2026-08-17 08:00", 80, base=24900.0)
    df = pd.concat([pad, rth], ignore_index=True)
    frame = _frame_from_df(df)
    spec = Spec(
        "ema_asia_20et", "window_ema", windows=WIN_ASIA,
        require_daily=True, require_60m=True, max_per_night=1,
    )
    assert simulate(frame, spec) == []


def test_merge_real_1m_slices(tmp_path):
    from scripts.backtest_overnight_edge import load_frame

    a = tmp_path / "MNQ_1m.csv"
    b = tmp_path / "MNQ_1m_august_databento.csv"
    pd.DataFrame({
        "datetime": pd.to_datetime(["2026-07-23 17:00:00+00:00", "2026-07-23 17:01:00+00:00"]),
        "open": [1, 1], "high": [2, 2], "low": [0, 0], "close": [1, 1], "volume": [1, 1],
    }).to_csv(a, index=False)
    pd.DataFrame({
        "datetime": pd.to_datetime(["2026-07-23 17:01:00+00:00", "2026-08-03 00:00:00+00:00"]),
        "open": [9, 3], "high": [9, 4], "low": [9, 2], "close": [9, 3], "volume": [1, 1],
    }).to_csv(b, index=False)
    df, note = load_frame(str(a), extra=[str(b)])
    assert len(df) == 3
    # overlapping 17:01 keeps the later slice
    assert float(df.loc[df["datetime"] == pd.Timestamp("2026-07-23 17:01:00+00:00"), "close"].iloc[0]) == 9.0
    assert "august_databento" in note


def test_split_is_oos_cutoff():
    from scripts.backtest_overnight_edge import OnTrade

    def t(ts, pnl=1.0):
        return OnTrade(
            entry_time=ts, exit_time=ts, direction="long", entry_price=1, exit_price=2,
            sl=0, tp=0, pnl=pnl, pts=0.5, exit_reason="FLAT", spec="x", session="s",
            hold_seconds=1.0,
        )

    ins, oos, aug = split_is_oos([
        t("2026-05-31 22:00:00+00:00"),
        t("2026-06-01 00:00:00+00:00"),
        t("2026-08-02 02:00:00+00:00"),
    ])
    assert len(ins) == 1
    assert len(oos) == 2
    assert len(aug) == 1
    empty = _summarize([], 10, "no_trade")
    assert empty["trades"] == 0
    assert empty["total_pnl"] == 0.0
