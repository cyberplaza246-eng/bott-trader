"""Deterministic WHY classification and advisory suggestion clustering."""
from datetime import datetime

import pandas as pd
import pytz

from src.ai.trade_review import (
    EXIT_EOD,
    EXIT_OTHER,
    EXIT_SL,
    build_suggestions,
    classify_why,
    compute_mae_mfe_pts,
    normalize_ema15_exit_reason,
)
from src.strategy.mnq_15m_ema_eod import capture_market_snapshot, window_label


def _snap(**kwargs):
    base = {
        "ts_et": "2026-07-06T09:40:00-04:00",
        "session": "RTH",
        "window_name": "09:35",
        "window": 0,
        "minutes_into_window": 5,
        "daily_permission": 1,
        "prior_rth_close": 20100.0,
        "daily_ema20": 20000.0,
        "ema15_fast": 20050.0,
        "ema15_slow": 20020.0,
        "trend_15m": 1,
        "atr15": 20.0,
        "ema60_fast": 20040.0,
        "ema60_slow": 20010.0,
        "trend_60m": 1,
        "sep15": 0.6,
        "price_1m": 20080.0,
        "atr_stop_pts": 40.0,
        "daily_agree": True,
        "tf15_agree": True,
        "tf60_agree": True,
        "high_confidence": True,
    }
    base.update(kwargs)
    return base


def _close(**kwargs):
    rec = {
        "event": "close",
        "side": "long",
        "direction": "long",
        "pts": -40.0,
        "pnl_usd": -80.0,
        "hold_minutes": 45.0,
        "exit_reason": EXIT_SL,
        "mae_pts": 40.0,
        "mfe_pts": 8.0,
        "atr_stop_pts": 40.0,
        "sl_pts": 40.0,
        "r_multiple": -1.0,
        "entry_snapshot": _snap(),
        "exit_snapshot": _snap(trend_15m=1, ts_et="2026-07-06T10:25:00-04:00"),
        "window_name": "09:35",
    }
    rec.update(kwargs)
    return rec


def test_normalize_never_invents_tp():
    assert normalize_ema15_exit_reason("EOD_1550") == EXIT_EOD
    assert normalize_ema15_exit_reason("SL") == EXIT_SL
    assert normalize_ema15_exit_reason("rejected") == "rejected"
    assert normalize_ema15_exit_reason("TP") == EXIT_OTHER
    assert normalize_ema15_exit_reason("TAKE_PROFIT") == EXIT_OTHER
    assert normalize_ema15_exit_reason("BROKER_BRACKET") == EXIT_OTHER


def test_why_stopped_before_trend_continued():
    why = classify_why(_close())
    assert why["outcome"] == "lose"
    assert why["primary_reason"] == "stopped before trend continued"
    assert any("ATR stop" in f for f in why["facts"])
    assert any("MAE" in f for f in why["facts"])


def test_why_eod_flattened_in_profit():
    why = classify_why(_close(
        pts=25.0,
        pnl_usd=50.0,
        exit_reason=EXIT_EOD,
        mae_pts=12.0,
        mfe_pts=30.0,
        r_multiple=0.625,
    ))
    assert why["outcome"] == "win"
    assert why["primary_reason"] == "EOD flattened in profit"
    assert why["exit_reason"] == EXIT_EOD
    assert "TP" not in why["primary_reason"]


def test_why_15m_flipped_after_entry():
    why = classify_why(_close(
        exit_snapshot=_snap(trend_15m=-1),
        mae_pts=22.0,
        mfe_pts=6.0,
    ))
    assert why["outcome"] == "lose"
    assert why["primary_reason"] == "15m flipped after entry"
    assert any("15m EMA8/21" in f for f in why["facts"])


def test_why_late_window_chop():
    why = classify_why(_close(
        pts=-4.0,
        pnl_usd=-8.0,
        exit_reason=EXIT_OTHER,
        mae_pts=10.0,
        mfe_pts=9.0,
        entry_snapshot=_snap(minutes_into_window=22, window_name="11:00", window=1),
    ))
    assert why["primary_reason"] == "entered late in window / chop"
    assert any("22 min" in f for f in why["facts"])


def test_why_daily_permission_weak():
    why = classify_why(_close(
        pts=-12.0,
        pnl_usd=-24.0,
        exit_reason=EXIT_OTHER,
        mae_pts=12.0,
        mfe_pts=4.0,
        entry_snapshot=_snap(daily_agree=False, daily_permission=-1),
        exit_snapshot=_snap(trend_15m=1),
    ))
    assert why["primary_reason"] == "daily permission weak"
    assert any("daily permission" in f for f in why["facts"])


def test_why_mae_hit_atr_stop_then_reversed():
    why = classify_why(_close(
        mae_pts=38.0,
        mfe_pts=18.0,
        atr_stop_pts=40.0,
        sl_pts=40.0,
        exit_snapshot=_snap(trend_15m=1),
    ))
    assert why["primary_reason"] == "MAE hit ATR stop then reversed"
    assert any("MFE 18.0 then MAE 38.0" in f for f in why["facts"])


def test_suggestions_gate_before_min_trades():
    rows = [_close(pnl_usd=-10.0, pts=-5.0) for _ in range(12)]
    payload = build_suggestions(rows, min_trades=30, min_losers=10)
    assert payload["ready"] is False
    assert payload["auto_apply"] is False
    assert payload["writes_live_config"] is False
    assert payload["suggestions"] == []
    assert "30" in payload["note"]


def _row(i, *, window, pnl, hour=10, tf60=True, mae=20.0, hold=40.0, exit_reason=EXIT_SL):
    day = 6 + (i % 5)
    ts = f"2026-07-{day:02d}T{hour:02d}:10:00-04:00"
    return _close(
        pnl_usd=pnl,
        pts=pnl / 2.0,
        exit_reason=exit_reason if pnl < 0 else EXIT_EOD,
        mae_pts=mae,
        mfe_pts=8.0 if pnl < 0 else 30.0,
        hold_minutes=hold,
        window_name=window,
        timestamp_et=ts,
        entry_snapshot=_snap(
            ts_et=ts,
            window_name=window,
            window=0 if window.startswith("09") else 1,
            tf60_agree=tf60,
            minutes_into_window=5,
        ),
    )


def test_suggestions_cluster_skip_weak_window():
    rows = []
    for i in range(20):
        rows.append(_row(i, window="09:35", pnl=100.0, hour=9, exit_reason=EXIT_EOD))
    for i in range(8):
        rows.append(_row(i, window="09:35", pnl=-40.0, hour=9))
    for i in range(12):
        rows.append(_row(i, window="11:00", pnl=-50.0, hour=11))
    for i in range(4):
        rows.append(_row(i, window="11:00", pnl=20.0, hour=11, exit_reason=EXIT_EOD))
    payload = build_suggestions(rows, min_trades=30, min_losers=10, min_cluster=5)
    assert payload["ready"] is True
    assert payload["auto_apply"] is False
    assert 3 <= len(payload["suggestions"]) <= 8
    weak = [s for s in payload["suggestions"] if s["cluster"] == "window" and s["bucket"] == "11:00"]
    assert weak
    assert weak[0]["n"] == 16
    assert weak[0]["pf"] is not None and payload["overall_pf"] is not None
    assert weak[0]["pf"] < payload["overall_pf"]
    assert "skip 11:00 window" in weak[0]["tweak"]
    assert f"{weak[0]['pf']:.2f}" in weak[0]["tweak"]


def test_suggestions_never_claim_auto_apply():
    rows = [_row(i, window="09:35", pnl=50.0 if i % 3 else -40.0) for i in range(40)]
    payload = build_suggestions(rows, min_trades=30, min_losers=10, min_cluster=5)
    assert payload["advisory_only"] is True
    assert payload["auto_apply"] is False
    assert payload["writes_live_config"] is False
    for s in payload["suggestions"]:
        assert s["user_applies"] is True


def test_mae_mfe_from_1m_bars():
    idx = pd.date_range("2026-07-06 13:40:00", periods=10, freq="1min", tz="UTC")
    close = pd.Series([100.0] * 10)
    df = pd.DataFrame({
        "datetime": idx,
        "open": close,
        "high": [100, 101, 103, 102, 99, 98, 97, 100, 101, 100],
        "low": [99, 99, 100, 98, 97, 96, 95, 98, 99, 99],
        "close": close,
    })
    mae, mfe = compute_mae_mfe_pts(df, idx[0], idx[-1], "long", 100.0)
    assert mae == 5.0
    assert mfe == 3.0


def test_window_label_matches_locked_windows():
    from src.strategy.mnq_15m_ema_eod import ENTRY_WINDOWS

    for i, (start, _end) in enumerate(ENTRY_WINDOWS):
        assert window_label(i) == f"{start // 60:02d}:{start % 60:02d}"
    assert window_label(None) == "none"


def test_snapshot_has_journal_fields():
    idx = pd.date_range("2026-07-01 13:40:00", periods=400, freq="1min", tz="UTC")
    close = pd.Series(range(400), dtype=float) + 20000.0
    df = pd.DataFrame({
        "datetime": idx,
        "open": close,
        "high": close + 2,
        "low": close - 2,
        "close": close,
    })
    now = datetime(2026, 7, 1, 13, 40, tzinfo=pytz.UTC)
    snap = capture_market_snapshot(df, now=now, side=1, atr_stop_pts=40.0, window=0)
    for key in (
        "daily_permission", "ema15_fast", "ema15_slow", "atr15",
        "ema60_fast", "trend_60m", "sep15", "price_1m", "session",
        "minutes_into_window", "atr_stop_pts", "daily_agree",
        "tf15_agree", "tf60_agree", "high_confidence",
    ):
        assert key in snap
    assert snap["session"] == "RTH"
    assert snap["window_name"] == "09:35"
    assert snap["atr_stop_pts"] == 40.0
