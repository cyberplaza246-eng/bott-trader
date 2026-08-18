"""Hybrid trigger fired but entry blocked — logging and retry helpers."""

import pandas as pd

from src.strategy.scalp_hybrid import (
    ScalpHybridState,
    check_hybrid_entry,
    hybrid_block_summary,
    hybrid_trigger_no_signal_reason,
    rsi_entry_block_reason,
)


def _bars(*, close_30s: float = 30243.0, prev_high: float = 30242.0, open_30s: float = 30242.5):
    row_30s = pd.Series(
        {
            "open": open_30s,
            "high": max(open_30s, close_30s) + 0.25,
            "low": min(open_30s, close_30s) - 0.25,
            "close": close_30s,
            "avg_body": 2.0,
            "datetime": "2026-06-15 14:30:30+00:00",
        }
    )
    prev_30s = pd.Series(
        {
            "open": 30242.0,
            "high": prev_high,
            "low": 30241.5,
            "close": 30242.25,
            "datetime": "2026-06-15 14:30:00+00:00",
        }
    )
    row_1m = pd.Series(
        {
            "open": 30242.0,
            "high": 30243.5,
            "low": 30241.0,
            "close": close_30s,
            "atr": 5.6,
            "ema_20": 30240.0,
            "datetime": "2026-06-15 14:30:00+00:00",
        }
    )
    row_5m = pd.Series(
        {
            "close": 30240.0,
            "adx": 22.0,
            "di_plus": 28.0,
            "di_minus": 18.0,
            "vwap": 30220.0,
            "ema_20": 30210.0,
        }
    )
    return row_1m, row_5m, row_30s, prev_30s


def test_aggressive_direct_trigger_fires_on_micro_break():
    """Micro-break above prev high with trend + flow should produce a signal."""
    row_1m, row_5m, row_30s, prev_30s = _bars()
    flow_snap = {"delta": 26, "buy_pct": 0.62}
    st = ScalpHybridState()
    signal, _ = check_hybrid_entry(
        "MNQ",
        row_1m,
        row_5m,
        None,
        row_30s,
        prev_30s,
        st,
        aggressive_mode=True,
        adx_min_pullback=15,
        trend_mode="vwap",
        momentum_burst_enabled=True,
        flow_snap=flow_snap,
        micro_break_pts=0.25,
    )
    assert signal is not None
    assert signal["direction"] == "long"
    assert signal.get("scalp_mode") == "trigger"


def test_block_summary_matches_entry_when_learner_params_applied():
    """Diagnostics and entry must agree when using the same kwargs."""
    row_1m, row_5m, row_30s, prev_30s = _bars()
    flow_snap = {"delta": 26, "buy_pct": 0.62}
    st = ScalpHybridState()
    kwargs = dict(
        aggressive_mode=True,
        adx_min_pullback=15,
        trend_mode="vwap",
        momentum_burst_enabled=True,
        flow_snap=flow_snap,
        micro_break_pts=0.25,
    )
    summary = hybrid_block_summary(
        row_1m, row_5m, None, row_30s, prev_30s, st, **kwargs,
    )
    signal, _ = check_hybrid_entry(
        "MNQ", row_1m, row_5m, None, row_30s, prev_30s, st, **kwargs,
    )
    assert signal is not None
    assert "READY LONG" in summary


def test_trigger_miss_reason_reports_chase_block():
    """When trigger fires but chase blocks, reason must name chase."""
    row_1m, row_5m, row_30s, prev_30s = _bars(
        close_30s=30243.0, prev_high=30242.0, open_30s=30242.5,
    )
    row_1m = row_1m.copy()
    row_1m["ema_20"] = 30216.31
    flow_snap = {"delta": 26, "buy_pct": 0.62}
    reason = hybrid_trigger_no_signal_reason(
        row_1m,
        row_5m,
        row_30s,
        prev_30s,
        direction=1,
        aggressive_mode=True,
        adx_min_pullback=15,
        trend_mode="vwap",
        chase_ema_atr=1.25,
        flow_snap=flow_snap,
        micro_break_pts=0.25,
    )
    assert reason is not None
    assert "chase" in reason.lower()


def test_rsi_gate_blocks_short_oversold_at_ema():
    """Short at/below EMA20 with oversold RSI must not enter."""
    row_1m, row_5m, row_30s, prev_30s = _bars(
        close_30s=30238.0, prev_high=30242.0, open_30s=30239.0,
    )
    row_1m = row_1m.copy()
    row_1m["close"] = 30238.0
    row_1m["ema_20"] = 30240.0
    row_1m["rsi"] = 32.0
    row_1m["rsi_prev"] = 34.0
    row_5m = row_5m.copy()
    row_5m["close"] = 30238.0
    row_5m["vwap"] = 30250.0
    row_5m["ema_20"] = 30245.0
    flow_snap = {"delta": -80, "buy_pct": 0.35}
    reason = rsi_entry_block_reason(
        -1, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=15, trend_mode="vwap",
    )
    assert reason is not None
    assert "SHORT RSI oversold" in reason


def test_rsi_gate_allows_short_with_trend_flow_relax():
    """Strong downtrend + flow may short with RSI down to relax floor."""
    row_1m, row_5m, row_30s, prev_30s = _bars(
        close_30s=30238.0, prev_high=30242.0, open_30s=30239.5,
    )
    row_1m = row_1m.copy()
    row_1m["close"] = 30241.0
    row_1m["ema_20"] = 30240.0
    row_1m["rsi"] = 38.0
    row_1m["rsi_prev"] = 42.0
    row_5m = row_5m.copy()
    row_5m["close"] = 30241.0
    row_5m["vwap"] = 30250.0
    row_5m["ema_20"] = 30245.0
    row_5m["adx"] = 28.0
    flow_snap = {"delta": -120, "buy_pct": 0.35}
    reason = rsi_entry_block_reason(
        -1, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=15, trend_mode="vwap",
    )
    assert reason is None
