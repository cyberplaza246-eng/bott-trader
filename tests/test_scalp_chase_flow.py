"""Chase filter relaxes when order flow confirms momentum direction."""

import pandas as pd

from src.strategy.scalp_hybrid import (
    CHASE_EMA_ATR,
    _chase_blocked,
    default_chase_flow_cfg,
)


def _bars(*, close_30s: float, ema20_1m: float, atr_1m: float):
    row_30s = pd.Series({"open": close_30s - 1.0, "close": close_30s, "avg_body": 2.0})
    row_1m = pd.Series({"ema_20": ema20_1m, "atr": atr_1m})
    return row_30s, row_1m


def test_chase_blocks_extended_price_without_flow():
    """Strict chase: price ~1.8× ATR from EMA20 is blocked."""
    row_30s, row_1m = _bars(close_30s=30229.0, ema20_1m=30216.31, atr_1m=5.608)
    blocked, reason = _chase_blocked(
        row_30s, row_1m, ema_atr=1.25, direction=1, flow_snap=None,
    )
    assert blocked is True
    assert "dist from 1M EMA20" in reason


def test_chase_relaxes_on_strong_flow_momentum():
    """Flow-confirmed LONG (Δ+56, 66% buy) passes with relaxed EMA cap."""
    row_30s, row_1m = _bars(close_30s=30229.0, ema20_1m=30216.31, atr_1m=5.608)
    flow_snap = {"delta": 56, "buy_pct": 0.66}
    cfg = default_chase_flow_cfg()
    cfg["chase_ema_atr_flow"] = 2.5

    blocked, reason = _chase_blocked(
        row_30s,
        row_1m,
        ema_atr=1.25,
        direction=1,
        flow_snap=flow_snap,
        chase_flow_cfg=cfg,
    )
    assert blocked is False, reason


def test_chase_skip_on_flow_disables_filter():
    row_30s, row_1m = _bars(close_30s=30229.0, ema20_1m=30216.31, atr_1m=5.608)
    flow_snap = {"delta": 56, "buy_pct": 0.66}
    cfg = default_chase_flow_cfg()
    cfg["chase_skip_on_flow"] = True

    blocked, _ = _chase_blocked(
        row_30s,
        row_1m,
        ema_atr=CHASE_EMA_ATR,
        direction=1,
        flow_snap=flow_snap,
        chase_flow_cfg=cfg,
    )
    assert blocked is False


def test_weak_flow_still_chase_blocked():
    row_30s, row_1m = _bars(close_30s=30229.0, ema20_1m=30216.31, atr_1m=5.608)
    flow_snap = {"delta": 10, "buy_pct": 0.52}

    blocked, reason = _chase_blocked(
        row_30s,
        row_1m,
        ema_atr=1.25,
        direction=1,
        flow_snap=flow_snap,
        chase_flow_cfg=default_chase_flow_cfg(),
    )
    assert blocked is True
    assert "dist from 1M EMA20" in reason
