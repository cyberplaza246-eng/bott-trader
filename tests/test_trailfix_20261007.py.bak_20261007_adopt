"""2026-10-07 trailfix: repaired/replaced SL legs (_rN_sl) stay protected,
dollar profit lock, short post-exit cooldown."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.test_orphan_working_orders import _conn, _row, _cancelled_targets, _sweep
import start_live_rithmic as slr
from src.broker.rithmic_connector import RithmicConnector


# ---- 1. orphan sweep / clear_chart must keep _rN_sl legs of a live trade ----

def test_entry_id_for_leg_tag_strips_replace_seq():
    f = RithmicConnector._entry_id_for_leg_tag
    assert f("bot_1825746db8_r14_sl") == "bot_1825746db8"
    assert f("bot_1825746db8_sl") == "bot_1825746db8"
    assert f("bot_1825746db8_tp_lit") == "bot_1825746db8"
    assert f("bot_1825746db8") == ""
    assert RithmicConnector._is_bot_entry_user_tag("bot_1825746db8_r14") is False
    assert RithmicConnector._is_bot_entry_user_tag("bot_1825746db8") is True


def test_sweep_keeps_repaired_rn_sl_of_live_trade():
    rows = [
        _row("bot_1825746db8_r14_sl", "245097997", "STOP_LIMIT", "SELL"),
        _row("bot_1825746db8_tp", "245088466", "LIMIT", "SELL"),
        _row("bot_old_r3_sl", "999", "STOP", "SELL"),  # closed trade's leftover
    ]
    conn = _conn(net=1, orders=rows)
    conn._orders = {"bot_1825746db8": {"symbol": "MNQ", "size": 1}}
    _sweep(conn, {"bot_1825746db8"})
    conn._orphan_sweep_ok_at["MNQ"] -= 25
    _sweep(conn, {"bot_1825746db8"})
    got = set(_cancelled_targets(conn))
    assert "bot_1825746db8_r14_sl" not in got
    assert "bot_1825746db8_tp" not in got
    assert "bot_old_r3_sl" in got


def test_clear_chart_keeps_blank_tag_leg_whose_basket_is_a_live_rn_leg():
    rows = [_row("", "245097997", "STOP_LIMIT", "SELL")]
    conn = _conn(net=1, orders=rows)
    conn._is_bot_owned_listed_order = lambda data: True
    conn._orders = {
        "bot_x": {"symbol": "MNQ", "size": 1, "sl_leg_history": ["bot_x_r2_sl"],
                  "protective_sl_order_id": "bot_x_r2_sl"},
        "bot_x_r2_sl": {"basket_id": "245097997"},
    }
    n = asyncio.run(conn._async_clear_chart_stops("MNQ", keep_entry_ids={"bot_x"}, rounds=1))
    assert n == 0
    assert _cancelled_targets(conn) == []


def test_closed_entry_snapshot_includes_rn_legs():
    conn = _conn(net=0)
    conn._orders = {"bot_x": {"symbol": "MNQ", "sl_leg_history": ["bot_x_r1_sl", "bot_x_r2_sl"]},
                    "bot_x_r2_sl": {"basket_id": "55"}}
    conn.remember_closed_entry("bot_x", "MNQ")
    rec = conn._closed_entry_registry()["bot_x"]
    assert {"bot_x_r1_sl", "bot_x_r2_sl"} <= rec["legs"]
    assert "55" in rec["baskets"]


# ---- 2. dollar profit lock ----

def _lock(direction="long", size=1, peak=0.0, live=0.0, cur=0.0):
    return slr.usd_lock_stop(direction, 31346.0, size, 2.0, peak, live, 0.25, cur,
                             activate_usd=100, keep_usd=20, trail_start_usd=150,
                             keep_pct=0.5, buffer_ticks=4)


def test_usd_lock_no_move_before_plus_100():
    # Today's trade: +31.75 pts on 1 MNQ = $63.50 -> old R-lock moved; now must not.
    assert _lock(peak=31.75, live=31357.0, cur=31309.5) is None
    assert _lock(peak=49.5, live=31395.0, cur=31309.5) is None  # $99
    assert _lock(peak=0, live=31340.0, cur=31309.5) is None  # losing


def test_usd_lock_plus_100_locks_plus_20_one_contract():
    assert _lock(peak=50.0, live=31396.0, cur=31309.5) == pytest.approx(31356.0)


def test_usd_lock_scales_with_contracts():
    # 4 MNQ: $100 = 12.5 pts; lock $20 = 2.5 pts
    assert _lock(size=4, peak=12.5, live=31358.5, cur=31300.0) == pytest.approx(31348.5)
    assert _lock(size=4, peak=12.0, live=31358.0, cur=31300.0) is None


def test_usd_lock_trails_half_of_peak_and_never_loosens():
    # peak $200 (100 pts) -> keep $100 = 50 pts
    sl = _lock(peak=100.0, live=31440.0, cur=31356.0)
    assert sl == pytest.approx(31396.0)
    # price pulls back, peak unchanged -> no change (never lowers)
    assert _lock(peak=100.0, live=31410.0, cur=sl) is None
    # between 100 and 150: still only the +$20 lock
    assert _lock(peak=70.0, live=31410.0, cur=31300.0) == pytest.approx(31356.0)


def test_usd_lock_short_side():
    sl = slr.usd_lock_stop("short", 31189.25, 2, 2.0, 25.0, 31160.0, 0.25, 31290.25,
                           activate_usd=100, keep_usd=20, trail_start_usd=150,
                           keep_pct=0.5, buffer_ticks=4)
    assert sl == pytest.approx(31184.25)  # $20 / ($2*2) = 5 pts


def _trader():
    from unittest.mock import MagicMock
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.paper_orders = True
    return t


def test_usd_mode_skips_early_atr_trail(monkeypatch):
    monkeypatch.setattr(slr, "LOCK_MODE", "usd")
    t = _trader()
    pos = slr.Position(order_id="bot_l", symbol="MNQ", direction="long", entry_price=25000.0,
                       size=1, sl=24960.0, tp=25100.0, entry_time=datetime.now(timezone.utc))
    t.positions[pos.order_id] = pos
    df = pd.DataFrame([{"high": 25030.0, "low": 25010.0, "close": 25025.0, "atr": 20.0}])
    t.update_trailing_stops("MNQ", df)
    assert pos.sl == 24960.0


def test_usd_lock_ignores_entry_bar_high(monkeypatch):
    monkeypatch.setattr(slr, "LOCK_MODE", "usd")
    t = _trader()
    now = datetime.now(timezone.utc)
    pos = slr.Position(order_id="bot_l", symbol="MNQ", direction="long", entry_price=25000.0,
                       size=1, sl=24960.0, tp=25200.0, entry_time=now)
    t.positions[pos.order_id] = pos
    t.get_price = lambda s: {"last": 25010.0}
    # bar started before the fill: its 25080 high may be pre-entry -> ignored
    df = pd.DataFrame([{"datetime": now - timedelta(minutes=3), "high": 25080.0,
                        "low": 24990.0, "close": 25010.0, "atr": 20.0}])
    t.update_dollar_profit_locks("MNQ", df)
    assert pos.sl == 24960.0
    # next bar fully after entry reaches +$100 -> lock +$20
    t.get_price = lambda s: {"last": 25030.0}
    df2 = pd.DataFrame([{"datetime": now + timedelta(minutes=2), "high": 25052.0,
                         "low": 25020.0, "close": 25030.0, "atr": 20.0}])
    t.update_dollar_profit_locks("MNQ", df2)
    assert pos.sl == pytest.approx(25010.0)


# ---- 3. cooldown ----

def test_cooldown_one_bar_after_win_two_after_loss(monkeypatch):
    t = _trader()
    monkeypatch.setattr(t, "_cancel_orders_for_closed_position", lambda *a, **k: None)
    monkeypatch.setattr(t, "_feed_learner_on_exit", lambda *a, **k: None)
    for oid, px, want in (("w", 25050.0, slr.COOLDOWN_BARS_AFTER_WIN),
                          ("l", 24950.0, slr.COOLDOWN_BARS_AFTER_LOSS)):
        t.positions[oid] = slr.Position(order_id=oid, symbol="MNQ", direction="long",
                                        entry_price=25000.0, size=1, sl=24900.0, tp=25100.0,
                                        entry_time=datetime.now(timezone.utc))
        t.process_exit(oid, "STOP_LOSS", px)
        assert t.cooldown == want
    assert slr.COOLDOWN_BARS_AFTER_WIN == 1 and slr.COOLDOWN_BARS_AFTER_LOSS == 2
