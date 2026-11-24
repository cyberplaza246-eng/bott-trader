"""Regression: 2026-10-05 20:16 — two MNQ longs hit TP on Lucid, bot booked LOSS $0.

Lucid sim route reports list_positions=0 even while trades are open, so an
empty position list must never by itself cancel SL/TP or drop a trade, and a
real TP/SL fill must always be carried through as real PnL.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import src.broker.rithmic_connector as rc
from tests.test_manual_adjust_tracking import (
    _cleanup_conn, _row, _two_longs, _trader, _pos,
)


def _filled_entries(conn):
    _two_longs(conn)
    for e in ("bot_a", "bot_b"):
        conn._orders[e]["fill_time"] = "2026-10-05T23:50:00+00:00"


def test_list_positions_zero_with_working_legs_never_cancels_or_drops(monkeypatch):
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_SEC", 0.0)
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_PASSES", 1)
    rows = [_row("bot_a_sl", "11", "STOP", trigger=95.0), _row("bot_a_tp", "12", price=110.0),
            _row("bot_b_sl", "21", "STOP", trigger=96.0), _row("bot_b_tp", "22", price=111.0)]
    conn = _cleanup_conn(rows, net=0)
    _filled_entries(conn)
    conn._async_is_order_working = AsyncMock(return_value=True)
    conn._async_get_order_snapshot = AsyncMock(return_value=None)   # Lucid can't resolve tag
    conn._async_listed_protective_fill_for_entry = AsyncMock(return_value=None)
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    for _ in range(5):
        conn._cleanup_closed_orders()
    assert "bot_a" in conn._orders and "bot_b" in conn._orders
    conn._async_cancel_remaining_bracket_legs.assert_not_awaited()
    conn._async_cancel_entry_protective_orders.assert_not_awaited()


def test_unresolvable_entry_tag_with_working_legs_is_still_open():
    rows = [_row("bot_a_sl", "11", "STOP", trigger=95.0), _row("bot_a_tp", "12", price=110.0)]
    conn = _cleanup_conn(rows, net=0)
    _filled_entries(conn)
    conn._async_get_order_snapshot = AsyncMock(return_value=None)
    conn._async_listed_protective_fill_for_entry = AsyncMock(return_value=None)
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._async_is_order_working = AsyncMock(side_effect=lambda t: t == "bot_a_tp")
    assert asyncio.run(conn._async_is_entry_still_open("bot_a", "MNQ")) is True


def test_tp_fill_is_recorded_with_real_price_before_release(monkeypatch):
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_SEC", 45.0)
    conn = _cleanup_conn([], net=0)
    _filled_entries(conn)
    conn._orders = {k: v for k, v in conn._orders.items() if k.startswith("bot_a")}
    conn._async_listed_protective_fill_for_entry = AsyncMock(
        return_value={"leg": "TP", "exit_price": 110.0, "user_tag": "bot_a_tp", "basket_id": "12"},
    )
    conn._cleanup_closed_orders()
    # TP fill is decisive: released immediately (no legs-gone wait), price kept.
    assert "bot_a" not in conn._orders
    fill = conn._confirmed_exit_fills["bot_a"]
    assert fill["confirmed"] and fill["leg"] == "TP" and fill["exit_price"] == 110.0


def test_legs_gone_needs_time_and_passes_before_release(monkeypatch):
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_SEC", 0.0)
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_PASSES", 3)
    conn = _cleanup_conn([], net=0)
    _filled_entries(conn)
    conn._orders = {k: v for k, v in conn._orders.items() if k.startswith("bot_a")}
    conn._async_is_entry_still_open = AsyncMock(return_value=False)
    conn._async_find_exit_fill_for_entry = AsyncMock(return_value=None)
    conn._cleanup_closed_orders(); conn._cleanup_closed_orders()
    assert "bot_a" in conn._orders
    conn._cleanup_closed_orders()
    assert "bot_a" not in conn._orders


def test_legs_reappear_resets_confirmation(monkeypatch):
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_SEC", 0.0)
    monkeypatch.setattr(rc, "_LEGS_GONE_CONFIRM_PASSES", 2)
    conn = _cleanup_conn([], net=0)
    _filled_entries(conn)
    conn._orders = {k: v for k, v in conn._orders.items() if k.startswith("bot_a")}
    conn._async_find_exit_fill_for_entry = AsyncMock(return_value=None)
    conn._async_is_entry_still_open = AsyncMock(side_effect=[False, True, False, False])
    for _ in range(3):
        conn._cleanup_closed_orders()
    assert "bot_a" in conn._orders, "a blip of visible legs must restart the count"
    conn._cleanup_closed_orders()
    assert "bot_a" not in conn._orders


def test_trader_books_tp_pnl_not_zero_loss_when_connector_dropped():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")          # long @100, TP 110
    t.broker.recover_exit_fill = MagicMock(
        return_value={"confirmed": True, "leg": "TP", "exit_price": 110.0},
    )
    t.sync_broker_position()
    assert "bot_a" not in t.positions
    tr = t.trades[-1]
    assert tr["exit_type"] == "TAKE_PROFIT" and tr["exit_price"] == 110.0 and tr["pnl"] > 0


def test_trader_keeps_trade_when_legs_still_working_even_if_flat():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.broker.recover_exit_fill = MagicMock(return_value=None)
    t.broker.entry_protection_live = MagicMock(return_value=True)
    t.sync_broker_position()
    assert "bot_a" in t.positions and not t.trades


def test_unknown_exit_not_fed_to_learner_as_loss():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.broker.recover_exit_fill = MagicMock(return_value=None)
    t.broker.entry_protection_live = MagicMock(return_value=False)
    t.sync_broker_position()
    assert "bot_a" not in t.positions
    assert t.trades[-1]["exit_type"] == "BROKER_SYNC_UNKNOWN"
    t._feed_learner_on_exit.assert_not_called()
