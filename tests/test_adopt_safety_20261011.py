"""2026-10-11b ADOPT-SAFE: protective orders on adopted (manual) trades can never
open or enlarge a position. All broker calls are mocked - nothing connects to Rithmic."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_orphan_working_orders import _conn, _row
import start_live_rithmic as slr


def _leg(tag, basket, ptype, side, qty, px):
    r = _row(tag, basket, ptype, side)
    r["quantity"] = qty
    if "STOP" in ptype:
        r["trigger_price"] = px
    else:
        r["price"] = px
    return r


def _adopted_conn(net, rows, side="BUY", size=4, oid="bot_adopt_aaa"):
    conn = _conn(net=net, orders=rows)
    conn._orders = {oid: {"ticket": oid, "symbol": "MNQ", "type": side, "size": size,
                          "adopted": True, "entry_price": 31200.0}}
    conn._live_market_price = MagicMock(return_value=31210.0)
    conn._async_place_fresh_sl = AsyncMock(return_value=f"{oid}_r9_sl")
    conn._async_submit_protective_orders = AsyncMock(
        side_effect=lambda **kw: {"protective_tp_order_id": f"{kw['prefix']}_tp"})
    return conn


def _cancelled_baskets(conn):
    return sorted(c.kwargs.get("basket_id") for c in conn._async_cancel_order_on_plant.await_args_list)


def _reconcile_confirmed(conn):
    asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    for k in list(conn._adopt_stale_seen):
        conn._adopt_stale_seen[k] -= 60
    return asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))


LONG_LEGS = lambda: [
    _leg("bot_adopt_aaa_r1_sl", "11", "STOP_LIMIT", "SELL", 4, 31180.0),
    _leg("bot_adopt_aaa_tp", "12", "LIMIT", "SELL", 4, 31240.0),
]


# 1. manual close -> no live adopted orders left
def test_manual_close_cancels_all_adopted_legs():
    conn = _adopted_conn(0, LONG_LEGS())
    first = asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert first["stale"] == [] and _cancelled_baskets(conn) == []  # one-pass flicker guard
    res = _reconcile_confirmed(conn)
    assert res["stale"] == ["bot_adopt_aaa"]
    assert _cancelled_baskets(conn) == ["11", "12"]
    conn._async_place_fresh_sl.assert_not_awaited()
    conn._async_submit_protective_orders.assert_not_awaited()


def test_flat_never_replaces_adopted_stop():
    conn = _adopted_conn(0, [])
    conn._async_working_sl_ids = AsyncMock(return_value=[])
    ok = asyncio.run(conn._async_repair_missing_stop("bot_adopt_aaa", "MNQ", "BUY", 4, 31180.0))
    assert ok is False
    conn._async_place_fresh_sl.assert_not_awaited()


def test_place_fresh_sl_refuses_when_flat_and_clamps_qty():
    from src.broker.rithmic_connector import RithmicConnector
    conn = _adopted_conn(0, [])
    conn._async_place_fresh_sl = RithmicConnector._async_place_fresh_sl.__get__(conn)
    conn._async_submit_protective_orders = AsyncMock(return_value={"protective_sl_order_id": "x"})
    assert asyncio.run(conn._async_place_fresh_sl("bot_adopt_aaa", "MNQ", "BUY", 4, 31180.0, 31210.0)) is None
    conn._async_submit_protective_orders.assert_not_awaited()
    conn._async_list_positions_net_or_none = AsyncMock(return_value=2)
    asyncio.run(conn._async_place_fresh_sl("bot_adopt_aaa", "MNQ", "BUY", 4, 31180.0, 31210.0))
    assert conn._async_submit_protective_orders.await_args.kwargs["qty"] == 2


# 2. partial close -> resized to what is really open
def test_partial_close_resizes_sl_and_tp(monkeypatch):
    monkeypatch.setenv("ADOPT_PLACE_TP", "0")
    conn = _adopted_conn(2, LONG_LEGS())
    res = asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert _cancelled_baskets(conn) == ["11", "12"]
    sl_args = conn._async_place_fresh_sl.await_args.args
    assert sl_args[:4] == ("bot_adopt_aaa", "MNQ", "BUY", 2) and sl_args[4] == pytest.approx(31180.0)
    tp_kw = conn._async_submit_protective_orders.await_args.kwargs
    assert tp_kw["qty"] == 2 and tp_kw["stop_loss"] == 0 and tp_kw["take_profit"] == pytest.approx(31240.0)
    assert tp_kw["prefix"].startswith("bot_adopt_aaa_r")
    assert conn._orders["bot_adopt_aaa"]["size"] == 2 and res["sizes"] == {"bot_adopt_aaa": 2}


def test_matching_size_left_alone():
    conn = _adopted_conn(4, LONG_LEGS())
    res = asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert res["resized"] == [] and _cancelled_baskets(conn) == []


# 3. flip -> adopted legs cancelled (they would add to the new position)
def test_flip_cancels_adopted_legs():
    conn = _adopted_conn(-4, LONG_LEGS())
    res = _reconcile_confirmed(conn)
    assert res["stale"] == ["bot_adopt_aaa"] and _cancelled_baskets(conn) == ["11", "12"]
    conn._async_place_fresh_sl.assert_not_awaited()
    conn2 = _adopted_conn(-4, LONG_LEGS())
    conn2._async_working_sl_ids = AsyncMock(return_value=[])
    assert asyncio.run(conn2._async_repair_missing_stop("bot_adopt_aaa", "MNQ", "BUY", 4, 31180.0)) is False


def test_unknown_net_does_nothing():
    conn = _adopted_conn(None, LONG_LEGS())
    res = asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert res.get("skipped") == "net unknown" and _cancelled_baskets(conn) == []


# 4. net includes the bot's own trade -> only the manual part is adopted
def _adopt_conn(rows, net, avg=31200.0):
    conn = _conn(net=net, orders=rows)
    conn._async_list_positions = AsyncMock(return_value=[
        {"symbol": "MNQ", "net_quantity": net, "avg_open_fill_price": avg}])
    conn._live_market_price = MagicMock(return_value=31210.0)
    conn._async_repair_missing_stop = AsyncMock(return_value=True)
    conn._async_submit_protective_orders = AsyncMock(
        side_effect=lambda **kw: {"protective_tp_order_id": f"{kw['prefix']}_tp"})
    return conn


def test_bot_trade_size_excluded_from_adoption():
    conn = _adopt_conn([_leg("bot_live_sl", "1", "STOP_LIMIT", "SELL", 1, 31160.0)], net=5)
    conn._orders = {"bot_live": {"symbol": "MNQ", "type": "BUY", "size": 1}}
    res = asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0))
    assert res["size"] == 4
    assert conn._async_repair_missing_stop.await_args.args[3] == 4  # bot's own SL not reused


def test_bot_trade_on_other_side_blocks_adoption():
    conn = _adopt_conn([], net=3)
    conn._orders = {"bot_live": {"symbol": "MNQ", "type": "SELL", "size": 1}}
    assert asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0)) is None
    conn._async_repair_missing_stop.assert_not_awaited()


def test_net_changing_during_adopt_aborts():
    conn = _adopt_conn([], net=4)
    conn._async_list_positions = AsyncMock(side_effect=[
        [{"symbol": "MNQ", "net_quantity": 4, "avg_open_fill_price": 31200.0}],
        [{"symbol": "MNQ", "net_quantity": 0}],
    ])
    assert asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0)) is None
    conn._async_repair_missing_stop.assert_not_awaited()


def test_growth_merges_into_existing_adopted_no_second_bracket():
    conn = _adopt_conn(LONG_LEGS(), net=8)
    conn._orders = {"bot_adopt_aaa": {"ticket": "bot_adopt_aaa", "symbol": "MNQ", "type": "BUY",
                                      "size": 4, "adopted": True, "entry_price": 31200.0}}
    conn._async_place_fresh_sl = AsyncMock(return_value="bot_adopt_aaa_r2_sl")
    res = asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0))
    assert res["merged"] is True and res["ticket"] == "bot_adopt_aaa" and res["size"] == 8
    conn._async_repair_missing_stop.assert_not_awaited()
    assert [k for k in conn._orders if k.startswith("bot_adopt_") and k != "bot_adopt_aaa"] == []
    # the one adopted SL is resized to the full manual size (8), never a second 4-lot stop
    assert conn._async_place_fresh_sl.await_args.args[3] == 8


# 5. the user's own TP / SL is never duplicated
def test_existing_user_tp_not_duplicated_on_adopt(monkeypatch):
    monkeypatch.setenv("ADOPT_PLACE_TP", "1")
    user_tp = _leg("bb6785e3-08fa-4c71-9965-daff6922890a", "248119685", "LIMIT", "SELL", 4, 31260.75)
    conn = _adopt_conn([user_tp], net=4)
    res = asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0, tp_r=1.5))
    conn._async_submit_protective_orders.assert_not_awaited()
    assert res["tp_existing"] is True


def test_user_tp_added_after_adopt_cancels_bot_tp():
    rows = LONG_LEGS() + [
        _leg("bb6785e3-08fa-4c71-9965-daff6922890a", "248119685", "LIMIT", "SELL", 4, 31260.75)]
    conn = _adopted_conn(4, rows)
    res = asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert _cancelled_baskets(conn) == ["12"]  # only the bot TP; user's TP + bot SL kept
    conn._async_submit_protective_orders.assert_not_awaited()
    assert ("bot_adopt_aaa", "tp", 4, 0) in res["resized"]


def test_user_stop_added_after_adopt_cancels_bot_stop():
    rows = LONG_LEGS() + [
        _leg("3a83c8a7-89fb-48c9-9e02-7c5b1f1f5f7b", "248119192", "STOP_MARKET", "SELL", 4, 31188.75)]
    conn = _adopted_conn(4, rows)
    asyncio.run(conn._async_reconcile_adopted_protection("MNQ"))
    assert _cancelled_baskets(conn) == ["11"]
    conn._async_place_fresh_sl.assert_not_awaited()


def test_adopt_tp_off_by_default(monkeypatch):
    monkeypatch.delenv("ADOPT_PLACE_TP", raising=False)
    conn = _adopt_conn([], net=4)
    res = asyncio.run(conn._async_adopt_untracked_position("MNQ", 30.0, tp_r=1.5))
    conn._async_submit_protective_orders.assert_not_awaited()
    assert res["tp_placed"] is False


# start_live side
def _trader(monkeypatch, net_signed):
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.broker._positions = {"MNQ": {"size": net_signed, "avg_price": 31200.0}}
    monkeypatch.setattr(t, "_live_broker", lambda: True)
    monkeypatch.setattr(t, "_broker_net_qty", lambda sym: abs(net_signed))
    monkeypatch.setattr(t, "_planned_sl_from_trades_log", lambda *a: 0.0)
    monkeypatch.setattr(t, "_planned_tp_from_trades_log", lambda *a: 0.0)
    return t


def _pos(oid, direction, size):
    return slr.Position(order_id=oid, symbol="MNQ", direction=direction, entry_price=31200.0,
                        size=size, sl=31180.0, tp=0.0, entry_time=datetime.now(timezone.utc))


def test_live_growth_merges_not_restacks(monkeypatch):
    t = _trader(monkeypatch, 8)
    t.positions = {"bot_adopt_aaa": _pos("bot_adopt_aaa", "long", 4)}
    t.broker.reconcile_adopted_protection.return_value = {"stale": [], "sizes": {}}
    t.broker.adopt_untracked_position.return_value = {
        "ticket": "bot_adopt_aaa", "type": "BUY", "size": 8, "open_price": 31205.0, "merged": True}
    assert t._adopt_untracked_exposure() == 0
    t._untracked_seen_at["MNQ"] -= slr.ADOPT_CONFIRM_SEC + 1
    t._adopt_reconcile_ts = {}
    assert t._adopt_untracked_exposure() == 0  # merged, not a new trade
    assert t.positions["bot_adopt_aaa"].size == 8 and len(t.positions) == 1


def test_live_stale_adopted_released(monkeypatch):
    t = _trader(monkeypatch, 0)
    t.positions = {"bot_adopt_aaa": _pos("bot_adopt_aaa", "long", 4)}
    t.broker.reconcile_adopted_protection.return_value = {"stale": ["bot_adopt_aaa"]}
    monkeypatch.setattr(t, "_cancel_orders_for_closed_position", lambda *a, **k: None)
    t._adopt_untracked_exposure()
    assert "bot_adopt_aaa" not in t.positions


def test_live_bot_trade_other_side_never_adopted(monkeypatch):
    t = _trader(monkeypatch, -4)
    t.positions = {"bot_live": _pos("bot_live", "long", 1)}
    t._adopt_untracked_exposure()
    t._untracked_seen_at = {"MNQ": time.time() - 999}
    t._adopt_untracked_exposure()
    t.broker.adopt_untracked_position.assert_not_called()


def test_adopt_manual_env_switch(monkeypatch):
    import importlib
    monkeypatch.setenv("ADOPT_MANUAL", "0")
    mod = importlib.reload(slr)
    try:
        assert mod.ADOPT_UNTRACKED is False
    finally:
        monkeypatch.delenv("ADOPT_MANUAL", raising=False)
        importlib.reload(slr)

