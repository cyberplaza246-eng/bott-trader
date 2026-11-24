"""Orphan working-order cleanup (no live orders; mocked Rithmic).

Covers the 'leftover sell limit on the chart' bug:
  * OCO sibling kept forever when a 2nd concurrent MNQ trade kept the symbol open
  * legs unreachable after _orders.pop (Lucid blanks user_tag; basket link lost)
  * start_live_rithmic dropped trades (broker sync / bar exit / shutdown)
    without cancelling their working orders
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import start_live_rithmic as slr
from src.broker.rithmic_connector import RithmicConnector


# ── connector helpers ──────────────────────────────────────────────

def _conn(net: Optional[int] = 0, orders: Optional[List[Dict[str, Any]]] = None) -> RithmicConnector:
    conn = object.__new__(RithmicConnector)
    conn._client = AsyncMock()
    conn._client.plants = {"ticker": object(), "order": object(), "pnl": object()}
    conn._plant_skip_logged = set()
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn._reverse_resolve = lambda sym: sym
    conn._order_lock = threading.Lock()
    conn._state_lock = threading.Lock()
    conn._orders = {}
    conn._positions = {}
    conn._confirmed_exit_fills = {}
    conn._connected = True
    conn._async_cancel_protective_leg = AsyncMock(return_value=True)
    conn._async_cancel_order_on_plant = AsyncMock(return_value=True)
    conn._async_is_order_working = AsyncMock(return_value=False)
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._async_list_positions_net_or_none = AsyncMock(return_value=net)
    conn._async_list_working_listed_orders = AsyncMock(return_value=list(orders or []))
    conn._listed_order_matches_symbol = MagicMock(return_value=True)
    return conn


def _row(tag: str, basket: str, ptype: str = "LIMIT", side: str = "SELL") -> Dict[str, Any]:
    return {
        "symbol": "MNQ", "user_tag": tag, "basket_id": basket,
        "status": "open", "price_type": ptype, "transaction_type": side,
    }


def _cancelled_targets(conn: RithmicConnector) -> List[str]:
    out = []
    for call in conn._async_cancel_protective_leg.await_args_list:
        out.append(str(call.args[0] if call.args else call.kwargs.get("leg_order_id")))
    return out


# ── 1. OCO sibling with two concurrent MNQ trades ───────────────────

def _fill_notification(tag: str):
    from async_rithmic.enums import ExchangeOrderNotificationType
    return {"user_tag": tag, "notify_type": ExchangeOrderNotificationType.FILL}


def _oco_conn(net: int) -> RithmicConnector:
    conn = _conn(net=net)
    conn._async_is_leg_filled = AsyncMock(return_value=True)
    conn._async_get_order_snapshot = AsyncMock(return_value=None)
    conn._fill_price_from_order = MagicMock(return_value=100.0)
    conn._record_exit_fill = MagicMock()
    conn.get_trade_route = MagicMock(return_value="simulator")
    conn._async_cancel_remaining_bracket_legs = AsyncMock(return_value=0)
    conn._orders = {
        "bot_a": {"symbol": "MNQ", "size": 1},
        "bot_b": {"symbol": "MNQ", "size": 1},
    }
    return conn


def test_oco_cancels_sibling_while_other_trade_keeps_symbol_open():
    conn = _oco_conn(net=1)  # bot_b still open → symbol not flat
    asyncio.run(conn._on_exchange_order_notification(_fill_notification("bot_a_sl")))
    assert "bot_a_tp" in _cancelled_targets(conn), "closed trade's sell limit must be cancelled"
    conn._async_cancel_remaining_bracket_legs.assert_awaited()
    assert "bot_a" in conn._closed_entry_registry()


def test_oco_cancels_sibling_even_when_position_net_lags():
    """After THIS entry's SL/TP fill, always cancel its sibling — net lag must not keep it."""
    conn = _oco_conn(net=2)  # plant still shows 2; only bot_b is the other live trade
    asyncio.run(conn._on_exchange_order_notification(_fill_notification("bot_a_sl")))
    assert "bot_a_tp" in _cancelled_targets(conn), "closed trade sibling must cancel regardless of net"
    assert "bot_a" not in conn._orders  # tracking dropped so STOP-LIMIT is not treated as active
    assert "bot_b" in conn._orders  # other concurrent trade untouched


def test_oco_tp_fill_cancels_stop_limit_sibling_when_net_still_open():
    """TP fill must cancel STOP-LIMIT SL even if list_positions still shows size."""
    conn = _oco_conn(net=1)  # lag / single-trade plant still showing 1 after TP
    conn._orders = {"bot_a": {"symbol": "MNQ", "size": 1, "protective_sl_order_id": "bot_a_sl"}}
    asyncio.run(conn._on_exchange_order_notification(_fill_notification("bot_a_tp")))
    assert "bot_a_sl" in _cancelled_targets(conn)
    conn._async_cancel_remaining_bracket_legs.assert_awaited()
    assert "bot_a" not in conn._orders


def test_oco_exchange_fill_cancels_even_if_leg_not_reconfirmed():
    """Lucid often blanks tags; exchange FILL alone must wipe the sibling STOP-LIMIT."""
    conn = _oco_conn(net=1)
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._remember_detected_exit_fill = MagicMock()
    asyncio.run(conn._on_exchange_order_notification(_fill_notification("bot_a_tp")))
    assert "bot_a_sl" in _cancelled_targets(conn)
    conn._remember_detected_exit_fill.assert_called()


# ── 2. legs still reachable after tracking is dropped ───────────────

def test_blank_tag_leg_cancelled_after_entry_popped_via_remembered_basket():
    orphan = _row("", "777001")  # Lucid blanked the tag on the TP sell limit
    conn = _conn(net=0, orders=[orphan])
    conn._orders["bot_a"] = {"symbol": "MNQ", "size": 1, "protective_tp_basket_id": "777001"}
    conn.remember_closed_entry("bot_a", "MNQ")
    conn._orders.pop("bot_a")
    n = asyncio.run(conn._async_cancel_remaining_bracket_legs(
        "bot_a", "MNQ", reason="t", force=True,
    ))
    assert n == 1
    assert conn._async_cancel_protective_leg.await_args.kwargs["basket_id"] == "777001"


def test_release_closed_entry_cancels_and_drops_tracking():
    conn = _conn(net=1, orders=[_row("bot_a_tp", "1"), _row("bot_b_tp", "2")])
    conn._run_sync = lambda coro, timeout=None: asyncio.run(coro)
    conn._orders = {"bot_a": {"symbol": "MNQ", "size": 1}, "bot_b": {"symbol": "MNQ", "size": 1}}
    n = conn.release_closed_entry("bot_a", "MNQ", reason="t")
    assert n == 1
    assert _cancelled_targets(conn) == ["bot_a_tp"], "other trade's TP must stay"
    assert "bot_a" not in conn._orders and "bot_b" in conn._orders


# ── 3. periodic orphan sweep ────────────────────────────────────────

def _sweep(conn, keep):
    return asyncio.run(conn._async_sweep_orphan_working_orders("MNQ", set(keep), reason="t"))


def test_sweep_keeps_live_trade_and_cancels_orphans():
    rows = [
        _row("bot_old_tp", "11"),                      # orphan sell limit
        _row("bot_old_sl", "12", "STOP", "SELL"),      # orphan sell stop
        _row("bot_b_sl", "21", "STOP", "SELL"),        # live trade legs
        _row("bot_b_tp", "22"),
        _row("manual_x", "31"),                        # tagged manual ticket
    ]
    conn = _conn(net=1, orders=rows)
    conn._orders = {"bot_b": {"symbol": "MNQ", "size": 1}}
    first = _sweep(conn, {"bot_b"})
    assert first["skipped"] == "awaiting_stable_net"
    assert _cancelled_targets(conn) == []
    conn._orphan_sweep_ok_at["MNQ"] -= 25  # next cycle
    res = _sweep(conn, {"bot_b"})
    got = set(_cancelled_targets(conn))
    assert {"bot_old_tp", "bot_old_sl"} <= got
    assert not got & {"bot_b_sl", "bot_b_tp", "manual_x"}
    assert res["cancelled"] >= 2


def test_sweep_never_strips_untracked_exposure():
    conn = _conn(net=2, orders=[_row("bot_old_tp", "11")])
    conn._orders = {"bot_b": {"symbol": "MNQ", "size": 1}}
    _sweep(conn, {"bot_b"})
    conn._orphan_sweep_ok_at = {"MNQ": time.time() - 25}
    res = _sweep(conn, {"bot_b"})
    assert res["skipped"].startswith("untracked_exposure")
    assert _cancelled_targets(conn) == []


def test_sweep_skips_broad_wipe_when_net_unknown():
    conn = _conn(net=None, orders=[_row("bot_old_tp", "11")])
    res = _sweep(conn, set())
    assert res["skipped"] == "net_unknown"
    assert _cancelled_targets(conn) == []


def test_sweep_cleans_known_closed_entry_even_when_net_unknown():
    conn = _conn(net=None, orders=[_row("bot_a_tp", "11")])
    conn._orders["bot_a"] = {"symbol": "MNQ", "size": 1}
    conn.remember_closed_entry("bot_a", "MNQ")
    conn._orders.pop("bot_a")
    _sweep(conn, set())
    assert "bot_a_tp" in _cancelled_targets(conn)


# ── 4. start_live_rithmic wiring ────────────────────────────────────

class SpyBroker:
    connected = True

    def __init__(self):
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._order_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._confirmed_exit_fills: Dict[str, Dict[str, Any]] = {}
        self.released: List[tuple] = []
        self.sweeps: List[tuple] = []
        self.confirm_result: Optional[Dict[str, Any]] = None

    def get_open_positions(self, symbol=None):
        return []

    def entry_protection_grace_active(self, order_id):
        return False

    def get_order_info(self, ticket):
        return self._orders.get(str(ticket))

    def release_closed_entry(self, order_id, symbol, reason=""):
        self.released.append((order_id, symbol, reason))
        return 2

    def sweep_orphan_working_orders(self, symbol, live_entry_ids=None, reason=""):
        self.sweeps.append((symbol, set(live_entry_ids or ()), reason))
        return {"cancelled": 0}

    def confirm_bracket_exit_fill(self, *a, **k):
        return self.confirm_result


def _pos(oid: str, age: float = 120.0) -> slr.Position:
    return slr.Position(
        order_id=oid, symbol="MNQ", direction="long", entry_price=100.0, size=1,
        sl=99.0, tp=101.0, entry_time=datetime.now(timezone.utc) - timedelta(seconds=age),
    )


def _trader() -> slr.LiveRithmicTrader:
    t = slr.LiveRithmicTrader(symbol="MNQ", symbols=["MNQ"], paper_mode=False, skip_confirm=True)
    t.broker = SpyBroker()
    t.paper_orders = False
    t._feed_learner_on_exit = MagicMock()
    t._write_board_status = MagicMock()
    return t


def test_max_concurrent_capped_at_three():
    assert slr.ENV_MAX_CONCURRENT_TRADES <= 3
    assert _trader().max_positions <= 3


def test_broker_sync_drop_cancels_that_trades_orders_and_keeps_other():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.positions["bot_b"] = _pos("bot_b")
    t.broker._orders["bot_b"] = {"symbol": "MNQ"}  # connector still tracks b
    t.sync_broker_position()
    assert "bot_a" not in t.positions and "bot_b" in t.positions
    assert ("bot_a", "MNQ", "broker_sync_flat") in t.broker.released
    assert t.broker.sweeps and t.broker.sweeps[-1][1] == {"bot_b"}


def test_broker_sync_uses_confirmed_fill_price():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.broker._confirmed_exit_fills["bot_a"] = {"confirmed": True, "leg": "TP", "exit_price": 101.0}
    t.sync_broker_position()
    assert t.trades[-1]["exit_type"] == "TAKE_PROFIT"
    assert t.broker.released and t.broker.released[0][0] == "bot_a"


def test_bar_touch_without_broker_fill_keeps_trade_and_orders():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.broker.confirm_result = None
    assert t._confirmed_live_exit("bot_a", "TAKE_PROFIT") is None
    assert "bot_a" in t.positions and not t.broker.released


def test_bar_touch_with_confirmed_fill_closes_and_cancels():
    t = _trader()
    t.positions["bot_a"] = _pos("bot_a")
    t.broker.confirm_result = {"confirmed": True, "leg": "SL", "exit_price": 98.75}
    et, px = t._confirmed_live_exit("bot_a", "TAKE_PROFIT")
    assert (et, px) == ("STOP_LOSS", 98.75)
    t.process_exit("bot_a", et, px)
    assert t.broker.released[0][:2] == ("bot_a", "MNQ")


def test_periodic_sweep_throttled_and_passes_live_ids():
    t = _trader()
    t.positions["bot_b"] = _pos("bot_b")
    t.sweep_orphan_orders()
    t.sweep_orphan_orders()  # throttled
    assert len(t.broker.sweeps) == 1 and t.broker.sweeps[0][1] == {"bot_b"}


def test_shutdown_sweeps_and_keeps_open_trade_protection(monkeypatch):
    monkeypatch.setattr(slr.time, "sleep", lambda s: None)
    t = _trader()
    t.positions["bot_b"] = _pos("bot_b")
    t.cleanup_orders_on_shutdown()
    reasons = [s[2] for s in t.broker.sweeps]
    assert reasons.count("shutdown") == 2
    assert all(s[1] == {"bot_b"} for s in t.broker.sweeps)


def test_paper_mode_never_touches_broker_orders():
    t = _trader()
    t.paper_orders = True
    t.positions["paper_1"] = _pos("paper_1")
    t.process_exit("paper_1", "TAKE_PROFIT", 101.0)
    t.sweep_orphan_orders(force=True)
    assert not t.broker.released and not t.broker.sweeps


def test_oco_resolves_blank_tag_fill_via_basket():
    conn = _oco_conn(net=1)
    conn._orders["bot_a"]["protective_sl_basket_id"] = "555"
    from async_rithmic.enums import ExchangeOrderNotificationType
    asyncio.run(conn._on_exchange_order_notification(
        {"user_tag": "", "basket_id": "555", "notify_type": ExchangeOrderNotificationType.FILL}
    ))
    assert "bot_a_tp" in _cancelled_targets(conn)
