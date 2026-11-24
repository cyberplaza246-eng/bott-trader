"""Duplicate closing stops (2026-10-19): repair must not double up, extras get trimmed.

Live bug: a 1-lot short ended up with _sl, _r1_sl, _r2_sl, _r3_sl BUY stops at one price.
When price hit it, every stop filled: one closed the short, the rest opened new longs.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

from src.broker.rithmic_connector import RithmicConnector


def _conn(net, rows):
    conn = object.__new__(RithmicConnector)
    conn._client = AsyncMock()
    conn._client.plants = {"ticker": object(), "order": object(), "pnl": object()}
    conn._plant_skip_logged = set()
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn._reverse_resolve = lambda sym: sym
    conn._order_lock = threading.RLock()
    conn._orders = {}
    conn._client.list_positions = AsyncMock(
        return_value=[] if net == 0 else [{"symbol": "MNQ", "net_qty": net}])
    if isinstance(rows, Exception):
        conn._client.list_orders = AsyncMock(side_effect=rows)
    else:
        conn._client.list_orders = AsyncMock(return_value=rows)
    conn._async_cancel_protective_leg = AsyncMock(return_value=True)
    conn._async_working_sl_ids = AsyncMock(return_value=[])
    conn._async_place_fresh_sl = AsyncMock(return_value="bot_s_r1_sl")
    conn._live_market_price = MagicMock(return_value=29600.0)
    return conn


def _stop(tag, basket, side="BUY", qty=1, px=29653.75):
    return {"symbol": "MNQ", "user_tag": tag, "basket_id": basket, "status": "open",
            "transaction_type": side, "price_type": "STOP_LIMIT", "quantity": qty,
            "trigger_price": px, "price": px + 0.5}


def test_repair_skips_when_broker_already_has_covering_stop():
    # tag lookup says "no SL" (Lucid blanked the tag), but a BUY stop for 1 is working
    conn = _conn(-1, [_stop("", "256474577")])
    conn._orders["bot_s"] = {"symbol": "MNQ", "type": "SELL", "size": 1, "stop_loss": 29653.75}
    ok = asyncio.run(conn._async_repair_missing_stop("bot_s", "MNQ", "SELL", 1, 29653.75))
    assert ok is True
    conn._async_place_fresh_sl.assert_not_called()


def test_repair_skips_when_order_list_unknown():
    conn = _conn(-1, RuntimeError("order plant reconnecting"))
    conn._orders["bot_s"] = {"symbol": "MNQ", "type": "SELL", "size": 1, "stop_loss": 29653.75}
    ok = asyncio.run(conn._async_repair_missing_stop("bot_s", "MNQ", "SELL", 1, 29653.75))
    assert ok is False
    conn._async_place_fresh_sl.assert_not_called()


def test_repair_skips_when_broker_flat():
    conn = _conn(0, [])
    conn._orders["bot_s"] = {"symbol": "MNQ", "type": "SELL", "size": 1, "stop_loss": 29653.75}
    asyncio.run(conn._async_repair_missing_stop("bot_s", "MNQ", "SELL", 1, 29653.75))
    conn._async_place_fresh_sl.assert_not_called()


def test_repair_places_only_missing_qty():
    conn = _conn(-2, [_stop("bot_a_sl", "b1")])
    conn._orders["bot_s"] = {"symbol": "MNQ", "type": "SELL", "size": 2, "stop_loss": 29653.75}
    asyncio.run(conn._async_repair_missing_stop("bot_s", "MNQ", "SELL", 2, 29653.75))
    conn._async_place_fresh_sl.assert_awaited_once()
    assert conn._async_place_fresh_sl.await_args.args[3] == 1


def test_trim_cancels_extra_stops_keeps_current():
    rows = [_stop("bot_s_sl", "b0"), _stop("bot_s_r1_sl", "b1"), _stop("bot_s_r2_sl", "b2"),
            _stop("bot_s_r3_sl", "b3")]
    conn = _conn(-1, rows)
    conn._orders["bot_s"] = {"symbol": "MNQ", "type": "SELL", "size": 1,
                             "protective_sl_order_id": "bot_s_r2_sl"}
    n = asyncio.run(conn._async_trim_excess_stops("MNQ"))
    assert n == 3
    cancelled = {c.args[0] for c in conn._async_cancel_protective_leg.await_args_list}
    assert cancelled == {"bot_s_sl", "bot_s_r1_sl", "bot_s_r3_sl"}


def test_trim_leaves_matching_coverage_and_other_side_alone():
    rows = [_stop("bot_s_sl", "b0"), _stop("manual_sell_stop", "m1", side="SELL")]
    conn = _conn(-1, rows)
    assert asyncio.run(conn._async_trim_excess_stops("MNQ")) == 0
    conn._async_cancel_protective_leg.assert_not_called()


def test_trim_waits_while_stop_is_being_moved():
    rows = [_stop("bot_s_sl", "b0"), _stop("bot_s_r1_sl", "b1")]
    conn = _conn(-1, rows)
    conn._orders["bot_s"] = {"symbol": "MNQ", "sl_change_ts": time.time()}
    assert asyncio.run(conn._async_trim_excess_stops("MNQ")) == 0
    conn._async_cancel_protective_leg.assert_not_called()


def test_trim_never_runs_when_flat_or_unknown():
    rows = [_stop("bot_s_sl", "b0"), _stop("bot_s_r1_sl", "b1")]
    conn = _conn(0, rows)
    assert asyncio.run(conn._async_trim_excess_stops("MNQ")) == 0
    conn._client.list_positions = AsyncMock(side_effect=RuntimeError("pnl plant down"))
    assert asyncio.run(conn._async_trim_excess_stops("MNQ")) == 0
    conn._async_cancel_protective_leg.assert_not_called()
