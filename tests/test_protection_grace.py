"""Protection verify grace period and cancel-order-id fixes."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.broker.rithmic_connector import RithmicConnector, plant_is_ready


def _bare_connector() -> RithmicConnector:
    conn = object.__new__(RithmicConnector)
    conn._client = AsyncMock()
    conn._client.plants = {"ticker": object(), "order": object(), "pnl": object()}
    conn._plant_skip_logged = set()
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn._reverse_resolve = lambda sym: sym
    conn._async_cancel_protective_leg = AsyncMock(return_value=True)
    conn._async_cancel_working_bot_order = AsyncMock(return_value=True)
    conn._async_is_order_working = AsyncMock(return_value=True)
    conn._order_lock = MagicMock()
    conn._orders = {}
    return conn


def test_entry_still_open_unknown_within_grace():
    conn = _bare_connector()
    now = datetime.now(timezone.utc).isoformat()
    conn._orders = {
        "bot_grace": {
            "symbol": "MNQ",
            "time": now,
            "fill_time": now,
        },
    }
    conn._client.list_positions = AsyncMock(return_value=[])
    conn._async_get_order_snapshot = AsyncMock(
        return_value=MagicMock(status="filled", avg_fill_price=30200.0),
    )
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_grace_sl"], ["bot_grace_tp"]))
    conn._async_is_leg_filled = AsyncMock(return_value=False)

    still_open = asyncio.run(conn._async_is_entry_still_open("bot_grace", "MNQ"))

    assert still_open is None


def test_cancel_protective_leg_order_not_found_is_success():
    conn = _bare_connector()
    conn._client.cancel_order = AsyncMock(
        side_effect=Exception("Order not found"),
    )

    ok = asyncio.run(
        conn._async_cancel_protective_leg(
            "bot_x_sl",
            symbol="MNQ",
            reason="test",
            require_flat=False,
            cancel_path="test",
        )
    )

    assert ok is True


def test_bulk_cancel_skips_during_entry_grace():
    conn = _bare_connector()
    now = datetime.now(timezone.utc).isoformat()
    conn._orders = {
        "bot_active": {
            "symbol": "MNQ",
            "time": now,
            "fill_time": now,
        },
    }
    conn._client.list_orders = AsyncMock(
        return_value=[
            {
                "symbol": "MNQ",
                "user_tag": "bot_active_sl",
                "order_id": "999001",
                "status": "open",
            },
        ],
    )

    cancelled = asyncio.run(
        conn._async_cancel_all_working_bot_orders_for_symbol("MNQ"),
    )

    assert cancelled == 0
    conn._async_cancel_protective_leg.assert_not_called()


def test_bulk_cancel_uses_user_tag_not_numeric_order_id():
    conn = _bare_connector()
    conn._orders = {}
    conn._client.list_orders = AsyncMock(
        return_value=[
            {
                "symbol": "MNQ",
                "user_tag": "bot_old_sl",
                "order_id": "128127785",
                "status": "open",
            },
        ],
    )

    asyncio.run(conn._async_cancel_all_working_bot_orders_for_symbol("MNQ"))

    conn._async_cancel_protective_leg.assert_awaited_once_with(
        "bot_old_sl",
        symbol="MNQ",
        reason="pre_flatten",
        require_flat=False,
        cancel_path="pre_flatten",
    )


def test_partial_bracket_keeps_sl_during_grace():
    conn = _bare_connector()
    now = datetime.now(timezone.utc).isoformat()
    conn._orders = {
        "bot_9e77d6f519": {
            "symbol": "MNQ",
            "time": now,
            "fill_time": now,
        },
    }
    conn._resolve_symbol = MagicMock(return_value=("MNQH6", "CME"))
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn._round_to_tick = lambda price, tick, mode="nearest": price
    conn._async_is_order_working = AsyncMock(return_value=False)
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._async_submit_protective_leg = AsyncMock(
        side_effect=lambda **kwargs: kwargs.get("leg") == "SL",
    )
    conn._async_try_attach_native_target = AsyncMock(return_value=False)
    conn._protective_chart_legend = MagicMock(return_value="")
    conn._async_cancel_protective_leg = AsyncMock(return_value=True)

    with patch("src.broker.rithmic_connector.get_instrument") as mock_inst:
        mock_inst.return_value = MagicMock(tick_size=0.25)
        out = asyncio.run(
            conn._async_submit_protective_orders(
                symbol="MNQ",
                entry_side="buy",
                qty=1,
                stop_loss=30200.0,
                take_profit=30256.0,
                prefix="bot_9e77d6f519",
                price_ref=30220.0,
            )
        )

    assert "protective_sl_order_id" in out
    conn._async_cancel_protective_leg.assert_not_called()


def test_verify_skips_flat_cancel_during_grace():
    conn = _bare_connector()
    now = datetime.now(timezone.utc).isoformat()
    conn._connected = True
    conn._orders = {
        "bot_entry": {
            "symbol": "MNQ",
            "time": now,
            "fill_time": now,
            "stop_loss": 30100.0,
            "take_profit": 30200.0,
        },
    }
    conn._broker_symbol_has_position = MagicMock(return_value=False)
    conn.cancel_all_bot_orders = MagicMock(return_value=2)
    conn._run_sync = MagicMock(return_value=False)
    conn.query_broker_protection = MagicMock(return_value=(True, True))
    conn._sync_protection_flags_from_broker = MagicMock()
    conn._log_protection_verified = MagicMock()

    sl_ok, tp_ok, closed = conn.verify_and_ensure_protection(
        ticket="bot_entry",
        symbol="MNQ",
        side="buy",
        size=1,
        stop_loss=30100.0,
        take_profit=30200.0,
        max_attempts=1,
    )

    assert sl_ok and tp_ok and not closed
    conn.cancel_all_bot_orders.assert_not_called()


def test_plant_is_ready_false_when_pnl_missing():
    class _Client:
        plants = {"ticker": object(), "order": object()}

    assert plant_is_ready(_Client(), "order") is True
    assert plant_is_ready(_Client(), "pnl") is False
    assert plant_is_ready(None, "pnl") is False


def test_list_positions_skips_quietly_without_pnl_plant():
    conn = _bare_connector()
    conn._client.plants = {"ticker": object(), "order": object()}
    conn._plant_skip_logged = set()
    conn._client.list_positions = AsyncMock(
        side_effect=AttributeError("'NoneType' object has no attribute 'send'")
    )
    result = asyncio.run(conn._async_list_positions())
    assert result is None
    conn._client.list_positions.assert_not_called()
