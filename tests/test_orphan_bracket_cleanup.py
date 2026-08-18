"""Orphan SL/TP cleanup after bracket exit (mock Rithmic connector)."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from src.broker.rithmic_connector import RithmicConnector


def _bare_connector() -> RithmicConnector:
    conn = object.__new__(RithmicConnector)
    conn._client = AsyncMock()
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn._reverse_resolve = lambda sym: sym
    conn._async_cancel_protective_leg = AsyncMock(return_value=True)
    conn._async_cancel_working_bot_order = AsyncMock(return_value=True)
    conn._async_is_order_working = AsyncMock(return_value=True)
    conn._order_lock = MagicMock()
    conn._orders = {}
    return conn


def test_orphan_cleanup_skips_when_position_open():
    conn = _bare_connector()
    conn._client.list_positions = AsyncMock(
        return_value=[{"symbol": "MNQ", "net_qty": 1}],
    )

    cancelled = asyncio.run(conn._async_cancel_orphan_protective_if_flat("MNQ"))

    assert cancelled == 0
    conn._async_cancel_protective_leg.assert_not_called()
    conn._async_cancel_working_bot_order.assert_not_called()


def test_orphan_cleanup_cancels_all_bot_orders_when_flat():
    conn = _bare_connector()
    conn._client.list_positions = AsyncMock(return_value=[])
    conn._client.list_orders = AsyncMock(
        return_value=[
            {
                "symbol": "MNQ",
                "user_tag": "bot_abc123_sl",
                "order_id": "leg-sl-1",
                "status": "open",
            },
            {
                "symbol": "MNQ",
                "user_tag": "bot_abc123_tp",
                "order_id": "leg-tp-1",
                "status": "open",
            },
            {
                "symbol": "MNQ",
                "user_tag": "bot_xyz789",
                "order_id": "entry-1",
                "status": "open",
            },
        ],
    )

    cancelled = asyncio.run(conn._async_cancel_orphan_protective_if_flat("MNQ"))

    assert cancelled == 3
    assert conn._async_cancel_protective_leg.await_count == 2
    conn._async_cancel_working_bot_order.assert_awaited_once_with(
        "bot_xyz789",
        symbol="MNQ",
        reason="pre_flatten",
    )


def test_cancel_remaining_bracket_legs_uses_require_flat_false():
    conn = _bare_connector()
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._async_is_entry_still_open = AsyncMock(return_value=False)
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_order_working = AsyncMock(side_effect=lambda oid: oid.endswith("_sl"))

    cancelled = asyncio.run(
        conn._async_cancel_remaining_bracket_legs(
            "bot_x", "MNQ", reason="position_closed",
        )
    )

    assert cancelled == 1
    conn._async_cancel_protective_leg.assert_awaited_once_with(
        "bot_x_sl",
        symbol="MNQ",
        reason="position_closed",
        require_flat=False,
        cancel_path="cancel_remaining_bracket",
    )


def test_entry_still_open_true_when_flat_with_working_protective_legs():
    conn = _bare_connector()
    conn._client.list_positions = AsyncMock(return_value=[])
    conn._async_get_order_snapshot = AsyncMock(
        return_value=MagicMock(status="filled", avg_fill_price=30200.0),
    )
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._async_is_order_working = AsyncMock(
        side_effect=lambda oid: str(oid).endswith("_sl") or str(oid).endswith("_tp"),
    )
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._orders = {"bot_x": {"symbol": "MNQ", "stop_loss": 30190.0, "take_profit": 30250.0}}
    conn._async_query_broker_protection = AsyncMock(return_value=(True, True))

    still_open = asyncio.run(conn._async_is_entry_still_open("bot_x", "MNQ"))

    assert still_open is True
    conn._async_cancel_protective_leg.assert_not_called()


def test_entry_still_open_false_when_flat_and_legs_terminal():
    conn = _bare_connector()
    conn._client.list_positions = AsyncMock(return_value=[])
    conn._async_get_order_snapshot = AsyncMock(
        return_value=MagicMock(status="filled", avg_fill_price=30200.0),
    )
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._async_is_order_working = AsyncMock(return_value=False)
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._orders = {"bot_x": {"symbol": "MNQ"}}
    conn._async_query_broker_protection = AsyncMock(return_value=(False, False))

    still_open = asyncio.run(conn._async_is_entry_still_open("bot_x", "MNQ"))

    assert still_open is False


def test_partial_bracket_cancels_sl_when_tp_submit_fails():
    conn = _bare_connector()
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

    assert "protective_sl_order_id" not in out
    conn._async_cancel_protective_leg.assert_awaited_once_with(
        "bot_9e77d6f519_sl",
        symbol="MNQ",
        reason="partial_bracket_tp_failed",
        require_flat=False,
        cancel_path="partial_bracket_cleanup",
    )


def test_cancel_remaining_bracket_legs_skips_during_entry_grace():
    conn = _bare_connector()
    conn._order_within_entry_grace = MagicMock(return_value=True)
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_order_working = AsyncMock(return_value=True)

    cancelled = asyncio.run(
        conn._async_cancel_remaining_bracket_legs(
            "bot_x", "MNQ", reason="flat_with_working_legs",
        )
    )

    assert cancelled == 0
    conn._async_cancel_protective_leg.assert_not_called()


def test_cancel_remaining_bracket_legs_skips_when_entry_still_open():
    conn = _bare_connector()
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._async_is_entry_still_open = AsyncMock(return_value=True)
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_order_working = AsyncMock(return_value=True)

    cancelled = asyncio.run(
        conn._async_cancel_remaining_bracket_legs(
            "bot_x", "MNQ", reason="bracket_exit_inferred",
        )
    )

    assert cancelled == 0
    conn._async_cancel_protective_leg.assert_not_called()


def test_confirm_bracket_exit_keeps_working_legs_when_broker_flat():
    conn = _bare_connector()
    conn._using_simulator_route = True
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._async_is_entry_still_open = AsyncMock(return_value=True)
    conn._confirmed_exit_fills = {}
    conn._orders = {"bot_dd48bacf89": {"symbol": "MNQ"}}
    conn._order_lock = MagicMock()
    conn._bracket_leg_ids = MagicMock(
        return_value=(["bot_dd48bacf89_sl"], ["bot_dd48bacf89_tp"]),
    )
    conn._async_is_leg_filled = AsyncMock(return_value=False)
    conn._async_cancel_remaining_bracket_legs = AsyncMock(return_value=0)
    conn._async_diagnose_bracket_legs = AsyncMock(
        return_value={"inferred_flat": False, "legs_working": True},
    )

    result = asyncio.run(
        conn._async_confirm_bracket_exit_fill(
            "bot_dd48bacf89",
            "MNQ",
            30238.50,
            30261.50,
        )
    )

    assert result is None
    conn._async_cancel_remaining_bracket_legs.assert_not_called()


def test_confirm_bracket_exit_cancels_remaining_leg_after_sl_fill():
    conn = _bare_connector()
    conn._order_within_entry_grace = MagicMock(return_value=False)
    conn._async_is_entry_still_open = AsyncMock(return_value=False)
    conn._confirmed_exit_fills = {}
    conn._orders = {"bot_x": {"symbol": "MNQ"}}
    conn._order_lock = MagicMock()
    conn._bracket_leg_ids = MagicMock(return_value=(["bot_x_sl"], ["bot_x_tp"]))
    conn._async_is_leg_filled = AsyncMock(
        side_effect=lambda oid: str(oid).endswith("_sl"),
    )
    conn._async_get_order_snapshot = AsyncMock(
        return_value=MagicMock(status="filled", avg_fill_price=30200.0),
    )
    conn._async_cancel_remaining_bracket_legs = AsyncMock(return_value=1)
    conn._get_account_id = MagicMock(return_value="TEST-ACCT")
    conn.get_trade_route = MagicMock(return_value="simulator")

    result = asyncio.run(
        conn._async_confirm_bracket_exit_fill(
            "bot_x", "MNQ", 30190.0, 30250.0,
        )
    )

    assert result is not None
    assert result["confirmed"] is True
    assert result["leg"] == "SL"
    conn._async_cancel_remaining_bracket_legs.assert_awaited_once_with(
        "bot_x", "MNQ", reason="bracket_exit_confirmed",
    )


def test_log_protection_verified_only_once():
    conn = _bare_connector()
    conn._order_lock = threading.Lock()
    conn._orders = {
        "bot_x": {
            "stop_loss": 30100.0,
            "take_profit": 30200.0,
            "bracket_mode": "protective_fallback",
        }
    }

    with patch("src.broker.rithmic_connector.bot_logger") as mock_logger:
        conn._log_protection_verified("bot_x", 30100.0, 30200.0)
        conn._log_protection_verified("bot_x", 30100.0, 30200.0)

    assert mock_logger.info.call_count == 1
    assert conn._orders["bot_x"]["protection_verified_logged"] is True


def test_ensure_protective_orders_cancels_all_when_flat():
    conn = _bare_connector()
    conn._async_broker_has_open_position = AsyncMock(return_value=False)
    conn._async_cancel_all_working_bot_orders_for_symbol = AsyncMock(return_value=2)

    result = asyncio.run(
        conn._async_ensure_protective_orders(
            "bot_entry", "MNQ", "buy", 1, 30100.0, 30200.0,
        )
    )

    assert result["bracket_mode"] == "position_flat"
    conn._async_cancel_all_working_bot_orders_for_symbol.assert_awaited_once_with("MNQ")
