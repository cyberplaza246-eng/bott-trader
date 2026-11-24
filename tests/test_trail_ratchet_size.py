"""2026-10-07: trail jump-back fix, ratchet-only broker SL, pure-confidence sizing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
import threading
import pytest

from src.broker.rithmic_connector import RithmicConnector as RC
from src.core.size_scale import compute_sized_contracts, load_size_scale_config


def _conn(order):
    c = RC.__new__(RC)
    c._connected = True
    c._order_lock = threading.RLock()
    c._orders = {"E1": order}
    c._client = MagicMock()
    c._run_sync = MagicMock(return_value=True)
    return c


def test_profit_side_detection():
    assert RC._stop_on_profit_side(True, 100.0, 100.0)
    assert RC._stop_on_profit_side(True, 100.0, 105.0)
    assert not RC._stop_on_profit_side(True, 100.0, 95.0)
    assert RC._stop_on_profit_side(False, 100.0, 95.0)
    assert not RC._stop_on_profit_side(False, 100.0, 105.0)


def test_modify_refuses_profit_side_native_ticks():
    """Root cause: abs(entry-sl) ticks mirrored a +5 lock to -5 loss stop."""
    c = _conn({"symbol": "MNQ", "type": "BUY", "entry_price": 100.0,
               "stop_loss": 90.0, "take_profit": 120.0})
    assert c.modify_position("E1", sl=105.0) is False
    c._run_sync.assert_not_called()
    assert c._orders["E1"]["stop_loss"] == 90.0


def test_modify_refuses_loosen():
    c = _conn({"symbol": "MNQ", "type": "SELL", "entry_price": 100.0,
               "stop_loss": 103.0, "take_profit": 80.0})
    assert c.modify_position("E1", sl=106.0) is False


def test_update_stop_loss_profit_side_uses_absolute_replace():
    c = _conn({"symbol": "MNQ", "type": "BUY", "entry_price": 100.0,
               "stop_loss": 90.0, "take_profit": 0})
    c.modify_position = MagicMock(return_value=True)
    c._async_replace_protective_sl = MagicMock(return_value="coro")
    assert c.update_stop_loss("E1", "MNQ", "BUY", 2, 105.0) is True
    c.modify_position.assert_not_called()
    c._async_replace_protective_sl.assert_called_once()
    assert c._orders["E1"]["stop_loss"] == 105.0


def test_update_stop_loss_below_entry_still_native():
    c = _conn({"symbol": "MNQ", "type": "BUY", "entry_price": 100.0,
               "stop_loss": 90.0, "take_profit": 0})
    c.modify_position = MagicMock(return_value=True)
    assert c.update_stop_loss("E1", "MNQ", "BUY", 1, 95.0) is True
    c.modify_position.assert_called_once()


def test_update_stop_loss_never_loosens():
    c = _conn({"symbol": "MNQ", "type": "BUY", "entry_price": 100.0,
               "stop_loss": 105.0, "take_profit": 0})
    c.modify_position = MagicMock(return_value=True)
    c._async_replace_protective_sl = MagicMock()
    assert c.update_stop_loss("E1", "MNQ", "BUY", 1, 102.0) is False
    c.modify_position.assert_not_called()
    c._async_replace_protective_sl.assert_not_called()


def test_sizing_pure_confidence(monkeypatch):
    for k in ("SIZE_WIN_STREAK_MIN", "SIZE_REQUIRE_DAY_GREEN"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_size_scale_config()
    assert cfg["win_streak_min"] == 0 and cfg["require_day_green"] is False
    kw = dict(daily_pnl=-50.0, win_streak=0, win_streak_min=cfg["win_streak_min"],
              require_day_green=cfg["require_day_green"], max_contracts=4)
    assert compute_sized_contracts(0.60, **kw)[0] == 1
    assert compute_sized_contracts(0.72, **kw)[0] == 2
    assert compute_sized_contracts(0.80, **kw)[0] == 3
    assert compute_sized_contracts(0.90, **kw)[0] == 4
    assert compute_sized_contracts(0.99, **{**kw, "max_contracts": 4})[0] == 4
