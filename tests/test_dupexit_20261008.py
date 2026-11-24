"""2026-10-08: a closed trade must book its P&L exactly once.

Live bug: revamp model_signals={'revamp_sleeve': 'breakout', ...} made the
adaptive learner crash ('str' object has no attribute 'get') inside
process_exit BEFORE the position was removed, so every 5s position sync
booked the same -$217 stop again (Daily -217, -434, -651 ...).
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import start_live_rithmic as slr

UTC = timezone.utc


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())   # keep live logs/trades.log clean
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t._write_board_status = lambda **k: None
    t._cancel_orders_for_closed_position = lambda *a, **k: 0
    t.daily_pnl = 0.0
    return t


def _short(t, oid="bot_16b412a862"):
    p = slr.Position(order_id=oid, symbol="MNQ", direction="short", entry_price=30976.25, size=1,
                     sl=31083.25, tp=30762.25, entry_time=datetime(2026, 10, 9, 0, 6, tzinfo=UTC),
                     exit_sleeve="breakout", confidence=0.70,
                     model_signals={"revamp_sleeve": "breakout", "adx": 28.4})
    t.positions[oid] = p
    return p


def test_exit_booked_once_even_if_learner_crashes(trader):
    trader.ensemble = MagicMock()
    trader.ensemble.record_trade_result.side_effect = AttributeError("'str' object has no attribute 'get'")
    _short(trader)
    trader.process_exit("bot_16b412a862", "STOP_LOSS", 31083.50)
    first = trader.daily_pnl
    assert first < 0 and "bot_16b412a862" not in trader.positions
    # next syncs see the same confirmed SL fill again
    trader.process_exit("bot_16b412a862", "STOP_LOSS", 31083.50)
    trader.process_exit("bot_16b412a862", "STOP_LOSS", 31083.50)
    assert trader.daily_pnl == first
    assert trader.daily_trades == 1 and len(trader.trades) == 1


def test_same_trade_readded_is_not_booked_twice(trader):
    p = _short(trader)
    trader.process_exit(p.order_id, "STOP_LOSS", 31083.50)
    first = trader.daily_pnl
    trader.positions[p.order_id] = p          # stale re-add of the same trade
    trader.process_exit(p.order_id, "STOP_LOSS", 31083.50)
    assert trader.daily_pnl == first and p.order_id not in trader.positions


def test_learner_payload_drops_non_dict_model_signals(trader):
    trader.ensemble = MagicMock()
    _short(trader)
    trader.process_exit("bot_16b412a862", "STOP_LOSS", 31083.50)
    payload = trader.ensemble.record_trade_result.call_args[0][0]
    assert payload["model_signals"] == {}


def test_adaptive_learner_tolerates_str_model_signals(monkeypatch, tmp_path):
    from src.ai import adaptive_learner as al
    monkeypatch.setattr(al, "LEARNING_DB_PATH", str(tmp_path / "learn.json"))
    learner = al.AdaptiveLearner()
    monkeypatch.setattr(learner, "_save", lambda: None)
    monkeypatch.setattr(learner, "_auto_learn_risk_params", lambda: None)
    learner.record_trade({"pair": "MNQ", "signal": "SELL", "profit_loss": -217.0,
                          "exit_type": "STOP_LOSS",
                          "model_signals": {"revamp_sleeve": "breakout", "adx": 28.4,
                                            "ml": {"signal": "SELL"}}})
    assert "revamp_sleeve" not in learner.model_accuracy
    assert learner.model_accuracy["ml"]["total"] == 1
