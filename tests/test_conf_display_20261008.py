"""2026-10-08: show each trade's confidence (percent) on entry and status lines."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import start_live_rithmic as slr
from src.core.conf_display import conf_pct, trade_conf_label, trade_status_line, extra_trade_line

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _legacy_rules(monkeypatch):
    """These tests cover the OLD rules (confidence extra-trade gate, calendar-day loss limit).
    Since 2026-10-09 both bots default to same_dir_profit + session limits (see
    tests/test_extra_rule_20261009.py and tests/test_session_limits_20261009.py)."""
    monkeypatch.setenv("EXTRA_TRADE_MODE", "conf")
    monkeypatch.setenv("SESSION_LIMITS", "0")


def _pos(**kw):
    base = dict(order_id="bot_1", symbol="MNQ", direction="short", entry_price=18500.25, size=1,
                sl=18520.0, tp=18460.0, entry_time=datetime(2026, 10, 8, 15, 0, tzinfo=UTC), confidence=0.74)
    base.update(kw)
    return slr.Position(**base)


def test_conf_pct_and_labels():
    assert conf_pct(0.736) == "74%" and conf_pct(0.7) == "70%"
    assert conf_pct(0) == "n/a" and conf_pct(None) == "n/a" and conf_pct(float("nan")) == "n/a"
    assert trade_conf_label(_pos()) == "conf 74%"
    assert trade_conf_label(_pos(reentry_gen=1, reentry_parent="bot_0")) == "conf 74% re-entry"
    assert trade_conf_label(_pos(regime="broker_adopt", confidence=0.9)) == "conf n/a"
    assert trade_conf_label(_pos(confidence=0.0)) == "conf n/a"   # manual / unknown


def test_trade_status_line():
    assert trade_status_line(1, _pos()) == "Trade 1 SHORT conf 74% | 1 MNQ @ 18500.25 SL 18520.00 TP 18460.00"
    assert trade_status_line(2, _pos(direction="long", tp=0.0, regime="broker_adopt")) == \
        "Trade 2 LONG conf n/a | 1 MNQ @ 18500.25 SL 18520.00"


def test_extra_trade_line():
    assert extra_trade_line(0, 3, 0.70, 0.66) is None
    assert extra_trade_line(3, 3, 0.70, 0.66) is None
    assert extra_trade_line(1, 3, 0.70, 0.66) == "2nd trade: conf 66%, needs 70%"
    assert extra_trade_line(2, 3, 0.70, 0.74) == "3rd trade: conf 74%, needs 70% - OK"
    assert extra_trade_line(1, 3, 0.70, None) == "2nd trade: no setup, needs 70%"
    assert extra_trade_line(1, 3, 0.70, None, note="cooldown 4 bars") == "2nd trade: cooldown 4 bars, needs 70%"


@pytest.fixture
def trader():
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.broker.place_order.return_value = {"ticket": "bot_e1", "entry_price": 18500.25}
    t._broker_can_update_stop = lambda r: True
    t._write_board_status = lambda **k: None
    t._contracts_for_entry = lambda sig: (1, float(sig.get("confidence", 0)), "test")
    t._effective_open_count = lambda: len(t.positions)
    t.max_risk_per_trade_usd = 500.0
    t._size_cfg = dict(t._size_cfg, extra_trade_min_conf=0.70)
    return t


def _sig(**kw):
    s = dict(symbol="MNQ", direction="short", entry=18500.25, sl=18520.0, tp=18460.0, atr=10.0,
             confidence=0.74, exit_sleeve="breakout", trailing_enabled=False)
    s.update(kw)
    return s


def test_entry_line_shows_conf_and_stores_it(trader, capsys):
    assert trader.place_order(_sig())
    out = capsys.readouterr().out
    assert "ENTRY SHORT 1 MNQ @ 18500.25 conf 74%" in out
    assert trader.positions["bot_e1"].confidence == pytest.approx(0.74)


def test_reentry_entry_line_labelled(trader, capsys):
    assert trader.place_order(_sig(confidence=0.75, reentry_gen=1, reentry_parent="bot_0"))
    assert "conf 75% re-entry" in capsys.readouterr().out


def test_extra_trade_status_uses_config(trader):
    trader.positions["bot_e1"] = _pos(order_id="bot_e1")
    trader.cooldown = 0
    assert trader._extra_trade_status(True, {"confidence": 0.66}) == "2nd trade: conf 66%, needs 70%"
    assert trader._extra_trade_status(True, None) == "2nd trade: no setup, needs 70%"
    trader.cooldown = 3
    assert trader._extra_trade_status(False, None) == "2nd trade: cooldown 3 bars, needs 70%"
    trader.positions.clear()
    assert trader._extra_trade_status(True, {"confidence": 0.8}) is None
