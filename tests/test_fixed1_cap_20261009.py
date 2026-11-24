"""2026-10-09: REVAMP live = 1 contract per trade and skip any trade whose 1-lot stop risk > $500.

Oct 8 live log: bot_8d7578e96c / bot_ecd8563b40 sent 2 MNQ ("reason=scale conf=0.76") and
bot_1f66b62857 / bot_7982038e6b sent $467 / $716 risk with no cap line - a bot session started
before the fixed-1 bat + MAX_RISK_PER_TRADE_USD code existed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import start_live_rithmic as slr


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", True)
    monkeypatch.delenv("REVAMP_MAX_CONTRACTS", raising=False)
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.broker.place_order.return_value = {"ticket": "bot_x", "entry_price": 30945.0}
    t._broker_can_update_stop = lambda r: True
    t._write_board_status = lambda **k: None
    t.max_risk_per_trade_usd = 500.0
    # old risky sizing config (.env MAX_CONTRACTS=4, conf bands 72/78/85)
    t._size_cfg = dict(t._size_cfg, base=1, max_contracts=4, conf_2=0.72, conf_3=0.78, conf_4=0.85,
                       win_streak_min=0, require_day_green=False)
    t.daily_pnl = 0.0
    return t


def _sig(sl, conf=0.765, entry=30945.0):
    return dict(symbol="MNQ", direction="short", entry=entry, sl=sl, tp=entry - 2 * (sl - entry), atr=20.0,
                confidence=conf, exit_sleeve="breakout", trailing_enabled=False)


def test_revamp_never_sizes_above_one(trader):
    n, _, why = trader._contracts_for_entry(_sig(31045.0, conf=0.90))
    assert n == 1 and "revamp_fixed1" in why
    trader.place_order(_sig(31045.0, conf=0.90))
    assert trader.broker.place_order.call_args.kwargs["size"] == 1


def test_revamp_clamp_env(trader, monkeypatch):
    monkeypatch.setenv("REVAMP_MAX_CONTRACTS", "0")       # 0 = no extra clamp (old behaviour)
    assert trader._contracts_for_entry(_sig(31045.0, conf=0.90))[0] > 1


def test_non_revamp_sizing_untouched(trader, monkeypatch):
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", False)
    assert trader._contracts_for_entry(_sig(31045.0, conf=0.90))[0] > 1


def test_716_risk_skipped_467_allowed(trader):
    assert trader.place_order(_sig(31303.0)) is False          # 358 pts x $2 = $716 > $500
    trader.broker.place_order.assert_not_called()
    assert trader.place_order(_sig(31178.5, entry=30945.0)) is True   # 233.5 pts = $467 <= $500
    assert trader.broker.place_order.call_args.kwargs["size"] == 1
