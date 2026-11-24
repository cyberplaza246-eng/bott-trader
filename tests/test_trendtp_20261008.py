"""2026-10-08: REVAMP trend sleeve gets a fixed TP at 1.5x its stop distance (no trail).

Mirrors box backtest next_ideas.py rule 2: tp = ref + dir * 1.5 * |ref - sl|.
REVAMP_TREND_TP_R=0 restores the old trail-only trend trade. Breakout / MR unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import start_live_rithmic as slr
from src.research_ensemble import live_signals as ls
from src.research_ensemble import revamp_exits as rx


def _df(n=260, sl_off=40.0, direction=1):
    idx = pd.date_range("2026-10-06 13:30", periods=n, freq="5min", tz="UTC")
    close = 31000 + np.arange(n) * 2.0 * direction
    df = pd.DataFrame({"datetime": idx, "open": close - direction, "high": close + 3, "low": close - 3,
                       "close": close, "volume": 500.0})
    return df


class _FakeStrat:
    """Routes every bar to the trend sleeve (adx_trend_min=0, no breakout)."""
    def __init__(self, direction=1, sl_off=40.0):
        self.direction, self.sl_off = direction, sl_off
        self.adx_trend_min, self.adx_range_max = 0.0, -1.0
        self.breakout_in_range = False
        self.revamp_exits, self.breakout_target_r, self.mr_target = True, 2.0, "bb_far"
        self.rth_only = False
        self._breakout = SimpleNamespace(generate_signals=lambda d: SimpleNamespace(
            entries=pd.Series(0, index=d.index)))

    def generate_signals(self, d):
        c = d["close"]
        return SimpleNamespace(entries=pd.Series(self.direction, index=d.index),
                               stop_price=c - self.direction * self.sl_off,
                               target_price=pd.Series(np.nan, index=d.index))


@pytest.mark.parametrize("direction", [1, -1])
def test_trend_hit_gets_1p5r_target(monkeypatch, direction):
    monkeypatch.setenv("REVAMP_EXITS", "1")
    monkeypatch.delenv("REVAMP_TREND_TP_R", raising=False)
    hit = ls.classify_last_sleeve(_df(direction=direction), _FakeStrat(direction, 40.0))
    assert hit["sleeve"] == "trend" and hit["trailing"] is False
    assert hit["tp"] == pytest.approx(hit["entry"] + direction * 1.5 * abs(hit["entry"] - hit["sl"]))
    assert hit["tp"] == pytest.approx(hit["entry"] + direction * 60.0)


def test_trend_target_env_and_off_switch(monkeypatch):
    monkeypatch.setenv("REVAMP_EXITS", "1")
    monkeypatch.setenv("REVAMP_TREND_TP_R", "2")
    hit = ls.classify_last_sleeve(_df(), _FakeStrat(1, 40.0))
    assert hit["tp"] == pytest.approx(hit["entry"] + 80.0)
    for off in ("0", "off", "none"):
        monkeypatch.setenv("REVAMP_TREND_TP_R", off)
        assert rx.revamp_trend_target_r() is None
        hit = ls.classify_last_sleeve(_df(), _FakeStrat(1, 40.0))
        assert hit["tp"] is None and hit["trailing"] is True
    monkeypatch.delenv("REVAMP_TREND_TP_R")
    assert rx.revamp_trend_target_r() == 1.5


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t._write_board_status = lambda **k: None
    return t


def _hit(tp, trailing):
    return {"direction": "short", "entry": 30976.25, "sl": 31083.25, "tp": tp, "sleeve": "trend",
            "trailing": trailing, "atr": 20.5, "adx": 31.0, "ensemble_signal": "SELL", "regime": "trending"}


def test_live_signal_trend_tp_no_trail(trader, monkeypatch):
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit(30976.25 - 1.5 * 107.0, False))
    sig = trader._check_revamp_research_signal("MNQ", _df(230))
    assert sig["exit_sleeve"] == "trend"
    assert sig["tp"] == pytest.approx(30815.75)          # 30976.25 - 160.5, on a 0.25 tick
    assert sig["trailing_enabled"] is False


def test_live_signal_trend_trail_only_when_off(trader, monkeypatch):
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit(None, True))
    sig = trader._check_revamp_research_signal("MNQ", _df(230))
    assert sig["tp"] == 0.0 and sig["trailing_enabled"] is True


def test_broker_gets_trend_tp_limit(trader, monkeypatch):
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit(30976.25 - 1.5 * 107.0, False))
    sig = trader._check_revamp_research_signal("MNQ", _df(230))
    trader.broker.place_order.return_value = {"ticket": "bot_trend1", "entry_price": 30976.25}
    trader.place_order(sig)
    kw = trader.broker.place_order.call_args.kwargs
    assert kw["take_profit"] == pytest.approx(30815.75) and kw["stop_loss"] == pytest.approx(31083.25)
    pos = trader.positions["bot_trend1"]
    assert pos.tp == pytest.approx(30815.75) and pos.trailing_enabled is False and pos.exit_sleeve == "trend"
