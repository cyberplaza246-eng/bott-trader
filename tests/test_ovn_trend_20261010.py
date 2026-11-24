"""2026-10-10: REVAMP overnight = trend-sleeve entries only (box ovn_fix.py variant 'trend only').

Box rule: for non-add-on revamp candidates whose ENTRY time is in the overnight session (outside
9:30-16:00 ET Mon-Fri, same rule as session_limits.session_of), skip sleeves other than 'trend'.
First trades and extra trades both come from the same revamp signal path. Add-ons (ONL, E1) are exempt.
REVAMP_OVERNIGHT_SLEEVES=all restores the old behaviour. The Ensemble bot is untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import start_live_rithmic as slr

UTC = timezone.utc
MON_DAY = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)      # Mon 10:00 ET
MON_NIGHT = datetime(2026, 10, 13, 0, 30, tzinfo=UTC)    # Mon 20:30 ET
SUN_6PM = datetime(2026, 10, 11, 22, 30, tzinfo=UTC)     # Sun 18:30 ET


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("REVAMP_OVERNIGHT_SLEEVES", raising=False)


@pytest.mark.parametrize("now,sleeve,blocked", [
    (MON_DAY, "breakout", False), (MON_DAY, "mean_reversion", False), (MON_DAY, "trend", False),
    (MON_NIGHT, "breakout", True), (MON_NIGHT, "mean_reversion", True), (MON_NIGHT, "trend", False),
    (SUN_6PM, "breakout", True), (SUN_6PM, "trend", False),
    (datetime(2026, 10, 12, 13, 29, tzinfo=UTC), "breakout", True),    # 9:29 ET = overnight
    (datetime(2026, 10, 12, 13, 30, tzinfo=UTC), "breakout", False),   # 9:30 ET = day
    (datetime(2026, 10, 12, 20, 0, tzinfo=UTC), "breakout", True),     # 16:00 ET = overnight
])
def test_block_rule(now, sleeve, blocked):
    assert slr.revamp_overnight_sleeves() == "trend"                   # default
    why = slr.revamp_overnight_block(sleeve, now)
    assert bool(why) == blocked
    if blocked:
        assert "trend trades only" in why and ("breakout" in why or "mean-reversion" in why)


def test_all_restores_old(monkeypatch):
    monkeypatch.setenv("REVAMP_OVERNIGHT_SLEEVES", "all")
    assert slr.revamp_overnight_block("breakout", MON_NIGHT) is None
    assert slr.revamp_overnight_block("mean_reversion", MON_NIGHT) is None


def _hit(sleeve):
    return dict(direction="long", entry=30000.0, sl=29960.0, tp=30080.0, sleeve=sleeve, trailing=False,
                atr=20.0, adx=30.0, ensemble_signal="BUY", regime="trending")


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", True)
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t._session_ledger = None
    t._effective_open_count = lambda: len(t.positions)
    t._is_trading_session_allowed = lambda s: True
    return t


def _df(n=230):
    ts = pd.date_range("2026-10-12 00:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"datetime": ts, "open": 30000.0, "high": 30010.0, "low": 29990.0, "close": 30000.0, "volume": 100})


def _at(monkeypatch, when):
    real = slr._session_limits.session_of
    monkeypatch.setattr(slr._session_limits, "session_of", lambda ts=None: real(when))


@pytest.mark.parametrize("sleeve", ["breakout", "mean_reversion"])
def test_overnight_skips_non_trend_signal(trader, monkeypatch, sleeve):
    _at(monkeypatch, MON_NIGHT)
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit(sleeve))
    assert trader._check_revamp_research_signal("MNQ", _df()) is None
    line = trader._overnight_sleeve_status_line()
    lab = "breakout" if sleeve == "breakout" else "mean-reversion"
    assert line == f"Overnight: trend trades only ({lab} skipped)"


def test_overnight_trend_still_trades(trader, monkeypatch):
    _at(monkeypatch, MON_NIGHT)
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit("trend"))
    sig = trader._check_revamp_research_signal("MNQ", _df())
    assert sig and sig["exit_sleeve"] == "trend"
    assert trader._overnight_sleeve_status_line().startswith("Overnight: trend trades only")


@pytest.mark.parametrize("sleeve", ["breakout", "mean_reversion", "trend"])
def test_day_session_unchanged(trader, monkeypatch, sleeve):
    _at(monkeypatch, MON_DAY)
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit(sleeve))
    assert trader._check_revamp_research_signal("MNQ", _df())["exit_sleeve"] == sleeve
    assert trader._overnight_sleeve_status_line() is None


def test_all_env_takes_breakout_overnight(trader, monkeypatch):
    monkeypatch.setenv("REVAMP_OVERNIGHT_SLEEVES", "all")
    _at(monkeypatch, MON_NIGHT)
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit("breakout"))
    assert trader._check_revamp_research_signal("MNQ", _df())["exit_sleeve"] == "breakout"
    assert trader._overnight_sleeve_status_line() is None


def test_extra_trade_overnight_also_trend_only(trader, monkeypatch):
    """A 2nd trade comes from the same check_entry_signal path, so a breakout extra is skipped overnight."""
    _at(monkeypatch, MON_NIGHT)
    trader.positions["t1"] = slr.Position("t1", "MNQ", "long", 29900.0, 1, 29850.0, 30000.0, MON_NIGHT,
                                          exit_sleeve="trend")
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit("breakout"))
    assert trader.check_entry_signal("MNQ", _df()) is None
    monkeypatch.setattr(slr, "classify_last_sleeve", lambda df: _hit("trend"))
    assert trader.check_entry_signal("MNQ", _df())["exit_sleeve"] == "trend"


def test_ensemble_bot_unchanged(trader, monkeypatch):
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", False)
    _at(monkeypatch, MON_NIGHT)
    assert trader._overnight_sleeve_status_line() is None
    called = MagicMock(return_value=None)
    monkeypatch.setattr(trader, "_check_revamp_research_signal", called)
    trader.ensemble = None
    trader._check_simple_breakout = lambda s, d: {"sentinel": True}
    assert trader.check_entry_signal("MNQ", _df()) == {"sentinel": True}
    called.assert_not_called()


def test_e1_addon_not_affected_by_overnight_gate(trader, monkeypatch):
    from tests import test_addons_20261010 as T
    c = T.EXP["e1"][0]
    b, now = T._feat(c["bar_start_et"])
    trader._addon_frame = lambda symbol: b
    sig, _ = trader.check_addon_signals("MNQ", now=now.to_pydatetime())
    assert sig and sig["addon"] == "e1" and sig["exit_sleeve"] == "addon"
