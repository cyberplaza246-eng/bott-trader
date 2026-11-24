"""2026-10-07: NQ launcher + NQ end-to-end specs (symbol map, $20/pt, costs, $-lock x10)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import start_live_rithmic as slr
from src.broker.rithmic_connector import RithmicConnector
from src.instruments.instrument_registry import get_instrument

ROOT = Path(__file__).resolve().parents[1]


def _conn():
    c = object.__new__(RithmicConnector)
    c._resolved_contracts = {}
    c._contract_overrides = {}
    c._contracts_seeded = False
    return c


def test_specs_nq_vs_mnq():
    assert slr.SYMBOL_SPECS["NQ"] == {"point_value": 20.0, "tick_size": 0.25}
    assert slr.SYMBOL_SPECS["MNQ"] == {"point_value": 2.0, "tick_size": 0.25}
    nq = get_instrument("NQ")
    assert nq.contract_multiplier == 20.0 and nq.tick_size == 0.25 and nq.tick_value_usd == 5.0
    assert get_instrument("MNQ").contract_multiplier == 2.0


def test_nq_maps_to_front_month_on_cme(monkeypatch):
    monkeypatch.setenv("RITHMIC_CONTRACT_OVERRIDES", "NQ=NQZ6")
    monkeypatch.setenv("TRADING_SYMBOLS", "NQ")
    c = _conn()
    c._seed_contract_overrides()
    assert c._resolve_symbol("NQ") == ("NQZ6", "CME")
    assert c._reverse_resolve("NQZ6") == "NQ"
    assert c._reverse_resolve("MNQZ6") == "MNQ"   # MNQ rows never look like NQ
    assert RithmicConnector._parse_explicit_contract("NQZ6") == ("NQ", "NQZ6")


def test_listed_orders_do_not_cross_mnq_and_nq(monkeypatch):
    monkeypatch.setenv("RITHMIC_CONTRACT_OVERRIDES", "NQ=NQZ6")
    c = _conn()
    c._seed_contract_overrides()
    m = c._listed_order_matches_symbol
    assert m({"symbol": "NQZ6"}, "NQ") is True
    assert m({"symbol": "MNQZ6"}, "NQ") is False
    assert m({"symbol": "NQZ6"}, "MNQ") is False


def test_round_trip_cost_nq_per_contract_mnq_unchanged():
    assert slr.round_trip_cost("NQ", 1, 20.0, 0.25) == pytest.approx(slr.COMMISSION_RT_NQ + 5.0)
    assert slr.round_trip_cost("NQ", 2, 20.0, 0.25) == pytest.approx(2 * (slr.COMMISSION_RT_NQ + 5.0))
    # legacy MNQ: flat $2 + 1 tick, regardless of size
    assert slr.round_trip_cost("MNQ", 1, 2.0, 0.25) == pytest.approx(2.50)
    assert slr.round_trip_cost("MNQ", 3, 2.0, 0.25) == pytest.approx(2.50)


def _trader(sym):
    t = slr.LiveRithmicTrader(symbol=sym, symbols=[sym], paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.paper_orders = True
    return t


@pytest.mark.parametrize("sym,size,exit_px,want", [
    ("NQ", 1, 20010.0, 10 * 20.0 - (4.50 + 5.0)),
    ("MNQ", 1, 20050.0, 50 * 2.0 - 2.50),
    ("MNQ", 2, 20050.0, 50 * 2.0 * 2 - 2.50),
])
def test_process_exit_pnl_uses_symbol_point_value(monkeypatch, sym, size, exit_px, want):
    monkeypatch.setattr(slr, "COMMISSION_RT_NQ", 4.50)
    t = _trader(sym)
    monkeypatch.setattr(t, "_cancel_orders_for_closed_position", lambda *a, **k: None)
    monkeypatch.setattr(t, "_feed_learner_on_exit", lambda *a, **k: None)
    t.positions["x"] = slr.Position(order_id="x", symbol=sym, direction="long", entry_price=20000.0,
                                    size=size, sl=19900.0, tp=20200.0,
                                    entry_time=datetime.now(timezone.utc))
    t.process_exit("x", "TAKE_PROFIT", exit_px)
    assert t.trades[-1]["pnl"] == pytest.approx(want)


def test_nq_usd_lock_scaled_x10(monkeypatch):
    monkeypatch.setattr(slr, "LOCK_MODE", "usd")
    monkeypatch.setattr(slr, "LOCK_ACTIVATE_USD", 1000.0)
    monkeypatch.setattr(slr, "LOCK_KEEP_USD", 200.0)
    monkeypatch.setattr(slr, "LOCK_TRAIL_START_USD", 1500.0)
    monkeypatch.setattr(slr, "LOCK_TRAIL_KEEP_PCT", 0.5)
    t = _trader("NQ")
    now = datetime.now(timezone.utc)
    pos = slr.Position(order_id="n", symbol="NQ", direction="long", entry_price=20000.0,
                       size=1, sl=19900.0, tp=20300.0, entry_time=now)
    t.positions["n"] = pos
    # +49.75 pts = $995 -> no move
    t.get_price = lambda s: {"last": 20040.0}
    t.update_dollar_profit_locks("NQ", pd.DataFrame([{"datetime": now + timedelta(minutes=1),
                                  "high": 20049.75, "low": 20020.0, "close": 20040.0, "atr": 20.0}]))
    assert pos.sl == 19900.0
    # +50 pts = $1000 -> lock +$200 = +10 pts
    t.update_dollar_profit_locks("NQ", pd.DataFrame([{"datetime": now + timedelta(minutes=2),
                                  "high": 20050.0, "low": 20030.0, "close": 20040.0, "atr": 20.0}]))
    assert pos.sl == pytest.approx(20010.0)
    # peak +100 pts = $2000 -> keep 50% = $1000 = +50 pts
    t.get_price = lambda s: {"last": 20090.0}
    t.update_dollar_profit_locks("NQ", pd.DataFrame([{"datetime": now + timedelta(minutes=3),
                                  "high": 20100.0, "low": 20060.0, "close": 20090.0, "atr": 20.0}]))
    assert pos.sl == pytest.approx(20050.0)


def test_nq_bat_settings_and_guard():
    bat = (ROOT / "start_nq_live_revamp.bat").read_text(encoding="ascii")
    assert '--symbols NQ --yes' in bat
    for kv in ('TRADING_SYMBOLS=NQ', 'RITHMIC_CONTRACT_OVERRIDES=NQ=NQZ6', 'MAX_CONTRACTS=1',
               'MAX_CONCURRENT_TRADES=1', 'LOCK_MODE=usd', 'LOCK_ACTIVATE_USD=1000',
               'LOCK_KEEP_USD=200', 'LOCK_TRAIL_START_USD=1500', 'LOCK_TRAIL_KEEP_PCT=0.5',
               'REVAMP_EXITS=1', 'TRADING_MODE=live', 'DAILY_LOSS_LIMIT_PERCENT=1.0'):
        assert f'set "{kv}"' in bat, kv
    # guard runs before python is launched
    assert bat.index("start_live_rithmic' }") < bat.index('"%~dp0venv\\Scripts\\python.exe" -u')
    assert "exit /b 1" in bat
    mnq = (ROOT / "start_mnq_live_revamp.bat").read_text(encoding="latin-1")
    assert "--symbols MNQ --yes" in mnq and "--symbols NQ" not in mnq
