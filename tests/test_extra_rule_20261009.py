"""2026-10-09: 2nd/3rd revamp trade = same direction as open trade(s) AND they are all in profit.

Mirrors box backtest second_trade2.py variant "same dir + open > 0.0R" (222 extras), which replaced
the 70% confidence check. EXTRA_TRADE_MODE=conf restores the old EXTRA_TRADE_MIN_CONF rule.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import start_live_rithmic as slr
from src.core import extra_rule as xr

UTC = timezone.utc
PV = 2.0  # MNQ $/pt


def _pos(oid="bot_1", direction="long", entry=31000.0, size=1, **kw):
    base = dict(order_id=oid, symbol="MNQ", direction=direction, entry_price=entry, size=size,
                sl=entry - 50 if direction == "long" else entry + 50,
                tp=entry + 100 if direction == "long" else entry - 100,
                entry_time=datetime(2026, 10, 9, 14, 0, tzinfo=UTC), confidence=0.64)
    base.update(kw)
    return slr.Position(**base)


def _backtest_rule(new_dir, opn, price):
    """Literal copy of second_trade2.py: sd(r, o) and all(state(q, T)[0] > 0 for q in o) (2nd trade)."""
    d = 1 if new_dir == "long" else -1
    sd = all((1 if q.direction == "long" else -1) == d for q in opn)
    ur = [((1 if q.direction == "long" else -1) * (price - q.entry_price)) / abs(q.entry_price - q.sl) for q in opn]
    return sd and all(u > 0 for u in ur)


# -- pure rule ------------------------------------------------------------------

def test_mode_defaults_and_env(monkeypatch):
    monkeypatch.delenv("EXTRA_TRADE_MODE", raising=False)
    assert xr.extra_trade_mode(True) == xr.SAME_DIR_PROFIT     # revamp bot
    assert xr.extra_trade_mode(False) == xr.SAME_DIR_PROFIT    # original Ensemble bot too (2026-10-09)
    monkeypatch.setenv("EXTRA_TRADE_MODE", "conf")
    assert xr.extra_trade_mode(True) == xr.CONF
    monkeypatch.setenv("EXTRA_TRADE_MODE", "same_dir_profit")
    assert xr.extra_trade_mode(False) == xr.SAME_DIR_PROFIT


@pytest.mark.parametrize("new_dir,opn,price", [
    ("long", [_pos()], 31000.25),                                   # in profit 1 tick, same dir -> allow
    ("long", [_pos()], 31000.0),                                    # flat -> block (needs > 0)
    ("long", [_pos()], 30980.0),                                    # losing -> block
    ("short", [_pos()], 31020.0),                                   # opposite -> block
    ("short", [_pos(direction="short")], 30990.0),                  # short in profit -> allow
])
def test_2nd_trade_matches_backtest_rule(new_dir, opn, price):
    allowed = xr.check_same_dir_profit(opn, len(opn), new_dir, price, PV) is None
    assert allowed == _backtest_rule(new_dir, opn, price)


def test_reasons_are_plain():
    assert xr.check_same_dir_profit([_pos()], 1, "long", 30980.0, PV) == \
        "trade 1 not in profit (down $40), needs trade 1 in profit + same direction"
    assert "against open trade 1 (LONG) - only LONG allowed" in xr.check_same_dir_profit([_pos()], 1, "short", 31050.0, PV)
    assert "not tracking" in xr.check_same_dir_profit([_pos()], 2, "long", 31050.0, PV)
    assert xr.check_same_dir_profit([], 0, "long", 1.0, PV) is None


def test_status_lines():
    assert xr.status_line([_pos()], 1, 3, 30980.0, PV) == \
        "2nd trade: needs trade 1 in profit + same direction (now: down $40)"
    assert xr.status_line([_pos()], 1, 3, 31012.5, PV) == "2nd trade: allowed, LONG setups only (now: up $25)"
    assert xr.status_line([_pos()], 1, 3, 31012.5, PV, {"direction": "long"}) == \
        "2nd trade: allowed, LONG setups only (now: up $25) - OK"
    assert xr.status_line([_pos()], 1, 3, 31012.5, PV, {"direction": "short"}).endswith("- SHORT setup skipped")
    two = [_pos(), _pos("bot_2", entry=31020.0)]
    assert xr.status_line([_pos()], 1, 3, 30980.0, PV, note="cooldown 3 bars") == \
        "2nd trade: cooldown 3 bars; needs trade 1 in profit + same direction (now: down $40)"
    assert xr.status_line([_pos()], 0, 3, 31000.0, PV) is None
    assert xr.status_line(two + [_pos("bot_3")], 3, 3, 31000.0, PV) is None


# -- live trader wiring -----------------------------------------------------------

@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t._write_board_status = lambda **k: None
    t._effective_open_count = lambda: len(t.positions)
    t._size_cfg = dict(t._size_cfg, extra_trade_min_conf=0.70)
    t.cooldown = 0
    return t


def test_trader_same_dir_profit_ignores_conf(trader, monkeypatch):
    monkeypatch.setenv("EXTRA_TRADE_MODE", "same_dir_profit")
    assert trader._extra_point_value("MNQ") == PV
    assert trader._extra_trade_block("long", 31050.0, 0.62) is None          # no open trade
    trader.positions["bot_1"] = _pos()
    assert trader._extra_trade_block("long", 31010.0, 0.62) is None          # 62% but trade 1 up -> allowed
    assert "not in profit" in trader._extra_trade_block("long", 30990.0, 0.90)  # 90% but trade 1 down -> blocked
    assert "against open trade 1" in trader._extra_trade_block("short", 31010.0, 0.90)
    assert trader._extra_trade_status(True, None, 30980.0) == \
        "2nd trade: needs trade 1 in profit + same direction (now: down $40)"
    assert trader._extra_trade_status(True, None, 31012.5) == "2nd trade: allowed, LONG setups only (now: up $25)"


def test_trader_conf_mode_restores_old_rule(trader, monkeypatch):
    monkeypatch.setenv("EXTRA_TRADE_MODE", "conf")
    trader.positions["bot_1"] = _pos()
    assert trader._extra_trade_block("short", 30990.0, 0.66) == "conf 66%, needs 70%"
    assert trader._extra_trade_block("short", 30990.0, 0.74) is None
    assert trader._extra_trade_status(True, {"confidence": 0.66}, 31012.5) == "2nd trade: conf 66%, needs 70%"


def test_reentry_uses_same_rule(trader, monkeypatch):
    monkeypatch.setenv("EXTRA_TRADE_MODE", "same_dir_profit")
    trader._is_trading_session_allowed = lambda s: True
    trader._symbol_in_consec_loss_pause = lambda s: False
    trader.daily_pnl = 0.0
    trader.positions["bot_1"] = _pos()
    arm = SimpleNamespace(symbol="MNQ", direction="long", trigger=31005.0, confidence=0.60, side="BUY")
    assert trader._reentry_block_reason(arm, 31005.0) is None
    assert "not in profit" in trader._reentry_block_reason(arm, 30995.0)
    arm_s = SimpleNamespace(symbol="MNQ", direction="short", trigger=31005.0, confidence=0.95, side="SELL")
    assert "against open trade 1" in trader._reentry_block_reason(arm_s, 31005.0)


def test_max_concurrent_and_size_untouched():
    assert slr._HARD_MAX_CONCURRENT == 3


# -- 3rd trade: same rule as 2nd (default); optional EXTRA_TRADE3_TP_PROGRESS (0 = off) ----

def test_3rd_trade_default_same_rule_as_2nd(monkeypatch):
    monkeypatch.delenv("EXTRA_TRADE3_TP_PROGRESS", raising=False)
    assert xr.trade3_tp_progress() == 0.0
    two = [_pos(), _pos("bot_2", entry=31010.0)]
    assert xr.check_same_dir_profit(two, 2, "long", 31010.25, PV) is None         # both in profit (1% / 0%+)
    assert xr.check_same_dir_profit(two, 2, "long", 31010.0, PV) == \
        "trade 2 not in profit (flat $0), needs both trades in profit + same direction"
    assert "against open trade 1" in xr.check_same_dir_profit(two, 2, "short", 30990.0, PV)
    no_tp = [_pos(), _pos("bot_adopt_1", entry=31010.0, tp=0.0)]
    assert xr.check_same_dir_profit(no_tp, 2, "long", 31020.0, PV) is None        # no TP is fine when off
    for opn, px in ((two, 31010.25), (two, 31005.0), (no_tp, 31020.0)):         # == backtest rule
        assert (xr.check_same_dir_profit(opn, 2, "long", px, PV) is None) == _backtest_rule("long", opn, px)
    assert xr.status_line(two, 2, 3, 31005.0, PV) == \
        "3rd trade: needs both trades in profit + same direction (now: trade 1 up $10, trade 2 down $10)"
    assert xr.status_line(two, 2, 3, 31020.0, PV) == "3rd trade: allowed, LONG setups only (now: trade 1 up $40, trade 2 up $20)"
    assert "both open trades in profit" in xr.describe(xr.SAME_DIR_PROFIT, 0.70)


# -- optional: EXTRA_TRADE3_TP_PROGRESS=0.5 (both open trades halfway to their own TP) ----

@pytest.fixture
def half(monkeypatch):
    monkeypatch.setenv("EXTRA_TRADE3_TP_PROGRESS", "0.5")

def _two():
    return [_pos(), _pos("bot_2", entry=31010.0)]        # TPs 31100 / 31110


def test_tp_progress():
    assert xr.tp_progress(_pos(), 31050.0) == pytest.approx(0.5)
    assert xr.tp_progress(_pos(direction="short"), 30970.0) == pytest.approx(0.3)
    assert xr.tp_progress(_pos(tp=0.0), 31050.0) is None
    cleared = _pos(tp=0.0, tp_cleared=True, original_tp=31100.0)
    assert xr.tp_progress(cleared, 31120.0) == pytest.approx(1.2)


def test_3rd_trade_needs_both_halfway(monkeypatch, half):
    assert xr.check_same_dir_profit(_two(), 2, "long", 31060.0, PV) is None          # 60% and 50%
    why = xr.check_same_dir_profit(_two(), 2, "long", 31040.0, PV)                  # 40% and 30%
    assert why == "trade 1 only 40% to TP, needs both trades 50% to TP (now 40% and 30%)"
    why = xr.check_same_dir_profit(_two(), 2, "long", 31055.0, PV)                  # 55% and 45%
    assert why.startswith("trade 2 only 45% to TP")
    assert "against open trade" in xr.check_same_dir_profit(_two(), 2, "short", 31080.0, PV)
    no_tp = [_pos(), _pos("bot_adopt_1", entry=31010.0, tp=0.0, regime="broker_adopt")]
    assert "has no TP" in xr.check_same_dir_profit(no_tp, 2, "long", 31090.0, PV)
    monkeypatch.setenv("EXTRA_TRADE3_TP_PROGRESS", "0.75")
    assert xr.check_same_dir_profit(_two(), 2, "long", 31060.0, PV) is not None
    assert xr.check_same_dir_profit(_two(), 2, "long", 31090.0, PV) is None        # 90% and 80%


def test_2nd_trade_still_only_needs_profit():
    assert xr.check_same_dir_profit([_pos()], 1, "long", 31000.25, PV) is None      # 0.25% to TP is fine


def test_3rd_trade_status_lines(half):
    assert xr.status_line(_two(), 2, 3, 31040.0, PV) == \
        "3rd trade: needs both trades halfway to TP + same direction (now 40% and 30%)"
    assert xr.status_line(_two(), 2, 3, 31060.0, PV) == \
        "3rd trade: allowed, LONG setups only (now 60% and 50% to TP)"
    assert xr.status_line(_two(), 2, 3, 31060.0, PV, {"direction": "long"}).endswith("- OK")
    no_tp = [_pos(), _pos("bot_adopt_1", entry=31010.0, tp=0.0)]
    assert xr.status_line(no_tp, 2, 3, 31060.0, PV) == "3rd trade: blocked, trade 2 has no TP"


def test_trader_3rd_trade(trader, monkeypatch):
    monkeypatch.setenv("EXTRA_TRADE_MODE", "same_dir_profit")
    for p in _two():
        trader.positions[p.order_id] = p
    monkeypatch.delenv("EXTRA_TRADE3_TP_PROGRESS", raising=False)
    assert trader._extra_trade_block("long", 31010.25, 0.62) is None              # default: just in profit
    assert "trade 2 not in profit" in trader._extra_trade_block("long", 31005.0, 0.95)
    monkeypatch.setenv("EXTRA_TRADE3_TP_PROGRESS", "0.5")
    assert trader._extra_trade_block("long", 31060.0, 0.62) is None
    assert "needs both trades 50% to TP" in trader._extra_trade_block("long", 31040.0, 0.95)
    assert trader._extra_trade_status(True, None, 31040.0) == \
        "3rd trade: needs both trades halfway to TP + same direction (now 40% and 30%)"


def test_ensemble_bot_gets_same_rule(trader, monkeypatch):
    """START_ENSEMBLE_VISIBLE.bat runs the same start_live_rithmic.py without REVAMP_EXITS."""
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", False)
    monkeypatch.delenv("EXTRA_TRADE_MODE", raising=False)
    assert trader._extra_mode() == xr.SAME_DIR_PROFIT
    trader.positions["bot_1"] = _pos()
    assert trader._extra_trade_block("long", 31010.0, 0.60) is None
    assert "against open trade 1" in trader._extra_trade_block("short", 30990.0, 0.95)
    assert trader._extra_trade_status(True, None, 30980.0) == \
        "2nd trade: needs trade 1 in profit + same direction (now: down $40)"
