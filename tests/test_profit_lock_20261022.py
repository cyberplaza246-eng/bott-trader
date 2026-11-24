"""2026-10-22 per-trade profit lock: +$400 open profit -> SL locks +$50."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import start_live_rithmic as slr


def _f(direction, size, pv, peak, live, cur, entry=25000.0):
    return slr.fixed_profit_lock_stop(direction, entry, size, pv, peak, live, 0.25, cur,
                                      trigger_usd=400, lock_usd=50, buffer_ticks=4)


def test_long_mnq_locks_plus_50():
    assert _f("long", 1, 2.0, 0, 25199.0, 24960.0) == ("wait", None)  # $398
    assert _f("long", 1, 2.0, 0, 25200.0, 24960.0) == ("move", pytest.approx(25025.0))


def test_short_mnq_locks_plus_50():
    assert _f("short", 1, 2.0, 0, 24800.0, 25040.0) == ("move", pytest.approx(24975.0))


def test_qty_gt_1_and_tick_rounding_safe_side():
    # 3 MNQ: $50/$6 = 8.33 pts -> 8.50 (rounds away from entry, locks >= $50)
    st, sl = _f("long", 3, 2.0, 0, 25067.0, 24960.0)
    assert st == "move" and sl == pytest.approx(25008.5)
    st, sl = _f("short", 3, 2.0, 0, 24933.0, 25040.0)
    assert st == "move" and sl == pytest.approx(24991.5)
    # 1 NQ $20/pt: $50 = 2.5 pts
    assert _f("long", 1, 20.0, 0, 25020.0, 24960.0) == ("move", pytest.approx(25002.5))


def test_already_tighter_stop_left_alone():
    assert _f("long", 1, 2.0, 0, 25250.0, 25100.0)[0] == "tighter"
    assert _f("short", 1, 2.0, 0, 24750.0, 24900.0)[0] == "tighter"


def test_disabled_with_zero():
    assert slr.fixed_profit_lock_stop("long", 25000.0, 1, 2.0, 0, 25300.0, 0.25, 24960.0,
                                      trigger_usd=0, lock_usd=50)[0] == "off"
    assert slr.fixed_profit_lock_stop("long", 25000.0, 1, 2.0, 0, 25300.0, 0.25, 24960.0,
                                      trigger_usd=400, lock_usd=0)[0] == "off"


def test_peak_hit_but_price_back_below_lock_waits():
    assert _f("long", 1, 2.0, 200.0, 25020.0, 24960.0) == ("wait", None)


def _trader(monkeypatch):
    monkeypatch.setattr(slr, "PROFIT_LOCK_TRIGGER", 400.0)
    monkeypatch.setattr(slr, "PROFIT_LOCK_AMOUNT", 50.0)
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.paper_orders = False
    t.paper_mode = False
    t.broker.update_stop_loss.return_value = True
    return t


def test_fires_once_through_broker_stop_path_keeps_tp(monkeypatch):
    t = _trader(monkeypatch)
    now = datetime.now(timezone.utc)
    pos = slr.Position(order_id="bot_adopt_x", symbol="MNQ", direction="long",
                       entry_price=25000.0, size=1, sl=24960.0, tp=25300.0, entry_time=now)
    t.positions[pos.order_id] = pos
    t.get_price = lambda s: {"last": 25210.0}
    df = pd.DataFrame([{"datetime": now + timedelta(minutes=1), "high": 25212.0,
                        "low": 25150.0, "close": 25210.0}])
    t.update_fixed_profit_lock("MNQ", df)
    assert pos.sl == pytest.approx(25025.0) and pos.tp == 25300.0 and pos.fixed_lock_done
    kw = t.broker.update_stop_loss.call_args.kwargs
    assert kw["new_sl"] == pytest.approx(25025.0) and kw["take_profit"] == 25300.0
    # Later bars: never fires again (even if stop was moved back externally)
    pos.sl = 24960.0
    t.update_fixed_profit_lock("MNQ", df)
    assert t.broker.update_stop_loss.call_count == 1
    assert pos.sl == 24960.0


def test_failed_broker_update_retries(monkeypatch):
    t = _trader(monkeypatch)
    t.broker.update_stop_loss.return_value = False
    now = datetime.now(timezone.utc)
    pos = slr.Position(order_id="bot_s", symbol="MNQ", direction="short",
                       entry_price=25000.0, size=1, sl=25040.0, tp=24700.0, entry_time=now)
    t.positions[pos.order_id] = pos
    t.get_price = lambda s: {"last": 24790.0}
    df = pd.DataFrame([{"datetime": now + timedelta(minutes=1), "high": 24850.0,
                        "low": 24788.0, "close": 24790.0}])
    t.update_fixed_profit_lock("MNQ", df)
    assert pos.sl == 25040.0 and not pos.fixed_lock_done
    t.broker.update_stop_loss.return_value = True
    t.update_fixed_profit_lock("MNQ", df)
    assert pos.sl == pytest.approx(24975.0) and pos.fixed_lock_done


def test_addon_trade_untouched(monkeypatch):
    t = _trader(monkeypatch)
    now = datetime.now(timezone.utc)
    pos = slr.Position(order_id="bot_e1", symbol="MNQ", direction="long", entry_price=25000.0,
                       size=1, sl=24960.0, tp=25300.0, entry_time=now, addon="e1")
    t.positions[pos.order_id] = pos
    t.get_price = lambda s: {"last": 25250.0}
    t.update_fixed_profit_lock("MNQ", pd.DataFrame([{"high": 25250.0, "low": 25200.0, "close": 25250.0}]))
    assert pos.sl == 24960.0
