import asyncio, sys, types, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from src.broker.rithmic_connector import RithmicConnector
import start_live_rithmic as slr


def _conn(quote_last, order):
    c = RithmicConnector.__new__(RithmicConnector)
    import threading
    c._order_lock = threading.RLock(); c._quote_lock = threading.RLock()
    c._quotes = {"MNQ": {"last": quote_last}}
    c._orders = {"bot_x": dict(order)}
    c._connected = True
    c.working = {"bot_x_sl"}
    c.placed, c.cancelled = [], []
    c.reject = False
    c._bracket_leg_ids = lambda oid: ([c._orders[oid].get("protective_sl_order_id"), f"{oid}_sl"], [])
    async def is_working(t): return t in c.working
    async def submit(**kw):
        c.placed.append(kw)
        if c.reject: return {}
        tag = kw["prefix"] + "_sl"; c.working.add(tag); return {"protective_sl_order_id": tag}
    async def cancel(t, **kw): c.cancelled.append(t); c.working.discard(t); return True
    c._async_is_order_working = is_working
    c._async_submit_protective_orders = submit
    c._async_cancel_protective_leg = cancel
    c._broker_symbol_has_position = lambda s: True
    return c

SHORT = {"entry_price": 31189.25, "stop_loss": 31290.25, "symbol": "MNQ", "type": "SELL",
         "size": 2, "protective_sl_order_id": "bot_x_sl"}

def run(c, sl): return asyncio.run(c._async_replace_protective_sl("bot_x", "MNQ", "SELL", 2, sl))

def test_reject_keeps_old_stop():
    c = _conn(31150.0, SHORT); c.reject = True
    assert run(c, 31176.75) is False
    assert c.cancelled == [] and "bot_x_sl" in c.working
    assert c._orders["bot_x"]["stop_loss"] == 31290.25

def test_place_first_then_cancel_old():
    c = _conn(31150.0, SHORT)
    assert run(c, 31176.75) is True
    assert c.placed and c.cancelled == ["bot_x_sl"]
    assert c._orders["bot_x"]["protective_sl_order_id"] == "bot_x_r1_sl"

def test_clamp_uses_live_price_not_entry():
    # live 31177.5 (in profit) ; stop 31176.75 below live for a short -> skip, keep stop
    c = _conn(31177.5, SHORT)
    assert run(c, 31176.75) is False
    assert c.placed == [] and c.cancelled == []

def test_no_move_when_losing():
    c = _conn(31200.0, SHORT)
    assert run(c, 31180.0) is False and c.placed == []

def test_no_move_to_loss_side_even_in_profit():
    c = _conn(31100.0, SHORT)
    assert run(c, 31250.0) is False and c.placed == []

def test_buffer_ticks():
    c = _conn(31150.0, SHORT)
    assert run(c, 31150.75) is False  # 3 ticks above live < 4

def test_repair_readds_stop():
    o = dict(SHORT); o.pop("protective_sl_order_id")
    c = _conn(31150.0, o); c.working = set()
    async def uncovered(sym, side): return 2
    c._async_uncovered_stop_qty = uncovered
    ok = asyncio.run(c._async_repair_missing_stop("bot_x", "MNQ", "SELL", 2, 31290.25))
    assert ok and c.placed[0]["stop_loss"] == 31290.25 and c.working

def test_repair_noop_when_stop_working():
    c = _conn(31150.0, SHORT)
    assert asyncio.run(c._async_repair_missing_stop("bot_x", "MNQ", "SELL", 2, 31290.25))
    assert c.placed == []

def test_modify_no_native_routes_to_replace():
    c = _conn(31150.0, dict(SHORT, supports_stop_modify=False))
    called = {}
    c.update_stop_loss = lambda **kw: called.update(kw) or True
    assert c.modify_position("bot_x", sl=31176.75) is True and called["new_sl"] == 31176.75

# ---- R-lock rule (short, entry 100, R=10, tick .25)
def rl(live, mfe, cur=110.0, d='short'):
    return slr.r_lock_stop(d, 100.0, 10.0, mfe, live, 0.25, cur,
                           r1=0.5, r1_keep=0.1, r2=1.0, r2_keep=0.5, keep_pct=0.5, buffer_ticks=4)

def test_rlock_none_when_losing_or_flat():
    assert rl(101.0, 6.0) is None and rl(100.0, 6.0) is None

def test_rlock_below_r1():
    assert rl(97.0, 4.0) is None

def test_rlock_r1():
    assert rl(95.0, 5.0) == 99.0

def test_rlock_r2():
    assert rl(90.0, 10.0) == 95.0

def test_rlock_trail_50pct_mfe():
    assert rl(80.0, 20.0) == 90.0

def test_rlock_tighten_only():
    assert rl(90.0, 10.0, cur=94.0) is None

def test_rlock_buffer():
    # MFE 20 but price back at 90.5: stop 90 is only 2 ticks from live -> skip
    assert rl(90.5, 20.0, cur=95.0) is None

def test_rlock_long():
    assert slr.r_lock_stop('long', 100.0, 10.0, 10.0, 110.0, 0.25, 90.0,
                           r1=0.5, r1_keep=0.1, r2=1.0, r2_keep=0.5, keep_pct=0.5, buffer_ticks=4) == 105.0
