import asyncio, logging, sys
import pytest
from src.broker import rithmic_stability as rs


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    from async_rithmic.plants.order import OrderPlant
    calls = {"n": 0, "mode": "ok"}

    async def fake(self, **kw):
        calls["n"] += 1
        if calls["mode"] == "hang":
            await asyncio.sleep(5)
        return [{"id": calls["n"]}]

    monkeypatch.setattr(rs, "_ORIG_LIST_ORDERS", None)
    monkeypatch.setattr(OrderPlant, "list_orders", fake)
    monkeypatch.setenv("RITHMIC_LIST_ORDERS_TIMEOUT", "0.2")
    rs.reset_list_orders_state()
    rs.install_order_plant_patches()
    yield calls, OrderPlant
    rs.reset_list_orders_state()


def test_throttle_dedupes(_patched):
    calls, OP = _patched
    async def go():
        p = object.__new__(OP)
        a = await asyncio.gather(*[p.list_orders(account_id="x") for _ in range(5)])
        b = await p.list_orders(account_id="x")
        return a, b
    a, b = asyncio.run(go())
    assert calls["n"] == 1 and b == [{"id": 1}]


def test_timeout_then_backoff_fast_fail_then_reconnect_resync(_patched):
    calls, OP = _patched
    calls["mode"] = "hang"
    p = object.__new__(OP)
    with pytest.raises(RuntimeError):
        asyncio.run(p.list_orders())
    n = calls["n"]
    with pytest.raises(RuntimeError, match="backing off"):
        asyncio.run(p.list_orders())
    assert calls["n"] == n  # no new broker hit during backoff, no stale data
    calls["mode"] = "ok"
    rs.mark_order_plant_reconnected()
    assert asyncio.run(p.list_orders())  # fresh resync right after reconnect
    assert rs.consume_resync_pending() and not rs.consume_resync_pending()


def test_backoff_grows_and_caps():
    assert rs.list_orders_backoff_delay(1) == 5
    assert rs.list_orders_backoff_delay(2) == 10
    assert rs.list_orders_backoff_delay(10) == 60


def test_log_filter_strips_traceback_and_dedupes():
    f = rs.RithmicDisconnectLogFilter()
    try:
        raise ConnectionError("x")
    except ConnectionError:
        ei = sys.exc_info()
    r = logging.LogRecord("rithmic.plant.order", logging.ERROR, "", 0, "WebSocket closed", (), ei)
    assert f.filter(r) and r.exc_info is None and r.levelname == "WARNING"
    r2 = logging.LogRecord("rithmic.plant.order", logging.ERROR, "", 0, "WebSocket closed", (), ei)
    assert not f.filter(r2)
    other = logging.LogRecord("TradingBot", logging.ERROR, "", 0, "keep", (), ei)
    assert f.filter(other) and other.exc_info is not None
