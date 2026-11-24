"""Sunday-night warm-up fix (2026-10-11).

Bot restarted Sun 11:37 PM ET: Rithmic returned 67 5m bars (only since the
6 PM open) because the replay window was 8h wall clock, and the local warm
CSV was rejected by a 6h wall-clock age check. All mocked, no broker.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

from src.broker.rithmic_stability import (
    filter_warm_history,
    filter_warm_history_market,
    history_start_for_bars,
    stitch_history,
    trading_minutes_between,
    trading_time_back,
)

ET = pytz.timezone("US/Eastern")


def et(y, mo, d, h, mi=0):
    return ET.localize(datetime(y, mo, d, h, mi)).astimezone(timezone.utc)


SUN_RESTART = et(2026, 10, 11, 23, 37)  # Sunday
FRI_CLOSE = et(2026, 10, 9, 17, 0)
SUN_OPEN = et(2026, 10, 11, 18, 0)


def bars(start_utc, n, step_min=5, px=31000.0):
    rows = []
    t = start_utc
    for i in range(n):
        rows.append({"datetime": t, "open": px, "high": px + 1, "low": px - 1,
                     "close": px + i * 0.25, "volume": 10})
        t = t + timedelta(minutes=step_min)
    return pd.DataFrame(rows)


def session_bars(end_utc, n, step_min=5):
    """n bars ending at end_utc, skipping closed Globex time."""
    out = []
    t = end_utc
    while len(out) < n:
        t = t - timedelta(minutes=step_min)
        if trading_minutes_between(t, t + timedelta(minutes=step_min)) >= step_min:
            out.append(t)
    out.reverse()
    return pd.DataFrame({"datetime": out, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": [float(i) for i in range(n)], "volume": 1})


def test_trading_minutes_skip_weekend_and_halt():
    assert trading_minutes_between(et(2026, 10, 9, 16, 0), et(2026, 10, 11, 19, 0)) == 120
    assert trading_minutes_between(et(2026, 10, 13, 16, 30), et(2026, 10, 13, 18, 30)) == 60
    assert trading_minutes_between(FRI_CLOSE, SUN_OPEN) == 0
    assert trading_minutes_between(SUN_OPEN, FRI_CLOSE) == 0


def test_trading_time_back_crosses_weekend():
    # 337 min Sunday + 413 min Friday = 750 min -> Fri 10:07 ET
    assert trading_time_back(SUN_RESTART, 750) == et(2026, 10, 9, 10, 7)
    # Across Tue daily halt: Tue 18:30 back 60 -> Tue 16:30
    assert trading_time_back(et(2026, 10, 13, 18, 30), 60) == et(2026, 10, 13, 16, 30)
    # Cap at max_days
    assert trading_time_back(SUN_RESTART, 10 ** 6, max_days=4) == SUN_RESTART - timedelta(days=4)


def test_history_start_sunday_night_reaches_friday():
    start = history_start_for_bars(SUN_RESTART, 5, 100, calendar_hours=8)
    assert start <= et(2026, 10, 9, 12, 0)
    # enough trading time for 100 bars
    assert trading_minutes_between(start, SUN_RESTART) >= 500


def test_history_start_midweek_unchanged():
    end = et(2026, 10, 14, 12, 0)  # Wednesday
    start = history_start_for_bars(end, 5, 100, calendar_hours=8)
    assert start == end - timedelta(minutes=5 * 100 * 3)


def test_async_get_candles_requests_window_over_weekend():
    from src.broker.rithmic_connector import RithmicConnector

    seen = {}

    class FakeClient:
        async def get_historical_time_bars(self, **kw):
            seen.update(kw)
            idx = session_bars(kw["end_time"], 150)
            return [{"bar_end_datetime": t.to_pydatetime(), "open_price": 1, "high_price": 2,
                     "low_price": 0.5, "close_price": 1.5, "volume": 3}
                    for t in idx["datetime"]]

    class Fake:
        _client = FakeClient()

        def _resolve_symbol(self, symbol):
            return "MNQZ6", "CME"

    df = asyncio.run(RithmicConnector._async_get_candles(Fake(), "MNQ", 5, 100, end_time=SUN_RESTART))
    assert seen["start_time"] <= et(2026, 10, 9, 12, 0)
    assert seen["end_time"] == SUN_RESTART
    assert len(df) == 100


def test_market_filter_keeps_friday_close_bars_on_sunday_night():
    fri = bars(et(2026, 10, 9, 16, 0), 12)  # 16:00..16:55 ET Friday
    assert filter_warm_history(fri, max_age_hours=6, now=SUN_RESTART) is None  # old rule
    fresh = filter_warm_history_market(fri, max_age_hours=6, now=SUN_RESTART)
    assert fresh is not None and len(fresh) == 4  # 16:40..16:55 within 6h trading time
    weeks_old = bars(et(2026, 8, 21, 16, 0), 12)
    assert filter_warm_history_market(weeks_old, max_age_hours=6, now=SUN_RESTART) is None


def test_stitch_friday_history_with_sunday_bars():
    fri = bars(et(2026, 10, 9, 13, 0), 48)  # 13:00..16:55 ET
    sun = bars(et(2026, 10, 11, 18, 5), 67)  # end-stamped 18:05..23:35
    out = stitch_history(fri, sun, 100)
    assert len(out) == 100
    assert out["datetime"].is_monotonic_increasing
    assert out["datetime"].iloc[-1] == sun["datetime"].iloc[-1]
    # overlap: fresh wins, no duplicates
    overlap = bars(et(2026, 10, 11, 18, 5), 3, px=1.0)
    out2 = stitch_history(pd.concat([fri, overlap]), sun, 200)
    assert out2["datetime"].is_unique
    assert len(out2) == 48 + 67
    # real trading gap -> history ignored
    thu = bars(et(2026, 10, 8, 10, 0), 20)
    assert len(stitch_history(thu, sun, 200)) == 67


class _Broker:
    def __init__(self):
        self.n = 67

    def get_candles(self, symbol, timeframe_minutes=5, num_candles=100):
        df = bars(SUN_OPEN + timedelta(minutes=5), self.n)
        return df.tail(num_candles).reset_index(drop=True)


def _fake_trader(broker):
    from start_live_rithmic import LiveRithmicTrader

    class T:
        pass

    t = T()
    t.broker = broker
    t.lookback = 10
    t.ema_len = 50
    t._warmup_log_state = {}
    t._get_local_history_candles = lambda symbol, count: None
    t._merge_candles = lambda h, l, c: LiveRithmicTrader._merge_candles(t, h, l, c)
    t.get_candles = lambda symbol, count=100: LiveRithmicTrader.get_candles(t, symbol, count)
    return t


def test_bar_count_grows_each_loop_and_reaches_75():
    b = _Broker()
    t = _fake_trader(b)
    counts = []
    for _ in range(9):
        df = t.get_candles("MNQ", 100)
        counts.append(len(df))
        b.n += 1  # one new 5m bar per 5 min
    assert counts == list(range(67, 76))
    assert counts[-1] >= 75  # 8 bars = ~40 min


def test_bar_count_never_shrinks_if_broker_returns_fewer():
    b = _Broker()
    t = _fake_trader(b)
    assert len(t.get_candles("MNQ", 100)) == 67

    class Short(_Broker):
        def get_candles(self, symbol, timeframe_minutes=5, num_candles=100):
            full = bars(SUN_OPEN + timedelta(minutes=5), 68)
            return full.tail(40).reset_index(drop=True)

    t.broker = Short()
    assert len(t.get_candles("MNQ", 100)) == 68
