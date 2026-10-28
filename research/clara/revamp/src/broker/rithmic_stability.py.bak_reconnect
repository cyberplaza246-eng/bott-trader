"""Keep Lucid history replays from fighting each other.

Rithmic allows one historical replay per symbol and period. The live loop and
the ensemble were both requesting MNQ 5m, and async_rithmic resent template
202 after a ForcedLogout reconnect. That produced "already in progress", then
permission denied on the new history socket.

This module is installed once, before RithmicClient is constructed.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

# RequestTimeBarReplay / RequestTickBarReplay. Resending these on a fresh
# socket keeps the previous replay "in progress" and draws a permission deny.
REPLAY_TEMPLATES = frozenset({202, 206})

_INSTALLED = False
_ORIG_SEND = None
_ORIG_GET_BARS = None
_ORIG_GET_DELAY = None


def is_in_progress_error(exc: Any) -> bool:
    """True when Rithmic rejected a second replay for the same bar key."""
    return "already in progress" in str(exc).lower()


def should_retry_send_after_reconnect(template_id: Any) -> bool:
    """Historical replays must not be resent onto a reconnected socket."""
    try:
        return int(template_id) not in REPLAY_TEMPLATES
    except (TypeError, ValueError):
        return True


class ThreadSingleFlight:
    """One in-flight call per key. Later callers wait and share the result.

    State is process-wide, so ``ThreadSingleFlight().do(key, fn)`` dedupes
    even when each caller constructs its own instance (live loop + ensemble).
    """

    _lock = threading.Lock()
    _slots: Dict[str, dict] = {}

    def do(self, key: str, fn: Callable[[], Any]) -> Any:
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = {
                    "event": threading.Event(),
                    "value": None,
                    "error": None,
                    "waiters": 0,
                }
                self._slots[key] = slot
                leader = True
            else:
                slot["waiters"] += 1
                leader = False

        if not leader:
            slot["event"].wait()
            error = slot["error"]
            value = slot["value"]
            with self._lock:
                slot["waiters"] -= 1
                if slot["waiters"] <= 0 and self._slots.get(key) is slot:
                    self._slots.pop(key, None)
            if error is not None:
                raise error
            return value

        try:
            value = fn()
            slot["value"] = value
            return value
        except BaseException as exc:
            slot["error"] = exc
            raise
        finally:
            slot["event"].set()
            with self._lock:
                if slot["waiters"] <= 0 and self._slots.get(key) is slot:
                    self._slots.pop(key, None)


def history_requests_paused(client: Any) -> bool:
    """True while the history plant is down or cooling off after a bad replay."""
    plant = _history_plant(client)
    if plant is None:
        return True
    try:
        connected = bool(plant.is_connected)
    except Exception:
        connected = False
    if not connected:
        return True
    event = getattr(plant, "_reconnect_event", None)
    if event is not None and not event.is_set():
        return True
    return time.time() < float(getattr(plant, "_replay_pause_until", 0) or 0)


def mark_history_replay_paused(plant: Any, seconds: Optional[float] = None) -> None:
    if plant is None:
        return
    if seconds is None:
        seconds = float(os.getenv("RITHMIC_HISTORY_PAUSE_SEC", "30"))
    plant._replay_pause_until = time.time() + max(0.0, seconds)


def filter_warm_history(
    df: Optional[pd.DataFrame],
    max_age_hours: float = 6.0,
    now: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """Drop warm-start bars older than ``max_age_hours``. None if nothing is fresh."""
    if df is None or len(df) == 0 or "datetime" not in df.columns:
        return None
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"])
    if out.empty:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    out = out.loc[out["datetime"] >= cutoff]
    if out.empty:
        return None
    return out.reset_index(drop=True)


def _history_plant(client: Any):
    if client is None:
        return None
    plants = getattr(client, "plants", None)
    if not plants:
        return None
    try:
        if hasattr(plants, "get"):
            return plants.get("history")
        return plants["history"]
    except Exception:
        return None


def _replay_key(symbol, exchange, bar_type, bar_type_periods) -> tuple:
    try:
        bar_id = int(bar_type)
    except (TypeError, ValueError):
        bar_id = str(bar_type)
    return (str(symbol), str(exchange), bar_id, int(bar_type_periods))


def install_async_rithmic_patches() -> None:
    """Patch async_rithmic once. Safe to call on every connect."""
    global _INSTALLED, _ORIG_SEND, _ORIG_GET_BARS, _ORIG_GET_DELAY
    if _INSTALLED:
        return

    from async_rithmic.objects import ReconnectionSettings
    from async_rithmic.plants.base import BasePlant
    from async_rithmic.plants.history import HistoryPlant

    _ORIG_SEND = BasePlant._send
    _ORIG_GET_BARS = HistoryPlant.get_historical_time_bars
    _ORIG_GET_DELAY = ReconnectionSettings.get_delay

    async def _send_no_replay_retry(self, message, template_id=None):
        if should_retry_send_after_reconnect(template_id):
            return await _ORIG_SEND(self, message, template_id=template_id)
        from async_rithmic.helpers.concurrency import try_acquire_lock

        # One shot. A closed socket must not reconnect and resend template 202.
        async with try_acquire_lock(self, context=f"send_{template_id}"):
            await self.ws.send(message)

    async def _get_bars_single_flight(
        self,
        symbol,
        exchange,
        start_time,
        end_time,
        bar_type,
        bar_type_periods,
        wait=True,
    ):
        if history_requests_paused(getattr(self, "client", None)) and not _plant_socket_open(self):
            raise RuntimeError("history plant down — pausing historical replay")
        if time.time() < float(getattr(self, "_replay_pause_until", 0) or 0):
            raise RuntimeError("history plant cooling down — pausing historical replay")
        event = getattr(self, "_reconnect_event", None)
        if event is not None and not event.is_set():
            raise RuntimeError("history plant reconnecting — pausing historical replay")

        key = _replay_key(symbol, exchange, bar_type, bar_type_periods)
        inflight = self.__dict__.setdefault("_replay_inflight", set())
        if key in inflight:
            raise RuntimeError(
                "Cannot start another historical time bar request with the same "
                "symbol, exchange, bar type, and period while one is already in progress."
            )
        inflight.add(key)
        try:
            return await _ORIG_GET_BARS(
                self,
                symbol,
                exchange,
                start_time,
                end_time,
                bar_type,
                bar_type_periods,
                wait,
            )
        except Exception as exc:
            if is_in_progress_error(exc) or "permission denied" in str(exc).lower():
                mark_history_replay_paused(self)
                self.logger.warning(
                    "Not retrying historical replay after %s",
                    "already in progress" if is_in_progress_error(exc) else "permission denied",
                )
            raise
        finally:
            inflight.discard(key)

    def _get_delay_with_floor(self, attempt: int) -> float:
        delay = _ORIG_GET_DELAY(self, attempt)
        floor = float(os.getenv("RITHMIC_RECONNECT_MIN_DELAY", "45"))
        return max(float(delay), floor)

    BasePlant._send = _send_no_replay_retry
    HistoryPlant.get_historical_time_bars = _get_bars_single_flight
    ReconnectionSettings.get_delay = _get_delay_with_floor
    _INSTALLED = True


def _plant_socket_open(plant: Any) -> bool:
    try:
        return bool(plant.is_connected)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Order-plant hardening (2026-10-07): list_orders (template 320) was polled by
# many loops with a 45s x2 library timeout. On a socket drop each call hung,
# printed a traceback, and stacked up behind the reconnect.
# ---------------------------------------------------------------------------
_ORIG_LIST_ORDERS = None
_LIST_STATE: Dict[str, Any] = {
    "value": None, "at": 0.0, "fail_until": 0.0, "fail_streak": 0,
    "inflight": None, "resync_pending": False, "generation": 0,
}


def list_orders_settings() -> Dict[str, float]:
    return {
        "cache_sec": float(os.getenv("RITHMIC_LIST_ORDERS_MIN_SEC", "2")),
        "timeout_sec": float(os.getenv("RITHMIC_LIST_ORDERS_TIMEOUT", "10")),
        "backoff_base": float(os.getenv("RITHMIC_LIST_ORDERS_BACKOFF", "5")),
        "backoff_max": float(os.getenv("RITHMIC_LIST_ORDERS_BACKOFF_MAX", "60")),
    }


def list_orders_backoff_delay(streak: int) -> float:
    s = list_orders_settings()
    return min(s["backoff_max"], s["backoff_base"] * (2 ** max(0, streak - 1)))


def mark_order_plant_reconnected() -> None:
    """Drop cached/backed-off state so the next call is a fresh broker resync."""
    _LIST_STATE.update(value=None, at=0.0, fail_until=0.0, fail_streak=0,
                       resync_pending=True)
    _LIST_STATE["generation"] += 1


def consume_resync_pending() -> bool:
    pending = bool(_LIST_STATE.get("resync_pending"))
    _LIST_STATE["resync_pending"] = False
    return pending


def reset_list_orders_state() -> None:
    _LIST_STATE.update(value=None, at=0.0, fail_until=0.0, fail_streak=0,
                       inflight=None, resync_pending=False)


def install_order_plant_patches() -> None:
    """Throttle + single-flight + short timeout + backoff for OrderPlant.list_orders.

    Never returns stale data after a failure (stop protection must not act on an
    old order list); during backoff it raises fast so callers treat it as unknown.
    """
    global _ORIG_LIST_ORDERS
    if _ORIG_LIST_ORDERS is not None:
        return
    import asyncio
    from async_rithmic.plants.order import OrderPlant

    _ORIG_LIST_ORDERS = OrderPlant.list_orders

    async def _list_orders_guarded(self, **kwargs):
        s = list_orders_settings()
        now = time.monotonic()
        st = _LIST_STATE
        if st["value"] is not None and now - st["at"] < s["cache_sec"]:
            return list(st["value"])
        if now < st["fail_until"]:
            raise RuntimeError(
                f"list_orders backing off {st['fail_until'] - now:.0f}s after order-plant timeout"
            )
        inflight = st.get("inflight")
        if inflight is not None and not inflight.done():
            return list(await asyncio.shield(inflight))

        gen = st["generation"]

        async def _do():
            return await asyncio.wait_for(_ORIG_LIST_ORDERS(self, **kwargs), timeout=s["timeout_sec"])

        task = asyncio.ensure_future(_do())
        st["inflight"] = task
        try:
            value = await task
        except Exception as exc:
            if st["generation"] == gen:
                st["fail_streak"] += 1
                delay = list_orders_backoff_delay(st["fail_streak"])
                st["fail_until"] = time.monotonic() + delay
                logging.getLogger("rithmic").warning(
                    "Rithmic list_orders unavailable (%s) - backing off %.0fs",
                    type(exc).__name__ if not str(exc) else str(exc)[:80], delay,
                )
            st["value"] = None
            raise RuntimeError(f"list_orders failed: {type(exc).__name__}") from None
        finally:
            if st.get("inflight") is task:
                st["inflight"] = None
        st.update(value=list(value or []), at=time.monotonic(), fail_streak=0, fail_until=0.0)
        return value

    OrderPlant.list_orders = _list_orders_guarded


class RithmicDisconnectLogFilter(logging.Filter):
    """One clean line per disconnect: strip tracebacks, dedupe repeats for 60s."""

    def __init__(self, window: float = 60.0):
        super().__init__()
        self.window = window
        self._seen: Dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.name).startswith(("rithmic", "async_rithmic")):
            return True
        if record.exc_info:
            exc = record.exc_info[1]
            record.exc_info = None
            record.exc_text = None
            if record.levelno >= logging.ERROR:
                record.levelno, record.levelname = logging.WARNING, "WARNING"
            if exc is not None:
                record.msg = f"{record.getMessage()} [{type(exc).__name__}]"
                record.args = ()
        key = f"{record.name}|{record.getMessage()[:120]}"
        now = time.monotonic()
        last = self._seen.get(key)
        if last is not None and now - last < self.window and "Reconnection successful" not in key:
            return False
        self._seen[key] = now
        return True


def install_rithmic_disconnect_log_filter() -> None:
    flt = RithmicDisconnectLogFilter()
    for name in ("rithmic", "async_rithmic"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, RithmicDisconnectLogFilter) for f in lg.filters):
            lg.addFilter(flt)
    for name in list(logging.Logger.manager.loggerDict):
        if str(name).startswith(("rithmic.", "async_rithmic.")):
            lg = logging.getLogger(name)
            if not any(isinstance(f, RithmicDisconnectLogFilter) for f in lg.filters):
                lg.addFilter(flt)
    for h in logging.getLogger().handlers:
        if not any(isinstance(f, RithmicDisconnectLogFilter) for f in h.filters):
            h.addFilter(flt)


# ---------------------------------------------------------------------------
# Market-time helpers (2026-10-11): weekend / daily-halt aware warm-up.
# CME Globex equity index: Sun 18:00 ET -> Fri 17:00 ET, halt 17:00-18:00 ET
# Mon-Thu. Wall-clock windows (8h lookback, 6h staleness) break right after
# the Sunday open and after the daily halt, so these count trading minutes.
# ---------------------------------------------------------------------------
try:  # pytz is already a project dependency (src/utils/trading_session.py)
    import pytz as _pytz

    _ET = _pytz.timezone("US/Eastern")

    def _et_midnight(d) -> datetime:
        return _ET.localize(datetime(d.year, d.month, d.day))

except Exception:  # pragma: no cover - fallback when pytz missing
    from zoneinfo import ZoneInfo as _ZoneInfo

    _ET = _ZoneInfo("America/New_York")

    def _et_midnight(d) -> datetime:
        return datetime(d.year, d.month, d.day, tzinfo=_ET)


def _as_utc(ts: Any) -> datetime:
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _globex_open_ranges_for_day(d) -> list:
    """UTC (start, end) open ranges for one US/Eastern calendar day."""
    wd = d.weekday()  # Mon=0 .. Sun=6
    mid = _et_midnight(d)
    nxt = _et_midnight(d + timedelta(days=1))

    def at(h: int) -> datetime:
        if _ET is not None and hasattr(_ET, "localize"):
            return _ET.localize(datetime(d.year, d.month, d.day, h))
        return datetime(d.year, d.month, d.day, h, tzinfo=_ET)

    if wd == 5:  # Saturday closed
        return []
    if wd == 6:  # Sunday opens 18:00
        rng = [(at(18), nxt)]
    elif wd == 4:  # Friday closes 17:00
        rng = [(mid, at(17))]
    else:  # Mon-Thu, halt 17-18
        rng = [(mid, at(17)), (at(18), nxt)]
    return [(a.astimezone(timezone.utc), b.astimezone(timezone.utc)) for a, b in rng]


def trading_minutes_between(start: Any, end: Any) -> float:
    """Globex trading minutes in [start, end). 0 if end <= start."""
    a = _as_utc(start)
    b = _as_utc(end)
    if b <= a:
        return 0.0
    d = a.astimezone(_ET).date()
    last = b.astimezone(_ET).date()
    total = 0.0
    while d <= last:
        for s, e in _globex_open_ranges_for_day(d):
            lo = max(s, a)
            hi = min(e, b)
            if hi > lo:
                total += (hi - lo).total_seconds() / 60.0
        d = d + timedelta(days=1)
    return total


def trading_time_back(end: Any, minutes: float, max_days: float = 4.0) -> datetime:
    """UTC time that is ``minutes`` of Globex trading time before ``end``.

    Skips weekends and the daily 5-6 PM ET halt. Never goes back more than
    ``max_days`` calendar days (holidays etc.).
    """
    b = _as_utc(end)
    floor = b - timedelta(days=max_days)
    need = max(0.0, float(minutes))
    if need <= 0:
        return b
    d = b.astimezone(_ET).date()
    floor_day = floor.astimezone(_ET).date()
    while d >= floor_day:
        for s, e in reversed(_globex_open_ranges_for_day(d)):
            hi = min(e, b)
            lo = max(s, floor)
            if hi <= lo:
                continue
            span = (hi - lo).total_seconds() / 60.0
            if span >= need:
                return hi - timedelta(minutes=need)
            need -= span
        d = d - timedelta(days=1)
    return floor


def history_start_for_bars(
    end_time: Any,
    timeframe_minutes: float,
    num_candles: int,
    calendar_hours: float = 8.0,
    pad: float = 1.5,
    max_days: float = 4.0,
) -> datetime:
    """Replay start that covers ``num_candles`` bars of *trading* time.

    Keeps the old wall-clock window (max of 3x bar span and calendar_hours)
    and extends it back across weekends / the daily halt when that window
    alone would hold too few trading minutes (e.g. Sunday night after the
    6 PM open: 8h back lands on Saturday, so only Sunday bars came back).
    """
    end = _as_utc(end_time)
    bar_span = timedelta(minutes=float(timeframe_minutes) * int(num_candles) * 3)
    old_start = end - max(bar_span, timedelta(hours=float(calendar_hours)))
    need_min = float(timeframe_minutes) * int(num_candles) * float(pad)
    span_days = max(float(max_days), (end - old_start).total_seconds() / 86400.0)
    mkt_start = trading_time_back(end, need_min, max_days=span_days)
    return min(old_start, mkt_start)


def filter_warm_history_market(
    df: Optional[pd.DataFrame],
    max_age_hours: float = 6.0,
    now: Optional[datetime] = None,
    max_days: float = 4.0,
) -> Optional[pd.DataFrame]:
    """Like filter_warm_history, but age counts Globex *trading* time only.

    A local bar stamped at Friday's 5 PM close is ~0 trading minutes old at
    Sunday's 6 PM open, so it stays usable. Weeks-old files still get dropped.
    """
    if df is None or len(df) == 0 or "datetime" not in df.columns:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = trading_time_back(now, float(max_age_hours) * 60.0, max_days=max_days)
    return filter_warm_history(df, max_age_hours=(_as_utc(now) - cutoff).total_seconds() / 3600.0, now=now)


def stitch_history(
    history_df: Optional[pd.DataFrame],
    live_df: Optional[pd.DataFrame],
    count: int,
    timeframe_minutes: float = 5.0,
    max_gap_bars: float = 3.0,
) -> Optional[pd.DataFrame]:
    """Join older bars in front of fresh bars without leaving a trading gap.

    History rows at/after the first fresh bar are dropped (fresh wins). If
    the trading-time gap between history's last bar and the first fresh bar
    is more than ``max_gap_bars`` bars, history is ignored (fresh only).
    """
    if history_df is None or len(history_df) == 0:
        if live_df is None or len(live_df) == 0:
            return None
        return live_df.tail(count).reset_index(drop=True)
    if live_df is None or len(live_df) == 0:
        return history_df.tail(count).reset_index(drop=True)
    if "datetime" not in history_df.columns or "datetime" not in live_df.columns:
        merged = pd.concat([history_df, live_df], ignore_index=True).drop_duplicates(keep="last")
        return merged.tail(count).reset_index(drop=True)

    hist = history_df.copy()
    live = live_df.copy()
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True, errors="coerce")
    live["datetime"] = pd.to_datetime(live["datetime"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["datetime"]).sort_values("datetime")
    live = live.dropna(subset=["datetime"]).sort_values("datetime")
    if live.empty:
        return hist.tail(count).reset_index(drop=True) if len(hist) else None
    first_live = live["datetime"].iloc[0]
    hist = hist.loc[hist["datetime"] < first_live]
    if hist.empty:
        return live.tail(count).reset_index(drop=True)
    gap = trading_minutes_between(hist["datetime"].iloc[-1], first_live)
    if gap > float(timeframe_minutes) * float(max_gap_bars):
        return live.tail(count).reset_index(drop=True)
    merged = pd.concat([hist, live], ignore_index=True)
    merged = merged.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
    return merged.tail(count).reset_index(drop=True)
