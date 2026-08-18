"""
Rithmic Broker Connector — Direct connection via R|Protocol (async_rithmic).

Architecture:  Bot → Rithmic R|Protocol → CME  (data + execution unified)

Uses the async_rithmic library which communicates via Protocol Buffers over
WebSocket.  Since the bot is synchronous (APScheduler + threads), we run a
dedicated asyncio event loop in a background daemon thread and expose a
synchronous BaseBroker interface on top.

Provides:
  - Real-time market data (BBO quotes, time bars)
  - Order execution with bracket support (stop + target)
  - Position & account PnL queries
  - Historical OHLCV candle retrieval
  - Yahoo Finance fallback when Rithmic is unavailable
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.broker.base_broker import BaseBroker
from src.instruments import get_instrument, is_futures
from src.ai.order_flow import TickFlowTracker
from src.utils.logger import bot_logger, error_logger

# Rithmic gateway URLs
RITHMIC_GATEWAYS = {
    # ── Test / Demo ──
    "Rithmic Test": "wss://rituz00100.rithmic.com:443",
    # ── Production ──
    "Rithmic 01": "wss://rprotocol.rithmic.com:443",
    "Rithmic Paper Trading": "wss://rprotocol.rithmic.com:443",
    "LucidTrading": "wss://rprotocol.rithmic.com:443",
    "tradesea": "wss://rprotocol.rithmic.com:443",
    "tradesea-d": "wss://rprotocol.rithmic.com:443",
    "Apex": "wss://rprotocol.rithmic.com:443",
    "TopstepTrader": "wss://rprotocol.rithmic.com:443",
    # Add more as needed; user can also pass a full URL
}

# Map our symbol names → (Rithmic symbol, exchange)
# Front-month contracts are resolved dynamically via get_front_month_contract()
RITHMIC_SYMBOL_MAP = {
    "MES": ("MES", "CME"),
    "MNQ": ("MNQ", "CME"),
    "ES": ("ES", "CME"),
    "NQ": ("NQ", "CME"),
    "CL": ("CL", "NYMEX"),
    "GC": ("GC", "COMEX"),
    # Forex (if ever needed via Rithmic)
    "EUR/USD": ("6E", "CME"),
    "GBP/USD": ("6B", "CME"),
    "USD/JPY": ("6J", "CME"),
}

# Explicit contract codes in env, e.g. NQM6 → base NQ
_FUTURES_CONTRACT_RE = re.compile(
    r"^(MES|MNQ|ES|NQ|CL|GC|MGC)([FGHJKMNQUVXZ]\d{1,2})$",
    re.IGNORECASE,
)

# Exchange order statuses meaning a protective leg is no longer working.
_PROTECTIVE_TERMINAL_STATUSES = frozenset({
    "complete", "completed", "filled", "cancelled", "canceled",
    "rejected", "expired", "failure", "failed", "inactive",
})
# After entry, PnL/position snapshots can lag — do not treat "no position" as flat yet.
_ENTRY_POSITION_GRACE_SEC = float(os.getenv("RITHMIC_ENTRY_POSITION_GRACE_SEC", "30"))
# When list_positions is flat but SL/TP fill is unconfirmed, force-sync tracked state.
_BROKER_FLAT_FORCE_SYNC_SEC = float(os.getenv("BROKER_FLAT_SYNC_SEC", "30"))
_SCAN_PROTECTION_REPAIR_RETRIES = int(os.getenv("SCAN_PROTECTION_REPAIR_RETRIES", "3"))


class SubMinuteBarAggregator:
    """Build OHLCV bars from LAST_TRADE ticks when SECOND_BAR history is unavailable."""

    def __init__(self, max_trades: int = 40000):
        self._lock = threading.Lock()
        self._trades: Dict[str, deque] = {}
        self._max_trades = max_trades

    def record_trade(
        self,
        symbol: str,
        price: float,
        size: float = 1.0,
        ts: Optional[Any] = None,
    ) -> None:
        if price <= 0:
            return
        if ts is None:
            epoch = time.time()
        elif hasattr(ts, "timestamp"):
            epoch = ts.timestamp()
        else:
            try:
                epoch = float(ts)
            except (TypeError, ValueError):
                epoch = time.time()
        with self._lock:
            if symbol not in self._trades:
                self._trades[symbol] = deque(maxlen=self._max_trades)
            self._trades[symbol].append((epoch, price, max(float(size or 0), 0.0)))

    def to_dataframe(
        self, symbol: str, period_seconds: int, count: int
    ) -> Optional[pd.DataFrame]:
        with self._lock:
            trades = list(self._trades.get(symbol, []))
        if not trades:
            return None
        period = max(1, int(period_seconds))
        buckets: Dict[int, dict] = {}
        for epoch, price, size in trades:
            bucket = int(epoch // period) * period
            bar = buckets.get(bucket)
            if bar is None:
                buckets[bucket] = {
                    "datetime": datetime.fromtimestamp(bucket, tz=timezone.utc),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": size,
                }
            else:
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] += size
        if not buckets:
            return None
        rows = sorted(buckets.values(), key=lambda r: r["datetime"])
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.tail(count).reset_index(drop=True)


def resample_1m_to_subminute(
    df_1m: pd.DataFrame, period_seconds: int = 30
) -> Optional[pd.DataFrame]:
    """Split each 1m bar into two synthetic sub-minute bars (imperfect intra-minute path)."""
    if df_1m is None or len(df_1m) < 1 or period_seconds != 30:
        return None
    n = len(df_1m)
    dt = pd.to_datetime(df_1m["datetime"], utc=True).dt.as_unit("ns")
    o = df_1m["open"].to_numpy(dtype=float)
    h = df_1m["high"].to_numpy(dtype=float)
    l = df_1m["low"].to_numpy(dtype=float)
    c = df_1m["close"].to_numpy(dtype=float)
    v = df_1m["volume"].fillna(0).to_numpy(dtype=int)
    mid = (o + c) / 2.0

    dt1 = dt.to_numpy()
    dt2 = (dt + pd.Timedelta(seconds=30)).to_numpy()

    out_o = np.repeat(mid, 2)
    out_o[0::2] = o
    out_c = np.repeat(mid, 2)
    out_c[1::2] = c

    hi_a = np.maximum(o, mid)
    lo_a = np.minimum(o, mid)
    hi_b = np.maximum(mid, c)
    lo_b = np.minimum(mid, c)
    out_h = np.empty(n * 2)
    out_l = np.empty(n * 2)
    out_h[0::2] = hi_a
    out_h[1::2] = hi_b
    out_l[0::2] = lo_a
    out_l[1::2] = lo_b

    vol_half = np.maximum(v // 2, 1)
    out_v = np.empty(n * 2, dtype=int)
    out_v[0::2] = vol_half
    out_v[1::2] = np.maximum(v - vol_half, 1)

    out_dt = np.empty(n * 2, dtype=dt1.dtype)
    out_dt[0::2] = dt1
    out_dt[1::2] = dt2

    df = pd.DataFrame(
        {
            "datetime": out_dt,
            "open": out_o,
            "high": out_h,
            "low": out_l,
            "close": out_c,
            "volume": out_v,
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.reset_index(drop=True)


def resample_subminute_to_1m(
    df_sub: pd.DataFrame, period_seconds: int = 30
) -> Optional[pd.DataFrame]:
    """Aggregate sub-minute OHLCV bars into 1M candles (inverse of resample_1m_to_subminute)."""
    if df_sub is None or len(df_sub) < 2:
        return None
    d = df_sub.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("datetime").sort_index()
    df_1m = d.resample("1min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    df_1m = df_1m.dropna(subset=["open", "close"]).reset_index()
    return df_1m if len(df_1m) >= 1 else None


class RithmicConnector(BaseBroker):
    """Direct Rithmic R|Protocol broker connector via async_rithmic."""

    def __init__(self, live_mode: bool = False):
        self._connected = False
        self._live_mode = live_mode
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client = None  # async_rithmic.RithmicClient

        # Thread-safe caches (written by async callbacks, read by sync methods)
        self._quote_lock = threading.Lock()
        self._quotes: Dict[str, Dict[str, float]] = {}  # symbol → {bid, ask, last, bid_size, ask_size}

        self._tick_flow = TickFlowTracker()
        self._subminute_agg = SubMinuteBarAggregator()
        self._second_bar_source: str = "unknown"  # rithmic | ticks | 1m_derived | none

        self._bar_lock = threading.Lock()
        self._bars: Dict[str, deque] = {}  # "symbol_tf" → deque of bar dicts

        # Position / account state
        self._state_lock = threading.Lock()
        self._positions: Dict[str, Any] = {}  # symbol → position info
        self._account_info: Dict[str, Any] = {"balance": 0.0, "equity": 0.0}

        # Resolved front-month contract symbols: "MES" → "MESH6"
        self._resolved_contracts: Dict[str, str] = {}
        self._contract_overrides: Dict[str, str] = {}
        self._contracts_seeded = False
        self._market_data_lock = threading.Lock()
        self._warned_front_month: Dict[str, float] = {}  # symbol → last warning time
        self._front_month_warn_cooldown = float(
            os.getenv("RITHMIC_FRONT_MONTH_WARN_COOLDOWN", "300")
        )

        # Track orders submitted by this bot
        self._order_lock = threading.Lock()
        self._orders: Dict[str, Dict[str, Any]] = {}  # order_id → order info

        # Yahoo Finance fallback (lazy-loaded)
        self._yf_fallback = None

        # Lock timeout / fallback mode tracking
        self._lock_error_count = 0
        self._max_lock_errors = 3  # Switch to fallback mode after this many errors
        self._fallback_until: Optional[float] = None  # timestamp when to retry Rithmic
        self._fallback_cooldown_secs = 120  # Stay in fallback mode for 2 minutes
        self._candle_cache: Dict[str, tuple] = {}  # "sym_tf" → (timestamp, DataFrame)
        self._candle_cache_ttl = 30  # Cache candles for 30 seconds
        self._candle_shortfall_warned: set = set()  # (symbol, tf) — once per session
        self._1m_from_30s_logged: set = set()  # symbol — once per session
        self._request_timeout = float(os.getenv("RITHMIC_REQUEST_TIMEOUT", "45"))
        self._request_retries = max(1, int(os.getenv("RITHMIC_REQUEST_RETRIES", "2")))
        self._market_data_timeout = float(os.getenv("RITHMIC_MARKET_DATA_TIMEOUT", "90"))
        self._candle_fetch_timeout = float(os.getenv("RITHMIC_CANDLE_TIMEOUT", "30"))
        self._second_bar_timeout = float(os.getenv("RITHMIC_SECOND_BAR_TIMEOUT", "60"))
        self._second_bar_retries = max(1, int(os.getenv("RITHMIC_SECOND_BAR_RETRIES", "3")))
        self._front_month_retries = max(1, int(os.getenv("RITHMIC_FRONT_MONTH_RETRIES", "2")))

        # Configuration from env
        self._user = os.getenv("RITHMIC_USER_ID", "")
        self._password = os.getenv("RITHMIC_PASSWORD", "")
        self._system = os.getenv("RITHMIC_SYSTEM", "Rithmic Paper Trading")
        self._gateway = os.getenv("RITHMIC_GATEWAY", "")
        self._app_name = "AiScalpBot"
        self._app_version = "1.0"
        self._disable_yahoo_fallback = (
            os.getenv("RITHMIC_DISABLE_YAHOO_FALLBACK", "false").lower() == "true"
        )
        self._account_id_override = os.getenv("RITHMIC_ACCOUNT_ID", "").strip()
        self._trade_route_override = os.getenv("RITHMIC_TRADE_ROUTE", "").strip()
        self._allow_simulator_route = (
            os.getenv("RITHMIC_ALLOW_SIMULATOR", "false").lower() == "true"
        )
        self._selected_account_id: Optional[str] = None
        self._routes_by_exchange: Dict[str, str] = {}
        self._using_simulator_route = False
        self._confirmed_exit_fills: Dict[str, Dict[str, Any]] = {}

        # Symbols to subscribe (populated from config at init time)
        self._symbols_to_watch: List[str] = []
        self._last_connect_error: Optional[str] = None
        self._market_data_ready = False

    # ── Connection lifecycle ──────────────────────────────────────

    @staticmethod
    def _format_connect_error(exc: BaseException) -> str:
        """Human-readable connect error (concurrent.futures.TimeoutError has empty str())."""
        if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
            return (
                "Connection timed out — Rithmic often rejects extra sessions "
                "(ForcedLogout). Close NinjaTrader/R|Trader/other bot instances, "
                "wait ~60s, then retry."
            )
        msg = str(exc).strip()
        if msg:
            return msg
        if exc.__cause__ is not None:
            cause_msg = str(exc.__cause__).strip()
            if cause_msg:
                return f"{type(exc).__name__}: {cause_msg}"
        return f"{type(exc).__module__}.{type(exc).__name__}"

    def initialize(self) -> None:
        if not self._user or not self._password:
            bot_logger.warning(
                "RITHMIC_USER_ID / RITHMIC_PASSWORD not set — "
                "running in data-fallback mode (Yahoo Finance)"
            )
            self._connected = False
            return

        url = self._gateway or RITHMIC_GATEWAYS.get(self._system, "")
        if not url:
            self._last_connect_error = f"No gateway URL for system {self._system!r}"
            error_logger.error(self._last_connect_error)
            self._connected = False
            return

        connect_timeout = float(os.getenv("RITHMIC_CONNECT_TIMEOUT", "90"))
        max_attempts = max(1, int(os.getenv("RITHMIC_CONNECT_RETRIES", "2")))

        for attempt in range(1, max_attempts + 1):
            try:
                self._start_io_thread()
                self._run_sync(self._async_connect(url), timeout=connect_timeout)
                self._connected = True
                self._last_connect_error = None
                self._market_data_ready = False
                self._seed_contract_overrides()
                bot_logger.info(
                    f"Rithmic connector initialized — system={self._system}, "
                    f"gateway={url}"
                )
                return
            except Exception as e:
                self._last_connect_error = self._format_connect_error(e)
                error_logger.error(
                    f"Rithmic connection failed (attempt {attempt}/{max_attempts}): "
                    f"{self._last_connect_error}",
                    exc_info=True,
                )
                self._connected = False
                self._teardown_io_thread()
                if attempt < max_attempts:
                    wait_secs = 5 * attempt
                    bot_logger.warning(
                        f"Rithmic connect retry in {wait_secs}s "
                        f"(attempt {attempt + 1}/{max_attempts})..."
                    )
                    time.sleep(wait_secs)

    def _start_io_thread(self) -> None:
        """Start (or restart) the background asyncio event loop thread."""
        if self._loop and self._loop.is_running():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="rithmic-io"
        )
        self._thread.start()

    def _teardown_io_thread(self) -> None:
        """Disconnect and stop the IO thread after a failed connect."""
        if self._client and self._loop and self._loop.is_running():
            try:
                self._run_sync(self._client.disconnect(), timeout=10)
            except Exception as e:
                error_logger.error(f"Rithmic disconnect after failed connect: {e}")
        self._client = None
        self._market_data_ready = False
        self._warned_front_month.clear()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    def shutdown(self) -> None:
        if self._client and self._loop and self._loop.is_running():
            try:
                self._run_sync(self._client.disconnect(), timeout=10)
            except Exception as e:
                error_logger.error(f"Rithmic disconnect error: {e}")
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        bot_logger.info("Rithmic connector shut down")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Account info ──────────────────────────────────────────────

    def get_balance(self) -> float:
        self._refresh_account_pnl()
        with self._state_lock:
            return self._account_info.get("balance", 0.0)

    def get_equity(self) -> float:
        self._refresh_account_pnl()
        with self._state_lock:
            return self._account_info.get("equity", 0.0)

    def get_account_info(self) -> Dict[str, Any]:
        self._refresh_account_pnl()
        with self._state_lock:
            return {
                "balance": self._account_info.get("balance", 0.0),
                "equity": self._account_info.get("equity", 0.0),
                "positions": len(self._positions),
                "broker": "rithmic",
                "system": self._system,
                "account_id": self._get_account_id(),
                "trade_routes": dict(self._routes_by_exchange),
                "simulator_route": self._using_simulator_route,
                "live_mode": self._live_mode,
            }

    @property
    def using_simulator_route(self) -> bool:
        return self._using_simulator_route

    def get_trade_route(self, exchange: str = "CME") -> str:
        return self._routes_by_exchange.get(exchange, "")

    # ── Market data ───────────────────────────────────────────────

    def _log_candle_shortfall_once(
        self,
        symbol: str,
        timeframe_minutes: int,
        got: int,
        suffix: str = "",
    ) -> None:
        key = (symbol, timeframe_minutes)
        if key in self._candle_shortfall_warned:
            return
        self._candle_shortfall_warned.add(key)
        bot_logger.warning(
            f"Rithmic candle fetch returned insufficient data for {symbol}: "
            f"{got} bars ({timeframe_minutes}m){suffix}"
        )

    def aggregate_1m_from_30s(
        self,
        symbol: str,
        num_candles: int = 300,
        period_seconds: int = 30,
        df_30s: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Build 1M OHLCV from 30s bars when Rithmic MINUTE_BAR history is empty."""
        if df_30s is None or len(df_30s) < 4:
            need_30s = max(num_candles * 2, 60)
            df_30s = self.get_candles_seconds(
                symbol, period_seconds=period_seconds, num_candles=need_30s,
            )
        derived = resample_subminute_to_1m(df_30s, period_seconds)
        if derived is None or len(derived) < 1:
            return None
        out = derived.tail(num_candles).reset_index(drop=True)
        if symbol not in self._1m_from_30s_logged:
            self._1m_from_30s_logged.add(symbol)
            bot_logger.info(
                f"1M fallback for {symbol}: aggregated {len(out)} bars "
                f"from {len(df_30s)}×{period_seconds}s bars "
                f"(MINUTE_BAR history unavailable)"
            )
        cache_key = f"{symbol}_1"
        self._candle_cache[cache_key] = (time.time(), out)
        return out

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
        end_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        # Historical pagination bypasses short-lived cache
        if end_time is None:
            cache_key = f"{symbol}_{timeframe_minutes}"
            now = time.time()
            latest_partial_df: Optional[pd.DataFrame] = None
            if cache_key in self._candle_cache:
                cached_time, cached_df = self._candle_cache[cache_key]
                if now - cached_time < self._candle_cache_ttl and cached_df is not None and len(cached_df) >= 10:
                    return cached_df
        else:
            cache_key = None
            latest_partial_df = None
            now = time.time()

        # Check if we're in fallback mode due to repeated lock errors
        if self._fallback_until is not None:
            if now < self._fallback_until:
                if self._disable_yahoo_fallback:
                    return None
                # Still in fallback mode — use Yahoo Finance
                df = self._yf_get_candles(symbol, timeframe_minutes, num_candles)
                if df is not None and len(df) >= 10:
                    self._candle_cache[cache_key] = (now, df)
                return df
            else:
                # Cooldown expired — try Rithmic again
                bot_logger.info("🔄 Rithmic history cooldown expired, attempting reconnect...")
                self._fallback_until = None
                self._lock_error_count = 0

        # Try Rithmic historical bars first with retry logic
        if self._connected:
            self._ensure_market_data()
            max_retries = 2
            count_retries = [num_candles, min(num_candles, 120), min(num_candles, 60)]
            seen_counts: set = set()
            for try_count in count_retries:
                if try_count in seen_counts:
                    continue
                seen_counts.add(try_count)
                for attempt in range(max_retries):
                    try:
                        df = self._run_sync(
                            self._async_get_candles(
                                symbol, timeframe_minutes, try_count, end_time,
                            ),
                            timeout=self._candle_fetch_timeout,
                        )
                        if df is not None and len(df) >= 10:
                            # Success! Reset error counter and cache result
                            self._lock_error_count = 0
                            if cache_key:
                                self._candle_cache[cache_key] = (now, df)
                            return df
                        else:
                            if df is not None and len(df) > 0:
                                latest_partial_df = df
                            if try_count == count_retries[-1] and attempt == max_retries - 1:
                                self._log_candle_shortfall_once(
                                    symbol, timeframe_minutes,
                                    0 if df is None else len(df),
                                )
                    except Exception as e:
                        error_msg = str(e).lower()
                        is_lock_error = "lock" in error_msg or "timeout" in error_msg

                        if is_lock_error:
                            self._lock_error_count += 1
                            # Check if we should enter fallback mode
                            if self._lock_error_count >= self._max_lock_errors:
                                self._fallback_until = now + self._fallback_cooldown_secs
                                if self._disable_yahoo_fallback:
                                    bot_logger.warning(
                                        f"⚠️ Rithmic history lock errors ({self._lock_error_count}x) — "
                                        f"Yahoo fallback disabled, retrying after {self._fallback_cooldown_secs}s"
                                    )
                                else:
                                    bot_logger.warning(
                                        f"⚠️ Rithmic history lock errors ({self._lock_error_count}x) — "
                                        f"switching to Yahoo Finance fallback for {self._fallback_cooldown_secs}s"
                                    )
                                break
                            elif attempt < max_retries - 1:
                                bot_logger.warning(f"Rithmic history lock timeout, retry {attempt+1}/{max_retries}")
                                time.sleep(0.5 + attempt)  # Brief backoff
                                continue

                        # KeyError from async_rithmic when no bars exist (e.g. market closed)
                        if isinstance(e, KeyError):
                            if self._disable_yahoo_fallback:
                                bot_logger.info(f"Rithmic no bars for {symbol} (market closed?)")
                            else:
                                bot_logger.info(f"Rithmic no bars for {symbol} (market closed?) — using Yahoo fallback")
                        else:
                            bot_logger.warning(f"Rithmic candles error for {symbol}: {e}")
                        break
                if latest_partial_df is not None and len(latest_partial_df) >= 10:
                    break

        if (
            timeframe_minutes == 1
            and end_time is None
            and (latest_partial_df is None or len(latest_partial_df) < 10)
        ):
            derived = self.aggregate_1m_from_30s(symbol, num_candles=num_candles)
            if derived is not None and len(derived) >= 10:
                return derived
            if derived is not None and len(derived) > 0:
                if latest_partial_df is None or len(derived) > len(latest_partial_df):
                    latest_partial_df = derived

        if self._disable_yahoo_fallback:
            if latest_partial_df is not None and len(latest_partial_df) > 0:
                return latest_partial_df
            return None

        # Fall back to Yahoo Finance
        df = self._yf_get_candles(symbol, timeframe_minutes, num_candles)
        if df is not None and len(df) >= 10 and cache_key:
            self._candle_cache[cache_key] = (now, df)
        return df

    def get_candles_deep(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 300,
        lookback_hours: float = 8.0,
    ) -> Optional[pd.DataFrame]:
        """Paginate Rithmic history backwards when session-open returns too few bars."""
        target = max(num_candles, 150 if timeframe_minutes == 1 else 50)
        df = self.get_candles(symbol, timeframe_minutes, num_candles=target)
        need = max(num_candles // 2, 30 if timeframe_minutes == 1 else 15)
        if df is not None and len(df) >= need:
            return df.tail(num_candles).reset_index(drop=True)

        if not self._connected:
            if df is not None and len(df) > 0:
                return df.tail(num_candles).reset_index(drop=True)
            if timeframe_minutes == 1:
                derived = self.aggregate_1m_from_30s(symbol, num_candles=num_candles)
                if derived is not None and len(derived) > 0:
                    return derived
            return df

        parts: List[pd.DataFrame] = []
        if df is not None and len(df) > 0:
            parts.append(df)
        end = datetime.now(timezone.utc)
        min_start = end - timedelta(hours=lookback_hours)
        max_pages = 6
        chunk = min(800, max(target, 200))
        empty_pages = 0

        while sum(len(p) for p in parts) < num_candles and end > min_start and max_pages > 0:
            page = self.get_candles(
                symbol, timeframe_minutes, num_candles=chunk, end_time=end,
            )
            max_pages -= 1
            if page is None or len(page) == 0:
                empty_pages += 1
                end = end - timedelta(hours=1)
                if empty_pages >= 3:
                    break
                time.sleep(0.15)
                continue
            empty_pages = 0
            oldest = pd.to_datetime(page["datetime"].iloc[0], utc=True)
            if len(parts) > 1:
                prior_oldest = pd.to_datetime(parts[0]["datetime"].iloc[0], utc=True)
                if oldest >= prior_oldest:
                    break
            parts.insert(0, page)
            if oldest <= min_start or len(page) < 10:
                break
            end = oldest - timedelta(minutes=max(timeframe_minutes, 1))
            time.sleep(0.25)

        if not parts:
            if timeframe_minutes == 1:
                derived = self.aggregate_1m_from_30s(symbol, num_candles=num_candles)
                if derived is not None and len(derived) > 0:
                    return derived
            return df
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        prev_len = len(df) if df is not None else 0
        if len(out) > prev_len:
            bot_logger.info(
                f"Deep {timeframe_minutes}m history for {symbol}: "
                f"{len(out)} bars (paginated, lookback {lookback_hours}h)"
            )
        result = out.tail(num_candles).reset_index(drop=True)
        if timeframe_minutes == 1 and len(result) < need:
            derived = self.aggregate_1m_from_30s(symbol, num_candles=num_candles)
            if derived is not None and len(derived) > len(result):
                return derived
        return result

    @property
    def second_bar_source(self) -> str:
        """How sub-minute bars were last resolved: rithmic, ticks, 1m_derived, or none."""
        return self._second_bar_source

    def _derive_subminute_from_1m(
        self,
        symbol: str,
        period_seconds: int,
        num_candles: int,
        min_bars: int,
        df_1m: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        if df_1m is None or len(df_1m) < 1:
            need_1m = max(num_candles // 2 + 5, min_bars)
            df_1m = self.get_candles(symbol, timeframe_minutes=1, num_candles=need_1m)
        derived = resample_1m_to_subminute(df_1m, period_seconds)
        if derived is None or len(derived) < min_bars:
            return None
        derived = derived.tail(num_candles).reset_index(drop=True)
        bot_logger.info(
            f"using 1M-derived {period_seconds}s fallback for {symbol} "
            f"({len(derived)} bars) — SECOND_BAR history unavailable"
        )
        self._second_bar_source = "1m_derived"
        return derived

    def get_candles_seconds(
        self,
        symbol: str,
        period_seconds: int = 30,
        num_candles: int = 100,
        end_time: Optional[datetime] = None,
        df_1m_fallback: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch sub-minute OHLCV bars (e.g. 30s) via Rithmic SECOND_BAR or tick aggregation."""
        period_seconds = max(1, int(period_seconds))
        min_bars = min(2, num_candles)
        if end_time is None:
            cache_key = f"{symbol}_{period_seconds}s"
            now = time.time()
            if cache_key in self._candle_cache:
                cached_time, cached_df = self._candle_cache[cache_key]
                if (
                    now - cached_time < self._candle_cache_ttl
                    and cached_df is not None
                    and len(cached_df) >= min_bars
                ):
                    return cached_df
        else:
            cache_key = None
            now = time.time()

        df: Optional[pd.DataFrame] = None
        if self._connected:
            self._ensure_market_data()
            for attempt in range(self._second_bar_retries):
                try:
                    df = self._run_sync(
                        self._async_get_candles_seconds(
                            symbol, period_seconds, num_candles, end_time
                        ),
                        timeout=self._second_bar_timeout,
                    )
                    if df is not None and len(df) >= min_bars:
                        self._second_bar_source = "rithmic"
                        if cache_key:
                            self._candle_cache[cache_key] = (now, df)
                        return df
                    got = 0 if df is None else len(df)
                    if attempt < self._second_bar_retries - 1:
                        bot_logger.info(
                            f"SECOND_BAR {symbol} ({period_seconds}s): {got} bars — "
                            f"retry {attempt + 1}/{self._second_bar_retries} "
                            f"(session-open / history plant slow)"
                        )
                        time.sleep(1.5 + attempt)
                except Exception as e:
                    bot_logger.warning(
                        f"Rithmic SECOND_BAR fetch failed for {symbol} "
                        f"({period_seconds}s): {e}"
                    )
                    if attempt < self._second_bar_retries - 1:
                        time.sleep(1.5 + attempt)
                        continue
                    break

        agg_df = self._subminute_agg.to_dataframe(symbol, period_seconds, num_candles)
        if agg_df is not None and len(agg_df) >= min_bars:
            self._second_bar_source = "ticks"
            if cache_key:
                self._candle_cache[cache_key] = (now, agg_df)
            return agg_df

        derived = self._derive_subminute_from_1m(
            symbol, period_seconds, num_candles, min_bars, df_1m_fallback
        )
        if derived is not None:
            if cache_key:
                self._candle_cache[cache_key] = (now, derived)
            return derived

        self._second_bar_source = "none"
        if df is not None and len(df) > 0:
            return df
        if agg_df is not None and len(agg_df) > 0:
            return agg_df
        return None

    def fetch_history_chunked(
        self,
        symbol: str,
        timeframe_minutes: int = 1,
        chunk_size: int = 8000,
        max_bars: int = 25000,
    ) -> Optional[pd.DataFrame]:
        """Paginate Rithmic history backwards for longer backtests."""
        if not self._connected:
            return None
        parts: List[pd.DataFrame] = []
        end = datetime.now(timezone.utc)
        while sum(len(p) for p in parts) < max_bars:
            df = self.get_candles(
                symbol, timeframe_minutes, num_candles=chunk_size, end_time=end
            )
            if df is None or len(df) < 50:
                break
            oldest = pd.to_datetime(df["datetime"].iloc[0], utc=True)
            parts.insert(0, df)
            end = oldest - timedelta(minutes=timeframe_minutes)
            if len(df) < chunk_size // 2:
                break
            time.sleep(0.3)
        if not parts:
            return None
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        return out.tail(max_bars).reset_index(drop=True)

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        if self._connected:
            self._ensure_market_data()
        # Try cached Rithmic quote
        with self._quote_lock:
            quote = self._quotes.get(symbol)
            if quote and quote.get("last", 0) > 0:
                return dict(quote)

        if self._disable_yahoo_fallback:
            return None

        # Fallback to Yahoo Finance
        return self._yf_get_latest_price(symbol)

    def get_spread(self, symbol: str) -> Optional[float]:
        with self._quote_lock:
            quote = self._quotes.get(symbol)
            if quote and quote.get("bid", 0) > 0 and quote.get("ask", 0) > 0:
                return quote["ask"] - quote["bid"]
        return get_instrument(symbol).spread_default

    # ── Order execution ───────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Dict[str, Any]]:
        if not self._connected:
            bot_logger.warning("Rithmic not connected — order rejected")
            return None

        self._ensure_market_data()

        try:
            spec = get_instrument(symbol)
            tick_size = spec.tick_size
            is_long = order_type.lower() in ("buy", "long")
            if stop_loss > 0:
                sl_mode = "down" if is_long else "up"
                stop_loss = self._round_to_tick(stop_loss, tick_size, mode=sl_mode)
            if take_profit > 0:
                tp_mode = "up" if is_long else "down"
                take_profit = self._round_to_tick(take_profit, tick_size, mode=tp_mode)

            pv = spec.contract_multiplier  # $/index point (NQ=20, MNQ=2)
            qty = int(size)
            risk_usd = abs(entry_price - stop_loss) * pv * qty if stop_loss > 0 else 0.0
            reward_usd = abs(take_profit - entry_price) * pv * qty if take_profit > 0 else 0.0

            order_id = f"bot_{uuid.uuid4().hex[:10]}"
            _, exchange = self._resolve_symbol(symbol)
            route = self.get_trade_route(exchange)
            acct = self._get_account_id() or "UNKNOWN"
            route_banner = (
                "PRO004 SIM MODE (simulator route)"
                if self._pro004_sim_mode_active() and self._live_mode
                else (
                    "🚨 ORDERS GOING TO SIMULATOR NOT LIVE 🚨"
                    if self._using_simulator_route and self._live_mode
                    else f"account={acct} trade_route={route or 'AUTO'}"
                )
            )
            bot_logger.warning(
                f"Rithmic ORDER SUBMIT: {order_id} {order_type} {qty} {symbol} "
                f"| ${pv:.0f}/pt tick={tick_size} risk~${risk_usd:.0f} reward~${reward_usd:.0f} "
                f"| {route_banner}"
            )
            if self._using_simulator_route and self._live_mode and not self._pro004_sim_mode_active():
                print(f"\n   🚨🚨🚨 ORDERS GOING TO SIMULATOR NOT LIVE 🚨🚨🚨")
                print(f"   account_id={acct} trade_route={route} exchange={exchange}")
                print(f"   Set RITHMIC_TRADE_ROUTE to your Lucid live route in .env\n")
                if not self._allow_simulator_route:
                    block_msg = (
                        f"Order {order_id} NOT placed on broker — blocked locally: "
                        f"live mode with simulator-only route (account={acct} route={route}). "
                        f"No order was sent to Rithmic. Ask Lucid to enable a live trade route, "
                        f"set RITHMIC_TRADE_ROUTE to that route in .env, or set "
                        f"RITHMIC_ALLOW_SIMULATOR=true only for deliberate sim testing on PRO."
                    )
                    bot_logger.error(block_msg)
                    print("\n   *** Order NOT placed on broker — blocked locally ***")
                    print(f"   {block_msg}\n")
                    return None
            result = self._run_sync(
                self._async_place_order(
                    symbol=symbol,
                    side=order_type,
                    qty=int(size),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=order_id,
                ),
                timeout=15,
            )
            if result is not None:
                raw_result = result.get("result") if isinstance(result, dict) else result
                native_stop_attached = bool(result.get("native_stop_attached")) if isinstance(result, dict) else False
                native_target_attached = bool(result.get("native_target_attached")) if isinstance(result, dict) else False
                order_info = {
                    "ticket": order_id,
                    "symbol": symbol,
                    "type": order_type,
                    "size": int(size),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "native_stop_attached": native_stop_attached,
                    "native_target_attached": native_target_attached,
                    "supports_stop_modify": native_stop_attached and not result.get("protective_sl_order_id"),
                    "bracket_mode": result.get("bracket_mode", "unknown") if isinstance(result, dict) else "unknown",
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                bracket_mode = result.get("bracket_mode", "none") if isinstance(result, dict) else "none"
                bot_logger.info(
                    f"Rithmic order placed: {order_id} {order_type} "
                    f"{int(size)} {symbol} | brackets={bracket_mode} "
                    f"SL={'yes' if native_stop_attached else 'no'} "
                    f"TP={'yes' if native_target_attached else 'no'}"
                )
                if bracket_mode == "protective_fallback":
                    bot_logger.info(self._protective_chart_legend(order_type))
                if stop_loss > 0 and not native_stop_attached:
                    bot_logger.error(
                        f"Rithmic order {order_id} has NO broker stop — "
                        f"position is only protected while bot is running"
                    )
                if take_profit > 0 and not native_target_attached:
                    bot_logger.warning(
                        f"Rithmic order {order_id} has NO broker take-profit"
                    )
                if isinstance(result, dict):
                    if result.get("stop_ticks"):
                        order_info["stop_ticks"] = result["stop_ticks"]
                    if result.get("target_ticks"):
                        order_info["target_ticks"] = result["target_ticks"]
                    if result.get("protective_sl_order_id"):
                        order_info["protective_sl_order_id"] = result["protective_sl_order_id"]
                    if result.get("protective_tp_order_id"):
                        order_info["protective_tp_order_id"] = result["protective_tp_order_id"]
                if raw_result is not None and isinstance(raw_result, dict):
                    order_info["broker_response"] = raw_result
                with self._order_lock:
                    existing = self._orders.get(order_id, {})
                    if existing.get("time"):
                        order_info["time"] = existing["time"]
                    if existing.get("price_ref"):
                        order_info["price_ref"] = existing["price_ref"]
                    if existing.get("entry_price") and result.get("price_ref"):
                        order_info["entry_price"] = result.get("price_ref", order_info["entry_price"])
                    self._orders[order_id] = order_info
                self.log_execution_diagnostics(symbol, order_id=order_id)
                return order_info
            return None
        except Exception as e:
            error_logger.error(f"Rithmic place_order error: {e}")
            return None

    def log_execution_diagnostics(
        self,
        symbol: str,
        *,
        order_id: Optional[str] = None,
        context: str = "post_order",
    ) -> Dict[str, Any]:
        """Log list_orders + list_positions counts for the configured account."""
        if not self._connected:
            return {}
        try:
            summary = self._run_sync(
                self._async_log_execution_diagnostics(symbol, order_id=order_id),
                timeout=15,
            )
        except Exception as e:
            error_logger.error(f"Execution diagnostics failed ({context}): {e}")
            return {}
        if not summary:
            return {}

        acct = summary.get("account_id", "?")
        route = summary.get("trade_route", "?")
        exchange = summary.get("exchange", "?")
        rith_sym = summary.get("rithmic_symbol", "?")
        order_count = summary.get("order_count", 0)
        pos_count = summary.get("position_count", 0)
        sym_pos = summary.get("symbol_position")
        related = summary.get("related_orders") or []
        position_source = summary.get("position_source", "list_positions")
        simulator = summary.get("simulator_route", False)

        line = (
            f"Rithmic diagnostics [{context}]: account={acct} route={route} "
            f"exchange={exchange} symbol={symbol} rithmic={rith_sym} | "
            f"list_orders={order_count} list_positions={pos_count}"
        )
        if simulator:
            line += " | SIMULATOR ROUTE (not live Lucid fills)"
        if sym_pos is not None:
            line += f" | {symbol} net_qty={sym_pos}"
        if position_source == "entry_fill_inference":
            line += (
                f" | position=OPEN via entry-fill inference "
                f"(list_positions empty — normal on simulator until Lucid live route)"
            )
        elif summary.get("inferred_open") is False and pos_count == 0:
            line += " | position=flat"
        if related:
            line += f" | tags: {', '.join(related[:6])}"
        bot_logger.info(line)
        print(f"   📋 {line}")
        return summary

    async def _async_log_execution_diagnostics(
        self,
        symbol: str,
        *,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        account_id = self._get_account_id()
        rith_sym, exchange = self._resolve_symbol(symbol)
        route = self.get_trade_route(exchange)

        orders = await self._client.list_orders(account_id=account_id)
        positions = await self._client.list_positions(account_id=account_id)

        sym_qty: Optional[int] = None
        open_positions: List[str] = []
        for p in positions or []:
            data = _response_to_dict_safe(p)
            our_sym = self._reverse_resolve(data.get("symbol", ""))
            qty = self._position_qty_from_broker_data(data)
            if qty != 0:
                open_positions.append(f"{our_sym}:{qty}")
                if self._position_matches_symbol(data, symbol):
                    sym_qty = qty

        list_positions_net = self._list_positions_net_qty(positions, symbol)
        inferred_open = await self._async_infer_open_from_filled_entries(
            symbol, order_id=order_id,
        )
        position_source = "list_positions"
        if list_positions_net == 0 and inferred_open is True:
            position_source = "entry_fill_inference"
            if sym_qty is None:
                with self._order_lock:
                    if order_id and order_id in self._orders:
                        sym_qty = int(self._orders[order_id].get("size", 1))
                    else:
                        for oid, info in self._orders.items():
                            if (
                                info.get("symbol") == symbol
                                and self._is_bot_entry_user_tag(oid)
                            ):
                                sym_qty = int(info.get("size", 1))
                                break

        related: List[str] = []
        prefix = f"{order_id}_" if order_id else None
        for o in orders or []:
            data = _response_to_dict_safe(o)
            tag = str(data.get("user_tag") or data.get("basket_id") or "?")
            status = str(data.get("status") or "?")
            if order_id and (tag == order_id or tag.startswith(prefix or "")):
                related.append(f"{tag}:{status}")

        return {
            "account_id": account_id,
            "trade_route": route,
            "exchange": exchange,
            "rithmic_symbol": rith_sym,
            "simulator_route": self._using_simulator_route,
            "order_count": len(orders or []),
            "position_count": len(open_positions),
            "open_positions": open_positions,
            "symbol_position": sym_qty,
            "position_source": position_source,
            "inferred_open": inferred_open is True,
            "related_orders": related,
        }

    def purge_orphan_protective_orders(
        self,
        symbols: List[str],
        *,
        threshold: int = 10,
    ) -> Dict[str, Dict[str, int]]:
        """Cancel working protective legs for flat symbols when order count exceeds threshold."""
        if not self._connected or threshold <= 0:
            return {}
        try:
            return self._run_sync(
                self._async_purge_orphan_protective_orders(symbols, threshold=threshold),
                timeout=45,
            ) or {}
        except Exception as e:
            error_logger.error(f"Orphan protective order purge failed: {e}")
            return {}

    async def _async_purge_orphan_protective_orders(
        self,
        symbols: List[str],
        *,
        threshold: int,
    ) -> Dict[str, Dict[str, int]]:
        account_id = self._get_account_id()
        orders = await self._client.list_orders(account_id=account_id) or []
        result: Dict[str, Dict[str, int]] = {}

        for symbol in symbols:
            rith_sym, _exchange = self._resolve_symbol(symbol)
            sym_orders = []
            for o in orders:
                data = _response_to_dict_safe(o)
                broker_sym = str(data.get("symbol") or "")
                if broker_sym == rith_sym or self._reverse_resolve(broker_sym) == symbol:
                    sym_orders.append(data)

            order_count = len(sym_orders)
            result[symbol] = {"order_count": order_count, "cancelled": 0}
            if order_count <= threshold:
                continue

            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
            list_net = self._list_positions_net_qty(positions, symbol)
            if list_net != 0:
                bot_logger.info(
                    f"Skipping orphan purge for {symbol}: "
                    f"orders={order_count} list_positions_net={list_net}"
                )
                continue

            cancelled = 0
            for data in sym_orders:
                tag = str(data.get("user_tag") or data.get("basket_id") or "")
                order_id = str(data.get("order_id") or data.get("basket_id") or tag)
                status = str(data.get("status") or "")
                if not self._is_working_order_status(status):
                    continue
                is_protective = (
                    tag.endswith("_sl")
                    or tag.endswith("_tp")
                    or tag.endswith("_tp_lit")
                    or "_sl" in tag
                    or "_tp" in tag
                )
                is_bot_entry = self._is_bot_entry_user_tag(tag)
                if not is_protective and not is_bot_entry:
                    continue
                if is_protective:
                    cancel_id = tag if tag.startswith("bot_") else order_id
                    ok = await self._async_cancel_protective_leg(
                        cancel_id,
                        symbol=symbol,
                        reason="orphan_purge_startup",
                        require_flat=False,
                        cancel_path="orphan_purge_startup",
                    )
                else:
                    cancel_id = tag if tag.startswith("bot_") else order_id
                    ok = await self._async_cancel_working_bot_order(
                        cancel_id,
                        symbol=symbol,
                        reason="orphan_purge_startup",
                    )
                if ok:
                    cancelled += 1
            result[symbol]["cancelled"] = cancelled
            if cancelled:
                bot_logger.info(
                    f"Purge orphan protective orders {symbol}: "
                    f"cancelled={cancelled} before={order_count} threshold={threshold}"
                )
        return result

    def get_order_info(self, ticket: Any) -> Optional[Dict[str, Any]]:
        order_id = str(ticket)
        with self._order_lock:
            order = self._orders.get(order_id)
            if not order:
                return None
            return dict(order)

    def ensure_protective_orders(
        self,
        ticket: str,
        symbol: str,
        side: str,
        size: int,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """Attach broker SL/TP if entry had no native bracket (safe to call repeatedly)."""
        if not self._connected:
            return False
        order_id = str(ticket)
        sl_ok, tp_ok = self.query_broker_protection(
            order_id, stop_loss=stop_loss, take_profit=take_profit,
        )
        self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
        if sl_ok and tp_ok:
            return True
        try:
            result = self._run_sync(
                self._async_ensure_protective_orders(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty=int(size),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                ),
                timeout=20,
            )
            if not result:
                return False
            with self._order_lock:
                order = self._orders.setdefault(order_id, {
                    "ticket": order_id,
                    "symbol": symbol,
                    "type": side,
                    "size": int(size),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                })
                order.update(result)
                order["native_stop_attached"] = bool(
                    result.get("native_stop_attached")
                    or result.get("protective_sl_order_id")
                )
                order["native_target_attached"] = bool(
                    result.get("native_target_attached")
                    or result.get("protective_tp_order_id")
                )
                order["supports_stop_modify"] = bool(result.get("native_stop_attached"))
            sl_ok, tp_ok = self.query_broker_protection(
                order_id, stop_loss=stop_loss, take_profit=take_profit,
            )
            self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
            bot_logger.info(
                f"Rithmic protective orders ensured for {order_id} {symbol}: "
                f"SL={'yes' if sl_ok else 'no'} TP={'yes' if tp_ok else 'no'} (broker-checked)"
            )
            return sl_ok and tp_ok
        except Exception as e:
            error_logger.error(f"Rithmic ensure_protective_orders error: {e}")
            return False

    def _protection_leg_status(self, ticket: str, *, broker: bool = False) -> tuple:
        """Return (sl_ok, tp_ok). When broker=True, query Rithmic working orders."""
        order_id = str(ticket)
        if broker and self._connected:
            with self._order_lock:
                order = self._orders.get(order_id, {})
                stop_loss = float(order.get("stop_loss") or 0)
                take_profit = float(order.get("take_profit") or 0)
            return self.query_broker_protection(
                order_id, stop_loss=stop_loss, take_profit=take_profit,
            )
        with self._order_lock:
            order = self._orders.get(order_id, {})
            sl_ok = bool(
                order.get("native_stop_attached") or order.get("protective_sl_order_id")
            )
            tp_ok = bool(
                order.get("native_target_attached") or order.get("protective_tp_order_id")
            )
            return sl_ok, tp_ok

    def query_broker_protection(
        self,
        ticket: str,
        stop_loss: float = 0,
        take_profit: float = 0,
    ) -> tuple:
        """Query Rithmic for live SL/TP legs (native bracket + fallback orders)."""
        if not self._connected:
            return False, False
        order_id = str(ticket)
        with self._order_lock:
            order = self._orders.get(order_id, {})
            if stop_loss <= 0:
                stop_loss = float(order.get("stop_loss") or 0)
            if take_profit <= 0:
                take_profit = float(order.get("take_profit") or 0)
        try:
            return self._run_sync(
                self._async_query_broker_protection(order_id, stop_loss, take_profit),
                timeout=12,
            )
        except Exception as e:
            error_logger.error(f"Rithmic query_broker_protection error for {order_id}: {e}")
            return self._protection_leg_status(order_id)

    def _sync_protection_flags_from_broker(
        self,
        order_id: str,
        sl_ok: bool,
        tp_ok: bool,
    ) -> None:
        """Align local SL/TP flags with what is actually working on the broker."""
        with self._order_lock:
            order = self._orders.get(str(order_id))
            if not order:
                return
            if not sl_ok:
                order["native_stop_attached"] = False
                order.pop("protective_sl_order_id", None)
            if not tp_ok:
                order["native_target_attached"] = False
                order.pop("protective_tp_order_id", None)

    def _log_protection_verified(
        self,
        ticket: str,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        order_id = str(ticket)
        with self._order_lock:
            order = self._orders.get(order_id, {})
            if order.get("protection_verified_logged"):
                return
            sl = float(order.get("stop_loss") or stop_loss)
            tp = float(order.get("take_profit") or take_profit)
            mode = order.get("bracket_mode", "unknown")
            if order:
                order["protection_verified_logged"] = True
        bot_logger.info(
            f"✅ VERIFIED: SL @ {sl:.2f} | TP @ {tp:.2f} on Rithmic "
            f"(order {order_id}, mode={mode})"
        )

    def verify_and_ensure_protection(
        self,
        ticket: str,
        symbol: str,
        side: str,
        size: int,
        stop_loss: float,
        take_profit: float,
        max_attempts: int = 5,
    ) -> Tuple[bool, bool, bool]:
        """Retry until broker SL/TP confirmed.

        Returns (sl_ok, tp_ok, already_closed). When already_closed is True the
        entry bracket completed on the broker during verify — caller must not
        track an open position.
        """
        if not self._connected:
            return False, False, False
        order_id = str(ticket)
        delays = (1.0, 1.5, 2.0, 2.5, 3.0)

        broker_pos = self._broker_symbol_has_position(symbol)
        if broker_pos is False and self._order_within_entry_grace(order_id):
            bot_logger.info(
                f"Rithmic {order_id} {symbol}: list_positions flat during entry grace — "
                f"continuing protection verify (SL/TP may still be settling)"
            )
            broker_pos = None
        if broker_pos is False:
            cancelled = self.cancel_all_bot_orders(symbol)
            if cancelled:
                bot_logger.info(
                    f"Rithmic {order_id} {symbol}: flat before verify — cancelled "
                    f"{cancelled} working bot_* order(s)"
                )
            return False, False, False

        for attempt in range(max_attempts):
            if self._run_sync(
                self._async_confirmed_flat_after_entry(
                    order_id, stop_loss=stop_loss, take_profit=take_profit,
                ),
                timeout=8,
            ):
                fill_info = self.confirm_bracket_exit_fill(
                    order_id, symbol, stop_loss, take_profit,
                )
                if fill_info and fill_info.get("confirmed"):
                    bot_logger.info(
                        f"Rithmic {order_id}: bracket exit already confirmed during verify "
                        f"({fill_info.get('leg')} @ {fill_info.get('exit_price'):.2f})"
                    )
                    return True, True, True
                bot_logger.warning(
                    f"Rithmic {order_id}: position flat during verify but bracket fill "
                    f"not confirmed — not treating as closed"
                )
                return False, False, False

            sl_ok, tp_ok = self.query_broker_protection(
                order_id, stop_loss=stop_loss, take_profit=take_profit,
            )
            self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
            if sl_ok and tp_ok:
                self._log_protection_verified(order_id, stop_loss, take_profit)
                return True, True, False

            bot_logger.warning(
                f"Rithmic protection verify {order_id} attempt {attempt + 1}/{max_attempts}: "
                f"SL={'yes' if sl_ok else 'MISSING'} TP={'yes' if tp_ok else 'MISSING'} — "
                f"submitting missing leg(s)"
            )
            self.ensure_protective_orders(
                ticket=order_id,
                symbol=symbol,
                side=side,
                size=int(size),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            if attempt < max_attempts - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])

        if self._run_sync(
            self._async_confirmed_flat_after_entry(
                order_id, stop_loss=stop_loss, take_profit=take_profit,
            ),
            timeout=8,
        ):
            fill_info = self.confirm_bracket_exit_fill(
                order_id, symbol, stop_loss, take_profit,
            )
            if fill_info and fill_info.get("confirmed"):
                bot_logger.info(
                    f"Rithmic {order_id}: bracket exit confirmed after verify retries "
                    f"({fill_info.get('leg')} @ {fill_info.get('exit_price'):.2f})"
                )
                return True, True, True
            bot_logger.warning(
                f"Rithmic {order_id}: flat after verify but bracket fill not confirmed"
            )
            return False, False, False

        sl_ok, tp_ok = self.query_broker_protection(
            order_id, stop_loss=stop_loss, take_profit=take_profit,
        )
        self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
        if sl_ok and tp_ok:
            self._log_protection_verified(order_id, stop_loss, take_profit)
            return True, True, False

        if sl_ok and not tp_ok:
            bot_logger.warning(
                f"Rithmic {symbol} order {order_id}: SL verified but TP still MISSING after "
                f"{max_attempts} attempts — position is stop-protected; TP retry continues in scan loop"
            )
            return True, False, False

        error_logger.error(
            f"🚨 CRITICAL: {symbol} order {order_id} UNPROTECTED after {max_attempts} attempts — "
            f"SL={'yes' if sl_ok else 'MISSING'} TP={'yes' if tp_ok else 'MISSING'} — "
            f"MANUAL INTERVENTION REQUIRED"
        )
        return sl_ok, tp_ok, False

    def aggressive_repair_protection(
        self,
        ticket: str,
        symbol: str,
        side: str,
        size: int,
        stop_loss: float,
        take_profit: float,
        max_attempts: int = _SCAN_PROTECTION_REPAIR_RETRIES,
    ) -> Tuple[bool, bool]:
        """Re-submit missing SL/TP when list_positions confirms exposure is still open."""
        if not self._connected:
            return False, False
        order_id = str(ticket)
        delays = (0.5, 1.0, 1.5, 2.0)

        for attempt in range(max(1, max_attempts)):
            pos_open = self._broker_symbol_has_position(symbol)
            if pos_open is False:
                bot_logger.info(
                    f"Rithmic repair {order_id} {symbol}: broker flat — skip protective repair"
                )
                return True, True

            sl_ok, tp_ok = self.query_broker_protection(
                order_id, stop_loss=stop_loss, take_profit=take_profit,
            )
            self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
            if sl_ok and tp_ok:
                return True, True

            missing = []
            if stop_loss > 0 and not sl_ok:
                missing.append("SL")
            if take_profit > 0 and not tp_ok:
                missing.append("TP")
            bot_logger.warning(
                f"⚠️ UNPROTECTED {symbol} order {order_id}: "
                f"missing {', '.join(missing) or 'legs'} — "
                f"re-submitting SL/TP (repair {attempt + 1}/{max_attempts})"
            )
            self.ensure_protective_orders(
                ticket=order_id,
                symbol=symbol,
                side=side,
                size=int(size),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            if attempt < max_attempts - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])

        sl_ok, tp_ok = self.query_broker_protection(
            order_id, stop_loss=stop_loss, take_profit=take_profit,
        )
        self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
        return sl_ok, tp_ok

    def has_broker_protection(self, ticket: str, *, verify_broker: bool = False) -> bool:
        """True when both SL and TP are confirmed (optionally via live broker query)."""
        order_id = str(ticket)
        if verify_broker and self._connected:
            sl_ok, tp_ok = self.query_broker_protection(order_id)
            self._sync_protection_flags_from_broker(order_id, sl_ok, tp_ok)
            return sl_ok and tp_ok
        sl_ok, tp_ok = self._protection_leg_status(order_id)
        return sl_ok and tp_ok

    @staticmethod
    def _rithmic_response_ok(response: Any) -> bool:
        """True when the last Rithmic response rp_code is success ('0')."""
        if response is None:
            return False
        items = response if isinstance(response, list) else [response]
        if not items:
            return False
        data = _response_to_dict_safe(items[-1])
        rp_code = data.get("rp_code")
        if rp_code is None:
            last = items[-1]
            rp_code = getattr(last, "rp_code", None)
        if not rp_code:
            return True
        if isinstance(rp_code, (list, tuple)):
            return len(rp_code) > 0 and str(rp_code[0]) == "0"
        return str(rp_code) == "0"

    async def _async_get_broker_position_for_symbol(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """Signed net qty and exact broker contract from list_positions."""
        empty = {"net": 0, "broker_symbol": "", "exchange": "CME"}
        if not self._client or not symbol:
            return empty
        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed for {symbol}: {e}")
            return {**empty, "error": str(e)}
        for p in positions or []:
            data = _response_to_dict_safe(p)
            if not self._position_matches_symbol(data, symbol):
                continue
            broker_sym = str(data.get("symbol") or "")
            exchange = str(data.get("exchange") or "CME")
            net = self._position_qty_from_broker_data(data)
            if broker_sym:
                self._resolved_contracts[symbol] = broker_sym
            return {
                "net": net,
                "broker_symbol": broker_sym,
                "exchange": exchange,
            }
        return empty

    async def _async_cancel_all_working_bot_orders_for_symbol(
        self,
        symbol: str,
    ) -> int:
        """Cancel working bot_* entry and protective legs for symbol."""
        if not self._client or not symbol:
            return 0
        if self.symbol_has_preserved_bot_orders(symbol):
            bot_logger.info(
                f"Skipping bulk cancel for {symbol} — entry grace or active bot entry tracking"
            )
            return 0
        try:
            orders = await self._client.list_orders(
                account_id=self._get_account_id(),
            ) or []
        except Exception as e:
            error_logger.error(
                f"Rithmic list_orders failed cancelling bot orders for {symbol}: {e}"
            )
            return 0

        cancelled = 0
        for order in orders:
            data = _response_to_dict_safe(order)
            our_sym = self._reverse_resolve(data.get("symbol", ""))
            if our_sym != symbol:
                continue
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            if not tag.startswith("bot_"):
                continue
            status = str(data.get("status") or "")
            if not self._is_working_order_status(status):
                continue
            parent_entry = self._protective_parent_entry_id(tag)
            if parent_entry and self._order_within_entry_grace(parent_entry):
                continue
            if self._is_bot_entry_user_tag(tag) and self._order_within_entry_grace(tag):
                continue
            cancel_id = tag if tag.startswith("bot_") else str(
                data.get("order_id") or data.get("basket_id") or tag
            )
            is_protective = (
                tag.endswith("_sl")
                or tag.endswith("_tp")
                or tag.endswith("_tp_lit")
                or "_sl" in tag
                or "_tp" in tag
            )
            if is_protective:
                ok = await self._async_cancel_protective_leg(
                    cancel_id,
                    symbol=symbol,
                    reason="pre_flatten",
                    require_flat=False,
                    cancel_path="pre_flatten",
                )
            else:
                ok = await self._async_cancel_working_bot_order(
                    cancel_id,
                    symbol=symbol,
                    reason="pre_flatten",
                )
            if ok:
                cancelled += 1
        if cancelled:
            bot_logger.info(
                f"Cancelled {cancelled} working bot_* order(s) for {symbol}"
            )
        return cancelled

    async def _async_try_exit_position(
        self,
        broker_sym: str,
        exchange: str,
    ) -> bool:
        if not broker_sym or not self._client:
            return False
        try:
            response = await self._client.exit_position(
                symbol=broker_sym,
                exchange=exchange,
                account_id=self._get_account_id(),
            )
            ok = self._rithmic_response_ok(response)
            if not ok:
                last = response[-1] if isinstance(response, list) and response else response
                error_logger.error(
                    f"Rithmic exit_position rejected for {broker_sym}@{exchange}: "
                    f"{_response_to_dict_safe(last)}"
                )
            return ok
        except Exception as e:
            error_logger.error(
                f"Rithmic exit_position failed for {broker_sym}@{exchange}: {e}"
            )
            return False

    async def _async_submit_flatten_market_order(
        self,
        *,
        symbol: str,
        broker_sym: str,
        exchange: str,
        net_qty: int,
    ) -> bool:
        """Flatten with an explicit market order on the closing side."""
        from async_rithmic import OrderType, TransactionType

        qty = abs(int(net_qty))
        if qty <= 0:
            return True
        if not broker_sym or not self._client:
            return False

        transaction = (
            TransactionType.BUY if net_qty < 0 else TransactionType.SELL
        )
        order_id = f"bot_flatten_{uuid.uuid4().hex[:10]}"
        close_label = "BUY" if net_qty < 0 else "SELL"
        bot_logger.warning(
            f"Rithmic FLATTEN MARKET: {order_id} {close_label} {qty} "
            f"{symbol} ({broker_sym}@{exchange})"
        )
        try:
            responses = await self._client.submit_order(
                order_id=order_id,
                symbol=broker_sym,
                exchange=exchange,
                qty=qty,
                transaction_type=transaction,
                order_type=OrderType.MARKET,
                account_id=self._get_account_id(),
            )
            if not self._rithmic_response_ok(responses):
                return False
            filled = await self._async_wait_entry_filled(order_id, max_wait=12.0)
            if filled is None:
                error_logger.error(
                    f"Flatten market order {order_id} for {symbol} did not fill"
                )
                return False
            return True
        except Exception as e:
            error_logger.error(
                f"Flatten market order failed for {symbol} ({broker_sym}): {e}"
            )
            return False

    async def _async_flatten_symbol(
        self,
        symbol: str,
        *,
        max_attempts: int = 5,
        verify_delay: float = 1.5,
        cancel_working_first: bool = True,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "symbol": symbol,
            "flat": False,
            "cancelled_bot": 0,
            "initial_net": 0,
            "final_net": 0,
            "broker_symbol": "",
            "exchange": "",
            "attempts": [],
        }
        if not self._client or not symbol:
            result["error"] = "not_connected"
            return result

        if cancel_working_first:
            result["cancelled_bot"] = (
                await self._async_cancel_all_working_bot_orders_for_symbol(symbol)
            )
            if result["cancelled_bot"]:
                await asyncio.sleep(min(verify_delay, 1.0))

        pos = await self._async_get_broker_position_for_symbol(symbol)
        result["initial_net"] = int(pos.get("net") or 0)
        result["final_net"] = result["initial_net"]
        if result["initial_net"] == 0:
            result["flat"] = True
            return result

        _, default_exchange = self._resolve_symbol(symbol)
        for attempt in range(1, max(1, max_attempts) + 1):
            pos = await self._async_get_broker_position_for_symbol(symbol)
            net = int(pos.get("net") or 0)
            result["final_net"] = net
            if net == 0:
                result["flat"] = True
                break

            broker_sym = str(pos.get("broker_symbol") or "")
            exchange = str(pos.get("exchange") or default_exchange)
            if not broker_sym:
                broker_sym, exchange = self._resolve_symbol(symbol)
            result["broker_symbol"] = broker_sym
            result["exchange"] = exchange

            attempt_info: Dict[str, Any] = {
                "attempt": attempt,
                "net_before": net,
                "broker_symbol": broker_sym,
                "close_side": "BUY" if net < 0 else "SELL",
                "qty": abs(net),
            }

            exit_ok = await self._async_try_exit_position(broker_sym, exchange)
            attempt_info["exit_position_ok"] = exit_ok
            await asyncio.sleep(verify_delay)

            pos_after = await self._async_get_broker_position_for_symbol(symbol)
            net_after = int(pos_after.get("net") or 0)
            attempt_info["net_after_exit"] = net_after
            if net_after == 0:
                result["flat"] = True
                result["final_net"] = 0
                result["attempts"].append(attempt_info)
                break

            market_ok = await self._async_submit_flatten_market_order(
                symbol=symbol,
                broker_sym=broker_sym,
                exchange=exchange,
                net_qty=net_after if net_after != 0 else net,
            )
            attempt_info["market_order_ok"] = market_ok
            await asyncio.sleep(verify_delay)

            pos_final = await self._async_get_broker_position_for_symbol(symbol)
            net_final = int(pos_final.get("net") or 0)
            attempt_info["net_after"] = net_final
            result["final_net"] = net_final
            result["attempts"].append(attempt_info)
            if net_final == 0:
                result["flat"] = True
                break

        if result["flat"]:
            with self._order_lock:
                self._orders = {
                    k: v for k, v in self._orders.items()
                    if v.get("symbol") != symbol
                }
        return result

    def cancel_working_bot_orders(self, symbol: str) -> int:
        """Cancel working bot_* orders (entry + protective) for symbol."""
        return self.cancel_all_bot_orders(symbol)

    def cancel_all_bot_orders(self, symbol: str) -> int:
        """Cancel every working order with bot_* user_tag for symbol."""
        if not self._connected or not symbol:
            return 0
        try:
            return int(
                self._run_sync(
                    self._async_cancel_all_working_bot_orders_for_symbol(symbol),
                    timeout=30,
                )
                or 0
            )
        except Exception as e:
            error_logger.error(
                f"Cancel all bot orders failed for {symbol}: {e}"
            )
            return 0

    def cancel_entry_bracket_legs(
        self,
        entry_order_id: str,
        symbol: str,
        *,
        reason: str = "position_closed",
    ) -> int:
        """Cancel working SL/TP legs for one bot entry (safe when position is flat)."""
        if not self._connected or not entry_order_id or not symbol:
            return 0
        try:
            return int(
                self._run_sync(
                    self._async_cancel_remaining_bracket_legs(
                        str(entry_order_id),
                        symbol,
                        reason=reason,
                    ),
                    timeout=20,
                )
                or 0
            )
        except Exception as e:
            error_logger.error(
                f"Cancel entry bracket legs failed for {entry_order_id} {symbol}: {e}"
            )
            return 0

    def cancel_orphan_stops_if_flat(self, symbol: str) -> int:
        """Cancel working bot SL/TP on a flat symbol (orphan bracket cleanup)."""
        if not self._connected or not symbol:
            return 0
        try:
            return int(
                self._run_sync(
                    self._async_cancel_orphan_protective_if_flat(symbol),
                    timeout=25,
                )
                or 0
            )
        except Exception as e:
            error_logger.error(
                f"Cancel orphan stops failed for {symbol}: {e}"
            )
            return 0

    def flatten_symbol(
        self,
        symbol: str,
        *,
        max_attempts: int = 5,
        verify_delay: float = 1.5,
        cancel_working_first: bool = True,
    ) -> Dict[str, Any]:
        """Cancel working bot orders, flatten net exposure, verify until flat."""
        if not self._connected or not symbol:
            return {"symbol": symbol, "flat": False, "error": "not_connected"}
        try:
            timeout = max(45.0, max_attempts * (verify_delay + 8) + 25)
            return (
                self._run_sync(
                    self._async_flatten_symbol(
                        symbol,
                        max_attempts=max_attempts,
                        verify_delay=verify_delay,
                        cancel_working_first=cancel_working_first,
                    ),
                    timeout=timeout,
                )
                or {"symbol": symbol, "flat": False}
            )
        except Exception as e:
            error_logger.error(f"Rithmic flatten_symbol error for {symbol}: {e}")
            return {"symbol": symbol, "flat": False, "error": str(e)}

    def close_position(
        self,
        symbol: Optional[str] = None,
        size: Optional[float] = None,
        ticket: Optional[int] = None,
    ) -> bool:
        if not self._connected:
            return False
        try:
            target_symbol = symbol
            if ticket:
                with self._order_lock:
                    order = self._orders.get(str(ticket))
                    if order:
                        target_symbol = order["symbol"]
            if not target_symbol:
                return False

            result = self.flatten_symbol(target_symbol)
            return bool(result.get("flat"))
        except Exception as e:
            error_logger.error(f"Rithmic close_position error: {e}")
            return False

    def close_all_positions(self) -> bool:
        if not self._connected:
            return False
        try:
            self._run_sync(
                self._client.exit_position(
                    account_id=self._get_account_id(),
                ),
                timeout=10,
            )
            with self._order_lock:
                self._orders.clear()
            return True
        except Exception as e:
            error_logger.error(f"Rithmic close_all error: {e}")
            return False

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        if not self._connected:
            bot_logger.warning("Rithmic modify_position rejected: connector not connected")
            return False
        try:
            order_id = str(ticket)
            kwargs: Dict[str, Any] = {"order_id": order_id}

            with self._order_lock:
                order = self._orders.get(order_id)
            if not order:
                bot_logger.warning(f"Rithmic modify_position rejected: unknown order_id {order_id}")
                return False

            if not order.get("supports_stop_modify", True):
                bot_logger.warning(
                    f"Rithmic modify_position skipped: order {order_id} has no native stop from entry"
                )
                return False

            if sl is None and tp is None:
                bot_logger.warning(f"Rithmic modify_position skipped: no SL/TP provided for {order_id}")
                return False

            spec = get_instrument(order["symbol"])
            if sl is not None:
                stop_ticks = abs(round(
                    (order["entry_price"] - sl) / spec.tick_size
                ))
                kwargs["stop_ticks"] = stop_ticks
                order["stop_loss"] = sl
            if tp is not None:
                target_ticks = abs(round(
                    (tp - order["entry_price"]) / spec.tick_size
                ))
                kwargs["target_ticks"] = target_ticks
                order["take_profit"] = tp

            self._run_sync(
                self._client.modify_order(**kwargs),
                timeout=10,
            )
            bot_logger.info(f"Rithmic order {order_id} modified: SL={sl}, TP={tp}")
            return True
        except Exception as e:
            error_msg = str(e)
            if "No stop loss was set at order creation" in error_msg:
                with self._order_lock:
                    order = self._orders.get(str(ticket))
                    if order is not None:
                        order["supports_stop_modify"] = False
                bot_logger.warning(
                    f"Rithmic order {ticket} does not support stop modification (stop missing at creation)"
                )
                return False
            error_logger.error(f"Rithmic modify_position error: {e}")
            return False

    def update_stop_loss(
        self,
        ticket: str,
        symbol: str,
        side: str,
        size: int,
        new_sl: float,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Tighten stop-loss on broker (native bracket modify or protective cancel/replace)."""
        if not self._connected or new_sl <= 0:
            return False
        order_id = str(ticket)
        with self._order_lock:
            order = self._orders.get(order_id, {})
            tp = float(take_profit if take_profit is not None else order.get("take_profit") or 0)
            supports_modify = bool(order.get("supports_stop_modify", True))
            has_protective = bool(
                order.get("protective_sl_order_id")
                or order.get("bracket_mode") == "protective_fallback"
            )

        if supports_modify and not has_protective:
            return self.modify_position(ticket=order_id, sl=new_sl, tp=tp if tp > 0 else None)

        try:
            ok = self._run_sync(
                self._async_replace_protective_sl(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty=int(size),
                    new_sl=new_sl,
                ),
                timeout=15,
            )
            if ok:
                with self._order_lock:
                    stored = self._orders.setdefault(order_id, {})
                    stored["stop_loss"] = new_sl
                bot_logger.info(
                    f"Rithmic protective SL replaced for {order_id} {symbol} @ {new_sl:.2f}"
                )
            return bool(ok)
        except Exception as e:
            error_logger.error(f"Rithmic update_stop_loss error for {order_id}: {e}")
            return False

    async def _async_replace_protective_sl(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        new_sl: float,
    ) -> bool:
        """Cancel working fallback SL and submit a tighter stop (simulator + protective_fallback)."""
        with self._order_lock:
            order = self._orders.get(order_id, {})
            sl_id = order.get("protective_sl_order_id") or f"{order_id}_sl"
            price_ref = float(order.get("entry_price") or 0)

        if await self._async_is_order_working(sl_id):
            await self._async_cancel_protective_leg(
                sl_id,
                symbol=symbol,
                reason="sl_tighten",
                require_flat=False,
                cancel_path="replace_sl",
            )
            await asyncio.sleep(0.3)

        protective = await self._async_submit_protective_orders(
            symbol=symbol,
            entry_side=side,
            qty=int(qty),
            stop_loss=new_sl,
            take_profit=0,
            prefix=order_id,
            price_ref=price_ref,
        )
        new_sl_id = protective.get("protective_sl_order_id")
        if not new_sl_id:
            return False

        with self._order_lock:
            stored = self._orders.setdefault(order_id, {})
            stored["stop_loss"] = new_sl
            stored["protective_sl_order_id"] = new_sl_id
            stored["native_stop_attached"] = True
        return True

    # ── Position queries ──────────────────────────────────────────

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions with ticket IDs for bot tracking.
        
        Rithmic returns aggregate positions by symbol, so we match against
        our _orders dict to reconstruct individual tickets.
        """
        self._refresh_positions()
        
        result = []
        with self._state_lock:
            # Get symbols that have open positions
            open_symbols = {sym for sym, pos in self._positions.items() if pos.get('size', 0) != 0}
        
        with self._order_lock:
            # Return individual orders as "positions" if their symbol still has net exposure
            for order_id, order_info in self._orders.items():
                order_symbol = order_info.get('symbol', '')
                if order_symbol in open_symbols:
                    # This order is still active (symbol has position)
                    pos_data = self._positions.get(order_symbol, {})
                    result.append({
                        'ticket': order_id,
                        'pair': order_symbol,
                        'symbol': order_symbol,
                        'type': order_info.get('type', 'BUY'),
                        'size': order_info.get('size', 1),
                        'open_price': order_info.get('entry_price', pos_data.get('avg_price', 0)),
                        'sl': order_info.get('stop_loss', 0),
                        'tp': order_info.get('take_profit', 0),
                        'open_time': order_info.get('time', ''),
                        'unrealized_pnl': pos_data.get('unrealized_pnl', 0) / max(1, len([o for o in self._orders.values() if o.get('symbol') == order_symbol])),
                    })
        
        if symbol:
            result = [p for p in result if p.get('symbol') == symbol]
        
        return result

    def get_bot_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.get_open_positions(symbol)

    def get_trade_history(
        self,
        hours: int = 24,
        include_all: bool = False,
    ) -> List[Dict[str, Any]]:
        # Return locally tracked orders for now
        with self._order_lock:
            return list(self._orders.values())

    # ══════════════════════════════════════════════════════════════
    #  INTERNAL: asyncio bridge
    # ══════════════════════════════════════════════════════════════

    def _run_event_loop(self) -> None:
        """Target for the background IO thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro, timeout: float = 15) -> Any:
        """Run an async coroutine from synchronous code, blocking until done."""
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("Rithmic event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ══════════════════════════════════════════════════════════════
    #  INTERNAL: async Rithmic operations
    # ══════════════════════════════════════════════════════════════

    async def _async_connect(self, url: str) -> None:
        from async_rithmic import RithmicClient, OrderPlacement, DataType
        from async_rithmic.objects import RetrySettings

        self._client = RithmicClient(
            user=self._user,
            password=self._password,
            system_name=self._system,
            app_name=self._app_name,
            app_version=self._app_version,
            url=url,
            manual_or_auto=OrderPlacement.AUTO,
            retry_settings=RetrySettings(
                max_retries=self._request_retries,
                timeout=self._request_timeout,
                jitter_range=(1.0, 2.5),
            ),
        )

        # Register callbacks for real-time data
        self._client.on_tick += self._on_tick
        self._client.on_time_bar += self._on_time_bar
        self._client.on_account_pnl_update += self._on_account_pnl
        self._client.on_instrument_pnl_update += self._on_instrument_pnl
        self._client.on_exchange_order_notification += self._on_exchange_order_notification

        # Connect all plants (ticker, order, history, pnl)
        await self._client.connect()

        await self._async_configure_execution()

        # Subscribe to PnL updates
        await self._client.subscribe_to_pnl_updates()

    def _ensure_market_data(self) -> None:
        """Resolve front-month contracts and subscribe to quotes (deferred from connect)."""
        if not self._connected or self._market_data_ready:
            return
        with self._market_data_lock:
            if self._market_data_ready:
                return
            self._seed_contract_overrides()
            try:
                self._run_sync(
                    self._async_ensure_market_data(),
                    timeout=self._market_data_timeout,
                )
            except Exception as e:
                self._warn_front_month_once(
                    "_market_data",
                    "cached/env contract",
                    e,
                    context="market data setup",
                )
                for sym in self._get_watch_symbols():
                    if sym not in self._resolved_contracts:
                        rith_sym, _ = self._resolve_symbol_raw(sym)
                        self._resolved_contracts[sym] = self._contract_overrides.get(
                            sym, rith_sym
                        )
                self._market_data_ready = True

    async def _async_ensure_market_data(self) -> None:
        from async_rithmic import DataType

        if self._market_data_ready:
            return

        self._seed_contract_overrides()

        for sym in self._get_watch_symbols():
            rith_sym, exchange = self._resolve_symbol_raw(sym)
            trading_sym = await self._resolve_front_month_contract(sym, rith_sym, exchange)
            try:
                await self._client.subscribe_to_market_data(
                    trading_sym, exchange,
                    DataType.BBO | DataType.LAST_TRADE,
                )
            except Exception as e:
                self._warn_front_month_once(
                    sym,
                    trading_sym,
                    e,
                    context="market data subscribe",
                )

        self._market_data_ready = True

    async def _resolve_front_month_contract(
        self,
        sym: str,
        rith_sym: str,
        exchange: str,
    ) -> str:
        """Resolve front month once per session; cache success and fallbacks."""
        cached = self._resolved_contracts.get(sym)
        if cached:
            return cached

        override = self._contract_overrides.get(sym)
        if override:
            self._resolved_contracts[sym] = override
            bot_logger.info(f"Using configured contract for {sym} → {override} on {exchange}")
            return override

        last_exc: Optional[BaseException] = None
        for attempt in range(self._front_month_retries):
            try:
                front = await self._client.get_front_month_contract(rith_sym, exchange)
                if front:
                    self._resolved_contracts[sym] = front
                    bot_logger.info(f"Resolved {sym} → {front} on {exchange}")
                    return front
            except Exception as e:
                last_exc = e
                if attempt < self._front_month_retries - 1:
                    await asyncio.sleep(1.0 + attempt)
                    continue

        fallback = rith_sym
        self._resolved_contracts[sym] = fallback
        self._warn_front_month_once(sym, fallback, last_exc, context="front month lookup")
        return fallback

    def _seed_contract_overrides(self) -> None:
        """Pre-load explicit contracts from env so ticker template-113 can be skipped."""
        if self._contracts_seeded:
            return
        self._contracts_seeded = True

        overrides_raw = os.getenv("RITHMIC_CONTRACT_OVERRIDES", "")
        for part in overrides_raw.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            base, contract = (p.strip().upper() for p in part.split("=", 1))
            if base and contract:
                self._contract_overrides[base] = contract

        for raw in os.getenv("TRADING_SYMBOLS", "").split(","):
            parsed = self._parse_explicit_contract(raw.strip())
            if parsed:
                base, contract = parsed
                self._contract_overrides.setdefault(base, contract)

        for sym, contract in self._contract_overrides.items():
            self._resolved_contracts.setdefault(sym, contract)

    @staticmethod
    def _parse_explicit_contract(raw: str) -> Optional[Tuple[str, str]]:
        """Parse NQM6-style symbols from TRADING_SYMBOLS into (base, contract)."""
        if not raw:
            return None
        token = raw.strip().upper()
        if token in RITHMIC_SYMBOL_MAP:
            return None
        match = _FUTURES_CONTRACT_RE.match(token)
        if not match:
            return None
        base = match.group(1).upper()
        contract = token
        if base not in RITHMIC_SYMBOL_MAP and base != "MGC":
            return None
        return base, contract

    def _warn_front_month_once(
        self,
        sym: str,
        fallback: str,
        exc: Optional[BaseException],
        context: str = "front month lookup",
    ) -> None:
        """Log a single user-friendly warning per symbol (no traceback spam)."""
        now = time.time()
        last = self._warned_front_month.get(sym, 0.0)
        if now - last < self._front_month_warn_cooldown:
            return
        self._warned_front_month[sym] = now

        detail = self._format_connect_error(exc) if exc else "unknown error"
        base = sym if sym in RITHMIC_SYMBOL_MAP else sym
        bot_logger.warning(
            f"Rithmic {context} failed for {sym} — using {fallback}. "
            f"Scans continue with cached contract. "
            f"Optional fix: RITHMIC_CONTRACT_OVERRIDES={base}={fallback}. "
            f"({detail})"
        )

    def get_tick_flow(self, symbol: str) -> Dict[str, Any]:
        """Rolling order-flow snapshot for a symbol (BBO + last-trade classified)."""
        return self._tick_flow.get_snapshot(symbol)

    def evaluate_order_flow(
        self,
        direction: str,
        symbol: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check whether rolling flow supports a proposed entry direction."""
        return self._tick_flow.evaluate_entry(direction, symbol, mode=mode)

    async def _on_tick(self, data: dict) -> None:
        """Callback for real-time quote updates (BBO + last trade)."""
        from async_rithmic import DataType
        symbol = data.get("symbol", "")
        # Reverse-map trading symbol back to our symbol
        our_sym = self._reverse_resolve(symbol)

        with self._quote_lock:
            if our_sym not in self._quotes:
                self._quotes[our_sym] = {
                    "bid": 0.0, "ask": 0.0, "last": 0.0,
                    "bid_size": 0.0, "ask_size": 0.0,
                }

            if data.get("data_type") == DataType.BBO:
                bid = data.get("best_bid_price", data.get("bid_price"))
                ask = data.get("best_ask_price", data.get("ask_price"))
                bid_size = data.get("best_bid_size", data.get("bid_size"))
                ask_size = data.get("best_ask_size", data.get("ask_size"))
                if bid is not None:
                    self._quotes[our_sym]["bid"] = float(bid)
                if ask is not None:
                    self._quotes[our_sym]["ask"] = float(ask)
                if bid_size is not None:
                    self._quotes[our_sym]["bid_size"] = float(bid_size)
                if ask_size is not None:
                    self._quotes[our_sym]["ask_size"] = float(ask_size)
            elif data.get("data_type") == DataType.LAST_TRADE:
                if "trade_price" in data:
                    self._quotes[our_sym]["last"] = float(data["trade_price"])

        if data.get("data_type") == DataType.BBO:
            self._tick_flow.record_bbo(our_sym, data)
        elif data.get("data_type") == DataType.LAST_TRADE:
            self._tick_flow.record_trade(our_sym, data)
            trade_price = data.get("trade_price")
            if trade_price is not None:
                try:
                    trade_size = float(
                        data.get("trade_size", data.get("size", 1)) or 1
                    )
                    trade_ts = (
                        data.get("datetime")
                        or data.get("trade_time")
                        or data.get("ssboe_datetime")
                    )
                    self._subminute_agg.record_trade(
                        our_sym,
                        float(trade_price),
                        trade_size,
                        trade_ts,
                    )
                except (TypeError, ValueError):
                    pass

    async def _on_time_bar(self, data: dict) -> None:
        """Callback for real-time time bar updates."""
        symbol = data.get("symbol", "")
        our_sym = self._reverse_resolve(symbol)
        bar_type = data.get("type", "")

        key = f"{our_sym}_{bar_type}"
        bar = {
            "datetime": data.get("bar_end_datetime", datetime.now(timezone.utc)),
            "open": float(data.get("open_price", 0)),
            "high": float(data.get("high_price", 0)),
            "low": float(data.get("low_price", 0)),
            "close": float(data.get("close_price", 0)),
            "volume": int(data.get("volume", 0)),
        }

        with self._bar_lock:
            if key not in self._bars:
                self._bars[key] = deque(maxlen=500)
            self._bars[key].append(bar)

    async def _on_account_pnl(self, response) -> None:
        """Callback for account PnL updates."""
        try:
            data = _response_to_dict_safe(response)
            with self._state_lock:
                if "account_balance" in data:
                    self._account_info["balance"] = float(data["account_balance"])
                if "cash_on_hand" in data:
                    self._account_info["equity"] = float(data["cash_on_hand"])
                elif "margin_balance" in data:
                    self._account_info["equity"] = float(data["margin_balance"])
        except Exception as e:
            error_logger.error(f"Account PnL callback error: {e}")

    async def _on_exchange_order_notification(self, response) -> None:
        """Cancel sibling protective leg when SL or TP fills (OCO-style cleanup)."""
        try:
            from async_rithmic.enums import ExchangeOrderNotificationType

            data = _response_to_dict_safe(response)
            user_tag = str(data.get("user_tag") or getattr(response, "user_tag", "") or "")
            notify_type = data.get("notify_type", getattr(response, "notify_type", None))
            if notify_type != ExchangeOrderNotificationType.FILL:
                return
            if user_tag.endswith("_sl"):
                sibling_id = user_tag[:-3] + "_tp"
                leg = "SL"
                entry_id = user_tag[:-3]
            elif user_tag.endswith("_tp"):
                sibling_id = user_tag[:-3] + "_sl"
                leg = "TP"
                entry_id = user_tag[:-3]
            else:
                return

            if not await self._async_is_leg_filled(user_tag):
                bot_logger.warning(
                    f"Keeping SL/TP — {user_tag} fill notification but leg not confirmed filled"
                )
                return

            fill_snap = await self._async_get_order_snapshot(user_tag)
            fill_price = self._fill_price_from_order(fill_snap, 0)
            self._record_exit_fill(entry_id, leg, fill_price, user_tag)
            bot_logger.info(
                f"Broker-confirmed bracket {leg} fill: entry={entry_id} leg={user_tag} "
                f"price={fill_price:.2f} account={self._get_account_id()} "
                f"route={self.get_trade_route('CME')}"
            )

            with self._order_lock:
                symbol = self._orders.get(entry_id, {}).get("symbol")
            if symbol:
                pos_open = await self._async_broker_has_open_position(symbol)
                if pos_open is None:
                    bot_logger.warning(
                        f"Keeping SL/TP {sibling_id} — broker position unknown after "
                        f"{leg} fill ({user_tag})"
                    )
                    return
                if pos_open:
                    bot_logger.info(
                        f"Keeping SL/TP {sibling_id} — position still open on broker "
                        f"after {leg} fill ({user_tag})"
                    )
                    return

            await self._async_cancel_protective_leg(
                sibling_id,
                symbol=symbol,
                reason=f"oco_{leg.lower()}_filled",
                require_flat=False,
                cancel_path="oco_sibling",
            )
        except Exception as e:
            error_logger.error(f"Protective OCO handler error: {e}")

    async def _async_cancel_protective_leg(
        self,
        leg_order_id: str,
        *,
        symbol: Optional[str] = None,
        reason: str = "unspecified",
        require_flat: bool = True,
        cancel_path: str = "unspecified",
    ) -> bool:
        """Cancel a standalone protective SL/TP order by user_tag."""
        if not leg_order_id or not self._client:
            return False

        if require_flat:
            if not symbol:
                with self._order_lock:
                    for info in self._orders.values():
                        prefix = leg_order_id.rsplit("_", 1)[0]
                        if info.get("protective_sl_order_id") == leg_order_id or (
                            info.get("protective_tp_order_id") == leg_order_id
                        ) or str(info.get("ticket", "")) == prefix:
                            symbol = info.get("symbol")
                            break
            if symbol:
                pos_open = await self._async_broker_has_open_position(symbol)
                if pos_open is None:
                    bot_logger.warning(
                        f"Keeping SL/TP {leg_order_id} — position unknown "
                        f"(cancel blocked, path={cancel_path}, reason={reason})"
                    )
                    return False
                if pos_open:
                    bot_logger.info(
                        f"Keeping SL/TP {leg_order_id} — position still open on broker "
                        f"(cancel blocked, path={cancel_path}, reason={reason})"
                    )
                    return False

        try:
            await self._client.cancel_order(
                order_id=leg_order_id,
                account_id=self._get_account_id(),
            )
            bot_logger.info(
                f"Cancelling SL/TP {leg_order_id} — path={cancel_path}, reason={reason}"
            )
            return True
        except Exception as e:
            if self._is_order_not_found_error(e):
                bot_logger.debug(
                    f"Rithmic cancel {leg_order_id} — already gone ({e})"
                )
                return True
            bot_logger.warning(f"Rithmic could not cancel {leg_order_id}: {e}")
            return False

    async def _async_cancel_remaining_bracket_legs(
        self,
        entry_order_id: str,
        symbol: str,
        order_info: Optional[Dict[str, Any]] = None,
        *,
        reason: str = "position_closed",
    ) -> int:
        """Cancel any working SL/TP legs still open for a bot entry."""
        if not entry_order_id or not symbol:
            return 0
        if self._order_within_entry_grace(entry_order_id):
            bot_logger.info(
                f"Keeping SL/TP for {entry_order_id} — entry grace active "
                f"(path=cancel_remaining_bracket, reason={reason})"
            )
            return 0
        still_open = await self._async_is_entry_still_open(entry_order_id, symbol)
        if still_open is True:
            bot_logger.info(
                f"Keeping SL/TP for {entry_order_id} — entry still open "
                f"(path=cancel_remaining_bracket, reason={reason})"
            )
            return 0
        if still_open is None:
            bot_logger.warning(
                f"Keeping SL/TP for {entry_order_id} — entry state unknown "
                f"(path=cancel_remaining_bracket, reason={reason})"
            )
            return 0
        cancelled = 0
        sl_ids, tp_ids = self._bracket_leg_ids(entry_order_id)
        if order_info:
            for key in ("protective_sl_order_id", "protective_tp_order_id"):
                leg_id = order_info.get(key)
                if leg_id and leg_id not in sl_ids + tp_ids:
                    if leg_id.endswith("_sl") or "_sl" in str(leg_id):
                        sl_ids.append(str(leg_id))
                    else:
                        tp_ids.append(str(leg_id))
        for leg_id in sl_ids + tp_ids:
            if not leg_id or not await self._async_is_order_working(leg_id):
                continue
            ok = await self._async_cancel_protective_leg(
                leg_id,
                symbol=symbol,
                reason=reason,
                require_flat=False,
                cancel_path="cancel_remaining_bracket",
            )
            if ok:
                cancelled += 1
                bot_logger.info(
                    f"Cancelled orphan stop order {leg_id} for {symbol} after exit"
                )
        return cancelled

    async def _async_cancel_orphan_protective_if_flat(self, symbol: str) -> int:
        """Cancel all working bot_* orders when list_positions is flat."""
        if not self._client or not symbol:
            return 0
        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(
                f"Rithmic list_positions failed orphan cleanup for {symbol}: {e}"
            )
            return 0
        if self._list_positions_net_qty(positions, symbol) != 0:
            return 0
        return await self._async_cancel_all_working_bot_orders_for_symbol(symbol)

    async def _async_cancel_entry_protective_orders(
        self,
        entry_order_id: str,
        order_info: Optional[Dict[str, Any]] = None,
        *,
        reason: str = "position_flat",
    ) -> None:
        """Cancel both fallback protective legs when broker confirms position is flat."""
        if order_info is None:
            with self._order_lock:
                order_info = self._orders.get(entry_order_id, {})
        symbol = order_info.get("symbol")
        if symbol:
            pos_open = await self._async_broker_has_open_position(symbol)
            if pos_open is None:
                bot_logger.warning(
                    f"Keeping SL/TP for {entry_order_id} — broker position unknown "
                    f"(cleanup blocked, path=cleanup_entry_protective, reason={reason})"
                )
                return
            if pos_open:
                bot_logger.info(
                    f"Keeping SL/TP for {entry_order_id} — position still open on broker "
                    f"(cleanup blocked, path=cleanup_entry_protective, reason={reason})"
                )
                return
        for key in ("protective_sl_order_id", "protective_tp_order_id"):
            leg_id = order_info.get(key)
            if leg_id:
                await self._async_cancel_protective_leg(
                    leg_id,
                    symbol=symbol,
                    reason=reason,
                    require_flat=False,
                    cancel_path="cleanup_entry_protective",
                )

    async def _on_instrument_pnl(self, response) -> None:
        """Callback for per-instrument PnL updates (positions)."""
        try:
            data = _response_to_dict_safe(response)
            symbol = data.get("symbol", "")
            our_sym = self._reverse_resolve(symbol)
            qty = int(data.get("buy_qty", 0)) - int(data.get("sell_qty", 0))

            with self._state_lock:
                if qty != 0:
                    self._positions[our_sym] = {
                        "symbol": our_sym,
                        "size": qty,
                        "avg_price": float(data.get("avg_open_fill_price", 0)),
                        "unrealized_pnl": float(data.get("open_pnl", 0)),
                        "realized_pnl": float(data.get("closed_pnl", 0)),
                    }
                elif our_sym in self._positions:
                    del self._positions[our_sym]
        except Exception as e:
            error_logger.error(f"Instrument PnL callback error: {e}")

    async def _async_get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int,
        end_time: Optional[datetime] = None,
        lookback_hours_override: Optional[float] = None,
    ) -> Optional[pd.DataFrame]:
        from async_rithmic import TimeBarType

        rith_sym, exchange = self._resolve_symbol(symbol)

        # Map timeframe to Rithmic bar type + period
        if timeframe_minutes < 60:
            bar_type = TimeBarType.MINUTE_BAR
            period = timeframe_minutes
        elif timeframe_minutes < 1440:
            bar_type = TimeBarType.MINUTE_BAR
            period = timeframe_minutes
        else:
            bar_type = TimeBarType.DAILY_BAR
            period = 1

        end_time = end_time or datetime.now(timezone.utc)
        lookback_hours = (
            lookback_hours_override
            if lookback_hours_override is not None
            else float(os.getenv("CANDLE_HISTORY_HOURS", "8"))
        )
        bar_span = timedelta(minutes=timeframe_minutes * num_candles * 3)
        calendar_span = timedelta(hours=lookback_hours)
        start_time = end_time - max(bar_span, calendar_span)

        bars = await self._client.get_historical_time_bars(
            symbol=rith_sym,
            exchange=exchange,
            start_time=start_time,
            end_time=end_time,
            bar_type=bar_type,
            bar_type_periods=period,
        )

        if not bars:
            return None

        rows = []
        for b in bars:
            rows.append({
                "datetime": b.get("bar_end_datetime", b.get("datetime")),
                "open": float(b.get("open_price", 0)),
                "high": float(b.get("high_price", 0)),
                "low": float(b.get("low_price", 0)),
                "close": float(b.get("close_price", 0)),
                "volume": int(b.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df.tail(num_candles).reset_index(drop=True)

    async def _async_get_candles_seconds(
        self,
        symbol: str,
        period_seconds: int,
        num_candles: int,
        end_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        from async_rithmic import TimeBarType

        rith_sym, exchange = self._resolve_symbol(symbol)
        end_time = end_time or datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=period_seconds * num_candles * 3)

        bars = await self._client.get_historical_time_bars(
            symbol=rith_sym,
            exchange=exchange,
            start_time=start_time,
            end_time=end_time,
            bar_type=TimeBarType.SECOND_BAR,
            bar_type_periods=period_seconds,
        )

        if not bars:
            return None

        rows = []
        for b in bars:
            rows.append({
                "datetime": b.get("bar_end_datetime", b.get("datetime")),
                "open": float(b.get("open_price", 0)),
                "high": float(b.get("high_price", 0)),
                "low": float(b.get("low_price", 0)),
                "close": float(b.get("close_price", 0)),
                "volume": int(b.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df.tail(num_candles).reset_index(drop=True)

    @staticmethod
    def _protective_chart_legend(entry_side: str) -> str:
        """Explain how Rithmic labels exit orders on the chart (not a bug)."""
        is_long = entry_side.lower() in ("buy", "long")
        if is_long:
            return (
                "Rithmic chart (LONG): SELL STOP = SL (exit if price falls), "
                "SELL LIMIT = TP (exit if price rises). "
                "Native bracket legs show as bracket; fallback shows as plain stops/limits."
            )
        return (
            "Rithmic chart (SHORT): BUY STOP = SL (exit if price rises), "
            "BUY LIMIT = TP (exit if price falls). "
            "Lucid may label these 'Buy Stop' / 'Buy Limit' — that is correct for a short."
        )

    @staticmethod
    def _bracket_ticks(
        price_ref: float,
        stop_loss: float,
        take_profit: float,
        tick_size: float,
    ) -> tuple:
        """Return (stop_ticks, target_ticks) as positive offsets from fill reference."""
        sl_ticks = 0
        tp_ticks = 0
        if price_ref > 0 and tick_size > 0:
            if stop_loss > 0:
                sl_ticks = max(1, int(round(abs(price_ref - stop_loss) / tick_size)))
            if take_profit > 0:
                tp_ticks = max(1, int(round(abs(take_profit - price_ref) / tick_size)))
        return sl_ticks, tp_ticks

    @staticmethod
    def _round_to_tick(price: float, tick_size: float, *, mode: str = "nearest") -> float:
        """Round price to a valid exchange tick (MNQ/NQ tick_size=0.25)."""
        if tick_size <= 0 or price <= 0:
            return price
        ticks = price / tick_size
        if mode == "up":
            rounded_ticks = math.ceil(ticks - 1e-9)
        elif mode == "down":
            rounded_ticks = math.floor(ticks + 1e-9)
        else:
            rounded_ticks = round(ticks)
        return round(rounded_ticks * tick_size, 6)

    @staticmethod
    def _format_order_snapshot(order: Any) -> str:
        """Compact broker order fields for reject/debug logging."""
        if not order:
            return "order not found"
        fields = (
            "status", "report_type", "text", "completion_reason",
            "price", "trigger_price", "quantity", "transaction_type", "price_type",
        )
        parts = []
        for name in fields:
            value = getattr(order, name, None)
            if value not in (None, "", 0):
                parts.append(f"{name}={value}")
        user_tag = getattr(order, "user_tag", None)
        if user_tag:
            parts.insert(0, f"user_tag={user_tag}")
        return ", ".join(parts) if parts else "no diagnostic fields"

    async def _async_get_order_snapshot(self, order_id: str) -> Optional[Any]:
        if not order_id or not self._client:
            return None
        try:
            return await self._client.get_order(
                order_id=order_id,
                account_id=self._get_account_id(),
            )
        except Exception:
            return None

    @staticmethod
    def _is_filled_order_snapshot(order: Any) -> bool:
        """True when Rithmic reports the order completed via fill (not rejected)."""
        if not order:
            return False
        status = str(getattr(order, "status", "") or "").lower()
        report_type = str(getattr(order, "report_type", "") or "").lower()
        if status in {"rejected", "failed", "failure", "cancelled", "canceled"}:
            return False
        if report_type in {"fill", "filled"}:
            return True
        return status in {"complete", "completed", "filled"}

    async def _async_leg_submission_outcome(
        self,
        order_id: str,
        *,
        wait_sec: float = 0.75,
    ) -> str:
        """Return 'working', 'filled', or 'failed' after a protective-leg submit."""
        if not order_id:
            return "failed"
        await asyncio.sleep(wait_sec)
        if await self._async_is_order_working(order_id):
            return "working"
        order = await self._async_get_order_snapshot(order_id)
        if self._is_filled_order_snapshot(order):
            return "filled"
        if order:
            status = str(getattr(order, "status", "") or "").lower()
            if status in {"rejected", "failed", "failure", "cancelled", "canceled"}:
                return "failed"
        return "failed"

    async def _async_verify_submitted_leg(
        self,
        order_id: str,
        *,
        wait_sec: float = 0.75,
        leg: str = "",
    ) -> bool:
        """Wait briefly, then confirm a protective leg is working or already filled."""
        outcome = await self._async_leg_submission_outcome(
            order_id, wait_sec=wait_sec,
        )
        if outcome == "working":
            return True
        if outcome == "filled":
            order = await self._async_get_order_snapshot(order_id)
            bot_logger.info(
                f"Rithmic protective {leg or 'leg'} {order_id} filled immediately on submit "
                f"({self._format_order_snapshot(order)})"
            )
            return True
        order = await self._async_get_order_snapshot(order_id)
        if order:
            error_logger.error(
                f"Rithmic protective leg {order_id} not working — "
                f"{self._format_order_snapshot(order)}"
            )
        else:
            error_logger.error(
                f"Rithmic protective leg {order_id} not found after submit "
                f"(may have been rejected before appearing in list_orders)"
            )
        return False

    async def _async_is_leg_filled(self, order_id: str) -> bool:
        """True when a protective order user_tag has filled (terminal fill state)."""
        if not order_id:
            return False
        order = await self._async_get_order_snapshot(order_id)
        return self._is_filled_order_snapshot(order)

    def _order_within_entry_grace(self, order_id: str) -> bool:
        """True shortly after order placement while broker position may not be visible yet."""
        ref = self._order_entry_grace_reference_time(order_id)
        if ref is None:
            return False
        age = (datetime.now(timezone.utc) - ref).total_seconds()
        return age < _ENTRY_POSITION_GRACE_SEC

    def entry_protection_grace_active(self, order_id: str) -> bool:
        """Public: entry fill + protective legs may still be settling on Rithmic."""
        return self._order_within_entry_grace(str(order_id))

    def symbol_has_preserved_bot_orders(self, symbol: str) -> bool:
        """True when bulk cancel must not run (grace window or active local entry)."""
        if not symbol:
            return False
        with self._order_lock:
            for oid, info in self._orders.items():
                if info.get("symbol") != symbol:
                    continue
                if self._order_within_entry_grace(oid):
                    return True
                if self._is_bot_entry_user_tag(oid):
                    return True
        return False

    def _order_entry_grace_reference_time(self, order_id: str) -> Optional[datetime]:
        with self._order_lock:
            order = self._orders.get(order_id, {})
        candidates: List[datetime] = []
        for key in ("time", "fill_time", "protective_submit_time"):
            time_str = order.get(key)
            if not time_str:
                continue
            try:
                placed = datetime.fromisoformat(str(time_str))
                if placed.tzinfo is None:
                    placed = placed.replace(tzinfo=timezone.utc)
                candidates.append(placed)
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates)

    @staticmethod
    def _protective_parent_entry_id(tag: str) -> Optional[str]:
        tag = str(tag or "")
        if not tag.startswith("bot_"):
            return None
        for suffix in ("_tp_lit", "_sl", "_tp"):
            if tag.endswith(suffix):
                return tag[:-len(suffix)]
        return None

    @staticmethod
    def _is_order_not_found_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "not found" in msg or "order not found" in msg

    @staticmethod
    def _position_qty_from_broker_data(data: Dict[str, Any]) -> int:
        buy = int(data.get("buy_qty", 0) or 0)
        sell = int(data.get("sell_qty", 0) or 0)
        if buy or sell:
            return buy - sell
        for key in ("net_qty", "quantity", "open_qty", "position_qty", "qty"):
            val = data.get(key)
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
        return 0

    def _position_matches_symbol(self, data: Dict[str, Any], symbol: str) -> bool:
        broker_sym = str(data.get("symbol", "") or "")
        if self._reverse_resolve(broker_sym) == symbol:
            return True
        rith_sym, _ = self._resolve_symbol(symbol)
        if broker_sym == rith_sym:
            return True
        base = RITHMIC_SYMBOL_MAP.get(symbol, (symbol,))[0]
        return (
            broker_sym.upper().startswith(base.upper())
            and len(broker_sym) > len(base)
        )

    def _list_positions_net_qty(self, positions: Any, symbol: str) -> int:
        """Signed net qty for symbol from a list_positions response."""
        for p in positions or []:
            data = _response_to_dict_safe(p)
            if self._position_matches_symbol(data, symbol):
                return self._position_qty_from_broker_data(data)
        return 0

    async def _async_collect_bot_entry_ids(self, symbol: str) -> List[str]:
        """Bot entry user_tags from local cache plus unfilled entries in list_orders."""
        entry_ids = set()
        with self._order_lock:
            for oid, info in self._orders.items():
                if info.get("symbol") == symbol and self._is_bot_entry_user_tag(oid):
                    entry_ids.add(str(oid))
        if not self._client:
            return sorted(entry_ids)
        try:
            orders = await self._client.list_orders(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_orders failed collecting entries for {symbol}: {e}")
            return sorted(entry_ids)
        for order in orders or []:
            data = _response_to_dict_safe(order)
            our_sym = self._reverse_resolve(data.get("symbol", ""))
            if our_sym != symbol:
                continue
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            if not self._is_bot_entry_user_tag(tag):
                continue
            # Filled/cancelled historical bot_* tags are not collected unless locally tracked.
            if self._is_working_order_status(data.get("status", "")):
                entry_ids.add(tag)
        return sorted(entry_ids)

    async def _async_is_entry_still_open(
        self,
        order_id: str,
        symbol: str,
    ) -> Optional[bool]:
        """Per-entry open check using list_positions + leg state (not symbol-wide inference)."""
        if not order_id or not symbol:
            return False

        entry = await self._async_get_order_snapshot(order_id)
        if entry and self._is_working_order_status(getattr(entry, "status", "")):
            return None
        if not entry or not self._is_filled_order_snapshot(entry):
            return False

        sl_ids, tp_ids = self._bracket_leg_ids(order_id)
        for leg_id in sl_ids + tp_ids:
            if leg_id and await self._async_is_leg_filled(leg_id):
                return False

        if self._order_within_entry_grace(order_id):
            return None

        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed for {symbol}: {e}")
            return None

        if self._list_positions_net_qty(positions, symbol) != 0:
            return True

        for leg_id in sl_ids + tp_ids:
            if leg_id and await self._async_is_order_working(leg_id):
                return True

        with self._order_lock:
            info = self._orders.get(order_id, {})
            sl = float(info.get("stop_loss") or 0)
            tp = float(info.get("take_profit") or 0)
        if sl > 0 or tp > 0:
            sl_ok, tp_ok = await self._async_query_broker_protection(order_id, sl, tp)
            if sl_ok or tp_ok:
                return True

        return False

    async def _async_cancel_stale_entry_legs(
        self,
        order_id: str,
        symbol: str,
    ) -> int:
        """Cancel working SL/TP legs left behind by a flat stale filled entry."""
        sl_ids, tp_ids = self._bracket_leg_ids(order_id)
        cancelled = 0
        for leg_id in sl_ids + tp_ids:
            if not leg_id or not await self._async_is_order_working(leg_id):
                continue
            ok = await self._async_cancel_protective_leg(
                leg_id,
                symbol=symbol,
                reason="stale_entry_reconcile",
                require_flat=False,
                cancel_path="reconcile_stale_entry",
            )
            if ok:
                cancelled += 1
        return cancelled

    async def _async_reconcile_symbol_exposure(self, symbol: str) -> Dict[str, Any]:
        """Reconcile broker net qty vs bot entry tags; drop stale local order tracking."""
        if not self._client or not symbol:
            return {
                "broker_net": 0,
                "list_positions_net": 0,
                "open_entries": 0,
                "open_order_ids": [],
                "working_entries": 0,
                "stale_cleared": 0,
                "exposure_source": "flat",
                "tag_inferred_entries": 0,
            }

        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed reconciling {symbol}: {e}")
            return {
                "broker_net": 0,
                "list_positions_net": 0,
                "open_entries": 0,
                "open_order_ids": [],
                "working_entries": 0,
                "stale_cleared": 0,
                "exposure_source": "flat",
                "tag_inferred_entries": 0,
                "error": str(e),
            }

        list_positions_net = self._list_positions_net_qty(positions, symbol)
        broker_net = abs(list_positions_net)
        open_order_ids: List[str] = []
        stale_cleared = 0
        legs_cancelled = 0

        for order_id in await self._async_collect_bot_entry_ids(symbol):
            still_open = await self._async_is_entry_still_open(order_id, symbol)
            if still_open is True:
                open_order_ids.append(order_id)
                continue
            if still_open is not False:
                continue
            legs_cancelled += await self._async_cancel_stale_entry_legs(order_id, symbol)
            with self._order_lock:
                removed = self._orders.pop(order_id, None)
            if removed:
                stale_cleared += 1
                bot_logger.info(
                    f"Reconcile {symbol}: cleared stale entry {order_id} "
                    f"(list_positions flat, no working SL/TP)"
                )

        working_entries = await self._async_count_working_entry_orders(symbol)
        open_entries = len(open_order_ids)

        if (
            broker_net == 0
            and open_entries == 0
            and working_entries == 0
            and not self.symbol_has_preserved_bot_orders(symbol)
        ):
            legs_cancelled += await self._async_cancel_orphan_protective_if_flat(symbol)

        if broker_net == 0 and open_entries == 0 and working_entries == 0:
            with self._state_lock:
                if symbol in self._positions:
                    del self._positions[symbol]

        if broker_net > 0:
            exposure_source = "list_positions"
        elif open_entries > 0 or working_entries > 0:
            exposure_source = "inferred_tags"
        else:
            exposure_source = "flat"

        return {
            "broker_net": broker_net,
            "list_positions_net": list_positions_net,
            "open_entries": open_entries,
            "open_order_ids": open_order_ids,
            "working_entries": working_entries,
            "stale_cleared": stale_cleared,
            "legs_cancelled": legs_cancelled,
            "exposure_source": exposure_source,
            "tag_inferred_entries": open_entries if broker_net == 0 else 0,
        }

    def reconcile_symbol_exposure(self, symbol: str) -> Dict[str, Any]:
        """Sync wrapper — reconcile list_positions + list_orders for one symbol."""
        if not self._connected or not symbol:
            return {
                "broker_net": 0,
                "list_positions_net": 0,
                "open_entries": 0,
                "open_order_ids": [],
                "working_entries": 0,
                "stale_cleared": 0,
                "exposure_source": "flat",
                "tag_inferred_entries": 0,
            }
        try:
            return self._run_sync(
                self._async_reconcile_symbol_exposure(symbol),
                timeout=20,
            ) or {}
        except Exception as e:
            error_logger.error(f"reconcile_symbol_exposure failed for {symbol}: {e}")
            return {
                "broker_net": 0,
                "list_positions_net": 0,
                "open_entries": 0,
                "open_order_ids": [],
                "working_entries": 0,
                "stale_cleared": 0,
                "exposure_source": "flat",
                "tag_inferred_entries": 0,
                "error": str(e),
            }

    async def _async_infer_open_from_filled_entries(
        self,
        symbol: str,
        order_id: Optional[str] = None,
    ) -> Optional[bool]:
        """Infer open exposure when list_positions is empty (simulator route fallback)."""
        if not self._client or not symbol:
            return None

        if order_id:
            still_open = await self._async_is_entry_still_open(order_id, symbol)
            if still_open is True:
                return True
            if still_open is False:
                return False
            return None

        open_count = 0
        unknown = False
        for oid in await self._async_collect_bot_entry_ids(symbol):
            still_open = await self._async_is_entry_still_open(oid, symbol)
            if still_open is True:
                open_count += 1
            elif still_open is None:
                unknown = True
        if open_count > 0:
            return True
        if unknown:
            return None
        return False

    def _broker_symbol_has_position(self, symbol: str) -> Optional[bool]:
        """Net exposure via list_positions, with filled-entry fallback when PnL plant is empty."""
        if not self._client or not symbol:
            return None
        try:
            positions = self._run_sync(
                self._client.list_positions(account_id=self._get_account_id()),
                timeout=10,
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed for {symbol}: {e}")
            return None

        if self._list_positions_net_qty(positions, symbol) != 0:
            return True

        if not self._using_simulator_route:
            return False

        try:
            inferred = self._run_sync(
                self._async_infer_open_from_filled_entries(symbol),
                timeout=10,
            )
            if inferred is True:
                return True
        except Exception as e:
            error_logger.error(
                f"Rithmic entry-fill position inference failed for {symbol}: {e}"
            )
        return False

    def get_symbol_list_positions_net(self, symbol: str) -> Optional[int]:
        """Signed net qty from list_positions only. None if query failed."""
        if not self._client or not symbol:
            return None
        try:
            positions = self._run_sync(
                self._client.list_positions(account_id=self._get_account_id()),
                timeout=10,
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed for {symbol}: {e}")
            return None
        return self._list_positions_net_qty(positions, symbol)

    def get_symbol_net_qty(self, symbol: str) -> Optional[int]:
        """Absolute net contract qty from list_positions (not inferred from stale bot tags)."""
        net = self.get_symbol_list_positions_net(symbol)
        if net is None:
            return None
        return abs(net)

    async def _async_cancel_working_bot_order(
        self,
        order_id: str,
        *,
        symbol: str,
        reason: str = "cancel",
    ) -> bool:
        """Cancel a working bot entry order (not SL/TP leg)."""
        if not order_id or not self._client:
            return False
        try:
            await self._client.cancel_order(
                order_id=order_id,
                account_id=self._get_account_id(),
            )
            bot_logger.info(
                f"Cancelled working bot order {order_id} {symbol} ({reason})"
            )
            return True
        except Exception as e:
            if self._is_order_not_found_error(e):
                bot_logger.debug(
                    f"Cancel bot order {order_id} — already gone ({reason}): {e}"
                )
                return True
            error_logger.error(
                f"Cancel working bot order {order_id} failed ({reason}): {e}"
            )
            return False

    @staticmethod
    def _is_bot_entry_user_tag(tag: str) -> bool:
        tag = str(tag or "")
        if not tag.startswith("bot_"):
            return False
        return not any(tag.endswith(suffix) for suffix in ("_sl", "_tp", "_tp_lit"))

    async def _async_count_working_entry_orders(self, symbol: str) -> int:
        """Unfilled bot entry parent orders for symbol (excludes SL/TP legs)."""
        if not self._client or not symbol:
            return 0
        try:
            orders = await self._client.list_orders(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_orders failed for {symbol}: {e}")
            return 0

        count = 0
        for order in orders or []:
            data = _response_to_dict_safe(order)
            our_sym = self._reverse_resolve(data.get("symbol", ""))
            if our_sym != symbol:
                continue
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            if not self._is_bot_entry_user_tag(tag):
                continue
            if self._is_filled_order_snapshot(order):
                continue
            if not self._is_working_order_status(data.get("status", "")):
                continue
            count += 1
        return count

    def count_working_entry_orders(self, symbol: str) -> int:
        """Sync wrapper — unfilled bot entry orders for symbol."""
        if not self._client or not symbol:
            return 0
        try:
            return self._run_sync(
                self._async_count_working_entry_orders(symbol),
                timeout=10,
            )
        except Exception as e:
            error_logger.error(f"Working entry order count failed for {symbol}: {e}")
            return 0

    async def _async_broker_has_open_position(self, symbol: str) -> Optional[bool]:
        """Async net exposure check via list_positions. None if query failed."""
        if not self._client or not symbol:
            return None
        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception as e:
            error_logger.error(f"Rithmic list_positions failed for {symbol}: {e}")
            return None

        if self._list_positions_net_qty(positions, symbol) != 0:
            return True

        if not self._using_simulator_route:
            return False

        inferred = await self._async_infer_open_from_filled_entries(symbol)
        return True if inferred is True else False

    @staticmethod
    def _fill_price_from_order(order: Any, fallback: float = 0) -> float:
        for attr in ("avg_fill_price", "fill_price", "price"):
            val = getattr(order, attr, None)
            if val in (None, "", 0):
                continue
            try:
                price = float(val)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return fallback

    async def _async_sync_positions_from_broker(self) -> bool:
        """Refresh cached net exposure from Rithmic PnL plant (not just callbacks)."""
        if not self._client:
            return False
        try:
            positions = await self._client.list_positions(
                account_id=self._get_account_id(),
            )
        except Exception:
            return False

        updated: Dict[str, Any] = {}
        for p in positions:
            data = _response_to_dict_safe(p)
            symbol = data.get("symbol", "")
            our_sym = self._reverse_resolve(symbol)
            qty = self._position_qty_from_broker_data(data)
            if qty != 0:
                updated[our_sym] = {
                    "symbol": our_sym,
                    "size": qty,
                    "avg_price": float(data.get("avg_open_fill_price", 0)),
                    "unrealized_pnl": float(data.get("open_pnl", 0)),
                }

        with self._state_lock:
            for sym in list(self._positions.keys()):
                if sym not in updated:
                    del self._positions[sym]
            self._positions.update(updated)
        return True

    async def _async_wait_entry_filled(
        self,
        order_id: str,
        max_wait: float = 10.0,
    ) -> Optional[Any]:
        """Wait until the entry user_tag reports a fill (or terminal failure)."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            order = await self._async_get_order_snapshot(order_id)
            if order and self._is_filled_order_snapshot(order):
                return order
            if order:
                status = str(getattr(order, "status", "") or "").lower()
                if status in {"rejected", "failed", "failure", "cancelled", "canceled"}:
                    return None
            await asyncio.sleep(0.25)
        return None

    async def _async_is_entry_position_open(self, order_id: str) -> Optional[bool]:
        """Return True/False for net exposure; None when fill/position state is still unknown."""
        with self._order_lock:
            symbol = self._orders.get(order_id, {}).get("symbol")
        if not symbol:
            return None

        if not await self._async_sync_positions_from_broker():
            return None

        with self._state_lock:
            pos = self._positions.get(symbol)
            if pos and int(pos.get("size", 0)) != 0:
                return True

        entry = await self._async_get_order_snapshot(order_id)
        if entry and self._is_working_order_status(getattr(entry, "status", "")):
            return None
        if entry and not self._is_filled_order_snapshot(entry):
            status = str(getattr(entry, "status", "") or "").lower()
            if status in {"rejected", "failed", "failure", "cancelled", "canceled"}:
                return False
            return None

        if self._order_within_entry_grace(order_id):
            return None

        inferred = await self._async_infer_open_from_filled_entries(symbol, order_id=order_id)
        if inferred is True:
            return True
        return False

    async def _async_confirmed_flat_after_entry(
        self,
        order_id: str,
        stop_loss: float = 0,
        take_profit: float = 0,
    ) -> bool:
        """True only when entry filled, grace elapsed, and broker shows no open position."""
        pos_open = await self._async_is_entry_position_open(order_id)
        if pos_open is not False:
            return False

        entry = await self._async_get_order_snapshot(order_id)
        if not self._is_filled_order_snapshot(entry):
            return False

        if self._order_within_entry_grace(order_id):
            return False

        if stop_loss > 0 or take_profit > 0:
            with self._order_lock:
                order = self._orders.get(order_id, {})
                sl_ids = [
                    order.get("protective_sl_order_id"),
                    f"{order_id}_sl",
                ]
                tp_ids = [
                    order.get("protective_tp_order_id"),
                    f"{order_id}_tp",
                    f"{order_id}_tp_lit",
                ]

            sl_filled = stop_loss <= 0
            tp_filled = take_profit <= 0
            if stop_loss > 0:
                sl_filled = any([
                    await self._async_is_leg_filled(leg_id)
                    for leg_id in sl_ids
                    if leg_id
                ])
            if take_profit > 0:
                tp_filled = any([
                    await self._async_is_leg_filled(leg_id)
                    for leg_id in tp_ids
                    if leg_id
                ])
            if stop_loss > 0 and take_profit > 0:
                return sl_filled or tp_filled
            if stop_loss > 0:
                return sl_filled
            if take_profit > 0:
                return tp_filled
        return True

    async def _async_submit_protective_leg(
        self,
        *,
        leg: str,
        order_id: str,
        symbol: str,
        exchange: str,
        qty: int,
        close_side: Any,
        close_label: str,
        is_long: bool,
        account_id: str,
        order_type: Any,
        price: float = 0,
        trigger_price: float = 0,
        duration: Any,
    ) -> bool:
        """Submit one protective leg; log Rithmic reject details; return True if working."""
        from async_rithmic import RithmicErrorResponse

        if await self._async_is_order_working(order_id):
            bot_logger.info(
                f"Rithmic {leg} {order_id} already working — skip duplicate submit"
            )
            return True
        if await self._async_is_leg_filled(order_id):
            bot_logger.info(
                f"Rithmic {leg} {order_id} already filled — skip duplicate submit"
            )
            return True

        kwargs: Dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "qty": qty,
            "transaction_type": close_side,
            "order_type": order_type,
            "account_id": account_id,
            "duration": duration,
        }
        if price > 0:
            kwargs["price"] = price
        if trigger_price > 0:
            kwargs["trigger_price"] = trigger_price

        try:
            responses = await self._client.submit_order(**kwargs)
            if responses:
                last = responses[-1]
                rp_code = getattr(last, "rp_code", None)
                if rp_code and rp_code[0] != "0":
                    error_logger.error(
                        f"Rithmic {leg} submit rp_code={list(rp_code)} for {order_id} "
                        f"{symbol}: {_response_to_dict_safe(last)}"
                    )
                    return False
        except RithmicErrorResponse as e:
            error_logger.error(
                f"Rithmic rejected {leg} {order_id} {symbol} "
                f"price={price:.2f} trigger={trigger_price:.2f}: {e}"
            )
            return False
        except Exception as e:
            error_logger.error(
                f"Failed to submit {leg} for {symbol} ({order_id}): {e}"
            )
            return False

        if await self._async_verify_submitted_leg(order_id, leg=leg):
            if leg == "SL":
                bot_logger.info(
                    f"SL STOP verified @ {trigger_price:.2f} — {order_id} {symbol} "
                    f"({close_label} STOP closes {'long' if is_long else 'short'})"
                )
            else:
                bot_logger.info(
                    f"TP LIMIT verified @ {price:.2f} — {order_id} {symbol} "
                    f"({close_label} LIMIT closes {'long' if is_long else 'short'})"
                )
            return True
        return False

    async def _async_try_attach_native_target(
        self,
        entry_order_id: str,
        symbol: str,
        take_profit: float,
        price_ref: float,
    ) -> bool:
        """Attach native target_ticks via modify API when a bracket target row already exists."""
        if not self._client or take_profit <= 0 or price_ref <= 0:
            return False
        order = await self._async_get_order_snapshot(entry_order_id)
        if not order or not self._is_filled_order_snapshot(order):
            return False

        _, current_target = await self._client.get_stop_and_target(
            basket_id=order.basket_id,
            account_id=order.account_id,
        )
        if current_target is not None:
            return True

        # Template 332 updates an existing bracket target row (level = current target_ticks).
        # There is no supported API to create the first target row post-fill — use LIMIT fallback.
        bot_logger.info(
            f"Native target not present for {entry_order_id} {symbol} — "
            f"standalone LIMIT fallback required (332 cannot create first target row)"
        )
        return False

    @staticmethod
    def _is_working_order_status(status: str) -> bool:
        normalized = str(status or "").lower().strip()
        if not normalized:
            return False
        return normalized not in _PROTECTIVE_TERMINAL_STATUSES

    async def _async_is_order_working(self, order_id: str) -> bool:
        """True when a user_tag order exists on Rithmic and is not terminal."""
        if not order_id or not self._client:
            return False
        try:
            order = await self._client.get_order(
                order_id=order_id,
                account_id=self._get_account_id(),
            )
            if not order:
                return False
            return self._is_working_order_status(getattr(order, "status", ""))
        except Exception:
            return False

    async def _async_native_bracket_legs(self, entry_order_id: str) -> tuple:
        """Return (sl_ok, tp_ok) from template-330 native bracket tables."""
        if not self._client:
            return False, False
        order = await self._client.get_order(
            order_id=entry_order_id,
            account_id=self._get_account_id(),
        )
        if not order:
            return False, False
        stop_ticks, target_ticks = await self._client.get_stop_and_target(
            basket_id=order.basket_id,
            account_id=order.account_id,
        )
        return stop_ticks is not None, target_ticks is not None

    async def _async_query_broker_protection(
        self,
        order_id: str,
        stop_loss: float,
        take_profit: float,
        *,
        assume_flat: bool = True,
    ) -> tuple:
        """Check Rithmic for working SL/TP (native bracket legs + fallback orders)."""
        sl_ok = stop_loss <= 0
        tp_ok = take_profit <= 0
        if sl_ok and tp_ok:
            return True, True

        native_sl, native_tp = await self._async_native_bracket_legs(order_id)
        if stop_loss > 0 and native_sl:
            sl_ok = True
        if take_profit > 0 and native_tp:
            tp_ok = True

        with self._order_lock:
            order = self._orders.get(order_id, {})
            sl_ids = [
                order.get("protective_sl_order_id"),
                f"{order_id}_sl",
            ]
            tp_ids = [
                order.get("protective_tp_order_id"),
                f"{order_id}_tp",
                f"{order_id}_tp_lit",
            ]

        if stop_loss > 0 and not sl_ok:
            for leg_id in sl_ids:
                if leg_id and await self._async_is_order_working(leg_id):
                    sl_ok = True
                    break

        if take_profit > 0 and not tp_ok:
            for leg_id in tp_ids:
                if leg_id and (
                    await self._async_is_order_working(leg_id)
                    or await self._async_is_leg_filled(leg_id)
                ):
                    tp_ok = True
                    break

        if stop_loss > 0 and not sl_ok:
            for leg_id in sl_ids:
                if leg_id and await self._async_is_leg_filled(leg_id):
                    sl_ok = True
                    break

        if assume_flat:
            pos_open = await self._async_is_entry_position_open(order_id)
            if pos_open is False and not self._order_within_entry_grace(order_id):
                entry = await self._async_get_order_snapshot(order_id)
                if self._is_filled_order_snapshot(entry):
                    if stop_loss > 0 and not sl_ok:
                        for leg_id in sl_ids:
                            if leg_id and await self._async_is_leg_filled(leg_id):
                                sl_ok = True
                                break
                    if take_profit > 0 and not tp_ok:
                        for leg_id in tp_ids:
                            if leg_id and await self._async_is_leg_filled(leg_id):
                                tp_ok = True
                                break

        return sl_ok, tp_ok

    async def _async_verify_native_bracket(
        self,
        order_id: str,
        max_attempts: int = 4,
    ) -> tuple:
        """Query Rithmic for template-330 bracket legs on a filled entry order."""
        delays = (0.75, 1.0, 1.5, 2.0)
        order = None
        for attempt in range(max_attempts):
            await asyncio.sleep(delays[min(attempt, len(delays) - 1)])
            order = await self._client.get_order(order_id=order_id)
            if not order:
                continue
            if not self._is_filled_order_snapshot(order):
                if attempt < max_attempts - 1:
                    bot_logger.info(
                        f"Rithmic bracket verify {order_id} attempt {attempt + 1}/{max_attempts}: "
                        f"entry not filled yet — retrying"
                    )
                continue
            stop_ticks, target_ticks = await self._client.get_stop_and_target(
                basket_id=order.basket_id,
                account_id=order.account_id,
            )
            stop_ok = stop_ticks is not None
            target_ok = target_ticks is not None
            if stop_ok and target_ok:
                return True, True, order
            if attempt < max_attempts - 1:
                bot_logger.info(
                    f"Rithmic bracket verify {order_id} attempt {attempt + 1}/{max_attempts}: "
                    f"SL={'yes' if stop_ok else 'no'} TP={'yes' if target_ok else 'no'} — retrying"
                )
        if order:
            stop_ticks, target_ticks = await self._client.get_stop_and_target(
                basket_id=order.basket_id,
                account_id=order.account_id,
            )
            return stop_ticks is not None, target_ticks is not None, order
        return False, False, None

    async def _async_submit_protective_orders(
        self,
        symbol: str,
        entry_side: str,
        qty: int,
        stop_loss: float,
        take_profit: float,
        prefix: str,
        price_ref: float = 0,
    ) -> Dict[str, Any]:
        """Standalone STOP + LIMIT orders to close an open position (chart-visible fallback)."""
        from async_rithmic import OrderDuration, OrderType, TransactionType

        rith_sym, exchange = self._resolve_symbol(symbol)
        account_id = self._get_account_id()
        spec = get_instrument(symbol)
        tick_size = spec.tick_size
        is_long = entry_side.lower() in ("buy", "long")
        close_side = TransactionType.SELL if is_long else TransactionType.BUY
        close_label = "SELL" if is_long else "BUY"
        out: Dict[str, Any] = {}
        sl_ok = stop_loss <= 0
        tp_ok = take_profit <= 0

        if stop_loss > 0:
            sl_mode = "down" if is_long else "up"
            sl_price = self._round_to_tick(stop_loss, tick_size, mode=sl_mode)
            if abs(sl_price - stop_loss) >= tick_size * 0.01:
                bot_logger.info(
                    f"SL rounded to tick: {stop_loss:.2f} → {sl_price:.2f} ({symbol})"
                )
            sl_id = f"{prefix}_sl"
            if not await self._async_is_order_working(sl_id) and not await self._async_is_leg_filled(sl_id):
                if await self._async_submit_protective_leg(
                    leg="SL",
                    order_id=sl_id,
                    symbol=rith_sym,
                    exchange=exchange,
                    qty=qty,
                    close_side=close_side,
                    close_label=close_label,
                    is_long=is_long,
                    account_id=account_id,
                    order_type=OrderType.STOP_MARKET,
                    trigger_price=sl_price,
                    duration=OrderDuration.GTC,
                ):
                    out["protective_sl_order_id"] = sl_id
                    sl_ok = True
            elif await self._async_is_order_working(sl_id) or await self._async_is_leg_filled(sl_id):
                out["protective_sl_order_id"] = sl_id
                sl_ok = True

        if take_profit > 0:
            tp_mode = "up" if is_long else "down"
            tp_price = self._round_to_tick(take_profit, tick_size, mode=tp_mode)
            if abs(tp_price - take_profit) >= tick_size * 0.01:
                bot_logger.info(
                    f"TP rounded to tick: {take_profit:.2f} → {tp_price:.2f} ({symbol})"
                )
            tp_id = f"{prefix}_tp"
            if await self._async_is_order_working(tp_id) or await self._async_is_leg_filled(tp_id):
                out["protective_tp_order_id"] = tp_id
                tp_ok = True
            else:
                tp_ok = await self._async_submit_protective_leg(
                    leg="TP",
                    order_id=tp_id,
                    symbol=rith_sym,
                    exchange=exchange,
                    qty=qty,
                    close_side=close_side,
                    close_label=close_label,
                    is_long=is_long,
                    account_id=account_id,
                    order_type=OrderType.LIMIT,
                    price=tp_price,
                    duration=OrderDuration.GTC,
                )
            if tp_ok:
                out["protective_tp_order_id"] = tp_id
                if await self._async_is_leg_filled(tp_id):
                    out["tp_filled_immediately"] = True
                    pos_open = await self._async_broker_has_open_position(symbol)
                    if pos_open is False:
                        await self._async_cancel_protective_leg(
                            f"{prefix}_sl",
                            symbol=symbol,
                            reason="tp_filled_immediately",
                            cancel_path="immediate_tp_fill",
                        )
                    elif pos_open is True:
                        bot_logger.info(
                            f"Keeping SL/TP {prefix}_sl — position still open on broker "
                            f"after immediate TP fill ({tp_id})"
                        )
                    else:
                        bot_logger.warning(
                            f"Keeping SL/TP {prefix}_sl — broker position unknown "
                            f"after immediate TP fill ({tp_id})"
                        )
            elif price_ref > 0 and await self._async_try_attach_native_target(
                prefix, symbol, tp_price, price_ref,
            ):
                out["native_target_attached"] = True
            else:
                bot_logger.warning(
                    f"TP LIMIT not resting for {symbol} @ {tp_price:.2f} — "
                    f"will retry on next protection verify"
                )

        if stop_loss > 0 and take_profit > 0 and sl_ok and not tp_ok and not out.get("native_target_attached"):
            if not self._order_within_entry_grace(prefix):
                sl_id = f"{prefix}_sl"
                cancelled = await self._async_cancel_protective_leg(
                    sl_id,
                    symbol=symbol,
                    reason="partial_bracket_tp_failed",
                    require_flat=False,
                    cancel_path="partial_bracket_cleanup",
                )
                if cancelled:
                    out.pop("protective_sl_order_id", None)
                    sl_ok = False
                    bot_logger.warning(
                        f"Cancelled orphan SL {sl_id} for {symbol} — TP submit failed "
                        f"(partial bracket cleanup)"
                    )
            else:
                bot_logger.info(
                    f"Keeping SL for {prefix} — TP submit pending during entry grace ({symbol})"
                )
        elif stop_loss > 0 and take_profit > 0 and tp_ok and not sl_ok:
            if not self._order_within_entry_grace(prefix):
                tp_id = f"{prefix}_tp"
                cancelled = await self._async_cancel_protective_leg(
                    tp_id,
                    symbol=symbol,
                    reason="partial_bracket_sl_failed",
                    require_flat=False,
                    cancel_path="partial_bracket_cleanup",
                )
                if cancelled:
                    out.pop("protective_tp_order_id", None)
                    tp_ok = False
                    bot_logger.warning(
                        f"Cancelled orphan TP {tp_id} for {symbol} — SL submit failed "
                        f"(partial bracket cleanup)"
                    )
            else:
                bot_logger.info(
                    f"Keeping TP for {prefix} — SL submit pending during entry grace ({symbol})"
                )

        if out:
            bot_logger.info(self._protective_chart_legend(entry_side))
        with self._order_lock:
            stored = self._orders.get(prefix, {})
            if stored and (sl_ok or tp_ok or out):
                stored["protective_submit_time"] = datetime.now(timezone.utc).isoformat()
                self._orders[prefix] = stored
        return out

    async def _async_ensure_protective_orders(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Dict[str, Any]]:
        broker_pos = await self._async_broker_has_open_position(symbol)
        if broker_pos is False and self._order_within_entry_grace(order_id):
            bot_logger.info(
                f"Rithmic {order_id} {symbol}: list_positions flat during entry grace — "
                f"continuing protective submit"
            )
            broker_pos = None
        if broker_pos is False:
            cancelled = await self._async_cancel_all_working_bot_orders_for_symbol(symbol)
            if cancelled:
                bot_logger.info(
                    f"Rithmic {order_id} {symbol}: flat — cancelled {cancelled} "
                    f"working bot_* order(s) before protective submit"
                )
            return {
                "native_stop_attached": False,
                "native_target_attached": False,
                "bracket_mode": "position_flat",
            }

        broker_sl, broker_tp = await self._async_query_broker_protection(
            order_id, stop_loss, take_profit,
        )
        if broker_sl and broker_tp:
            return {
                "native_stop_attached": broker_sl,
                "native_target_attached": broker_tp,
                "bracket_mode": "broker_verified",
            }

        pos_open = await self._async_is_entry_position_open(order_id)
        if pos_open is False and not self._order_within_entry_grace(order_id):
            broker_pos = await self._async_broker_has_open_position(symbol)
            if broker_pos is True:
                bot_logger.warning(
                    f"Rithmic {order_id} {symbol}: position cache stale (flat) but "
                    f"list_positions shows open — submitting protective orders"
                )
                pos_open = True
            elif broker_pos is None:
                bot_logger.warning(
                    f"Rithmic {order_id} {symbol}: position unknown after flat cache — "
                    f"will attempt protective submit"
                )
            else:
                entry = await self._async_get_order_snapshot(order_id)
                if self._is_filled_order_snapshot(entry):
                    bot_logger.info(
                        f"Rithmic {order_id}: position flat — skip protective submit"
                    )
                    return {
                        "native_stop_attached": broker_sl,
                        "native_target_attached": broker_tp,
                        "bracket_mode": "position_flat",
                    }

        need_sl = stop_loss > 0 and not broker_sl
        need_tp = take_profit > 0 and not broker_tp
        extra: Dict[str, Any] = {}
        if need_sl or need_tp:
            with self._order_lock:
                stored = self._orders.get(order_id, {})
                price_ref = float(
                    stored.get("entry_price") or stored.get("price_ref") or 0
                )
            protective = await self._async_submit_protective_orders(
                symbol=symbol,
                entry_side=side,
                qty=qty,
                stop_loss=stop_loss if need_sl else 0,
                take_profit=take_profit if need_tp else 0,
                prefix=order_id,
                price_ref=price_ref,
            )
            extra.update(protective)

        broker_sl, broker_tp = await self._async_query_broker_protection(
            order_id, stop_loss, take_profit,
        )
        return {
            "native_stop_attached": broker_sl,
            "native_target_attached": broker_tp,
            "bracket_mode": "protective_fallback" if extra else "broker_verified",
            **extra,
        }

    async def _async_place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_id: str,
    ) -> Any:
        from async_rithmic import OrderType, TransactionType

        rith_sym, exchange = self._resolve_symbol(symbol)
        transaction = (
            TransactionType.BUY
            if side.lower() in ("buy", "long")
            else TransactionType.SELL
        )

        kwargs: Dict[str, Any] = {
            "order_id": order_id,
            "symbol": rith_sym,
            "exchange": exchange,
            "qty": qty,
            "transaction_type": transaction,
            "order_type": OrderType.MARKET,
            "account_id": self._get_account_id(),
        }

        spec = get_instrument(symbol)
        tick_size = spec.tick_size
        is_long = side.lower() in ("buy", "long")
        if stop_loss > 0:
            sl_mode = "down" if is_long else "up"
            stop_loss = self._round_to_tick(stop_loss, tick_size, mode=sl_mode)
        if take_profit > 0:
            tp_mode = "up" if is_long else "down"
            take_profit = self._round_to_tick(take_profit, tick_size, mode=tp_mode)
        # Prefer signal entry over cached quote — cache may be empty or stale at entry time.
        price_ref = entry_price if entry_price > 0 else self._get_cached_price(symbol)
        if price_ref <= 0:
            price_ref = self._get_cached_price(symbol)

        placed_at = datetime.now(timezone.utc).isoformat()
        with self._order_lock:
            self._orders[order_id] = {
                "ticket": order_id,
                "symbol": symbol,
                "type": side,
                "size": qty,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "price_ref": price_ref,
                "time": placed_at,
            }

        sl_ticks, tp_ticks = self._bracket_ticks(
            price_ref, stop_loss, take_profit, spec.tick_size,
        )
        if stop_loss > 0 and sl_ticks > 0:
            kwargs["stop_ticks"] = sl_ticks
        if take_profit > 0 and tp_ticks > 0:
            kwargs["target_ticks"] = tp_ticks

        if stop_loss > 0 and "stop_ticks" not in kwargs:
            bot_logger.error(
                f"Cannot compute stop_ticks for {symbol} (price_ref={price_ref}) — "
                f"will attempt protective orders after fill"
            )
        if take_profit > 0 and "target_ticks" not in kwargs:
            bot_logger.warning(
                f"Cannot compute target_ticks for {symbol} (price_ref={price_ref})"
            )

        result = await self._client.submit_order(**kwargs)
        bracket_mode = "native_bracket" if kwargs.get("stop_ticks") or kwargs.get("target_ticks") else "entry_only"

        filled_order = await self._async_wait_entry_filled(order_id)
        if filled_order:
            fill_price = self._fill_price_from_order(filled_order, price_ref)
            if fill_price > 0 and abs(fill_price - price_ref) >= tick_size * 0.01:
                bot_logger.info(
                    f"Entry fill price for {order_id} {symbol}: "
                    f"signal {price_ref:.2f} → fill {fill_price:.2f} (using fill for bracket ref)"
                )
                price_ref = fill_price
                sl_ticks, tp_ticks = self._bracket_ticks(
                    price_ref, stop_loss, take_profit, tick_size,
                )
            with self._order_lock:
                stored = self._orders.get(order_id, {})
                stored["entry_price"] = price_ref
                stored["price_ref"] = price_ref
                stored["fill_time"] = datetime.now(timezone.utc).isoformat()
                if sl_ticks > 0:
                    stored["stop_ticks"] = sl_ticks
                if tp_ticks > 0:
                    stored["target_ticks"] = tp_ticks
                self._orders[order_id] = stored
            await self._async_sync_positions_from_broker()
        else:
            bot_logger.warning(
                f"Rithmic entry {order_id} {symbol} not confirmed filled within wait window — "
                f"will still attempt bracket/protective attach"
            )

        stop_ok, target_ok, _ = await self._async_verify_native_bracket(order_id)
        native_stop = stop_ok
        native_target = target_ok
        extra: Dict[str, Any] = {}

        if (stop_loss > 0 and not stop_ok) or (take_profit > 0 and not target_ok):
            bot_logger.warning(
                f"Rithmic native bracket (template 330) not visible for {order_id} "
                f"after retries (SL={'yes' if stop_ok else 'no'} TP={'yes' if target_ok else 'no'}) "
                f"— submitting separate protective exit orders. "
                f"{self._protective_chart_legend(side)}"
            )
            protective = await self._async_submit_protective_orders(
                symbol=symbol,
                entry_side=side,
                qty=qty,
                stop_loss=stop_loss if not stop_ok else 0,
                take_profit=take_profit if not target_ok else 0,
                prefix=order_id,
                price_ref=price_ref,
            )
            extra.update(protective)
            bracket_mode = "protective_fallback"

        broker_sl, broker_tp = await self._async_query_broker_protection(
            order_id, stop_loss, take_profit,
        )
        native_stop = broker_sl
        native_target = broker_tp

        return {
            "result": result,
            "native_stop_attached": native_stop,
            "native_target_attached": native_target,
            "price_ref": price_ref,
            "stop_ticks": kwargs.get("stop_ticks"),
            "target_ticks": kwargs.get("target_ticks"),
            "bracket_mode": bracket_mode,
            **extra,
        }

    # ══════════════════════════════════════════════════════════════
    #  INTERNAL: helpers
    # ══════════════════════════════════════════════════════════════

    def _get_watch_symbols(self) -> List[str]:
        """Return symbols to subscribe, from config or defaults."""
        if self._symbols_to_watch:
            return self._symbols_to_watch
        from config.strategy_config import PAIRS, ASSET_CLASS
        if ASSET_CLASS == "futures":
            return [s for s in PAIRS if s in RITHMIC_SYMBOL_MAP]
        return []

    def _resolve_symbol_raw(self, symbol: str) -> tuple:
        """Map our symbol name → (rithmic_base_symbol, exchange)."""
        entry = RITHMIC_SYMBOL_MAP.get(symbol)
        if entry:
            return entry
        # Pass through if already a rithmic symbol
        return (symbol, "CME")

    def _resolve_symbol(self, symbol: str) -> tuple:
        """Map our symbol → (resolved_trading_symbol, exchange)."""
        _, exchange = self._resolve_symbol_raw(symbol)
        if symbol in self._resolved_contracts:
            return (self._resolved_contracts[symbol], exchange)

        override = self._contract_overrides.get(symbol)
        if override:
            self._resolved_contracts[symbol] = override
            return (override, exchange)

        parsed = self._parse_explicit_contract(symbol)
        if parsed:
            base, contract = parsed
            if base == symbol or symbol in RITHMIC_SYMBOL_MAP:
                self._resolved_contracts[symbol] = contract
                return (contract, exchange)

        trading_sym = RITHMIC_SYMBOL_MAP.get(symbol, (symbol,))[0]
        return (trading_sym, exchange)

    def _reverse_resolve(self, trading_symbol: str) -> str:
        """Reverse-map a Rithmic trading symbol back to our symbol."""
        for our_sym, rith_sym in self._resolved_contracts.items():
            if rith_sym == trading_symbol:
                return our_sym
        # Try matching base symbol (e.g., "MESH6" starts with "MES")
        for our_sym, (base, _) in RITHMIC_SYMBOL_MAP.items():
            if trading_symbol.startswith(base) and len(trading_symbol) > len(base):
                return our_sym
        return trading_symbol

    @staticmethod
    def _is_simulator_route_name(route: str) -> bool:
        name = str(route or "").lower()
        return "simulator" in name or name in {"sim", "paper", "demo"}

    @staticmethod
    def _is_pro004_account(account_id: str) -> bool:
        return "PRO004" in str(account_id or "").upper()

    def _pro004_sim_mode_active(self) -> bool:
        acct = self._get_account_id() or self._account_id_override or ""
        return (
            self._using_simulator_route
            and self._allow_simulator_route
            and self._is_pro004_account(acct)
        )

    def _print_pro004_sim_mode_banner(self) -> None:
        msg = (
            "PRO004 SIM MODE — orders via Rithmic simulator until Lucid enables live route"
        )
        banner = f"\n   {'═' * 60}\n   {msg}\n   {'═' * 60}\n"
        bot_logger.warning(msg)
        print(banner)

    @staticmethod
    def _route_sort_key(route_obj: Any, *, preferred: str = "", reject_simulator: bool) -> tuple:
        route_name = str(getattr(route_obj, "trade_route", "") or "")
        lower = route_name.lower()
        is_sim = RithmicConnector._is_simulator_route_name(route_name)
        is_default = bool(getattr(route_obj, "is_default", False))
        if preferred and lower == preferred.lower():
            return (0, 0 if not is_sim else 1, route_name)
        if is_default and not is_sim:
            return (1, 0, route_name)
        if not is_sim and reject_simulator:
            return (2, 0, route_name)
        if is_default:
            return (3, 1 if is_sim else 0, route_name)
        if not is_sim:
            return (4, 0, route_name)
        return (5, 0, route_name)

    def _prioritize_trade_routes(self, routes: List[Any]) -> List[Any]:
        """Reorder async_rithmic trade routes so submit_order picks the intended route."""
        preferred = self._trade_route_override
        reject_sim = self._live_mode and not self._allow_simulator_route and not preferred
        by_exchange: Dict[str, List[Any]] = {}
        for route_obj in routes or []:
            exchange = str(getattr(route_obj, "exchange", "") or "")
            by_exchange.setdefault(exchange, []).append(route_obj)

        prioritized: List[Any] = []
        for exchange in sorted(by_exchange):
            sorted_routes = sorted(
                by_exchange[exchange],
                key=lambda r: self._route_sort_key(
                    r, preferred=preferred, reject_simulator=reject_sim,
                ),
            )
            prioritized.extend(sorted_routes)
            if sorted_routes:
                chosen = str(getattr(sorted_routes[0], "trade_route", "") or "")
                self._routes_by_exchange[exchange] = chosen
        return prioritized

    def _select_account_id(self, accounts: List[Any]) -> Optional[str]:
        if not accounts:
            return None
        if self._account_id_override:
            for acct in accounts:
                acct_id = str(getattr(acct, "account_id", "") or "")
                if acct_id == self._account_id_override:
                    return acct_id
            bot_logger.error(
                f"RITHMIC_ACCOUNT_ID={self._account_id_override!r} not found — "
                f"available: {[getattr(a, 'account_id', '') for a in accounts]}"
            )
        if len(accounts) == 1:
            return str(getattr(accounts[0], "account_id", "") or "") or None

        if self._live_mode:
            for acct in accounts:
                acct_id = str(getattr(acct, "account_id", "") or "")
                acct_name = str(getattr(acct, "account_name", "") or "").lower()
                if "sim" not in acct_id.lower() and "sim" not in acct_name:
                    return acct_id
        return str(getattr(accounts[0], "account_id", "") or "") or None

    async def _async_configure_execution(self) -> None:
        """Resolve account + trade routes after connect; async_rithmic defaults to routes[0]."""
        order_plant = self._client.plants["order"]
        accounts = list(order_plant.accounts or [])
        routes = list(order_plant.trade_routes or [])

        acct_lines = [
            f"{getattr(a, 'account_id', '?')} "
            f"({getattr(a, 'account_name', '') or 'unnamed'})"
            for a in accounts
        ]
        route_lines = [
            f"{getattr(r, 'exchange', '?')}: {getattr(r, 'trade_route', '?')} "
            f"default={getattr(r, 'is_default', False)} "
            f"status={getattr(r, 'status', '')}"
            for r in routes
        ]
        bot_logger.info(
            f"Rithmic accounts ({len(accounts)}): {acct_lines or ['none']}"
        )
        bot_logger.info(
            f"Rithmic trade routes ({len(routes)}): {route_lines or ['none']}"
        )

        self._selected_account_id = self._select_account_id(accounts)
        if self._selected_account_id and accounts:
            order_plant.accounts = sorted(
                accounts,
                key=lambda a: 0 if getattr(a, "account_id", "") == self._selected_account_id else 1,
            )

        if routes:
            order_plant.trade_routes = self._prioritize_trade_routes(routes)

        self._using_simulator_route = any(
            self._is_simulator_route_name(route_name)
            for route_name in self._routes_by_exchange.values()
        )

        primary_route = self._routes_by_exchange.get("CME") or next(
            iter(self._routes_by_exchange.values()), ""
        )
        bot_logger.info(
            f"Rithmic execution: account_id={self._selected_account_id} "
            f"trade_route(CME)={primary_route} live_mode={self._live_mode}"
        )
        if self._using_simulator_route and self._live_mode:
            if self._pro004_sim_mode_active():
                self._print_pro004_sim_mode_banner()
            else:
                msg = (
                    "ORDERS GOING TO SIMULATOR NOT LIVE — "
                    f"account_id={self._selected_account_id} trade_route={primary_route}. "
                    "Set RITHMIC_TRADE_ROUTE to your Lucid live route (not 'simulator')."
                )
                bot_logger.error(msg)
                print(f"\n   🚨🚨🚨 {msg} 🚨🚨🚨\n")

    def _get_account_id(self) -> Optional[str]:
        """Return configured account ID (RITHMIC_ACCOUNT_ID or resolved default)."""
        if self._selected_account_id:
            return self._selected_account_id
        if self._client and self._client.accounts:
            return self._client.accounts[0].account_id
        return None

    def confirm_bracket_exit_fill(
        self,
        order_id: str,
        symbol: str,
        stop_loss: float,
        take_profit: float,
        *,
        force_sync: bool = False,
        flat_duration_sec: float = 0.0,
        entry_price: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Return fill details when broker confirms bracket exit; None if unconfirmed."""
        if not self._connected:
            return None
        try:
            return self._run_sync(
                self._async_confirm_bracket_exit_fill(
                    order_id,
                    symbol,
                    stop_loss,
                    take_profit,
                    force_sync=force_sync,
                    flat_duration_sec=flat_duration_sec,
                    entry_price=entry_price,
                ),
                timeout=15,
            )
        except Exception as e:
            error_logger.error(f"confirm_bracket_exit_fill error for {order_id}: {e}")
            return None

    def acknowledge_flat_position(self, order_id: str, symbol: str) -> None:
        """Drop local order tracking after bot force-syncs a broker-flat position."""
        oid = str(order_id)
        try:
            self._run_sync(
                self._async_cancel_remaining_bracket_legs(
                    oid,
                    symbol,
                    reason="ack_flat",
                ),
                timeout=15,
            )
        except Exception as e:
            error_logger.error(
                f"Cancel bracket legs on ack_flat failed for {oid} {symbol}: {e}"
            )
        with self._order_lock:
            removed = self._orders.pop(oid, None)
        if removed:
            bot_logger.info(
                f"🔄 Order {oid} ({symbol}) cleared — bot force-synced flat broker state"
            )
        try:
            cancelled = self._run_sync(
                self._async_cancel_orphan_protective_if_flat(symbol),
                timeout=15,
            )
            if cancelled:
                bot_logger.info(
                    f"{symbol}: cancelled {cancelled} orphan protective order(s) on ack_flat"
                )
        except Exception as e:
            error_logger.error(
                f"Orphan protective cleanup on ack_flat failed for {symbol}: {e}"
            )
        self._refresh_positions()

    def _bracket_leg_ids(self, order_id: str) -> Tuple[List[str], List[str]]:
        with self._order_lock:
            order = self._orders.get(order_id, {})
            sl_ids = [
                order.get("protective_sl_order_id"),
                f"{order_id}_sl",
            ]
            tp_ids = [
                order.get("protective_tp_order_id"),
                f"{order_id}_tp",
                f"{order_id}_tp_lit",
            ]
        return (
            [leg_id for leg_id in sl_ids if leg_id],
            [leg_id for leg_id in tp_ids if leg_id],
        )

    async def _async_classify_protective_leg(self, leg_id: str) -> str:
        """Return filled, working, cancelled, terminal, or missing."""
        if not leg_id:
            return "missing"
        order = await self._async_get_order_snapshot(leg_id)
        if not order:
            return "missing"
        if self._is_filled_order_snapshot(order):
            return "filled"
        status = str(getattr(order, "status", "") or "").lower()
        if status in {"cancelled", "canceled", "rejected", "expired", "inactive"}:
            return "cancelled"
        if self._is_working_order_status(status):
            return "working"
        return "terminal"

    async def _async_diagnose_bracket_legs(
        self,
        order_id: str,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """Inspect SL/TP legs when broker is flat but fill is not confirmed."""
        sl_ids, tp_ids = self._bracket_leg_ids(order_id)
        leg_statuses: Dict[str, str] = {}
        for leg_id in sl_ids + tp_ids:
            leg_statuses[leg_id] = await self._async_classify_protective_leg(leg_id)

        if any(status == "filled" for status in leg_statuses.values()):
            return {"inferred_flat": False}

        if any(status == "working" for status in leg_statuses.values()):
            return {"inferred_flat": False, "legs_working": True}

        if not leg_statuses:
            entry = await self._async_get_order_snapshot(order_id)
            if entry and self._is_filled_order_snapshot(entry):
                with self._order_lock:
                    info = self._orders.get(order_id, {})
                    sl = float(info.get("stop_loss") or stop_loss or 0)
                    tp = float(info.get("take_profit") or take_profit or 0)
                sl_ok, tp_ok = await self._async_query_broker_protection(order_id, sl, tp)
                if sl_ok or tp_ok:
                    return {"inferred_flat": False, "legs_working": True}
                if self._order_within_entry_grace(order_id):
                    return {"inferred_flat": False}
                return {
                    "inferred_flat": True,
                    "sync_reason": "BROKER_SYNC",
                    "leg": "SYNC",
                    "detail": "entry filled, protective legs absent, broker flat",
                }
            return {"inferred_flat": False}

        cancelled = sum(1 for s in leg_statuses.values() if s == "cancelled")
        terminal = sum(1 for s in leg_statuses.values() if s in {"cancelled", "terminal", "missing"})
        if terminal >= len(leg_statuses):
            with self._order_lock:
                info = self._orders.get(order_id, {})
                sl = float(info.get("stop_loss") or stop_loss or 0)
                tp = float(info.get("take_profit") or take_profit or 0)
            sl_ok, tp_ok = await self._async_query_broker_protection(order_id, sl, tp)
            if sl_ok or tp_ok:
                return {"inferred_flat": False, "legs_working": True}
            detail = (
                f"legs cancelled/terminal ({cancelled} cancelled of {len(leg_statuses)})"
            )
            return {
                "inferred_flat": True,
                "sync_reason": "BROKER_SYNC",
                "leg": "SYNC",
                "detail": detail,
                "leg_statuses": leg_statuses,
            }

        return {"inferred_flat": False, "leg_statuses": leg_statuses}

    def _resolve_inferred_exit_price(
        self,
        symbol: str,
        stop_loss: float,
        take_profit: float,
        *,
        entry_price: float = 0.0,
    ) -> float:
        exit_price = self._get_cached_price(symbol)
        if exit_price > 0:
            return exit_price
        if entry_price > 0:
            return entry_price
        if stop_loss > 0 and take_profit > 0:
            return (stop_loss + take_profit) / 2.0
        return stop_loss or take_profit or 0.0

    async def _async_confirm_bracket_exit_fill(
        self,
        order_id: str,
        symbol: str,
        stop_loss: float,
        take_profit: float,
        *,
        force_sync: bool = False,
        flat_duration_sec: float = 0.0,
        entry_price: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        if self._order_within_entry_grace(order_id):
            bot_logger.debug(
                f"Bracket exit {order_id} {symbol}: entry grace active — "
                f"not confirming exit or cancelling SL/TP"
            )
            return None

        entry_open = await self._async_is_entry_still_open(order_id, symbol)
        if entry_open is None:
            return None
        if entry_open:
            return None

        with self._order_lock:
            cached = dict(self._confirmed_exit_fills.get(order_id, {}))
        if cached.get("confirmed"):
            return cached

        exit_price = 0.0
        leg = ""
        sl_ids, tp_ids = self._bracket_leg_ids(order_id)

        for leg_id in sl_ids:
            if leg_id and await self._async_is_leg_filled(leg_id):
                snap = await self._async_get_order_snapshot(leg_id)
                exit_price = self._fill_price_from_order(snap, stop_loss)
                leg = "SL"
                break

        if not leg:
            for leg_id in tp_ids:
                if leg_id and await self._async_is_leg_filled(leg_id):
                    snap = await self._async_get_order_snapshot(leg_id)
                    exit_price = self._fill_price_from_order(snap, take_profit)
                    leg = "TP"
                    break

        if leg:
            fill_from_broker = exit_price > 0
            if not fill_from_broker:
                bot_logger.warning(
                    f"Bracket exit {order_id} {symbol}: leg {leg} filled but no fill price on "
                    f"order snapshot — using bracket level fallback"
                )
                exit_price = take_profit if leg == "TP" else stop_loss

            result = {
                "confirmed": True,
                "inferred": False,
                "exit_price": exit_price,
                "leg": leg,
                "sync_reason": "BROKER_BRACKET",
                "fill_from_broker": fill_from_broker,
                "account_id": self._get_account_id(),
                "trade_route": self.get_trade_route("CME"),
            }
            with self._order_lock:
                self._confirmed_exit_fills[order_id] = result
            await self._async_cancel_remaining_bracket_legs(
                order_id,
                symbol,
                reason="bracket_exit_confirmed",
            )
            return result

        diagnosis = await self._async_diagnose_bracket_legs(
            order_id, stop_loss, take_profit,
        )
        if diagnosis.get("legs_working"):
            route_note = (
                "simulator list_positions often empty despite open fill + SL/TP"
                if self._using_simulator_route
                else "working SL/TP without confirmed fill"
            )
            bot_logger.info(
                f"Bracket exit {order_id} {symbol}: broker flat with working legs — "
                f"keeping protection ({route_note})"
            )
            return None
        if diagnosis.get("inferred_flat"):
            exit_price = self._resolve_inferred_exit_price(
                symbol, stop_loss, take_profit, entry_price=entry_price,
            )
            sync_reason = str(diagnosis.get("sync_reason") or "BROKER_SYNC")
            result = {
                "confirmed": True,
                "inferred": True,
                "exit_price": exit_price,
                "leg": diagnosis.get("leg", "SYNC"),
                "sync_reason": sync_reason,
                "fill_from_broker": exit_price > 0 and self._get_cached_price(symbol) > 0,
                "detail": diagnosis.get("detail", ""),
                "account_id": self._get_account_id(),
                "trade_route": self.get_trade_route("CME"),
            }
            bot_logger.warning(
                f"Bracket exit {order_id} {symbol}: inferred flat — {sync_reason} "
                f"@ {exit_price:.2f} ({diagnosis.get('detail', 'legs terminal without fill')})"
            )
            with self._order_lock:
                self._confirmed_exit_fills[order_id] = result
            await self._async_cancel_remaining_bracket_legs(
                order_id,
                symbol,
                reason="bracket_exit_inferred",
            )
            return result

        if force_sync and flat_duration_sec >= _BROKER_FLAT_FORCE_SYNC_SEC:
            still_open = await self._async_is_entry_still_open(order_id, symbol)
            if still_open is True:
                route_note = (
                    "simulator route — list_positions often empty despite open sim fills"
                    if self._using_simulator_route
                    else "entry filled with working SL/TP"
                )
                bot_logger.warning(
                    f"Bracket exit {order_id} {symbol}: skipping force-sync — "
                    f"{route_note} (flat {flat_duration_sec:.0f}s)"
                )
                return None
            if still_open is None:
                bot_logger.info(
                    f"Bracket exit {order_id} {symbol}: force-sync pending — "
                    f"entry state unknown (flat {flat_duration_sec:.0f}s)"
                )
                return None
            exit_price = self._resolve_inferred_exit_price(
                symbol, stop_loss, take_profit, entry_price=entry_price,
            )
            result = {
                "confirmed": True,
                "inferred": True,
                "force_sync": True,
                "exit_price": exit_price,
                "leg": "UNKNOWN",
                "sync_reason": "UNKNOWN",
                "fill_from_broker": exit_price > 0 and self._get_cached_price(symbol) > 0,
                "detail": f"force-sync after {flat_duration_sec:.0f}s broker-flat",
                "account_id": self._get_account_id(),
                "trade_route": self.get_trade_route("CME"),
            }
            bot_logger.warning(
                f"Bracket exit {order_id} {symbol}: force-sync UNKNOWN @ {exit_price:.2f} "
                f"after {flat_duration_sec:.0f}s flat (no SL/TP fill confirmed)"
            )
            with self._order_lock:
                self._confirmed_exit_fills[order_id] = result
            await self._async_cancel_remaining_bracket_legs(
                order_id,
                symbol,
                reason="bracket_exit_force_sync",
            )
            return result

        bot_logger.warning(
            f"Bracket exit for {order_id} {symbol} — position flat but no SL/TP fill "
            f"confirmed on broker; not reporting BROKER_BRACKET close "
            f"(flat {flat_duration_sec:.0f}s, force_sync={force_sync})"
        )
        return None

    def _record_exit_fill(
        self,
        entry_id: str,
        leg: str,
        fill_price: float,
        leg_order_id: str,
    ) -> None:
        with self._order_lock:
            self._confirmed_exit_fills[entry_id] = {
                "confirmed": True,
                "exit_price": fill_price,
                "leg": leg,
                "leg_order_id": leg_order_id,
                "account_id": self._get_account_id(),
                "trade_route": self.get_trade_route("CME"),
            }

    def _get_cached_price(self, symbol: str) -> float:
        """Get current cached price for a symbol."""
        with self._quote_lock:
            quote = self._quotes.get(symbol, {})
            if quote.get("last", 0) > 0:
                return quote["last"]
            if quote.get("bid", 0) > 0 and quote.get("ask", 0) > 0:
                return (quote["bid"] + quote["ask"]) / 2
        return 0.0

    def _refresh_positions(self) -> None:
        """Sync positions from Rithmic PnL plant."""
        if not self._connected:
            return
        try:
            positions = self._run_sync(
                self._client.list_positions(account_id=self._get_account_id()),
                timeout=10,
            )
            updated: Dict[str, Any] = {}
            for p in positions:
                data = _response_to_dict_safe(p)
                symbol = data.get("symbol", "")
                our_sym = self._reverse_resolve(symbol)
                qty = self._position_qty_from_broker_data(data)
                if qty != 0:
                    updated[our_sym] = {
                        "symbol": our_sym,
                        "size": qty,
                        "avg_price": float(data.get("avg_open_fill_price", 0)),
                        "unrealized_pnl": float(data.get("open_pnl", 0)),
                    }

            with self._state_lock:
                for sym in list(self._positions.keys()):
                    if sym not in updated:
                        del self._positions[sym]
                self._positions.update(updated)

            self._cleanup_closed_orders()

        except Exception as e:
            error_logger.error(f"Rithmic refresh positions error: {e}")

    def _cleanup_closed_orders(self) -> None:
        """Remove local order tracking only when broker confirms symbol is flat."""
        with self._order_lock:
            candidates = list(self._orders.items())

        for oid, info in candidates:
            symbol = info.get("symbol", "")
            if not symbol:
                continue

            if self._order_within_entry_grace(oid):
                bot_logger.debug(
                    f"Keeping SL/TP for {oid} — entry grace period active ({symbol})"
                )
                continue

            try:
                positions = self._run_sync(
                    self._client.list_positions(account_id=self._get_account_id()),
                    timeout=10,
                )
                broker_net = self._list_positions_net_qty(positions, symbol)
            except Exception as e:
                error_logger.error(
                    f"list_positions failed during cleanup for {symbol}: {e}"
                )
                bot_logger.info(
                    f"Keeping SL/TP for {oid} — broker position unknown, skipping cleanup ({symbol})"
                )
                continue

            if broker_net != 0:
                bot_logger.debug(
                    f"Keeping SL/TP for {oid} — position still open on broker ({symbol})"
                )
                continue

            try:
                still_open = self._run_sync(
                    self._async_is_entry_still_open(oid, symbol),
                    timeout=10,
                )
            except Exception as e:
                error_logger.error(
                    f"Entry open check failed during cleanup for {oid} {symbol}: {e}"
                )
                still_open = None
            if still_open is True or still_open is None:
                bot_logger.debug(
                    f"Keeping SL/TP for {oid} — entry still open or unknown ({symbol})"
                )
                continue

            with self._order_lock:
                removed = self._orders.pop(oid, None)
            if not removed:
                continue

            bot_logger.info(
                f"🔄 Order {oid} ({symbol}) removed — position confirmed flat on broker "
                f"(path=cleanup_closed_orders)"
            )
            if removed.get("protective_sl_order_id") or removed.get("protective_tp_order_id"):
                try:
                    self._run_sync(
                        self._async_cancel_entry_protective_orders(
                            oid,
                            removed,
                            reason="position_confirmed_flat_on_broker",
                        ),
                        timeout=10,
                    )
                except Exception as e:
                    error_logger.error(
                        f"Failed to cancel orphan protective orders for {oid} "
                        f"(path=cleanup_closed_orders): {e}"
                    )

    def _refresh_account_pnl(self) -> None:
        """Sync account balance/equity from Rithmic PnL plant."""
        if not self._connected:
            return
        try:
            summaries = self._run_sync(
                self._client.list_account_summary(
                    account_id=self._get_account_id()
                ),
                timeout=10,
            )
            if summaries:
                data = _response_to_dict_safe(summaries[0])
                with self._state_lock:
                    if "account_balance" in data:
                        self._account_info["balance"] = float(data["account_balance"])
                    if "cash_on_hand" in data:
                        self._account_info["equity"] = float(data["cash_on_hand"])
                    elif "margin_balance" in data:
                        self._account_info["equity"] = float(data["margin_balance"])
        except Exception as e:
            error_logger.error(f"Rithmic account PnL refresh error: {e}")

    # ══════════════════════════════════════════════════════════════
    #  INTERNAL: Yahoo Finance fallback
    # ══════════════════════════════════════════════════════════════

    def _get_yf(self):
        """Lazy-load Yahoo Finance fallback."""
        if self._yf_fallback is None:
            from src.data.yahoo_finance_live import YahooFinanceLive
            self._yf_fallback = YahooFinanceLive()
        return self._yf_fallback

    def _yf_get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int,
    ) -> Optional[pd.DataFrame]:
        try:
            df = self._get_yf().get_candles(symbol, timeframe_minutes, num_candles)
            if df is not None and len(df) > 0:
                bot_logger.info(f"Yahoo Finance fallback: {symbol} {len(df)} candles ({timeframe_minutes}m)")
            return df
        except Exception as e:
            error_logger.error(f"Yahoo Finance candles fallback error: {e}")
            return None

    def _yf_get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            return self._get_yf().get_latest_price(symbol)
        except Exception as e:
            error_logger.error(f"Yahoo Finance price fallback error: {e}")
            return None


def _response_to_dict_safe(response) -> dict:
    """Convert a protobuf response to a dict, handling both raw objects and dicts."""
    if isinstance(response, dict):
        return response
    try:
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(response, preserving_proto_field_name=True)
    except Exception:
        # If it's a proto-like object with attributes
        result = {}
        for attr in dir(response):
            if not attr.startswith("_"):
                try:
                    result[attr] = getattr(response, attr)
                except Exception:
                    pass
        return result
