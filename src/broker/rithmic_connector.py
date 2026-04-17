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
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.broker.base_broker import BaseBroker
from src.instruments import get_instrument, is_futures
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


class RithmicConnector(BaseBroker):
    """Direct Rithmic R|Protocol broker connector via async_rithmic."""

    def __init__(self):
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_started = threading.Event()
        self._client = None  # async_rithmic.RithmicClient

        # Thread-safe caches (written by async callbacks, read by sync methods)
        self._quote_lock = threading.Lock()
        self._quotes: Dict[str, Dict[str, float]] = {}  # symbol → {bid, ask, last}

        self._bar_lock = threading.Lock()
        self._bars: Dict[str, deque] = {}  # "symbol_tf" → deque of bar dicts

        # Position / account state
        self._state_lock = threading.Lock()
        self._positions: Dict[str, Any] = {}  # symbol → position info
        self._account_info: Dict[str, Any] = {"balance": 0.0, "equity": 0.0}
        self._pnl_permission_denied = False

        # Resolved front-month contract symbols: "MES" → "MESH6"
        self._resolved_contracts: Dict[str, str] = {}

        # Track orders submitted by this bot
        self._order_lock = threading.Lock()
        self._orders: Dict[str, Dict[str, Any]] = {}  # order_id → order info

        # Serialize history requests to avoid async_rithmic internal lock contention.
        # The library's history plant uses a single lock shared between _recv_loop
        # and send_and_recv_immediate; concurrent requests cause lock timeouts.
        self._history_gate = threading.Lock()
        self._history_async_lock: Optional[asyncio.Lock] = None  # created on event loop

        # Yahoo Finance fallback (lazy-loaded)
        self._yf_fallback = None

        # Lock timeout / fallback mode tracking
        self._lock_error_count = 0
        self._consecutive_lock_failures = 0  # tracks back-to-back failures for progressive backoff
        self._max_lock_errors = int(
            os.getenv("RITHMIC_HISTORY_MAX_LOCK_ERRORS", "1")
        )  # Switch to fallback mode quickly on lock contention
        self._fallback_until: Optional[float] = None  # timestamp when to retry Rithmic
        self._fallback_cooldown_secs = int(
            os.getenv("RITHMIC_HISTORY_FALLBACK_COOLDOWN_SECS", "60")
        )
        self._candle_cache: Dict[str, tuple] = {}  # "sym_tf" → (timestamp, DataFrame)
        self._candle_cache_ttl = int(
            os.getenv("RITHMIC_CANDLE_CACHE_TTL_SECS", "90")
        )

        # Configuration from env
        self._user = os.getenv("RITHMIC_USER_ID", "")
        self._password = os.getenv("RITHMIC_PASSWORD", "")
        self._system = os.getenv("RITHMIC_SYSTEM", "Rithmic Paper Trading")
        self._gateway = os.getenv("RITHMIC_GATEWAY", "")
        self._requested_account_id = os.getenv("RITHMIC_ACCOUNT_ID", "").strip()
        self._strict_account_match = (
            os.getenv("RITHMIC_STRICT_ACCOUNT_MATCH", "false").lower() == "true"
        )
        self._selected_account_id: Optional[str] = None
        self._app_name = "AiScalpBot"
        self._app_version = "1.0"
        self._connection_dead = threading.Event()  # set when reconnect exhausted
        self._disable_yahoo_fallback = (
            os.getenv("RITHMIC_DISABLE_YAHOO_FALLBACK", "false").lower() == "true"
        )

        # Symbols to subscribe (populated from config at init time)
        self._symbols_to_watch: List[str] = []

    # ── Connection lifecycle ──────────────────────────────────────

    def initialize(self) -> None:
        if not self._user or not self._password:
            bot_logger.warning(
                "RITHMIC_USER_ID / RITHMIC_PASSWORD not set — "
                "running in data-fallback mode (Yahoo Finance)"
            )
            self._connected = False
            return

        try:
            if os.name == 'nt' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            # Resolve gateway URL
            url = self._gateway or RITHMIC_GATEWAYS.get(self._system, "")
            if not url:
                error_logger.error(f"No gateway URL for system {self._system!r}")
                self._connected = False
                return

            # Start asyncio event loop in background thread
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_event_loop, daemon=True, name="rithmic-io"
            )
            self._thread.start()

            if not self._loop_started.wait(timeout=5):
                raise RuntimeError("Rithmic event loop thread failed to start")

            # Connect (blocks until done or timeout)
            self._run_sync(self._async_connect(url), timeout=30)
            self._connected = True
            bot_logger.info(
                f"Rithmic connector initialized — system={self._system}, "
                f"gateway={url}"
            )
        except Exception as e:
            error_logger.error(f"Rithmic connection failed: {e}")
            self._connected = False

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
            }

    # ── Market data ───────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
    ) -> Optional[pd.DataFrame]:
        # Check cache first
        cache_key = f"{symbol}_{timeframe_minutes}"
        now = time.time()
        latest_partial_df: Optional[pd.DataFrame] = None
        stale_cached_df: Optional[pd.DataFrame] = None
        if cache_key in self._candle_cache:
            cached_time, cached_df = self._candle_cache[cache_key]
            if cached_df is not None and len(cached_df) >= 10:
                stale_cached_df = cached_df
            if now - cached_time < self._candle_cache_ttl and cached_df is not None and len(cached_df) >= 10:
                return cached_df

        # Check if we're in fallback mode due to repeated lock errors
        if self._fallback_until is not None:
            if now < self._fallback_until:
                # During cooldown prefer stale cache to avoid hammering history plant.
                if stale_cached_df is not None:
                    return stale_cached_df
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
            max_retries = int(os.getenv("RITHMIC_HISTORY_MAX_RETRIES", "2"))
            inner_timeout = float(os.getenv("RITHMIC_HISTORY_INNER_TIMEOUT_SECS", "15"))
            for attempt in range(max_retries):
                try:
                    # IMPORTANT: outer timeout must exceed inner async timeout,
                    # otherwise run_sync can timeout first and leave overlapping
                    # history requests that contend on async_rithmic's lock.
                    rpc_timeout = inner_timeout + 5 + attempt * 4
                    df = self._run_sync(
                        self._async_get_candles(symbol, timeframe_minutes, num_candles),
                        timeout=rpc_timeout,
                    )
                    if df is not None and len(df) >= 10:
                        # Success! Reset error counter and cache result
                        self._lock_error_count = 0
                        self._consecutive_lock_failures = 0
                        self._candle_cache[cache_key] = (now, df)
                        return df
                    else:
                        if df is not None and len(df) > 0:
                            latest_partial_df = df
                            if self._disable_yahoo_fallback:
                                # In strict Rithmic mode, return partial bars immediately.
                                return latest_partial_df
                        bot_logger.warning(
                            f"Rithmic candle fetch returned insufficient data for {symbol}: "
                            f"{0 if df is None else len(df)} bars"
                        )
                except Exception as e:
                    error_msg = str(e).lower()
                    is_lock_error = "lock" in error_msg or "timeout" in error_msg

                    if is_lock_error:
                        self._lock_error_count += 1
                        bot_logger.warning(
                            f"History lock error for {symbol} (attempt {attempt+1}/{max_retries}): {e}"
                        )
                        if attempt < max_retries - 1:
                            # Wait for _recv_loop to recover, then retry
                            wait = 2.0 + attempt * 2.0  # 2s, 4s
                            bot_logger.info(
                                f"⏳ Waiting {wait:.0f}s for history plant lock to clear..."
                            )
                            time.sleep(wait)
                            continue
                        # Final attempt failed — enter fallback
                        if self._lock_error_count >= self._max_lock_errors:
                            self._consecutive_lock_failures += 1
                            # Progressive backoff: 60s, 120s, 240s, 480s, cap at 600s
                            backoff = min(
                                self._fallback_cooldown_secs * (2 ** (self._consecutive_lock_failures - 1)),
                                600,
                            )
                            self._fallback_until = now + backoff
                            if self._disable_yahoo_fallback:
                                bot_logger.warning(
                                    f"⚠️ Rithmic history lock errors ({self._lock_error_count}x, "
                                    f"streak {self._consecutive_lock_failures}) — "
                                    f"Yahoo fallback disabled, retrying after {backoff}s"
                                )
                            else:
                                bot_logger.warning(
                                    f"⚠️ Rithmic history lock errors ({self._lock_error_count}x, "
                                    f"streak {self._consecutive_lock_failures}) — "
                                    f"switching to Yahoo Finance fallback for {backoff}s"
                                )
                        break

                    # KeyError from async_rithmic when no bars exist (e.g. market closed)
                    if isinstance(e, KeyError):
                        if self._disable_yahoo_fallback:
                            bot_logger.info(f"Rithmic no bars for {symbol} (market closed?)")
                        else:
                            bot_logger.info(f"Rithmic no bars for {symbol} (market closed?) — using Yahoo fallback")
                    else:
                        bot_logger.error(f"Rithmic candles error for {symbol}: {e}")
                    break

        if self._disable_yahoo_fallback:
            if latest_partial_df is not None and len(latest_partial_df) > 0:
                return latest_partial_df
            if stale_cached_df is not None:
                return stale_cached_df
            return None

        # Fall back to Yahoo Finance
        df = self._yf_get_candles(symbol, timeframe_minutes, num_candles)
        if df is not None and len(df) >= 10:
            self._candle_cache[cache_key] = (now, df)
        return df

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
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
        symbol: str = "",
        order_type: str = "",
        size: float = 1,
        entry_price: float = 0,
        stop_loss: float = 0,
        take_profit: float = 0,
        # Accept bot.py's alternate keyword names
        pair: str = "",
        lot_size: float = 0,
    ) -> Optional[Dict[str, Any]]:
        # Normalize alternate parameter names from bot.py
        if not symbol and pair:
            symbol = pair
        if size <= 0 and lot_size > 0:
            size = lot_size
        if not self._connected:
            bot_logger.warning("Rithmic not connected — order rejected")
            return None

        try:
            order_id = f"bot_{uuid.uuid4().hex[:10]}"
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
                    "supports_stop_modify": native_stop_attached,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                with self._order_lock:
                    self._orders[order_id] = order_info
                bot_logger.info(
                    f"Rithmic order placed: {order_id} {order_type} "
                    f"{int(size)} {symbol}"
                )
                if stop_loss > 0 and not native_stop_attached:
                    bot_logger.error(
                        f"Rithmic order {order_id} was submitted without a native stop; "
                        f"later stop modification will not be available"
                    )
                if take_profit > 0 and not native_target_attached:
                    bot_logger.warning(
                        f"Rithmic order {order_id} was submitted without a native target"
                    )
                if raw_result is not None and isinstance(raw_result, dict):
                    order_info["broker_response"] = raw_result
                return order_info
            return None
        except Exception as e:
            error_logger.error(f"Rithmic place_order error: {e}")
            return None

    def get_order_info(self, ticket: Any) -> Optional[Dict[str, Any]]:
        order_id = str(ticket)
        with self._order_lock:
            order = self._orders.get(order_id)
            if not order:
                return None
            return dict(order)

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

            rith_sym, exchange = self._resolve_symbol(target_symbol)
            self._run_sync(
                self._client.exit_position(
                    symbol=rith_sym,
                    exchange=exchange,
                    account_id=self._get_account_id(),
                ),
                timeout=10,
            )
            # Remove from local tracking
            with self._order_lock:
                self._orders = {
                    k: v for k, v in self._orders.items()
                    if v.get("symbol") != target_symbol
                }
            return True
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

    def get_net_position_size(self, symbol: str) -> int:
        """Return live net quantity for a symbol from Rithmic position data.

        Positive = net long, negative = net short, zero = flat.
        """
        self._refresh_positions()
        with self._state_lock:
            return int(self._positions.get(symbol, {}).get('size', 0))

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
        self._loop_started.set()
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
        from async_rithmic import RithmicClient, OrderPlacement, DataType, ReconnectionSettings

        self._client = RithmicClient(
            user=self._user,
            password=self._password,
            system_name=self._system,
            app_name=self._app_name,
            app_version=self._app_version,
            url=url,
            manual_or_auto=OrderPlacement.AUTO,
            reconnection_settings=ReconnectionSettings(
                max_retries=2,
                backoff_type="linear",
                interval=30,
                max_delay=60,
            ),
        )

        # Register callbacks for real-time data
        self._client.on_tick += self._on_tick
        self._client.on_time_bar += self._on_time_bar
        self._client.on_account_pnl_update += self._on_account_pnl
        self._client.on_instrument_pnl_update += self._on_instrument_pnl
        self._client.on_disconnected += self._on_disconnected

        # Connect all plants (ticker, order, history, pnl)
        await self._client.connect()

        # Let all plants (especially history) finish their handshake
        # before we start sending requests.  The history plant's _recv_loop
        # holds its internal lock while processing connection messages;
        # requesting bars too early causes the 5-second lock timeout.
        await asyncio.sleep(3)

        # Install ForcedLogout storm detector on each plant.
        self._forced_logout_times: List[float] = []
        self._patch_forced_logout_detection()

        # Subscribe to PnL updates (some prop accounts do not grant this permission).
        try:
            await self._client.subscribe_to_pnl_updates()
        except Exception as e:
            msg = str(e).lower()
            if "no permission to this account" in msg or "1088" in msg:
                self._pnl_permission_denied = True
                bot_logger.warning(
                    "Rithmic PnL subscription not permitted for this account (rpCode 1088). "
                    "Continuing without account/instrument PnL updates."
                )
            else:
                raise

        # Resolve front-month contracts and subscribe to market data
        from async_rithmic import DataType

        # Choose account once per connection so order routing is deterministic.
        self._selected_account_id = self._select_account_id()
        for sym in self._get_watch_symbols():
            rith_sym, exchange = self._resolve_symbol_raw(sym)
            # Resolve front-month contract (e.g., "MES" → "MESH6")
            try:
                front = await self._client.get_front_month_contract(rith_sym, exchange)
                if front:
                    self._resolved_contracts[sym] = front
                    bot_logger.info(f"Resolved {sym} → {front} on {exchange}")
                else:
                    self._resolved_contracts[sym] = rith_sym
            except Exception as e:
                error_logger.error(f"Failed to resolve front month for {sym}: {e}")
                self._resolved_contracts[sym] = rith_sym

            # Subscribe to BBO + last trade
            trading_sym = self._resolved_contracts[sym]
            await self._client.subscribe_to_market_data(
                trading_sym, exchange,
                DataType.BBO | DataType.LAST_TRADE,
            )

    async def _on_tick(self, data: dict) -> None:
        """Callback for real-time quote updates (BBO + last trade)."""
        from async_rithmic import DataType
        symbol = data.get("symbol", "")
        # Reverse-map trading symbol back to our symbol
        our_sym = self._reverse_resolve(symbol)

        with self._quote_lock:
            if our_sym not in self._quotes:
                self._quotes[our_sym] = {"bid": 0.0, "ask": 0.0, "last": 0.0}

            if data.get("data_type") == DataType.BBO:
                if "best_bid_price" in data:
                    self._quotes[our_sym]["bid"] = float(data["best_bid_price"])
                if "best_ask_price" in data:
                    self._quotes[our_sym]["ask"] = float(data["best_ask_price"])
            elif data.get("data_type") == DataType.LAST_TRADE:
                if "trade_price" in data:
                    self._quotes[our_sym]["last"] = float(data["trade_price"])

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

    async def _on_disconnected(self, plant_type) -> None:
        """Called when a plant's WebSocket disconnects after exhausting retries."""
        error_logger.error(
            f"Rithmic {plant_type} plant disconnected (retries exhausted). "
            "Signalling bot to shut down."
        )
        self._connected = False
        self._connection_dead.set()

    @property
    def is_connection_dead(self) -> bool:
        """True when reconnection retries are exhausted and bot should stop."""
        return self._connection_dead.is_set()

    def _patch_forced_logout_detection(self) -> None:
        """Monkey-patch each plant to detect ForcedLogout storms.

        If >5 ForcedLogout messages arrive within 30 seconds we set
        ``_connection_dead`` so the main loop exits cleanly.
        We also force-close each plant's WebSocket to stop the
        reconnect loop from inside the library.
        """
        MAX_LOGOUTS = 5
        WINDOW_SECS = 30.0

        for name, plant in self._client.plants.items():
            original = plant._process_response

            async def _wrapped(response, _orig=original, _name=name):
                if getattr(response, "template_id", None) == 77:
                    now = time.time()
                    self._forced_logout_times.append(now)
                    # Trim old entries outside the window
                    cutoff = now - WINDOW_SECS
                    self._forced_logout_times = [
                        t for t in self._forced_logout_times if t > cutoff
                    ]
                    if len(self._forced_logout_times) >= MAX_LOGOUTS:
                        if not self._connection_dead.is_set():
                            error_logger.error(
                                f"ForcedLogout storm detected ({len(self._forced_logout_times)} "
                                f"in {WINDOW_SECS}s). Another session is using these credentials. "
                                "Shutting down."
                            )
                            self._connected = False
                            self._connection_dead.set()
                        return True
                try:
                    return await _orig(response)
                except Exception as e:
                    # Some accounts cannot access PnL templates for the selected account.
                    # Ignore this specific entitlement error so the rest of the connector
                    # (market data, history, orders) stays healthy.
                    msg = str(e).lower()
                    if _name == "pnl" and (
                        "no permission to this account" in msg or "1088" in msg
                    ):
                        if not self._pnl_permission_denied:
                            bot_logger.warning(
                                "Rithmic PnL plant denied account access (rpCode 1088). "
                                "Disabling account PnL refresh for this session."
                            )
                        self._pnl_permission_denied = True
                        return True
                    raise

            plant._process_response = _wrapped

    async def _async_get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int,
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

        end_time = datetime.now(timezone.utc)
        # Request extra bars to account for non-trading hours
        start_time = end_time - timedelta(minutes=timeframe_minutes * num_candles * 3)

        # Serialize history requests — the async_rithmic history plant has a
        # single internal lock shared by _recv_loop and send_and_recv_immediate.
        # Concurrent history calls cause the 5-second lock timeout.
        # We also wrap in wait_for() to CANCEL hang if the library deadlocks;
        # cancellation releases the internal lock so _recv_loop can recover.
        if self._history_async_lock is None:
            self._history_async_lock = asyncio.Lock()
        inner_timeout = float(os.getenv("RITHMIC_HISTORY_INNER_TIMEOUT_SECS", "15"))
        async with self._history_async_lock:
            try:
                bars = await asyncio.wait_for(
                    self._client.get_historical_time_bars(
                        symbol=rith_sym,
                        exchange=exchange,
                        start_time=start_time,
                        end_time=end_time,
                        bar_type=bar_type,
                        bar_type_periods=period,
                    ),
                    timeout=inner_timeout,  # Allow time for library lock wait + data transfer
                )
            except asyncio.TimeoutError:
                bot_logger.warning(
                    f"⏱️ History request for {symbol} timed out ({inner_timeout:.0f}s) — "
                    f"likely async_rithmic internal lock deadlock"
                )
                # Give _recv_loop time to reacquire the now-released lock
                await asyncio.sleep(1.0)
                raise TimeoutError("Rithmic history lock timeout — cancelled")

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

        # Add bracket (SL/TP) in ticks
        spec = get_instrument(symbol)
        price_ref = self._get_cached_price(symbol)
        if price_ref <= 0 and entry_price > 0:
            # Fallback to caller-provided entry estimate so bracket orders are still attached.
            price_ref = entry_price

        if stop_loss > 0:
            # We need the distance in ticks from an approximate fill reference.
            if price_ref > 0:
                sl_ticks = abs(round((price_ref - stop_loss) / spec.tick_size))
                if sl_ticks > 0:
                    kwargs["stop_ticks"] = sl_ticks
                else:
                    bot_logger.warning(
                        f"SL tick calc = 0: price_ref={price_ref}, SL={stop_loss}, "
                        f"tick_size={spec.tick_size}"
                    )
            else:
                bot_logger.warning(
                    f"Cannot attach SL: no price reference (cached=0, entry={entry_price})"
                )

        if take_profit > 0:
            if price_ref > 0:
                tp_ticks = abs(round((take_profit - price_ref) / spec.tick_size))
                if tp_ticks > 0:
                    kwargs["target_ticks"] = tp_ticks

        bot_logger.info(
            f"Rithmic submit_order: {symbol} {transaction} qty={qty} "
            f"price_ref={price_ref} stop_ticks={kwargs.get('stop_ticks', 'NONE')} "
            f"target_ticks={kwargs.get('target_ticks', 'NONE')}"
        )

        result = await self._client.submit_order(**kwargs)
        return {
            "result": result,
            "native_stop_attached": bool(kwargs.get("stop_ticks", 0) > 0),
            "native_target_attached": bool(kwargs.get("target_ticks", 0) > 0),
            "price_ref": price_ref,
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
        trading_sym = self._resolved_contracts.get(
            symbol,
            RITHMIC_SYMBOL_MAP.get(symbol, (symbol,))[0],
        )
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

    def _get_account_id(self) -> Optional[str]:
        """Return selected account id for all account-scoped broker calls."""
        if self._selected_account_id:
            return self._selected_account_id
        if self._client and self._client.accounts:
            self._selected_account_id = self._client.accounts[0].account_id
            return self._selected_account_id
        return None

    def _select_account_id(self) -> Optional[str]:
        """Select account id from env (if provided) or default to first account."""
        if not self._client or not getattr(self._client, "accounts", None):
            bot_logger.warning("No Rithmic accounts available after connect")
            return None

        account_ids: List[str] = []
        for acct in self._client.accounts:
            acct_id = getattr(acct, "account_id", "")
            if acct_id:
                account_ids.append(str(acct_id))

        if not account_ids:
            bot_logger.warning("No valid Rithmic account IDs returned by API")
            return None

        if self._requested_account_id:
            for acct_id in account_ids:
                if acct_id == self._requested_account_id:
                    bot_logger.info(
                        f"Rithmic account selected from RITHMIC_ACCOUNT_ID: {acct_id}"
                    )
                    return acct_id
            if self._strict_account_match:
                raise RuntimeError(
                    "RITHMIC_ACCOUNT_ID strict match enabled but requested account was not found. "
                    f"requested={self._requested_account_id}, available={account_ids}"
                )
            bot_logger.warning(
                "RITHMIC_ACCOUNT_ID not found in connected accounts. "
                f"requested={self._requested_account_id}, available={account_ids}. "
                f"Defaulting to first account={account_ids[0]}"
            )
        else:
            bot_logger.info(
                f"Rithmic account defaulting to first available: {account_ids[0]} "
                f"(available={account_ids})"
            )

        return account_ids[0]

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

            # Log raw position data for debugging phantom removals
            if positions:
                for p in positions:
                    raw = _response_to_dict_safe(p)
                    bot_logger.debug(
                        f"[POS_RAW] symbol={raw.get('symbol')} "
                        f"buy_qty={raw.get('buy_qty')} sell_qty={raw.get('sell_qty')} "
                        f"net_qty={raw.get('net_qty')} quantity={raw.get('quantity')} "
                        f"keys={list(raw.keys())[:20]}"
                    )

            new_positions: Dict[str, Any] = {}
            if positions:
                for p in positions:
                    data = _response_to_dict_safe(p)
                    symbol = data.get("symbol", "")
                    our_sym = self._reverse_resolve(symbol)

                    # Try multiple field name patterns — async_rithmic versions differ
                    qty = int(data.get("buy_qty", 0)) - int(data.get("sell_qty", 0))
                    if qty == 0:
                        # Fallback: some versions use net_qty or quantity
                        qty = int(data.get("net_qty", 0))
                    if qty == 0:
                        qty = int(data.get("quantity", 0))

                    if qty != 0:
                        new_positions[our_sym] = {
                            "symbol": our_sym,
                            "size": qty,
                            "avg_price": float(data.get("avg_open_fill_price", data.get("avg_price", 0))),
                            "unrealized_pnl": float(data.get("open_pnl", data.get("pnl", 0))),
                        }

            with self._state_lock:
                # Only clear if we got a valid (non-empty) response, or keep stale
                # data to prevent phantom removal on transient API failures.
                if positions is not None:
                    self._positions = new_positions
                # If positions is None (timeout/error), keep existing _positions.

            # Clean up orders for closed symbols (with age guard)
            self._cleanup_closed_orders()

        except Exception as e:
            error_logger.error(f"Rithmic refresh positions error: {e}")

    def _cleanup_closed_orders(self) -> None:
        """Remove orders from _orders dict when their symbol has no position.

        Only removes orders older than 120 seconds to avoid cleaning up orders
        whose fill hasn't propagated to the PnL plant yet.
        """
        with self._state_lock:
            open_symbols = {sym for sym, pos in self._positions.items() if pos.get('size', 0) != 0}

        min_age_secs = 120  # Don't remove orders placed less than 2 min ago
        now_str = datetime.now(timezone.utc).isoformat()

        with self._order_lock:
            closed_order_ids = []
            for oid, info in self._orders.items():
                if info.get('symbol', '') in open_symbols:
                    continue
                # Check order age — skip recent orders
                order_time_str = info.get('time', '')
                if order_time_str:
                    try:
                        order_time = datetime.fromisoformat(order_time_str)
                        age = (datetime.now(timezone.utc) - order_time).total_seconds()
                        if age < min_age_secs:
                            continue
                    except (ValueError, TypeError):
                        pass
                closed_order_ids.append(oid)

            for oid in closed_order_ids:
                removed = self._orders.pop(oid, None)
                if removed:
                    bot_logger.info(f"🔄 Order {oid} ({removed.get('symbol')}) removed — position closed")

    def _refresh_account_pnl(self) -> None:
        """Sync account balance/equity from Rithmic PnL plant."""
        if not self._connected or self._pnl_permission_denied:
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
            msg = str(e).lower()
            if "no permission to this account" in msg or "1088" in msg:
                self._pnl_permission_denied = True
                bot_logger.warning(
                    "Rithmic account summary not permitted for this account (rpCode 1088). "
                    "Using cached/default balance and equity values."
                )
                return
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
