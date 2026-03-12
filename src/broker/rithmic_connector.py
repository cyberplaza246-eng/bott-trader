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

        # Resolved front-month contract symbols: "MES" → "MESH6"
        self._resolved_contracts: Dict[str, str] = {}

        # Track orders submitted by this bot
        self._order_lock = threading.Lock()
        self._orders: Dict[str, Dict[str, Any]] = {}  # order_id → order info

        # Yahoo Finance fallback (lazy-loaded)
        self._yf_fallback = None

        # Configuration from env
        self._user = os.getenv("RITHMIC_USER_ID", "")
        self._password = os.getenv("RITHMIC_PASSWORD", "")
        self._system = os.getenv("RITHMIC_SYSTEM", "Rithmic Paper Trading")
        self._gateway = os.getenv("RITHMIC_GATEWAY", "")
        self._app_name = "AiScalpBot"
        self._app_version = "1.0"

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
            }

    # ── Market data ───────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
    ) -> Optional[pd.DataFrame]:
        # Try Rithmic historical bars first with retry logic
        if self._connected:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    df = self._run_sync(
                        self._async_get_candles(symbol, timeframe_minutes, num_candles),
                        timeout=25,  # Increased timeout
                    )
                    if df is not None and len(df) >= 10:
                        return df
                except Exception as e:
                    error_msg = str(e).lower()
                    # Log and retry on lock timeout
                    if "lock" in error_msg or "timeout" in error_msg:
                        if attempt < max_retries - 1:
                            bot_logger.warning(f"Rithmic history lock timeout, retry {attempt+1}/{max_retries}")
                            time.sleep(1.0 + attempt)  # Brief backoff
                            continue
                    error_logger.error(f"Rithmic candles error for {symbol}: {e}")
                    break

        # Fallback to Yahoo Finance for futures
        bot_logger.info(f"Using Yahoo Finance fallback for {symbol} candles")
        return self._yf_get_candles(symbol, timeframe_minutes, num_candles)

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        # Try cached Rithmic quote
        with self._quote_lock:
            quote = self._quotes.get(symbol)
            if quote and quote.get("last", 0) > 0:
                return dict(quote)

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

        try:
            order_id = f"bot_{uuid.uuid4().hex[:10]}"
            result = self._run_sync(
                self._async_place_order(
                    symbol=symbol,
                    side=order_type,
                    qty=int(size),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=order_id,
                ),
                timeout=15,
            )
            if result is not None:
                order_info = {
                    "ticket": order_id,
                    "symbol": symbol,
                    "type": order_type,
                    "size": int(size),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                with self._order_lock:
                    self._orders[order_id] = order_info
                bot_logger.info(
                    f"Rithmic order placed: {order_id} {order_type} "
                    f"{int(size)} {symbol}"
                )
                return order_info
            return None
        except Exception as e:
            error_logger.error(f"Rithmic place_order error: {e}")
            return None

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
            return False
        try:
            order_id = str(ticket)
            kwargs: Dict[str, Any] = {"order_id": order_id}

            with self._order_lock:
                order = self._orders.get(order_id)
            if not order:
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
            error_logger.error(f"Rithmic modify_position error: {e}")
            return False

    # ── Position queries ──────────────────────────────────────────

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self._refresh_positions()
        with self._state_lock:
            positions = list(self._positions.values())
        if symbol:
            return [p for p in positions if p.get("symbol") == symbol]
        return positions

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

        self._client = RithmicClient(
            user=self._user,
            password=self._password,
            system_name=self._system,
            app_name=self._app_name,
            app_version=self._app_version,
            url=url,
            manual_or_auto=OrderPlacement.AUTO,
        )

        # Register callbacks for real-time data
        self._client.on_tick += self._on_tick
        self._client.on_time_bar += self._on_time_bar
        self._client.on_account_pnl_update += self._on_account_pnl
        self._client.on_instrument_pnl_update += self._on_instrument_pnl

        # Connect all plants (ticker, order, history, pnl)
        await self._client.connect()

        # Subscribe to PnL updates
        await self._client.subscribe_to_pnl_updates()

        # Resolve front-month contracts and subscribe to market data
        from async_rithmic import DataType
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

    async def _async_place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
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
        if stop_loss > 0:
            # For a buy, SL is below entry; for sell, SL is above
            # We need distance in ticks from fill price
            # Use current price as estimate for tick distance
            price = self._get_cached_price(symbol)
            if price > 0:
                sl_ticks = abs(round((price - stop_loss) / spec.tick_size))
                if sl_ticks > 0:
                    kwargs["stop_ticks"] = sl_ticks

        if take_profit > 0:
            price = self._get_cached_price(symbol)
            if price > 0:
                tp_ticks = abs(round((take_profit - price) / spec.tick_size))
                if tp_ticks > 0:
                    kwargs["target_ticks"] = tp_ticks

        result = await self._client.submit_order(**kwargs)
        return result

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
        """Return the first available account ID from Rithmic."""
        if self._client and self._client.accounts:
            return self._client.accounts[0].account_id
        return None

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
            with self._state_lock:
                self._positions.clear()
                for p in positions:
                    data = _response_to_dict_safe(p)
                    symbol = data.get("symbol", "")
                    our_sym = self._reverse_resolve(symbol)
                    qty = int(data.get("buy_qty", 0)) - int(data.get("sell_qty", 0))
                    if qty != 0:
                        self._positions[our_sym] = {
                            "symbol": our_sym,
                            "size": qty,
                            "avg_price": float(data.get("avg_open_fill_price", 0)),
                            "unrealized_pnl": float(data.get("open_pnl", 0)),
                        }
        except Exception as e:
            error_logger.error(f"Rithmic refresh positions error: {e}")

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
            return self._get_yf().get_candles(symbol, timeframe_minutes, num_candles)
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
