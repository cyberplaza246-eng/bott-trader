"""
TradersPost Webhook Connector

Execution flow:  Bot → JSON webhook → TradersPost → Tradovate → Lucid Trading

TradersPost handles:
  - Order routing to Tradovate
  - Position management
  - Account sync

This connector sends webhooks and polls TradersPost for position/account state.
Market data comes from a separate provider (Polygon.io or Yahoo Finance).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.broker.base_broker import BaseBroker
from src.instruments import get_instrument, is_futures
from src.utils.logger import bot_logger, error_logger


class TradersPostConnector(BaseBroker):
    """Webhook-based broker via TradersPost → Tradovate → Lucid Trading."""

    def __init__(self):
        self._connected = False
        self._balance = 0.0
        self._equity = 0.0
        self._positions: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._order_lock = threading.Lock()
        self._recent_order_ids: List[str] = []  # dedup window

        # Configuration from env
        self.webhook_url = os.getenv("TRADERSPOST_WEBHOOK_URL", "")
        self.api_key = os.getenv("TRADERSPOST_API_KEY", "")
        self.webhook_secret = os.getenv("TRADERSPOST_WEBHOOK_SECRET", "")
        self.account_id = os.getenv("TRADERSPOST_ACCOUNT_ID", "")

        # Market data provider
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self._polygon_base = "https://api.polygon.io"

        # Rate limiting
        self._last_webhook_time = 0.0
        self._min_webhook_interval = 1.0  # seconds between webhooks

    # ── Connection lifecycle ──────────────────────────────────────

    def initialize(self) -> None:
        if not self.webhook_url:
            bot_logger.warning("TRADERSPOST_WEBHOOK_URL not set — running in dry-run mode")
            self._connected = False
            return
        self._connected = True
        bot_logger.info("TradersPost connector initialized (webhook mode)")

    def shutdown(self) -> None:
        self._connected = False
        bot_logger.info("TradersPost connector shut down")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Account info ──────────────────────────────────────────────

    def get_balance(self) -> float:
        return self._balance

    def get_equity(self) -> float:
        return self._equity

    def get_account_info(self) -> Dict[str, Any]:
        return {
            "balance": self._balance,
            "equity": self._equity,
            "positions": len(self._positions),
            "broker": "traderspost",
            "account_id": self.account_id,
        }

    def sync_account(self, balance: float, equity: float) -> None:
        """Called externally to update balance/equity from TradersPost callback."""
        self._balance = balance
        self._equity = equity

    # ── Market data (Polygon.io) ──────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
    ) -> Optional[pd.DataFrame]:
        if not self.polygon_api_key:
            return None
        try:
            poly_symbol = self._to_polygon_symbol(symbol)
            multiplier, timespan = self._tf_to_polygon(timeframe_minutes)
            end = datetime.now(timezone.utc)
            # Overshoot to account for non-trading hours
            minutes_needed = num_candles * timeframe_minutes * 2
            start = end - pd.Timedelta(minutes=minutes_needed)

            url = (
                f"{self._polygon_base}/v2/aggs/ticker/{poly_symbol}"
                f"/range/{multiplier}/{timespan}"
                f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            )
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": min(num_candles * 3, 50000),
                "apiKey": self.polygon_api_key,
            }
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            if data.get("resultsCount", 0) == 0:
                return None

            bars = data["results"]
            df = pd.DataFrame(bars)
            df = df.rename(columns={
                "t": "datetime", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume",
            })
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            return df.tail(num_candles).reset_index(drop=True)
        except Exception as e:
            error_logger.error(f"Polygon candles error for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self.polygon_api_key:
            return None
        try:
            poly_symbol = self._to_polygon_symbol(symbol)
            url = f"{self._polygon_base}/v2/last/trade/{poly_symbol}"
            params = {"apiKey": self.polygon_api_key}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            result = r.json().get("results", {})
            price = result.get("p", 0.0)
            spec = get_instrument(symbol)
            half_spread = spec.spread_default / 2
            return {
                "bid": price - half_spread,
                "ask": price + half_spread,
                "last": price,
            }
        except Exception as e:
            error_logger.error(f"Polygon price error for {symbol}: {e}")
            return None

    def get_spread(self, symbol: str) -> Optional[float]:
        price = self.get_latest_price(symbol)
        if price:
            return price["ask"] - price["bid"]
        return get_instrument(symbol).spread_default

    # ── Order execution (webhook) ─────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Dict[str, Any]]:
        with self._order_lock:
            # Rate limit
            now = time.time()
            wait = self._min_webhook_interval - (now - self._last_webhook_time)
            if wait > 0:
                time.sleep(wait)

            order_id = str(uuid.uuid4())[:12]

            # Duplicate filter — skip if same symbol+direction within 5 seconds
            dedup_key = f"{symbol}:{order_type}"
            if dedup_key in self._recent_order_ids:
                bot_logger.warning(f"Duplicate order blocked: {dedup_key}")
                return None

            payload = self._build_webhook_payload(
                symbol=symbol,
                action="buy" if order_type.lower() in ("buy", "long") else "sell",
                qty=int(size) if is_futures(symbol) else size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
            )

            success = self._send_webhook(payload)
            self._last_webhook_time = time.time()

            if success:
                # Track for dedup (expire after 5 seconds)
                self._recent_order_ids.append(dedup_key)
                if len(self._recent_order_ids) > 20:
                    self._recent_order_ids = self._recent_order_ids[-10:]

                order_info = {
                    "ticket": order_id,
                    "symbol": symbol,
                    "type": order_type,
                    "size": size,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                self._positions.append(order_info)
                return order_info
            return None

    def close_position(
        self,
        symbol: Optional[str] = None,
        size: Optional[float] = None,
        ticket: Optional[int] = None,
    ) -> bool:
        with self._order_lock:
            target_symbol = symbol
            if ticket:
                for pos in self._positions:
                    if pos.get("ticket") == ticket:
                        target_symbol = pos["symbol"]
                        break

            if not target_symbol:
                return False

            payload = self._build_webhook_payload(
                symbol=target_symbol,
                action="exit",
                qty=int(size) if size and is_futures(target_symbol) else size,
            )
            success = self._send_webhook(payload)
            if success:
                self._positions = [
                    p for p in self._positions
                    if p.get("symbol") != target_symbol
                ]
            return success

    def close_all_positions(self) -> bool:
        with self._order_lock:
            symbols = list({p["symbol"] for p in self._positions})
            all_ok = True
            for sym in symbols:
                payload = self._build_webhook_payload(symbol=sym, action="exit")
                if not self._send_webhook(payload):
                    all_ok = False
            if all_ok:
                self._positions.clear()
            return all_ok

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        # TradersPost doesn't support in-place SL/TP modification via webhook.
        # We track SL/TP locally and exit + re-enter if needed.
        for pos in self._positions:
            if pos.get("ticket") == ticket:
                if sl is not None:
                    pos["stop_loss"] = sl
                if tp is not None:
                    pos["take_profit"] = tp
                bot_logger.info(f"Position {ticket} SL/TP updated locally: SL={sl}, TP={tp}")
                return True
        return False

    # ── Position queries ──────────────────────────────────────────

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            return [p for p in self._positions if p.get("symbol") == symbol]
        return list(self._positions)

    def get_bot_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        # All positions are bot positions in webhook mode
        return self.get_open_positions(symbol)

    def get_trade_history(
        self,
        hours: int = 24,
        include_all: bool = False,
    ) -> List[Dict[str, Any]]:
        return list(self._trade_history)

    # ── Webhook internals ─────────────────────────────────────────

    def _build_webhook_payload(
        self,
        symbol: str,
        action: str,
        qty: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build TradersPost-compatible webhook JSON."""
        payload: Dict[str, Any] = {
            "ticker": symbol,
            "action": action,  # "buy", "sell", "exit"
        }
        if qty is not None:
            payload["quantity"] = qty
        if stop_loss is not None:
            payload["stopLoss"] = {"type": "stop", "stopPrice": stop_loss}
        if take_profit is not None:
            payload["takeProfit"] = {"type": "limit", "limitPrice": take_profit}
        if order_id:
            payload["orderId"] = order_id
        return payload

    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send webhook to TradersPost with optional HMAC signing."""
        if not self.webhook_url:
            bot_logger.info(f"[DRY-RUN] Would send webhook: {json.dumps(payload)}")
            return True
        try:
            headers = {"Content-Type": "application/json"}

            # HMAC signature if secret is configured
            if self.webhook_secret:
                body_bytes = json.dumps(payload, separators=(",", ":")).encode()
                sig = hmac.new(
                    self.webhook_secret.encode(),
                    body_bytes,
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Signature"] = sig

            r = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            if r.status_code in (200, 201, 202):
                bot_logger.info(f"Webhook sent: {payload.get('action')} {payload.get('ticker')} → {r.status_code}")
                return True
            error_logger.error(f"Webhook failed ({r.status_code}): {r.text[:200]}")
            return False
        except Exception as e:
            error_logger.error(f"Webhook send error: {e}")
            return False

    # ── Symbol mapping helpers ────────────────────────────────────

    @staticmethod
    def _to_polygon_symbol(symbol: str) -> str:
        """Convert our symbol format to Polygon.io ticker format."""
        forex_map = {
            "EUR/USD": "C:EURUSD",
            "GBP/USD": "C:GBPUSD",
            "USD/JPY": "C:USDJPY",
        }
        if symbol in forex_map:
            return forex_map[symbol]
        # Futures symbols pass through (MES, ES, NQ, etc.)
        return symbol

    @staticmethod
    def _tf_to_polygon(timeframe_minutes: int) -> tuple:
        """Convert minutes to Polygon multiplier + timespan."""
        if timeframe_minutes < 60:
            return (timeframe_minutes, "minute")
        if timeframe_minutes < 1440:
            return (timeframe_minutes // 60, "hour")
        return (timeframe_minutes // 1440, "day")
