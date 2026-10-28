"""
Base Broker Interface — Abstract contract all broker connectors must implement.

Provides a unified API so the bot can swap between MT5, TradersPost, or any
future broker without touching strategy code.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseBroker(abc.ABC):
    """Abstract broker interface.

    Every concrete connector (MT5Connector, TradersPostConnector) must
    implement these methods.  The bot only talks through this interface.
    """

    # ── Connection lifecycle ──────────────────────────────────────

    @abc.abstractmethod
    def initialize(self) -> None:
        """Establish connection to the broker / exchange."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Gracefully close the broker connection."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool:  # noqa: D401
        """Whether the broker connection is currently alive."""

    # ── Account info ──────────────────────────────────────────────

    @abc.abstractmethod
    def get_balance(self) -> float:
        """Current account cash balance (USD)."""

    @abc.abstractmethod
    def get_equity(self) -> float:
        """Current equity including unrealised P&L."""

    @abc.abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Full account snapshot (balance, equity, margin, etc.)."""

    # ── Market data ───────────────────────────────────────────────

    @abc.abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV candles."""

    @abc.abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return {'bid': ..., 'ask': ..., 'last': ...} for *symbol*."""

    @abc.abstractmethod
    def get_spread(self, symbol: str) -> Optional[float]:
        """Current spread in price units."""

    # ── Order execution ───────────────────────────────────────────

    @abc.abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: str,       # "buy" | "sell"
        size: float,           # lots (forex) or contracts (futures)
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Dict[str, Any]]:
        """Submit a new order.  Returns order info dict or None on failure."""

    @abc.abstractmethod
    def close_position(
        self,
        symbol: Optional[str] = None,
        size: Optional[float] = None,
        ticket: Optional[int] = None,
    ) -> bool:
        """Close an existing position.  Returns True on success."""

    @abc.abstractmethod
    def close_all_positions(self) -> bool:
        """Flatten all open positions.  Returns True on success."""

    @abc.abstractmethod
    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Modify SL / TP on an open position."""

    # ── Position queries ──────────────────────────────────────────

    @abc.abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List open positions, optionally filtered by symbol."""

    @abc.abstractmethod
    def get_bot_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List positions opened by this bot (magic-number filtered)."""

    @abc.abstractmethod
    def get_trade_history(
        self,
        hours: int = 24,
        include_all: bool = False,
    ) -> List[Dict[str, Any]]:
        """Recent closed trades."""
