"""
Order flow tracker — rolling tick/trade delta from Rithmic BBO + LAST_TRADE.

Tier 1 advisory: log flow each scan; block mode gates entries by buy/sell pressure.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class _SymbolFlow:
    trades: Deque[Tuple[float, str, float]] = field(default_factory=deque)
    last_trade_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0


class TickFlowTracker:
    """Thread-safe rolling buy/sell volume from classified trades."""

    def __init__(self, window_sec: Optional[float] = None):
        self._lock = threading.Lock()
        self._window_sec = float(
            window_sec if window_sec is not None else os.getenv("ORDER_FLOW_WINDOW_SEC", "60")
        )
        self._symbols: Dict[str, _SymbolFlow] = {}

    def _state(self, symbol: str) -> _SymbolFlow:
        if symbol not in self._symbols:
            self._symbols[symbol] = _SymbolFlow()
        return self._symbols[symbol]

    @staticmethod
    def _first_float(data: dict, *keys: str) -> Optional[float]:
        for key in keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _classify_side(
        price: float,
        size: float,
        aggressor: Optional[Any],
        bid: float,
        ask: float,
        last_price: float,
    ) -> Optional[str]:
        """Return 'buy' or 'sell' for a trade, or None if unknown."""
        if aggressor is not None:
            ag = str(aggressor).upper()
            if ag in ("BUY", "1", "TRANSACTIONTYPE_BUY"):
                return "buy"
            if ag in ("SELL", "2", "TRANSACTIONTYPE_SELL"):
                return "sell"

        if bid > 0 and ask > 0 and price > 0:
            mid = (bid + ask) / 2.0
            if price >= ask - 1e-9:
                return "buy"
            if price <= bid + 1e-9:
                return "sell"
            if price > mid:
                return "buy"
            if price < mid:
                return "sell"

        if last_price > 0 and price > 0:
            if price > last_price:
                return "buy"
            if price < last_price:
                return "sell"

        return None

    def record_bbo(self, symbol: str, data: dict) -> None:
        """Capture best bid/offer sizes when present."""
        bid = self._first_float(
            data, "best_bid_price", "bid_price", "best_bid", "bid",
        )
        ask = self._first_float(
            data, "best_ask_price", "ask_price", "best_ask", "ask",
        )
        bid_size = self._first_float(
            data, "best_bid_size", "bid_size",
        )
        ask_size = self._first_float(
            data, "best_ask_size", "ask_size",
        )
        if bid is None and ask is None and bid_size is None and ask_size is None:
            return

        with self._lock:
            st = self._state(symbol)
            if bid is not None:
                st.bid = bid
            if ask is not None:
                st.ask = ask
            if bid_size is not None:
                st.bid_size = bid_size
            if ask_size is not None:
                st.ask_size = ask_size

    def record_trade(self, symbol: str, data: dict) -> None:
        """Classify and append a last-trade tick."""
        price = self._first_float(data, "trade_price", "last", "price")
        if price is None or price <= 0:
            return

        size = self._first_float(data, "trade_size", "trade_qty", "size", "qty")
        if size is None or size <= 0:
            size = 1.0

        aggressor = data.get("aggressor")

        with self._lock:
            st = self._state(symbol)
            side = self._classify_side(
                price, size, aggressor, st.bid, st.ask, st.last_trade_price,
            )
            st.last_trade_price = price
            if side is None:
                return

            now = time.time()
            st.trades.append((now, side, size))
            cutoff = now - self._window_sec
            while st.trades and st.trades[0][0] < cutoff:
                st.trades.popleft()

    def _prune(self, st: _SymbolFlow, now: float) -> None:
        cutoff = now - self._window_sec
        while st.trades and st.trades[0][0] < cutoff:
            st.trades.popleft()

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Rolling-window flow metrics for scan loop / entry gating."""
        now = time.time()
        with self._lock:
            st = self._state(symbol)
            self._prune(st, now)

            buy_vol = 0.0
            sell_vol = 0.0
            tick_count = 0
            for _, side, qty in st.trades:
                tick_count += 1
                if side == "buy":
                    buy_vol += qty
                else:
                    sell_vol += qty

            total = buy_vol + sell_vol
            delta = buy_vol - sell_vol
            buy_pct = (buy_vol / total) if total > 0 else 0.5
            sell_pct = 1.0 - buy_pct

            bbo_imbalance: Optional[float] = None
            if st.bid_size > 0 or st.ask_size > 0:
                bbo_total = st.bid_size + st.ask_size
                if bbo_total > 0:
                    bbo_imbalance = (st.bid_size - st.ask_size) / bbo_total

            return {
                "symbol": symbol,
                "window_sec": self._window_sec,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "delta": delta,
                "tick_count": tick_count,
                "buy_pct": buy_pct,
                "sell_pct": sell_pct,
                "bid_size": st.bid_size,
                "ask_size": st.ask_size,
                "bbo_imbalance": bbo_imbalance,
                "last_price": st.last_trade_price,
            }

    @staticmethod
    def format_scan_line(snapshot: Optional[Dict[str, Any]] = None) -> str:
        """e.g. Flow 60s (MNQ): delta=-420 | 62% sell"""
        if not snapshot:
            return "Flow: no data"
        sym = snapshot.get("symbol", "?")
        window = int(snapshot.get("window_sec", 60))
        ticks = int(snapshot.get("tick_count", 0))
        if ticks == 0:
            return f"Flow {window}s ({sym}): warming up — no classified trades yet"

        delta = snapshot.get("delta", 0)
        buy_pct = float(snapshot.get("buy_pct", 0.5))
        sell_pct = float(snapshot.get("sell_pct", 0.5))
        dominant = "buy" if buy_pct >= sell_pct else "sell"
        dominant_pct = max(buy_pct, sell_pct) * 100.0

        line = (
            f"Flow {window}s ({sym}): delta={delta:+.0f} | "
            f"{dominant_pct:.0f}% {dominant}"
        )
        imb = snapshot.get("bbo_imbalance")
        if imb is not None:
            line += f" | BBO imb={imb:+.2f}"
        return line

    @staticmethod
    def confirms_direction(
        direction: str,
        snapshot: Optional[Dict[str, Any]],
        min_ticks: int = 3,
    ) -> bool:
        """True when rolling flow agrees with the proposed entry direction."""
        if not snapshot or int(snapshot.get("tick_count", 0)) < min_ticks:
            return False
        delta = float(snapshot.get("delta", 0))
        buy_pct = float(snapshot.get("buy_pct", 0.5))
        wants_long = direction.lower() in ("long", "buy")
        if wants_long:
            return delta > 0 or buy_pct >= 0.52
        return delta < 0 or buy_pct <= 0.48

    def evaluate_entry(
        self,
        direction: str,
        symbol: str,
        mode: Optional[str] = None,
        min_ticks: int = 3,
    ) -> Dict[str, Any]:
        """
        Check whether order flow supports the proposed entry.

        Long: delta > 0 or buy_pct > 0.55
        Short: delta < 0 or buy_pct < 0.45
        Advisory mode: never block; block mode: reject when flow disagrees.
        """
        mode = (mode or os.getenv("ORDER_FLOW_MODE", "advisory")).lower()
        snap = self.get_snapshot(symbol)
        ticks = int(snap.get("tick_count", 0))

        if ticks < min_ticks:
            return {
                "allowed": True,
                "reason": "insufficient flow data",
                "advisory_note": f"Flow {int(snap['window_sec'])}s: only {ticks} trades — not gating",
                "snapshot": snap,
            }

        delta = float(snap.get("delta", 0))
        buy_pct = float(snap.get("buy_pct", 0.5))
        wants_long = direction.lower() in ("long", "buy")

        if wants_long:
            flow_ok = delta > 0 or buy_pct > 0.55
            conflict = not flow_ok
            advisory_note = (
                f"Flow supports LONG (delta={delta:+.0f}, {buy_pct:.0%} buy)"
                if flow_ok
                else f"Flow weak for LONG (delta={delta:+.0f}, {buy_pct:.0%} buy)"
            )
        else:
            flow_ok = delta < 0 or buy_pct < 0.45
            conflict = not flow_ok
            advisory_note = (
                f"Flow supports SHORT (delta={delta:+.0f}, {buy_pct:.0%} buy)"
                if flow_ok
                else f"Flow weak for SHORT (delta={delta:+.0f}, {buy_pct:.0%} buy)"
            )

        allowed = True
        reason = "advisory only" if mode == "advisory" else "passed"
        if mode == "block" and conflict:
            allowed = False
            dir_label = direction.upper()
            reason = (
                f"order flow conflicts with {dir_label}: "
                f"delta={delta:+.0f}, buy={buy_pct:.0%}"
            )

        return {
            "allowed": allowed,
            "reason": reason,
            "advisory_note": advisory_note,
            "snapshot": snap,
            "flow_ok": flow_ok,
        }
