"""
Prop Firm Guard — Hard safety layer for Lucid Trading evaluation account.

Rules enforced:
  1. Trailing drawdown kill switch ($48K floor on $50K account)
  2. Max daily loss ($400 default, configurable)
  3. Max contracts per trade (default 2)
  4. Max open positions (default 3)
  5. Maintenance window block (CME daily 4-5pm CT)
  6. Consecutive loss cooldown (pause after N losses in a row)
  7. End-of-day flatten (all positions closed before session close)

If ANY check fails, the guard returns can_trade=False and the bot MUST NOT
open new positions.  The kill switch is irreversible for the session — once
equity touches the floor, trading stops until manual reset.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, time as dt_time, timezone
from typing import Optional

from src.instruments import is_maintenance_window
from src.utils.logger import bot_logger, error_logger


class PropGuard:
    """Hard safety guard for prop firm evaluation accounts."""

    def __init__(
        self,
        account_size: float = 50_000.0,
        max_drawdown: float = 2_000.0,
        daily_loss_limit: float = 400.0,
        max_contracts: int = 2,
        max_positions: int = 3,
        consec_loss_pause: int = 3,
        flatten_before: dt_time = dt_time(21, 45),  # UTC — 15 min before CME close
    ):
        self.account_size = account_size
        self.max_drawdown = max_drawdown
        self.kill_floor = account_size - max_drawdown  # $48,000
        self.daily_loss_limit = daily_loss_limit
        self.max_contracts = max_contracts
        self.max_positions = max_positions
        self.consec_loss_pause = consec_loss_pause
        self.flatten_before = flatten_before

        # Runtime state
        self._high_water_mark = account_size
        self._trailing_floor = self.kill_floor
        self._daily_pnl = 0.0
        self._daily_start_equity = account_size
        self._consecutive_losses = 0
        self._kill_switch_active = False
        self._open_positions = 0
        self._lock = threading.Lock()

        bot_logger.info(
            f"PropGuard initialized: ${account_size:,.0f} account | "
            f"Kill floor ${self.kill_floor:,.0f} | "
            f"Daily loss limit ${daily_loss_limit:,.0f} | "
            f"Max {max_contracts} contracts, {max_positions} positions"
        )

    # ── Core gate — call before every trade ───────────────────────

    def can_trade(self, symbol: str, equity: float, contracts: int = 1) -> tuple[bool, str]:
        """Check all 7 safety rules. Returns (allowed, reason)."""
        with self._lock:
            # Update high water mark (trailing drawdown)
            if equity > self._high_water_mark:
                self._high_water_mark = equity
                self._trailing_floor = self._high_water_mark - self.max_drawdown

            # 1. Kill switch (irreversible)
            if self._kill_switch_active:
                return False, "KILL SWITCH ACTIVE — session halted"

            if equity <= self._trailing_floor:
                self._kill_switch_active = True
                error_logger.error(
                    f"🛑 KILL SWITCH TRIGGERED: equity ${equity:,.2f} "
                    f"<= floor ${self._trailing_floor:,.2f}"
                )
                return False, f"KILL SWITCH: equity ${equity:,.2f} hit trailing floor"

            # 2. Daily loss limit
            self._daily_pnl = equity - self._daily_start_equity
            if self._daily_pnl <= -self.daily_loss_limit:
                return False, f"Daily loss limit: ${self._daily_pnl:,.2f} / -${self.daily_loss_limit:,.0f}"

            # 3. Max contracts
            if contracts > self.max_contracts:
                return False, f"Max contracts exceeded: {contracts} > {self.max_contracts}"

            # 4. Max positions
            if self._open_positions >= self.max_positions:
                return False, f"Max positions: {self._open_positions}/{self.max_positions}"

            # 5. Maintenance window
            now_utc = datetime.now(timezone.utc)
            if is_maintenance_window(symbol, now_utc):
                return False, f"{symbol} in maintenance window"

            # 6. Consecutive loss cooldown
            if self._consecutive_losses >= self.consec_loss_pause:
                return False, f"Consecutive losses: {self._consecutive_losses} — cooling down"

            # 7. End-of-day flatten window
            if now_utc.time() >= self.flatten_before:
                return False, "EOD flatten window — no new positions"

            return True, "OK"

    # ── State updates ─────────────────────────────────────────────

    def on_trade_opened(self, contracts: int = 1) -> None:
        with self._lock:
            self._open_positions += contracts

    def on_trade_closed(self, pnl: float, contracts: int = 1) -> None:
        with self._lock:
            self._open_positions = max(0, self._open_positions - contracts)
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def sync_positions(self, count: int) -> None:
        with self._lock:
            self._open_positions = count

    def reset_daily(self, equity: float) -> None:
        """Call at start of each trading day."""
        with self._lock:
            self._daily_start_equity = equity
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._kill_switch_active = False
            bot_logger.info(f"PropGuard daily reset: start equity ${equity:,.2f}")

    def should_flatten(self) -> bool:
        """Check if we're in the EOD flatten window."""
        return datetime.now(timezone.utc).time() >= self.flatten_before

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            return {
                "kill_switch": self._kill_switch_active,
                "high_water_mark": self._high_water_mark,
                "trailing_floor": self._trailing_floor,
                "daily_pnl": self._daily_pnl,
                "daily_loss_limit": self.daily_loss_limit,
                "consecutive_losses": self._consecutive_losses,
                "open_positions": self._open_positions,
                "max_positions": self.max_positions,
                "max_contracts": self.max_contracts,
            }
