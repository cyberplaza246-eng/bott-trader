"""
Evaluation Passing Algorithm — Adaptive mode system for prop firm evaluation.

Two-layer system:
  Layer 1 — Daily Mode (selected at session start based on P&L trajectory)
  Layer 2 — Intraday Scaling (contract management within the day)

Daily Modes:
  AGGRESSIVE  — On track or ahead of target. Full signal set, wider TP.
  NORMAL      — Default. Standard parameters.
  DEFENSIVE   — Behind target or recent losses. Tighter filters, smaller risk.
  PROTECT     — Near profit target ($53K). Micro-risk to cross the finish line.

Intraday Scaling (Hybrid $500/day):
  - Start with 1 contract per trade
  - After +$250 intraday ("house money"), allow 2 contracts on high-confidence
  - Stop trading for the day at +$500 hit or -$400 loss
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import bot_logger


@dataclass
class EvalConfig:
    """Tunable evaluation parameters."""
    account_size: float = 50_000.0
    profit_target: float = 53_000.0      # Pass evaluation at this equity
    daily_target: float = 500.0          # Ideal daily profit
    daily_stop_loss: float = 400.0       # Stop trading after this daily loss
    house_money_threshold: float = 250.0 # Allow scale-up after this intraday profit
    scale_up_confidence: float = 0.70    # Min confidence to use 2 contracts
    protect_zone_buffer: float = 300.0   # Enter PROTECT when within this of target
    defensive_drawdown: float = 500.0    # Enter DEFENSIVE when trailing HWM by this


class EvalAlgorithm:
    """Adaptive evaluation passing algorithm.

    Call `update()` each cycle with current equity. Query `get_mode()` and
    `get_contracts()` before each trade.
    """

    MODES = ("AGGRESSIVE", "NORMAL", "DEFENSIVE", "PROTECT")

    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()
        self._mode = "NORMAL"
        self._day_start_equity = self.cfg.account_size
        self._session_hwm = self.cfg.account_size
        self._intraday_pnl = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0

        bot_logger.info(
            f"EvalAlgorithm initialized: target ${self.cfg.profit_target:,.0f} | "
            f"daily +${self.cfg.daily_target:,.0f} / -${self.cfg.daily_stop_loss:,.0f}"
        )

    # ── Core API ──────────────────────────────────────────────────

    def update(self, equity: float) -> str:
        """Update state and recalculate mode. Returns current mode."""
        self._intraday_pnl = equity - self._day_start_equity
        if equity > self._session_hwm:
            self._session_hwm = equity

        self._mode = self._select_mode(equity)
        return self._mode

    def get_mode(self) -> str:
        return self._mode

    def get_contracts(self, base_contracts: int, confidence: float) -> int:
        """Apply intraday scaling logic to determine actual contracts.

        Rules:
          - PROTECT mode: always 1
          - DEFENSIVE mode: always 1
          - If intraday P&L >= house_money_threshold AND confidence >= threshold: allow 2
          - Otherwise: base_contracts (usually 1)
        """
        if self._mode in ("PROTECT", "DEFENSIVE"):
            return 1

        if (
            self._intraday_pnl >= self.cfg.house_money_threshold
            and confidence >= self.cfg.scale_up_confidence
            and base_contracts >= 1
        ):
            return min(base_contracts + 1, 2)

        return min(base_contracts, 1)

    def should_stop_trading(self) -> tuple[bool, str]:
        """Check if daily target or loss limit has been hit."""
        if self._intraday_pnl >= self.cfg.daily_target:
            return True, f"Daily target hit: +${self._intraday_pnl:,.2f}"
        if self._intraday_pnl <= -self.cfg.daily_stop_loss:
            return True, f"Daily loss limit: ${self._intraday_pnl:,.2f}"
        return False, ""

    def on_trade_closed(self, pnl: float) -> None:
        self._trades_today += 1
        if pnl >= 0:
            self._wins_today += 1
        else:
            self._losses_today += 1

    def reset_daily(self, equity: float) -> None:
        """Call at start of each trading day."""
        self._day_start_equity = equity
        self._session_hwm = equity
        self._intraday_pnl = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0
        self._mode = self._select_mode(equity)
        bot_logger.info(
            f"EvalAlgorithm daily reset: equity ${equity:,.2f}, mode={self._mode}"
        )

    # ── Risk multiplier for the current mode ──────────────────────

    def risk_multiplier(self) -> float:
        """Scale risk % based on current mode."""
        return {
            "AGGRESSIVE": 1.0,
            "NORMAL": 1.0,
            "DEFENSIVE": 0.5,
            "PROTECT": 0.25,
        }.get(self._mode, 1.0)

    def tp_multiplier(self) -> float:
        """Scale TP target based on current mode."""
        return {
            "AGGRESSIVE": 1.2,
            "NORMAL": 1.0,
            "DEFENSIVE": 0.8,
            "PROTECT": 0.6,
        }.get(self._mode, 1.0)

    # ── Mode selection logic ──────────────────────────────────────

    def _select_mode(self, equity: float) -> str:
        # PROTECT: near the profit target finish line
        if equity >= self.cfg.profit_target - self.cfg.protect_zone_buffer:
            return "PROTECT"

        # DEFENSIVE: trailing from session HWM or daily loss building up
        hwm_drawdown = self._session_hwm - equity
        if (
            hwm_drawdown >= self.cfg.defensive_drawdown
            or self._intraday_pnl <= -(self.cfg.daily_stop_loss * 0.5)
        ):
            return "DEFENSIVE"

        # AGGRESSIVE: up nicely on the day (past house money)
        if self._intraday_pnl >= self.cfg.house_money_threshold:
            return "AGGRESSIVE"

        return "NORMAL"

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "mode": self._mode,
            "intraday_pnl": round(self._intraday_pnl, 2),
            "day_start_equity": self._day_start_equity,
            "session_hwm": self._session_hwm,
            "trades_today": self._trades_today,
            "wins_today": self._wins_today,
            "losses_today": self._losses_today,
            "risk_multiplier": self.risk_multiplier(),
            "tp_multiplier": self.tp_multiplier(),
        }
