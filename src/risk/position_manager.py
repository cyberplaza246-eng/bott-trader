"""
Risk Management System with Dynamic Account Scaling

As your account grows from $50 upward, the bot automatically adjusts:
  - Position sizes (lot sizes scale with balance)
  - Max concurrent trades (more trades as account grows)
  - Daily loss limits (recalculated from live balance)
  - Lot size caps (grow with account tier)
  - Pair-specific pip values (JPY pairs vs standard pairs)
  - Minimum balance thresholds (proportional to account size)
"""
import numpy as np
from src.utils.logger import bot_logger, error_logger
from config.strategy_config import (
    RISK_PER_TRADE_PERCENT,
    STOP_LOSS_MULTIPLIER,
    TAKE_PROFIT_RATIO,
    INITIAL_BALANCE,
    MAX_CONCURRENT_TRADES,
    DAILY_LOSS_LIMIT_PERCENT
)


# ── Account Tier Definitions ──────────────────────────────────────────
# Each tier unlocks more capacity as your balance grows
ACCOUNT_TIERS = {
    'micro': {
        'min_balance': 0,
        'max_balance': 200,
        'max_concurrent_trades': 2,
        'max_lot_size': 0.05,
        'risk_percent': 1.0,       # Conservative at small size
        'description': 'Micro ($0-$200)',
    },
    'mini': {
        'min_balance': 200,
        'max_balance': 1000,
        'max_concurrent_trades': 3,
        'max_lot_size': 0.5,
        'risk_percent': 1.0,
        'description': 'Mini ($200-$1K)',
    },
    'standard': {
        'min_balance': 1000,
        'max_balance': 5000,
        'max_concurrent_trades': 5,
        'max_lot_size': 2.0,
        'risk_percent': 1.5,       # Can afford slightly more risk
        'description': 'Standard ($1K-$5K)',
    },
    'professional': {
        'min_balance': 5000,
        'max_balance': 25000,
        'max_concurrent_trades': 8,
        'max_lot_size': 5.0,
        'risk_percent': 2.0,
        'description': 'Professional ($5K-$25K)',
    },
    'elite': {
        'min_balance': 25000,
        'max_balance': float('inf'),
        'max_concurrent_trades': 10,
        'max_lot_size': 10.0,
        'risk_percent': 2.0,
        'description': 'Elite ($25K+)',
    },
}

# ── Pair-specific pip values ──────────────────────────────────────────
# pip value per standard lot (100,000 units) in USD
PIP_VALUES = {
    'EUR/USD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0},
    'GBP/USD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0},
    'USD/JPY': {'pip_size': 0.01,   'pip_value_per_lot': 6.5},   # ~$6.50 per pip at ~153 JPY
    'AUD/USD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0},
    'NZD/USD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0},
    'USD/CHF': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0},
    'USD/CAD': {'pip_size': 0.0001, 'pip_value_per_lot': 7.5},
}
DEFAULT_PIP = {'pip_size': 0.0001, 'pip_value_per_lot': 10.0}


class RiskManager:
    """Manages position sizing, stop-loss, take-profit, and risk limits.

    Dynamically scales all parameters as the account balance grows.
    """

    def __init__(self, initial_balance=INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_starting_balance = initial_balance
        self.open_trades = 0
        self.daily_loss = 0.0

        # Set initial tier & limits (will be recalculated each cycle)
        self._current_tier_name = None
        self._current_tier = None
        self._refresh_tier()

        bot_logger.info(
            f"RiskManager initialized: ${initial_balance:.2f} | "
            f"Tier: {self._current_tier['description']}"
        )

    # ── Tier Management ───────────────────────────────────────────────

    def _refresh_tier(self):
        """Recalculate the account tier based on current balance."""
        old_tier = self._current_tier_name
        for name, tier in ACCOUNT_TIERS.items():
            if tier['min_balance'] <= self.current_balance < tier['max_balance']:
                self._current_tier_name = name
                self._current_tier = tier
                break

        # Recalculate daily loss limit from live balance
        self.daily_loss_limit = self.current_balance * (DAILY_LOSS_LIMIT_PERCENT / 100)

        # Minimum balance to keep trading (20% of current balance, floor $5)
        self.min_balance_threshold = max(5.0, self.current_balance * 0.20)

        if old_tier and old_tier != self._current_tier_name:
            bot_logger.info(
                f"🎯 ACCOUNT TIER UPGRADE: {old_tier} → {self._current_tier_name} "
                f"| Balance: ${self.current_balance:.2f} "
                f"| Max lots: {self._current_tier['max_lot_size']} "
                f"| Max trades: {self._current_tier['max_concurrent_trades']}"
            )

    def get_tier_info(self):
        """Return current tier details for dashboard / logging."""
        growth = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        next_tier = self._next_tier()
        return {
            'tier_name': self._current_tier_name,
            'tier_description': self._current_tier['description'],
            'max_lot_size': self._current_tier['max_lot_size'],
            'max_concurrent_trades': self._current_tier['max_concurrent_trades'],
            'risk_percent': self._current_tier['risk_percent'],
            'account_growth': round(growth, 2),
            'next_tier': next_tier['description'] if next_tier else 'MAX',
            'next_tier_at': next_tier['min_balance'] if next_tier else None,
            'balance_to_next': round(next_tier['min_balance'] - self.current_balance, 2) if next_tier else 0,
        }

    def _next_tier(self):
        """Return the next tier above the current one, or None if at max."""
        tier_order = list(ACCOUNT_TIERS.keys())
        idx = tier_order.index(self._current_tier_name)
        if idx + 1 < len(tier_order):
            return ACCOUNT_TIERS[tier_order[idx + 1]]
        return None

    # ── Balance Sync ──────────────────────────────────────────────────

    def sync_balance(self, broker_balance):
        """Sync balance from broker and refresh tier (call each cycle)."""
        if broker_balance and broker_balance > 0:
            self.current_balance = broker_balance
            self._refresh_tier()

    def update_balance(self, new_balance):
        """Update current account balance"""
        self.current_balance = new_balance
        daily_loss = self.daily_starting_balance - new_balance
        self.daily_loss = max(0, daily_loss)
        self._refresh_tier()

    # ── Trade Gating ──────────────────────────────────────────────────

    def can_trade(self):
        """Check if we can open new trades (uses dynamic tier limits)."""
        max_trades = self._current_tier['max_concurrent_trades']

        can_open = (
            self.open_trades < max_trades and
            self.daily_loss < self.daily_loss_limit and
            self.current_balance > self.min_balance_threshold
        )

        if not can_open:
            if self.open_trades >= max_trades:
                bot_logger.warning(
                    f"Max concurrent trades ({max_trades}) reached "
                    f"[Tier: {self._current_tier_name}]"
                )
            if self.daily_loss >= self.daily_loss_limit:
                bot_logger.warning(
                    f"Daily loss limit (${self.daily_loss_limit:.2f} / "
                    f"{DAILY_LOSS_LIMIT_PERCENT}%) reached - pausing trades"
                )
            if self.current_balance <= self.min_balance_threshold:
                bot_logger.warning(
                    f"Balance ${self.current_balance:.2f} below minimum "
                    f"${self.min_balance_threshold:.2f} - pausing trades"
                )

        return can_open

    # ── Position Sizing ───────────────────────────────────────────────

    def calculate_position_size(self, entry_price, stop_loss_price, pair=None):
        """
        Calculate position size based on dynamic risk management.

        Scales lot size with account tier, uses pair-specific pip values.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            pair: Currency pair (e.g. 'EUR/USD') for pip value lookup

        Returns:
            dict with lot_size, risk_amount, risk_percent, pip_distance, tier
        """
        # Use tier-appropriate risk percentage
        risk_pct = self._current_tier['risk_percent']
        risk_amount = self.current_balance * (risk_pct / 100)

        # Get pair-specific pip info
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP) if pair else DEFAULT_PIP
        pip_value_per_lot = pip_info['pip_value_per_lot']
        pip_size = pip_info['pip_size']

        # Calculate distance to stop-loss in price units
        pip_distance = abs(entry_price - stop_loss_price)

        if pip_distance < pip_size * 0.1:  # Prevent division by zero
            error_logger.error("Invalid stop-loss distance")
            return None

        # Convert price distance to pip count, then to dollar risk
        pip_count = pip_distance / pip_size
        risk_per_lot = pip_count * pip_value_per_lot
        position_size = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01

        # Apply tier-based lot cap
        max_lots = self._current_tier['max_lot_size']
        position_size = max(0.01, min(position_size, max_lots))

        bot_logger.info(
            f"Position sizing [{self._current_tier_name}]: "
            f"Balance ${self.current_balance:.2f} | "
            f"Risk {risk_pct}% = ${risk_amount:.2f} | "
            f"Lot: {position_size:.2f} (max {max_lots})"
        )

        return {
            'lot_size': round(position_size, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percent': risk_pct,
            'pip_distance': pip_distance,
            'pip_count': round(pip_count, 1),
            'tier': self._current_tier_name,
            'max_lot_size': max_lots,
        }

    def calculate_stop_loss(self, entry_price, atr, trade_type):
        """
        Calculate stop-loss based on ATR

        Args:
            entry_price: Entry price
            atr: Average True Range
            trade_type: 'BUY' or 'SELL'

        Returns:
            Stop-loss price
        """
        if trade_type == 'BUY':
            stop_loss = entry_price - (atr * STOP_LOSS_MULTIPLIER)
        else:  # SELL
            stop_loss = entry_price + (atr * STOP_LOSS_MULTIPLIER)

        return stop_loss

    def calculate_take_profit(self, entry_price, stop_loss_price, trade_type):
        """
        Calculate take-profit based on risk:reward ratio

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            trade_type: 'BUY' or 'SELL'

        Returns:
            Take-profit price
        """
        # Risk is distance to stop-loss
        risk = abs(entry_price - stop_loss_price)

        # Reward is risk * ratio
        reward = risk * TAKE_PROFIT_RATIO

        if trade_type == 'BUY':
            take_profit = entry_price + reward
        else:  # SELL
            take_profit = entry_price - reward

        return take_profit

    def on_trade_opened(self):
        """Called when a new trade is opened"""
        self.open_trades += 1
        max_trades = self._current_tier['max_concurrent_trades']
        bot_logger.info(
            f"Trade opened. Open positions: {self.open_trades}/{max_trades} "
            f"[Tier: {self._current_tier_name}]"
        )

    def on_trade_closed(self, profit_loss):
        """Called when a trade is closed"""
        self.open_trades = max(0, self.open_trades - 1)
        self.current_balance += profit_loss
        self.daily_loss -= profit_loss  # Reduce loss if profitable

        # Refresh tier in case balance crossed a boundary
        self._refresh_tier()

        bot_logger.info(
            f"Trade closed - P/L: {profit_loss:+.2f} | "
            f"Balance: {self.current_balance:.2f} | "
            f"Tier: {self._current_tier_name} | "
            f"Open positions: {self.open_trades}"
        )

    def get_daily_status(self):
        """Get daily trading status (includes tier info)."""
        daily_loss_percent = (self.daily_loss / max(self.daily_starting_balance, 1)) * 100
        tier_info = self.get_tier_info()

        return {
            'current_balance': self.current_balance,
            'daily_starting_balance': self.daily_starting_balance,
            'daily_loss': self.daily_loss,
            'daily_loss_percent': daily_loss_percent,
            'daily_loss_limit': self.daily_loss_limit,
            'open_trades': self.open_trades,
            'max_concurrent_trades': self._current_tier['max_concurrent_trades'],
            'can_trade': self.can_trade(),
            'margin_available': self.current_balance * 0.95,
            # Scaling info
            'tier': tier_info['tier_name'],
            'tier_description': tier_info['tier_description'],
            'max_lot_size': tier_info['max_lot_size'],
            'risk_percent': tier_info['risk_percent'],
            'account_growth': tier_info['account_growth'],
            'next_tier': tier_info['next_tier'],
            'balance_to_next': tier_info['balance_to_next'],
        }

    def reset_daily_limits(self, new_balance):
        """Reset daily limits at end of day"""
        self.current_balance = new_balance
        self.daily_starting_balance = new_balance
        self.daily_loss = 0.0
        self._refresh_tier()
        bot_logger.info(
            f"Daily limits reset. Balance: ${new_balance:.2f} | "
            f"Tier: {self._current_tier_name} | "
            f"Daily loss limit: ${self.daily_loss_limit:.2f}"
        )
