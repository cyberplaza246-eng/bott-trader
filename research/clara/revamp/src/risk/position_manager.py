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
import os
from src.utils.logger import bot_logger, error_logger
from src.instruments import get_instrument, is_futures, dollar_risk, REGISTRY
from config.strategy_config import (
    RISK_PER_TRADE_PERCENT,
    STOP_LOSS_MULTIPLIER,
    TAKE_PROFIT_RATIO,
    MICRO_TAKE_PROFIT_RATIO,
    INITIAL_BALANCE,
    MAX_CONCURRENT_TRADES,
    DAILY_LOSS_LIMIT_PERCENT,
    SCALPING_PAIRS,
)


# ── Account Tier Definitions ──────────────────────────────────────────
# Each tier unlocks more capacity as your balance grows
ACCOUNT_TIERS = {
    'micro': {
        'min_balance': 0,
        'max_balance': 200,
        'max_concurrent_trades': 3,    # 3 concurrent trades allowed
        'max_lot_size': 0.01,
        'max_contracts': 1,
        'risk_percent': 1.0,       # Conservative at small size
        'description': 'Micro ($0-$200)',
    },
    'mini': {
        'min_balance': 200,
        'max_balance': 1000,
        'max_concurrent_trades': 3,
        'max_lot_size': 0.03,
        'max_contracts': 1,
        'risk_percent': 1.0,
        'description': 'Mini ($200-$1K)',
    },
    'standard': {
        'min_balance': 1000,
        'max_balance': 5000,
        'max_concurrent_trades': 3,
        'max_lot_size': 0.04,
        'max_contracts': 1,
        'risk_percent': 1.5,       # Can afford slightly more risk
        'description': 'Standard ($1K-$5K)',
    },
    'professional': {
        'min_balance': 5000,
        'max_balance': 25000,
        'max_concurrent_trades': 3,
        'max_lot_size': 0.05,
        'max_contracts': 1,
        'risk_percent': 2.0,
        'description': 'Professional ($5K-$25K)',
    },
    'elite': {
        'min_balance': 25000,
        'max_balance': float('inf'),
        'max_concurrent_trades': 3,
        'max_lot_size': 0.05,
        'max_contracts': 2,         # Scale up only at elite tier
        'risk_percent': 2.0,
        'description': 'Elite ($25K+)',
    },
}

# ── Backward-compat aliases — use instrument registry instead ─────────
# Kept so any stale imports (from position_manager import PIP_VALUES) still work.
PIP_VALUES = {
    sym: {'pip_size': spec.tick_size, 'pip_value_per_lot': spec.tick_value_usd}
    for sym, spec in REGISTRY.items()
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
        self.account_leverage = 100  # Will be updated from broker
        self.free_margin = initial_balance * 0.95  # Will be updated from broker
        self.slot3_confidence_threshold = float(os.getenv('SLOT3_CONFIDENCE_THRESHOLD', '0.58'))
        self.slot3_min_agreement = int(os.getenv('SLOT3_MIN_AGREEMENT', '3'))

        # ── Hard Risk Envelope ────────────────────────────────────────
        # Per-trade wallet exposure cap (% of balance risked per trade)
        self.max_trade_exposure_pct = float(os.getenv('MAX_TRADE_EXPOSURE_PCT', '2.0'))
        # Aggregate open exposure cap (% of balance across all open trades)
        self.max_aggregate_exposure_pct = float(os.getenv('MAX_AGGREGATE_EXPOSURE_PCT', '6.0'))
        # Session realized-loss gate: pause trading after losing this % in a session
        self.session_loss_gate_pct = float(os.getenv('SESSION_LOSS_GATE_PCT', '4.0'))
        # Cooldown after session loss gate (seconds)
        self.session_loss_cooldown_sec = int(os.getenv('SESSION_LOSS_COOLDOWN_SEC', '1800'))
        # Equity drawdown hard-stop: enter safe mode if equity drops this % from peak
        self.equity_hardstop_pct = float(os.getenv('EQUITY_HARDSTOP_PCT', '8.0'))
        # Per-symbol kill switch (track consecutive losses per symbol)
        self.symbol_consecutive_loss_limit = int(os.getenv('SYMBOL_CONSEC_LOSS_LIMIT', '4'))

        # Runtime state for hard risk gates
        self._session_realized_loss = 0.0
        self._session_loss_gate_active = False
        self._session_loss_gate_until = 0.0  # timestamp
        self._equity_peak = initial_balance
        self._safe_mode = False
        self._symbol_consec_losses = {}  # symbol -> consecutive loss count
        self._killed_symbols = set()  # symbols disabled by kill switch
        self._aggregate_risk_usd = 0.0  # total $ at risk across open trades

        # Set initial tier & limits (will be recalculated each cycle)
        self._current_tier_name = None
        self._current_tier = None
        self._refresh_tier()

        bot_logger.info(
            f"RiskManager initialized: ${initial_balance:.2f} | "
            f"Tier: {self._current_tier['description']}"
        )
        bot_logger.info(
            f"  Hard risk: exposure cap {self.max_trade_exposure_pct}%/trade, "
            f"{self.max_aggregate_exposure_pct}% agg | "
            f"session gate {self.session_loss_gate_pct}% | "
            f"equity hard-stop {self.equity_hardstop_pct}%"
        )

    # ── Tier Management ───────────────────────────────────────────────

    def _refresh_tier(self):
        """Recalculate the account tier based on current balance.
        
        Uses hysteresis to prevent flapping at tier boundaries:
        - Upgrade: balance must reach tier's min_balance
        - Downgrade: balance must drop 2% below current tier's min_balance
        """
        old_tier = self._current_tier_name
        tier_order = list(ACCOUNT_TIERS.keys())

        if old_tier:
            # Apply hysteresis: require 2% drop below current tier floor to downgrade
            current_min = self._current_tier['min_balance']
            downgrade_threshold = current_min * 0.98  # Must drop 2% below boundary

            if self.current_balance < downgrade_threshold:
                # Need to downgrade — find the correct lower tier
                for name, tier in ACCOUNT_TIERS.items():
                    if tier['min_balance'] <= self.current_balance < tier['max_balance']:
                        self._current_tier_name = name
                        self._current_tier = tier
                        break
            elif self.current_balance >= self._current_tier['max_balance']:
                # Upgrade — find the correct higher tier
                for name, tier in ACCOUNT_TIERS.items():
                    if tier['min_balance'] <= self.current_balance < tier['max_balance']:
                        self._current_tier_name = name
                        self._current_tier = tier
                        break
            # else: stay in current tier (within hysteresis band)
        else:
            # First-time initialization
            for name, tier in ACCOUNT_TIERS.items():
                if tier['min_balance'] <= self.current_balance < tier['max_balance']:
                    self._current_tier_name = name
                    self._current_tier = tier
                    break

        # Recalculate daily loss limit from live balance
        self.daily_loss_limit = self.current_balance * (DAILY_LOSS_LIMIT_PERCENT / 100)

        # Minimum balance to keep trading (20% of current balance, floor $1)
        self.min_balance_threshold = max(1.0, self.current_balance * 0.20)

        if old_tier and old_tier != self._current_tier_name:
            old_idx = tier_order.index(old_tier) if old_tier in tier_order else 0
            new_idx = tier_order.index(self._current_tier_name) if self._current_tier_name in tier_order else 0
            direction = 'UPGRADE' if new_idx > old_idx else 'DOWNGRADE'
            emoji = '🎯' if direction == 'UPGRADE' else '⚠️'
            bot_logger.info(
                f"{emoji} ACCOUNT TIER {direction}: {old_tier} → {self._current_tier_name} "
                f"| Balance: ${self.current_balance:.2f} "
                f"| Max lots: {self._current_tier['max_lot_size']} "
                f"| Max trades: {self._current_tier['max_concurrent_trades']}"
            )

    def get_tier_info(self):
        """Return current tier details for dashboard / logging."""
        growth = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        next_tier = self._next_tier()
        max_trades = self._max_trades_cap()
        return {
            'tier_name': self._current_tier_name,
            'tier_description': self._current_tier['description'],
            'max_lot_size': self._current_tier['max_lot_size'],
            'max_concurrent_trades': max_trades,
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

    def _max_trades_cap(self):
        """Absolute concurrent-trade cap policy.

        Requirement: keep max at 3 while account is below $200.
        """
        if self.current_balance < 200:
            return 3
        return self._current_tier['max_concurrent_trades']

    # ── Balance Sync ──────────────────────────────────────────────────

    def sync_balance(self, broker_balance, leverage=None, free_margin=None):
        """Sync balance, leverage, free margin from broker (call each cycle)."""
        if broker_balance and broker_balance > 0:
            self.current_balance = broker_balance
            if leverage and leverage > 0:
                self.account_leverage = leverage
            if free_margin is not None:
                self.free_margin = free_margin
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
        max_trades = self._max_trades_cap()

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

    def can_trade_with_market(self, signal_result=None):
        """Check if a new trade can be opened with market-aware trade capacity.

        Micro accounts are capped to 3 concurrent trades (or lower if configured).
        """
        effective_cap, _ = self.get_trade_capacity(signal_result)

        can_open = (
            self.open_trades < effective_cap and
            self.daily_loss < self.daily_loss_limit and
            self.current_balance > self.min_balance_threshold
        )

        if not can_open:
            if self.open_trades >= effective_cap:
                bot_logger.warning(
                    f"Max concurrent trades ({effective_cap}) reached "
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

    def get_trade_capacity(self, signal_result=None):
        """Return (effective_cap, available_slots) for current market/account context."""
        # Hard business rule: while account < $200, cap is always exactly 3.
        effective_cap = self._max_trades_cap()

        available_slots = max(0, effective_cap - self.open_trades)
        return effective_cap, available_slots

    # ── Hard Risk Envelope ────────────────────────────────────────────

    def check_hard_risk_gates(self, symbol=None, risk_amount_usd=0.0):
        """Check all hard risk gates before allowing a new trade.

        Returns (allowed: bool, reason: str).  Call BEFORE opening any trade.
        """
        import time

        # 1. Safe-mode hard-stop (equity drawdown)
        if self._safe_mode:
            return False, "SAFE_MODE: equity drawdown hard-stop active"

        # Update equity peak
        if self.current_balance > self._equity_peak:
            self._equity_peak = self.current_balance
        dd_pct = ((self._equity_peak - self.current_balance) / self._equity_peak) * 100 if self._equity_peak > 0 else 0
        if dd_pct >= self.equity_hardstop_pct:
            self._safe_mode = True
            bot_logger.warning(
                f"🛑 EQUITY HARD-STOP: DD {dd_pct:.1f}% >= {self.equity_hardstop_pct}% "
                f"(peak ${self._equity_peak:.2f} → ${self.current_balance:.2f}) — entering safe mode"
            )
            return False, f"EQUITY_HARDSTOP: {dd_pct:.1f}% drawdown"

        # 2. Session realized-loss gate
        if self._session_loss_gate_active:
            if time.time() < self._session_loss_gate_until:
                remaining = int(self._session_loss_gate_until - time.time())
                return False, f"SESSION_LOSS_GATE: cooldown {remaining}s remaining"
            else:
                self._session_loss_gate_active = False
                bot_logger.info("✅ Session loss gate cooldown expired — trading resumed")

        session_loss_pct = (self._session_realized_loss / self.daily_starting_balance) * 100 if self.daily_starting_balance > 0 else 0
        if session_loss_pct >= self.session_loss_gate_pct:
            self._session_loss_gate_active = True
            self._session_loss_gate_until = time.time() + self.session_loss_cooldown_sec
            bot_logger.warning(
                f"⛔ SESSION LOSS GATE: lost {session_loss_pct:.1f}% >= {self.session_loss_gate_pct}% "
                f"— pausing {self.session_loss_cooldown_sec}s"
            )
            return False, f"SESSION_LOSS_GATE: {session_loss_pct:.1f}% session loss"

        # 3. Per-trade exposure cap
        if risk_amount_usd > 0 and self.current_balance > 0:
            trade_exposure_pct = (risk_amount_usd / self.current_balance) * 100
            if trade_exposure_pct > self.max_trade_exposure_pct:
                return False, (
                    f"TRADE_EXPOSURE: {trade_exposure_pct:.1f}% > "
                    f"{self.max_trade_exposure_pct}% cap"
                )

        # 4. Aggregate exposure cap
        if self.current_balance > 0:
            new_agg = self._aggregate_risk_usd + risk_amount_usd
            agg_pct = (new_agg / self.current_balance) * 100
            if agg_pct > self.max_aggregate_exposure_pct:
                return False, (
                    f"AGGREGATE_EXPOSURE: {agg_pct:.1f}% > "
                    f"{self.max_aggregate_exposure_pct}% cap"
                )

        # 5. Per-symbol kill switch
        if symbol and symbol in self._killed_symbols:
            return False, f"SYMBOL_KILLED: {symbol} disabled after {self.symbol_consecutive_loss_limit} consecutive losses"

        return True, "OK"

    def record_trade_pnl(self, symbol, pnl):
        """Record a completed trade's PnL for hard risk gate tracking."""
        # Session loss tracking
        if pnl < 0:
            self._session_realized_loss += abs(pnl)

        # Per-symbol consecutive loss tracking
        if symbol:
            if pnl < 0:
                self._symbol_consec_losses[symbol] = self._symbol_consec_losses.get(symbol, 0) + 1
                if self._symbol_consec_losses[symbol] >= self.symbol_consecutive_loss_limit:
                    self._killed_symbols.add(symbol)
                    bot_logger.warning(
                        f"🔴 SYMBOL KILL SWITCH: {symbol} disabled after "
                        f"{self._symbol_consec_losses[symbol]} consecutive losses"
                    )
            else:
                # Reset on win
                self._symbol_consec_losses[symbol] = 0
                # Re-enable if previously killed and now winning
                if symbol in self._killed_symbols:
                    self._killed_symbols.discard(symbol)
                    bot_logger.info(f"✅ {symbol} re-enabled after winning trade")

    def add_open_risk(self, risk_usd):
        """Track aggregate open risk when a new position is opened."""
        self._aggregate_risk_usd += risk_usd

    def remove_open_risk(self, risk_usd):
        """Remove aggregate open risk when a position is closed."""
        self._aggregate_risk_usd = max(0.0, self._aggregate_risk_usd - risk_usd)

    def reset_session(self):
        """Reset session-level risk counters (call at start of each trading day)."""
        self._session_realized_loss = 0.0
        self._session_loss_gate_active = False
        self.daily_starting_balance = self.current_balance
        self.daily_loss = 0.0
        bot_logger.info(f"📅 Session reset: starting balance ${self.current_balance:.2f}")

    def exit_safe_mode(self):
        """Manually exit safe mode after review."""
        self._safe_mode = False
        self._equity_peak = self.current_balance
        bot_logger.info(f"✅ Safe mode exited. New equity peak: ${self.current_balance:.2f}")

    @property
    def is_safe_mode(self):
        return self._safe_mode

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

        # Fixed lot size override (from env)
        fixed_lot = float(os.getenv('FIXED_LOT_SIZE', '0'))
        if fixed_lot > 0:
            position_size = min(fixed_lot, max_lots)
            bot_logger.info(
                f"Position sizing [{self._current_tier_name}]: "
                f"Balance ${self.current_balance:.2f} | "
                f"FIXED lot {position_size:.2f} (max {max_lots})"
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

        position_size = max(0.01, min(position_size, max_lots))

        # Apply margin safety cap using ACTUAL account leverage & free margin
        leverage = self.account_leverage
        if pair and is_futures(pair):
            # Futures: margin is per-contract, not per-lot
            spec = get_instrument(pair)
            margin_per_lot = spec.margin_per_contract if spec else 1000.0
        elif pair and pair.upper().startswith('USD/'):
            margin_per_lot = 100_000 / leverage
        else:
            margin_per_lot = (entry_price * 100_000) / leverage
        if margin_per_lot > 0:
            # Use 80% of free margin (leave 20% buffer for spread/slippage)
            usable_margin = self.free_margin * 0.80
            max_lots_by_margin = usable_margin / margin_per_lot
            if position_size > max_lots_by_margin:
                bot_logger.info(
                    f"Margin cap: {position_size:.2f} lots → {max_lots_by_margin:.2f} lots "
                    f"(free margin ${self.free_margin:.2f}, leverage {leverage}:1)"
                )
                position_size = max(0.01, min(position_size, max_lots_by_margin))

        # Round to 2 decimal places
        position_size = round(position_size, 2)

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

    # ── Futures Position Sizing ───────────────────────────────────────

    def calculate_contracts(self, symbol, entry_price, stop_loss_price):
        """Calculate position size in contracts for futures instruments.

        Safety caps (applied in order, most restrictive wins):
          1. Risk-based: risk_amount / dollar_risk_per_contract
          2. Tier max_contracts cap
          3. Margin cap: 80% of free_margin / margin_intraday
          4. Hard cap: max 2 contracts (Lucid $50K eval rule)

        Returns:
            dict with contracts, risk_amount, dollar_risk, tick_distance, tier
            or None if the trade should be skipped.
        """
        spec = get_instrument(symbol)

        risk_pct = self._current_tier['risk_percent']
        risk_amount = self.current_balance * (risk_pct / 100)

        # Dollar risk per contract
        risk_per_contract = dollar_risk(symbol, entry_price, stop_loss_price, contracts=1)
        if risk_per_contract <= 0:
            error_logger.error(f"Invalid stop distance for {symbol}")
            return None

        # Cap 1: risk-based sizing
        contracts = int(risk_amount / risk_per_contract)
        contracts = max(1, contracts)

        # Cap 2: tier-based
        max_contracts = self._current_tier.get('max_contracts', 1)
        contracts = min(contracts, max_contracts)

        # Cap 3: margin-based (80% of free margin)
        if spec.margin_intraday > 0:
            usable_margin = self.free_margin * 0.80
            margin_cap = int(usable_margin / spec.margin_intraday)
            margin_cap = max(1, margin_cap)
            if contracts > margin_cap:
                bot_logger.info(
                    f"Margin cap: {contracts} → {margin_cap} contracts "
                    f"(free margin ${self.free_margin:.2f})"
                )
                contracts = margin_cap

        # Cap 4: hard absolute cap (Lucid eval safety)
        hard_cap = int(os.getenv('MAX_CONTRACTS', '2'))
        contracts = min(contracts, hard_cap)

        actual_risk = risk_per_contract * contracts

        bot_logger.info(
            f"Futures sizing [{self._current_tier_name}]: "
            f"{symbol} {contracts} contract(s) | "
            f"Risk ${actual_risk:.2f} ({risk_pct}% of ${self.current_balance:.2f}) | "
            f"${risk_per_contract:.2f}/contract"
        )

        return {
            'contracts': contracts,
            'risk_amount': round(actual_risk, 2),
            'risk_percent': risk_pct,
            'risk_per_contract': round(risk_per_contract, 2),
            'tier': self._current_tier_name,
            'max_contracts': max_contracts,
        }

    # Minimum stop distance per pair type (in price units)
    # Most brokers require 3-5 pips; we use 8 pips as safe default
    MIN_STOP_DISTANCE = {
        'JPY':     0.080,    # 8 pips (3-digit pairs)
        'DEFAULT': 0.00080,  # 8 pips (5-digit pairs)
    }

    @staticmethod
    def _price_digits(pair=None):
        """Return decimal places for instrument (uses registry when available)."""
        if pair and pair in REGISTRY:
            return REGISTRY[pair].decimal_places
        if pair and 'JPY' in pair.upper():
            return 3
        return 5

    def _min_stop_distance(self, pair=None):
        """Return the minimum allowed stop distance for a pair."""
        if pair and pair in REGISTRY:
            # 8 ticks minimum for any instrument
            return REGISTRY[pair].tick_size * 8
        if pair and 'JPY' in pair.upper():
            return self.MIN_STOP_DISTANCE['JPY']
        return self.MIN_STOP_DISTANCE['DEFAULT']

    def calculate_stop_loss(self, entry_price, atr, trade_type, pair=None):
        """
        Calculate stop-loss based on ATR

        Args:
            entry_price: Entry price
            atr: Average True Range
            trade_type: 'BUY' or 'SELL'
            pair: Currency pair for rounding precision

        Returns:
            Stop-loss price (rounded to broker precision)
        """
        # Tier-aware SL multiplier — give trades room to breathe
        if 'micro' in (self._current_tier_name or 'micro'):
            sl_mult = STOP_LOSS_MULTIPLIER     # 1.5 — same as standard (was 1.2, too tight)
        elif 'mini' in (self._current_tier_name or ''):
            sl_mult = STOP_LOSS_MULTIPLIER     # 1.5
        else:
            sl_mult = 1.8                       # more room for larger accounts
        sl_distance = atr * sl_mult
        min_dist = self._min_stop_distance(pair)
        if sl_distance < min_dist:
            _spec = REGISTRY.get(pair)
            _ts = _spec.tick_size if _spec else (0.01 if pair and 'JPY' in pair.upper() else 0.0001)
            bot_logger.info(
                f"SL distance widened: {sl_distance:.5f} → {min_dist:.5f} "
                f"(minimum {min_dist / _ts:.0f} ticks)"
            )
            sl_distance = min_dist

        if trade_type == 'BUY':
            stop_loss = entry_price - sl_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance

        return round(stop_loss, self._price_digits(pair))

    def calculate_take_profit(self, entry_price, stop_loss_price, trade_type, pair=None):
        """
        Calculate take-profit based on risk:reward ratio

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            trade_type: 'BUY' or 'SELL'
            pair: Currency pair for rounding precision

        Returns:
            Take-profit price (rounded to broker precision)
        """
        # Risk is distance to stop-loss
        risk = abs(entry_price - stop_loss_price)

        # Reward is risk * ratio (micro tier uses tighter TP)
        if 'micro' in (self._current_tier_name or 'micro'):
            tp_ratio = MICRO_TAKE_PROFIT_RATIO
        else:
            tp_ratio = TAKE_PROFIT_RATIO
        reward = risk * tp_ratio

        # TP buffer: pull TP back ~1.5 ticks so the order fills before
        # price stalls at S/R zones / round numbers and reverses.
        _spec = REGISTRY.get(pair)
        pip_size = _spec.tick_size if _spec else (0.01 if (pair and 'JPY' in pair.upper()) else 0.0001)
        tp_buffer = pip_size * 1.5

        if trade_type == 'BUY':
            take_profit = entry_price + reward - tp_buffer
        else:  # SELL
            take_profit = entry_price - reward + tp_buffer

        return round(take_profit, self._price_digits(pair))

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

    def sync_open_trades(self, actual_count: int):
        """Sync internal open_trades counter with actual broker positions.
        
        This prevents drift between the internal counter and MT5 reality.
        """
        if actual_count != self.open_trades:
            bot_logger.info(f"📊 Syncing open_trades: {self.open_trades} → {actual_count}")
            self.open_trades = actual_count

    def get_daily_status(self, actual_open_trades: int = None):
        """Get daily trading status (includes tier info).
        
        Args:
            actual_open_trades: If provided, use this instead of internal counter
        """
        # Use actual count if provided, otherwise fall back to internal counter
        open_trades = actual_open_trades if actual_open_trades is not None else self.open_trades
        
        daily_loss_percent = (self.daily_loss / max(self.daily_starting_balance, 1)) * 100
        tier_info = self.get_tier_info()
        
        # Calculate capacity using actual open trades
        max_trades = self._current_tier['max_concurrent_trades']
        effective_cap = max_trades
        available_slots = max(0, effective_cap - open_trades)
        
        # can_trade should use actual position count
        can_trade = (
            open_trades < max_trades and
            self.daily_loss <= self.daily_loss_limit and
            self.current_balance > 0
        )

        return {
            'current_balance': self.current_balance,
            'daily_starting_balance': self.daily_starting_balance,
            'daily_loss': self.daily_loss,
            'daily_loss_percent': daily_loss_percent,
            'daily_loss_limit': self.daily_loss_limit,
            'open_trades': open_trades,  # Use actual count
            'max_concurrent_trades': max_trades,
            'effective_trade_cap': effective_cap,
            'available_trade_slots': available_slots,
            'can_trade': can_trade,  # Use actual count
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

    # ── Scalping-Specific SL/TP ──────────────────────────────────────

    def calculate_scalping_stop_loss(self, pair, timeframe_key, entry_price, trade_type,
                                     atr_value=None):
        """Calculate ATR-based stop loss for scalping.

        Uses 0.8 × ATR(14) when atr_value is provided; otherwise falls back to
        a conservative default based on the pair's pip_size from SCALPING_PAIRS.

        Args:
            pair: Currency pair (e.g. 'EUR/USD')
            timeframe_key: '1m' or '5m'
            entry_price: Entry price
            trade_type: 'BUY' or 'SELL'
            atr_value: Current ATR(14) value (preferred)

        Returns:
            Stop-loss price (rounded to 5 decimals)
        """
        pair_config = SCALPING_PAIRS.get(pair, {})
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)

        if atr_value and atr_value > 0:
            sl_distance = 0.8 * atr_value
        else:
            # Conservative fallback: 8 pips
            sl_distance = 8 * pip_info['pip_size']

        if trade_type == 'BUY':
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

        if is_futures(pair):
            spec = get_instrument(pair)
            digits = 2 if spec else 5
        else:
            digits = 3 if 'JPY' in pair.upper() else 5
        return round(stop_loss, digits)

    def calculate_scalping_take_profit(self, pair, timeframe_key, entry_price, stop_loss, trade_type):
        """Calculate take profit for scalping using R:R ratio from config.

        Args:
            pair: Currency pair
            timeframe_key: '1m' or '5m'
            entry_price: Entry price
            stop_loss: Stop-loss price
            trade_type: 'BUY' or 'SELL'

        Returns:
            Take-profit price (rounded to 5 decimals)
        """
        pair_config = SCALPING_PAIRS.get(pair, {})
        tp_ratio = 2.0  # ATR base TP ratio (was 1.4 — too tight for sweep entries)

        risk_distance = abs(entry_price - stop_loss)
        tp_distance = risk_distance * tp_ratio

        # Minimal buffer (0.3 pips slippage)
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
        tp_buffer = pip_info['pip_size'] * 0.3

        if trade_type == 'BUY':
            take_profit = entry_price + tp_distance - tp_buffer
        else:
            take_profit = entry_price - tp_distance + tp_buffer

        if is_futures(pair):
            spec = get_instrument(pair)
            digits = 2 if spec else 5
        else:
            digits = 3 if 'JPY' in pair.upper() else 5
        return round(take_profit, digits)
