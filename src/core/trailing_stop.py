"""
Trailing Stop Manager

Manages active positions with dynamic stop-loss movement:
  1. After price moves 1× SL distance in profit → move SL to breakeven
  2. Then trail SL behind price at 1.5× ATR distance
  3. Locks in profit as trade runs; lets winners ride

Works with both relay (live) and simulation modes.
"""
from src.utils.logger import bot_logger


class TrailingStopManager:
    """Track and update trailing stops for open positions."""

    def __init__(self, breakeven_r=1.0, trail_atr_mult=1.5):
        """
        Args:
            breakeven_r:    Move SL to breakeven after price moves this many R
                            (1.0 = once profit equals the original risk distance).
            trail_atr_mult: Once at breakeven, trail SL at this × ATR behind price.
        """
        self.breakeven_r = breakeven_r
        self.trail_atr_mult = trail_atr_mult

        # Track per-ticket metadata: {ticket: {entry, sl, direction, atr, at_breakeven}}
        self._tracking = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, ticket, entry_price, stop_loss, direction, atr, pair):
        """Register a new position for trailing stop management.

        Args:
            ticket:      Broker ticket ID
            entry_price: Entry price
            stop_loss:   Original stop-loss price
            direction:   'BUY' or 'SELL'
            atr:         ATR at time of entry
            pair:        Currency pair (for pip rounding)
        """
        self._tracking[ticket] = {
            'entry': entry_price,
            'original_sl': stop_loss,
            'current_sl': stop_loss,
            'direction': direction,
            'atr': atr,
            'pair': pair,
            'at_breakeven': False,
            'best_price': entry_price,  # Best price seen so far
        }
        bot_logger.info(
            f"📌 Trailing registered: ticket={ticket} {pair} {direction} "
            f"entry={entry_price:.5f} SL={stop_loss:.5f} ATR={atr:.5f}"
        )

    def unregister(self, ticket):
        """Remove a position from tracking (e.g., after it closes)."""
        self._tracking.pop(ticket, None)

    # ── Main Update Loop ─────────────────────────────────────────────

    def update(self, broker):
        """Check all tracked positions and update SL where appropriate.

        Args:
            broker: MT5Connector instance (needs get_open_positions + modify_position)

        Returns:
            List of modification results
        """
        positions = broker.get_open_positions()
        if not positions:
            # All positions may have closed — clean up tracking
            if self._tracking:
                closed_tickets = list(self._tracking.keys())
                for t in closed_tickets:
                    self.unregister(t)
            return []

        # Build a set of currently open tickets
        open_tickets = set()
        for pos in positions:
            open_tickets.add(pos.get('ticket'))

        # Clean up any tracked tickets that are no longer open
        stale = [t for t in self._tracking if t not in open_tickets]
        for t in stale:
            bot_logger.info(f"🗑️ Trailing: ticket {t} closed — removing from tracking")
            self.unregister(t)

        results = []
        for pos in positions:
            ticket = pos.get('ticket')
            if ticket not in self._tracking:
                continue  # Not tracked (existed before bot started)

            info = self._tracking[ticket]
            current_price = pos.get('current_price', 0)
            if not current_price:
                continue

            new_sl = self._compute_new_sl(info, current_price)
            if new_sl and new_sl != info['current_sl']:
                digits = 3 if 'JPY' in info['pair'] else 5
                new_sl = round(new_sl, digits)

                # Only move SL in the profitable direction (never widen risk)
                if info['direction'] == 'BUY' and new_sl <= info['current_sl']:
                    continue
                if info['direction'] == 'SELL' and new_sl >= info['current_sl']:
                    continue

                result = broker.modify_position(ticket, sl=new_sl)
                if result:
                    old_sl = info['current_sl']
                    info['current_sl'] = new_sl
                    pip_div = 0.01 if 'JPY' in info['pair'] else 0.0001
                    bot_logger.info(
                        f"📈 Trailing SL moved: ticket={ticket} {info['pair']} "
                        f"{old_sl:.{digits}f} → {new_sl:.{digits}f} "
                        f"({abs(new_sl - old_sl) / pip_div:.1f} pips)"
                    )
                    results.append(result)

        return results

    # ── SL Computation ───────────────────────────────────────────────

    def _compute_new_sl(self, info, current_price):
        """Compute the new SL based on breakeven and trailing logic."""
        entry = info['entry']
        direction = info['direction']
        atr = info['atr']
        original_risk = abs(entry - info['original_sl'])

        if direction == 'BUY':
            profit_distance = current_price - entry
            # Track best price
            if current_price > info['best_price']:
                info['best_price'] = current_price

            # Phase 1: Move to breakeven after 1R profit
            if not info['at_breakeven'] and profit_distance >= original_risk * self.breakeven_r:
                info['at_breakeven'] = True
                # Set SL to entry + small buffer (1 pip profit)
                pip = 0.01 if 'JPY' in info.get('pair', '') else 0.0001
                return entry + pip
            
            # Phase 2: Trail behind the best price
            if info['at_breakeven']:
                trail_distance = atr * self.trail_atr_mult
                trail_sl = info['best_price'] - trail_distance
                # Never go below breakeven
                pip = 0.01 if 'JPY' in info.get('pair', '') else 0.0001
                breakeven_sl = entry + pip
                return max(trail_sl, breakeven_sl)

        elif direction == 'SELL':
            profit_distance = entry - current_price
            if current_price < info['best_price']:
                info['best_price'] = current_price

            # Phase 1: Breakeven
            if not info['at_breakeven'] and profit_distance >= original_risk * self.breakeven_r:
                info['at_breakeven'] = True
                pip = 0.01 if 'JPY' in info.get('pair', '') else 0.0001
                return entry - pip

            # Phase 2: Trail above best price
            if info['at_breakeven']:
                trail_distance = atr * self.trail_atr_mult
                trail_sl = info['best_price'] + trail_distance
                pip = 0.01 if 'JPY' in info.get('pair', '') else 0.0001
                breakeven_sl = entry - pip
                return min(trail_sl, breakeven_sl)

        return None

    @property
    def tracked_count(self):
        return len(self._tracking)
