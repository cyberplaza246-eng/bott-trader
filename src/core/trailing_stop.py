"""
Trailing Stop Manager

Manages active positions with dynamic stop-loss movement:
  1. After price moves 1× SL distance in profit → move SL to breakeven
  2. After price reaches ~75% of TP distance → partial close (lock in profit)
  3. Then trail SL behind price at 1.5× ATR distance
  4. Locks in profit as trade runs; lets winners ride

Works with both relay (live) and simulation modes.
"""
from src.utils.logger import bot_logger


class TrailingStopManager:
    """Track and update trailing stops for open positions."""

    def __init__(self, breakeven_r=0.8, trail_atr_mult=0.8, partial_close_pct=0.50):
        """
        Args:
            breakeven_r:      Move SL to breakeven after price moves this many R
                              (0.8 = once profit equals 80% of the original risk).
            trail_atr_mult:   Once at breakeven, trail SL at this × ATR behind price.
            partial_close_pct: Close half the position when price reaches this fraction
                               of the TP distance (0.50 = 50% = 1.0R on a 1.4R target).
        """
        self.breakeven_r = breakeven_r
        self.trail_atr_mult = trail_atr_mult
        self.partial_close_pct = partial_close_pct

        # Track per-ticket metadata: {ticket: {entry, sl, direction, atr, at_breakeven}}
        self._tracking = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, ticket, entry_price, stop_loss, direction, atr, pair, take_profit=None, volume=None, scalping_mode=False, quick_wins=False):
        """Register a new position for trailing stop management.

        Args:
            ticket:      Broker ticket ID
            entry_price: Entry price
            stop_loss:   Original stop-loss price
            direction:   'BUY' or 'SELL'
            atr:         ATR at time of entry
            pair:        Currency pair (for pip rounding)
            take_profit: Take-profit price (for partial close logic)
            volume:      Position volume/lot size (for partial close)
            scalping_mode: If True, use tighter breakeven/trail/partial settings
            quick_wins:  If True, use ULTRA tight settings (takes small wins fast)
        """
        # Enhanced profit protection mode: More aggressive settings for 5m scalping
        if quick_wins:
            effective_breakeven_r = 1.0       # Breakeven at 1.0R (full risk covered)
            effective_trail_mult = 1.5        # Trail at 1.5× ATR (room to breathe)
            effective_partial_pct = 0.50      # Partial close at 50% of TP
        # Scalping mode: Improved settings for better profit protection
        elif scalping_mode:
            effective_breakeven_r = 1.0       # Breakeven at 1.0R (full risk covered)
            effective_trail_mult = 1.2        # Trail at 1.2× ATR
            effective_partial_pct = 0.50      # Partial close at 50% of TP
        else:
            effective_breakeven_r = self.breakeven_r
            effective_trail_mult = self.trail_atr_mult
            effective_partial_pct = self.partial_close_pct

        self._tracking[ticket] = {
            'ticket': ticket,            # Add ticket for logging
            'entry': entry_price,
            'original_sl': stop_loss,
            'current_sl': stop_loss,
            'direction': direction,
            'atr': atr,
            'pair': pair,
            'take_profit': take_profit,
            'volume': volume,
            'at_breakeven': False,
            'partial_closed': False,     # True after first partial close
            'best_price': entry_price,   # Best price seen so far
            'scalping_mode': scalping_mode,
            'quick_wins': quick_wins,
            'breakeven_r': effective_breakeven_r,
            'trail_atr_mult': effective_trail_mult,
            'partial_close_pct': effective_partial_pct,
        }
        mode_label = " [QUICK_WINS]" if quick_wins else (" [SCALP]" if scalping_mode else "")
        bot_logger.info(
            f"📌 Trailing registered{mode_label}: ticket={ticket} {pair} {direction} "
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
        if positions is None:
            # Relay/broker failure — do NOT clear tracking; positions may still be open
            bot_logger.warning("Trailing: broker returned None — skipping update (keeping tracking)")
            return []
        if not positions:
            # Genuinely no open positions — clean up tracking
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

            # ── Partial close check ──────────────────────────────────
            # When price reaches ~75% of TP distance, close half to lock profit
            partial_result = self._check_partial_close(info, current_price, ticket, broker)
            if partial_result:
                results.append(partial_result)

            # ── Trailing SL check ────────────────────────────────────
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

    # ── Partial Close Logic ─────────────────────────────────────────

    def _check_partial_close(self, info, current_price, ticket, broker):
        """Close half the position when price reaches ~75% of TP distance.

        This locks in profit before price can reverse from near-TP levels
        (common at S/R zones and round numbers).

        Returns:
            Close result dict if partial close was executed, else None
        """
        # Skip if already partially closed, no TP set, or volume too small
        if info['partial_closed']:
            return None
        tp = info.get('take_profit')
        volume = info.get('volume')
        if not tp or not volume:
            return None

        # Need at least 0.02 lots to split (0.01 minimum per side)
        if volume < 0.02:
            return None

        entry = info['entry']
        direction = info['direction']
        tp_distance = abs(tp - entry)

        # Use per-position partial close threshold
        partial_pct = info.get('partial_close_pct', self.partial_close_pct)

        if direction == 'BUY':
            profit_distance = current_price - entry
        else:
            profit_distance = entry - current_price

        # Check if price has reached the partial close threshold
        if tp_distance > 0 and profit_distance >= tp_distance * partial_pct:
            close_volume = round(volume / 2, 2)
            close_volume = max(0.01, close_volume)  # Broker minimum

            pair = info.get('pair', '')
            pip_div = 0.01 if 'JPY' in pair.upper() else 0.0001
            digits = 3 if 'JPY' in pair.upper() else 5

            bot_logger.info(
                f"💰 Partial close triggered: ticket={ticket} {pair} {direction} "
                f"price={current_price:.{digits}f} reached {partial_pct*100:.0f}% of TP "
                f"({profit_distance/pip_div:.1f}/{tp_distance/pip_div:.1f} pips) — "
                f"closing {close_volume} of {volume} lots"
            )

            result = broker.close_position(pair=pair, volume=close_volume, ticket=ticket)
            if result:
                info['partial_closed'] = True
                info['volume'] = round(volume - close_volume, 2)
                bot_logger.info(
                    f"✅ Partial close filled: ticket={ticket} closed {close_volume} lots, "
                    f"{info['volume']} lots remaining"
                )
                return {'ticket': ticket, 'action': 'partial_close', 'closed_volume': close_volume}
            else:
                bot_logger.warning(f"⚠️ Partial close failed: ticket={ticket}")

        return None

    # ── SL Computation ───────────────────────────────────────────────

    def _compute_new_sl(self, info, current_price):
        """Compute the new SL based on breakeven and trailing logic."""
        entry = info['entry']
        direction = info['direction']
        atr = info['atr']
        original_risk = abs(entry - info['original_sl'])

        # Use per-position overrides (scalping vs swing)
        breakeven_r = info.get('breakeven_r', self.breakeven_r)
        trail_mult = info.get('trail_atr_mult', self.trail_atr_mult)

        pip = 0.01 if 'JPY' in info.get('pair', '') else 0.0001

        if direction == 'BUY':
            profit_distance = current_price - entry
            # Track best price
            if current_price > info['best_price']:
                info['best_price'] = current_price

            # Phase 1: Move to breakeven after profit meets breakeven_r
            if not info['at_breakeven'] and profit_distance >= original_risk * breakeven_r:
                info['at_breakeven'] = True
                bot_logger.info(f"🔒 Breakeven: {info['pair']} ticket={info.get('ticket', 'unknown')} - securing +2 pips")
                return entry + 2 * pip
            
            # Phase 2: Trail behind the best price
            if info['at_breakeven']:
                trail_distance = atr * trail_mult
                trail_sl = info['best_price'] - trail_distance
                # Never go below breakeven
                breakeven_sl = entry + 2 * pip
                return max(trail_sl, breakeven_sl)

        elif direction == 'SELL':
            profit_distance = entry - current_price
            if current_price < info['best_price']:
                info['best_price'] = current_price

            # Phase 1: Breakeven
            if not info['at_breakeven'] and profit_distance >= original_risk * breakeven_r:
                info['at_breakeven'] = True
                bot_logger.info(f"🔒 Breakeven: {info['pair']} ticket={info.get('ticket', 'unknown')} - securing +2 pips")
                return entry - 2 * pip

            # Phase 2: Trail above best price
            if info['at_breakeven']:
                trail_distance = atr * trail_mult
                trail_sl = info['best_price'] + trail_distance
                breakeven_sl = entry - 2 * pip
                return min(trail_sl, breakeven_sl)

        return None

    @property
    def tracked_count(self):
        return len(self._tracking)
