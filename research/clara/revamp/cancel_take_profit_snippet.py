    def cancel_take_profit(
        self,
        ticket: str,
        symbol: str,
        *,
        reason: str = "past_tp_trail",
    ) -> bool:
        """Cancel working TP leg(s) for an open position; never cancel SL.

        Used when price trades beyond the original take-profit so winners can
        run under ATR trailing with SL floored at the old TP price.
        """
        if not self._connected or not ticket or not symbol:
            return False
        try:
            return bool(
                self._run_sync(
                    self._async_cancel_take_profit(
                        str(ticket),
                        str(symbol),
                        reason=reason,
                    ),
                    timeout=20,
                )
            )
        except Exception as e:
            error_logger.error(
                f"Rithmic cancel_take_profit failed for {ticket} {symbol}: {e}"
            )
            return False

    async def _async_cancel_take_profit(
        self,
        order_id: str,
        symbol: str,
        *,
        reason: str = "past_tp_trail",
    ) -> bool:
        """Cancel protective/native TP tags only; leave SL working."""
        _, tp_ids = self._bracket_leg_ids(order_id)
        with self._order_lock:
            order = dict(self._orders.get(order_id, {}) or {})
            extra = order.get("protective_tp_order_id")
            if extra and extra not in tp_ids:
                tp_ids = list(tp_ids) + [str(extra)]

        # Deduplicate while preserving order
        seen = set()
        unique_tp_ids = []
        for tp_id in tp_ids:
            if not tp_id or tp_id in seen:
                continue
            seen.add(tp_id)
            unique_tp_ids.append(tp_id)

        if not unique_tp_ids:
            with self._order_lock:
                stored = self._orders.get(order_id)
                if stored is not None:
                    stored["take_profit"] = 0
                    stored["native_target_attached"] = False
                    stored.pop("protective_tp_order_id", None)
            return True

        cancelled_or_gone = False
        cancel_failed = False
        for tp_id in unique_tp_ids:
            try:
                working = await self._async_is_order_working(tp_id)
            except Exception:
                working = True  # attempt cancel anyway
            if not working:
                cancelled_or_gone = True
                continue
            ok = await self._async_cancel_protective_leg(
                tp_id,
                symbol=symbol,
                reason=reason,
                require_flat=False,
                cancel_path="past_tp_clear",
            )
            if ok:
                cancelled_or_gone = True
            else:
                cancel_failed = True

        if cancelled_or_gone and not cancel_failed:
            with self._order_lock:
                stored = self._orders.get(order_id)
                if stored is not None:
                    stored["take_profit"] = 0
                    stored["native_target_attached"] = False
                    stored.pop("protective_tp_order_id", None)
            bot_logger.info(
                f"Rithmic cancelled TP for open {order_id} {symbol} "
                f"(reason={reason}); SL left working"
            )
            return True

        if cancelled_or_gone:
            # Partial success — still clear local TP state so we do not re-arm.
            with self._order_lock:
                stored = self._orders.get(order_id)
                if stored is not None:
                    stored["take_profit"] = 0
                    stored["native_target_attached"] = False
                    stored.pop("protective_tp_order_id", None)
            bot_logger.warning(
                f"Rithmic partial TP cancel for {order_id} {symbol} (reason={reason})"
            )
            return True

        bot_logger.error(
            f"Rithmic could not cancel any TP leg for {order_id} {symbol} "
            f"(reason={reason})"
        )
        return False

