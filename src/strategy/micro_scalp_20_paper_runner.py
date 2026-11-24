"""Lucid paper runner for MNQ micro-scalp $20 / min-hold 10s."""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz

from src.ai.action_log import ActionLog, write_status
from src.broker.rithmic_connector import RithmicConnector, apply_lucid_ticker_order_plants
from src.data.paper_csv_feed import paper_rithmic_brackets_enabled
from src.strategy.micro_scalp_20 import (
    EMA_FAST,
    EMA_SLOW,
    IMPULSE_PTS,
    LOOKBACK_SEC,
    MIN_HOLD_SEC,
    OpenPos,
    ScalpState,
    TP_USD,
    check_entry,
    check_exit,
    on_closed,
    recipe_banner_lines,
    rth_only_enabled,
    in_rth_et,
    unrealized_usd,
    real_tp_price,
)
from src.strategy.overnight_research import day_bot_owns_lucid, skip_local_fake_fill
from src.utils.redact import redact_secrets

ET = pytz.timezone("US/Eastern")
SESSION_NAME = "micro_scalp_20"
SCAN_SEC = float(os.getenv("MICRO_SCALP20_SCAN_SEC", "0.75"))
HEARTBEAT_SEC = float(os.getenv("MICRO_SCALP20_HEARTBEAT_SEC", "30"))
FLATTEN_ATTEMPTS = int(os.getenv("MICRO_SCALP20_FLATTEN_ATTEMPTS", "10"))
CLEAR_ROUNDS = int(os.getenv("MICRO_SCALP20_CLEAR_ROUNDS", "4"))
PROTECT_ATTEMPTS = int(os.getenv("MICRO_SCALP20_PROTECT_ATTEMPTS", "8"))


class MicroScalp20PaperEngine:
    def __init__(self) -> None:
        self.rithmic: Optional[RithmicConnector] = None
        self.log = ActionLog()
        self.state = ScalpState()
        self._last_hb = 0.0
        self._last_protect_check = 0.0

    def _connect(self) -> None:
        os.environ["PAPER_USE_RITHMIC"] = "true"
        os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
        _ov = os.getenv("RITHMIC_CONTRACT_OVERRIDES", "")
        _up = _ov.upper()
        if "MNQ=MNQM6" in _up or "MNQ=MNQU6" in _up:
            os.environ["RITHMIC_CONTRACT_OVERRIDES"] = "MNQ=MNQZ6"
            print("Rolled RITHMIC_CONTRACT_OVERRIDES MNQ → MNQZ6 (front month)")
        if day_bot_owns_lucid():
            print("⚠️  Stop day hybrid / overnight paper first — one Lucid session only.")
        if not paper_rithmic_brackets_enabled():
            print("Set PAPER_RITHMIC_BRACKETS=true for Lucid sim orders")
            return
        try:
            apply_lucid_ticker_order_plants()
            raw = RithmicConnector(live_mode=False, quotes_only=False)
            raw._symbols_to_watch = ["MNQ"]
            raw._disable_yahoo_fallback = True
            raw.initialize()
            if raw.connected:
                self.rithmic = raw
                try:
                    ensure = getattr(raw, "_ensure_market_data", None)
                    if callable(ensure):
                        ensure()
                except Exception as exc:
                    print(f"⚠️  Market data ensure: {redact_secrets(exc)}")
                got = False
                for _ in range(40):
                    if self._quote() is not None:
                        got = True
                        break
                    time.sleep(0.5)
                if got:
                    print("🔗 Lucid sim connected (TEST) — MNQ ticker live. Close R|Trader if logged in elsewhere.")
                else:
                    print(
                        "🔗 Lucid sim connected (TEST) but no MNQ ticker yet after ~20s. "
                        "Close R|Trader / other Lucid sessions, check RITHMIC_CONTRACT_OVERRIDES → MNQZ6."
                    )
            else:
                err = redact_secrets(getattr(raw, "_last_connect_error", "") or "not connected")
                print(f"❌ Lucid connect failed ({err})")
                try:
                    raw.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            print(f"❌ Rithmic error: {redact_secrets(exc)}")

    def _quote(self) -> Optional[float]:
        if self.rithmic is None or not self.rithmic.connected:
            return None
        try:
            q = self.rithmic.get_latest_price("MNQ") or {}
            px = float(q.get("last") or q.get("bid") or 0)
            if px > 0:
                return px
            cached = getattr(self.rithmic, "_get_cached_price", None)
            if callable(cached):
                px2 = float(cached("MNQ") or 0)
                if px2 > 0:
                    return px2
            lock = getattr(self.rithmic, "_quote_lock", None)
            quotes = getattr(self.rithmic, "_quotes", None) or {}
            if lock is not None:
                with lock:
                    raw_q = dict(quotes.get("MNQ") or {})
            else:
                raw_q = dict(quotes.get("MNQ") or {})
            px3 = float(raw_q.get("last") or raw_q.get("bid") or raw_q.get("ask") or 0)
            return px3 if px3 > 0 else None
        except Exception:
            return None

    def _contract_label(self) -> str:
        if self.rithmic is None:
            return "MNQZ6"
        try:
            sym, _ = self.rithmic._resolve_symbol("MNQ")  # noqa: SLF001
            return str(sym or "MNQZ6")
        except Exception:
            return os.getenv("RITHMIC_CONTRACT_OVERRIDES", "MNQ=MNQZ6").split("=")[-1] or "MNQZ6"

    def _chart_howto(self, direction: str, sl: float, tp: float) -> None:
        """Lucid TEST sim labels SL/TP as stop + limit working orders."""
        contract = self._contract_label()
        if direction == "long":
            sl_txt = f"SELL STOP @ {sl:.2f}  = Stop Loss"
            tp_txt = f"SELL LIMIT @ {tp:.2f}  = Take Profit (${TP_USD:.0f})"
        else:
            sl_txt = f"BUY STOP @ {sl:.2f}  = Stop Loss"
            tp_txt = f"BUY LIMIT @ {tp:.2f}  = Take Profit (${TP_USD:.0f})"
        print("", flush=True)
        print("=" * 60, flush=True)
        print(f"  LUCID Working Orders — {contract}", flush=True)
        print(f"  {sl_txt}", flush=True)
        print(f"  {tp_txt}", flush=True)
        print(
            "  Lucid TEST has no Stop Loss / Take Profit labels — those two orders ARE SL/TP.",
            flush=True,
        )
        print("=" * 60, flush=True)
        print("", flush=True)

    def _legs_ok(self, ticket: str, sl: float, tp: float) -> Tuple[bool, bool]:
        if self.rithmic is None or not self.rithmic.connected:
            return False, False
        try:
            sl_ok, tp_ok = self.rithmic.query_broker_protection(
                ticket, stop_loss=sl, take_profit=tp
            )
        except Exception as exc:
            print(f"  query_broker_protection: {redact_secrets(exc)}", flush=True)
            sl_ok, tp_ok = False, False
        # Only trust working protective ids — never fake SL/TP from unused stop_ticks.
        if not sl_ok or not tp_ok:
            try:
                with self.rithmic._order_lock:  # noqa: SLF001
                    order = dict(self.rithmic._orders.get(str(ticket), {}) or {})  # noqa: SLF001
                if not sl_ok and order.get("protective_sl_order_id"):
                    sl_ok = True
                if not tp_ok and order.get("protective_tp_order_id"):
                    tp_ok = True
            except Exception:
                pass
        return bool(sl_ok), bool(tp_ok)

    def _force_protect(
        self,
        ticket: str,
        side: str,
        sl: float,
        tp: float,
        *,
        attempts: int = PROTECT_ATTEMPTS,
    ) -> bool:
        """
        Attach SL+TP and confirm both resting.

        Do NOT call verify_and_ensure_protection here — on Lucid ticker+order
        (no PnL plant) it can falsely decide 'flat' and CANCEL the legs.
        """
        if self.rithmic is None or not self.rithmic.connected:
            return False
        # Keep fill-anchored prices on the order record before every ensure.
        try:
            with self.rithmic._order_lock:  # noqa: SLF001
                stored = self.rithmic._orders.setdefault(str(ticket), {"ticket": ticket})  # noqa: SLF001
                stored["stop_loss"] = float(sl)
                stored["take_profit"] = float(tp)
                stored["symbol"] = "MNQ"
                stored["type"] = side
        except Exception:
            pass
        for i in range(1, max(1, attempts) + 1):
            try:
                self.rithmic.ensure_protective_orders(
                    ticket, "MNQ", side, 1, sl, tp
                )
            except Exception as exc:
                print(f"  ensure_protective[{i}]: {redact_secrets(exc)}", flush=True)
            time.sleep(0.8 + 0.2 * i)
            sl_ok, tp_ok = self._legs_ok(ticket, sl, tp)
            print(
                f"  protect[{i}/{attempts}] SL={'yes' if sl_ok else 'NO'} "
                f"TP={'yes' if tp_ok else 'NO'} @ SL {sl:.2f} / TP {tp:.2f}",
                flush=True,
            )
            if sl_ok and tp_ok:
                return True
        return False

    def _guard_open_protection(self) -> None:
        """Re-attach missing SL/TP while a trade is open (every ~5s)."""
        pos = self.state.open
        if pos is None or pos.exit_pending or self.rithmic is None:
            return
        now = time.time()
        if now - self._last_protect_check < 5.0:
            return
        self._last_protect_check = now
        side = "buy" if pos.direction == "long" else "sell"
        sl_ok, tp_ok = self._legs_ok(pos.ticket, pos.sl, pos.tp_far)
        if sl_ok and tp_ok:
            return
        print(
            f"⚠️  Missing Lucid legs (SL={'ok' if sl_ok else 'MISSING'} "
            f"TP={'ok' if tp_ok else 'MISSING'}) — re-attaching…",
            flush=True,
        )
        self._force_protect(pos.ticket, side, pos.sl, pos.tp_far, attempts=4)
        self._chart_howto(pos.direction, pos.sl, pos.tp_far)

    def _working_bot_count(self) -> Tuple[int, List[str]]:
        if self.rithmic is None or not self.rithmic.connected:
            return -1, []
        try:
            rows = self.rithmic.list_working_orders("MNQ") or []
        except Exception as exc:
            print(f"  list_working failed: {redact_secrets(exc)}", flush=True)
            return -1, []
        kinds: List[str] = []
        n = 0
        for row in rows:
            try:
                owned = self.rithmic._is_bot_owned_listed_order(row)  # noqa: SLF001
            except Exception:
                owned = True
            if not owned:
                continue
            n += 1
            kind = str(
                row.get("price_type")
                or row.get("order_type")
                or row.get("transaction_type")
                or "?"
            )
            kinds.append(kind)
        return n, kinds

    def _clear_chart(self, ticket: str = "", *, reason: str = "exit") -> bool:
        """Cancel bot_* tickets only — never cancel/flatten your manual Lucid trades."""
        if self.rithmic is None or not self.rithmic.connected:
            return False
        working_left = -1
        for round_i in range(1, max(1, CLEAR_ROUNDS) + 1):
            try:
                n = self.rithmic.cancel_all_bot_orders("MNQ")
                print(
                    f"  clear[{round_i}] cancelled_bot={n} "
                    f"(your manual orders/positions left alone)",
                    flush=True,
                )
            except Exception as exc:
                print(f"  clear[{round_i}] cancel_bot: {redact_secrets(exc)}", flush=True)
            if ticket:
                try:
                    self.rithmic.cancel_entry_bracket_legs(
                        ticket, "MNQ", reason=f"micro_scalp_{reason}"
                    )
                except Exception:
                    pass
            try:
                self.rithmic.cancel_working_orders_after_exit(
                    "MNQ",
                    ticket or "",
                    reason=f"micro_scalp_{reason}",
                    already_flat=True,
                )
            except Exception:
                pass
            time.sleep(0.5)
            working_left, kinds = self._working_bot_count()
            print(
                f"  clear[{round_i}] working_bot={working_left} kinds={kinds}",
                flush=True,
            )
            if working_left == 0:
                return True
        return working_left == 0

    def _place(self, sig: Dict[str, Any], ref: float) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print("❌ Lucid brackets required — PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true")
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("⏳ No Lucid order plant — skip entry")
            return False
        left, kinds = self._working_bot_count()
        if left and left > 0:
            print(
                f"🧹 {left} leftover bot order(s) {kinds} — clearing bot tickets only "
                f"(manual trades untouched)…",
                flush=True,
            )
            if not self._clear_chart(reason="pre_entry"):
                print("❌ Bot chart not clear — skip entry", flush=True)
                return False

        side = "buy" if sig["direction"] == "long" else "sell"
        # Pre-compute fill-anchored brackets from signal ref; re-anchor after fill.
        sl = float(sig["sl"])
        tp = float(sig["tp_far"])
        try:
            result = self.rithmic.place_order(
                symbol="MNQ",
                order_type=side,
                size=1,
                entry_price=ref,
                stop_loss=sl,
                take_profit=tp,
            )
        except Exception as exc:
            print(f"❌ place_order failed: {redact_secrets(exc)}")
            return False
        if not result or not (result.get("ticket") or result.get("order_id")):
            print("❌ Lucid order not booked")
            return False

        fill = float(result.get("entry_price") or result.get("price_ref") or ref)
        ticket = str(result.get("ticket") or result.get("order_id"))
        # Always re-anchor both legs to the real fill + live SL/TP distances.
        from src.strategy.micro_scalp_20 import brackets_from_fill

        sl, tp = brackets_from_fill(sig["direction"], fill)
        # Force nearby $20 TP even if connector returned a far placeholder.
        tp = real_tp_price(sig["direction"], fill)
        # Prefer fill-anchored prices returned by place_order when present.
        if float(result.get("stop_loss") or 0) > 0:
            sl = float(result["stop_loss"])
        if float(result.get("take_profit") or 0) > 0:
            tp = float(result["take_profit"])

        already = bool(
            result.get("protective_sl_order_id") and result.get("protective_tp_order_id")
        )
        mode = str(result.get("bracket_mode") or "")
        if already:
            print(
                f"🛡️  Lucid SL {sl:.2f} + TP {tp:.2f} submitted with fill {fill:.2f} "
                f"(mode={mode})",
                flush=True,
            )
            ok = self._force_protect(ticket, side, sl, tp, attempts=2)
        else:
            print(
                f"🛡️  Attaching Lucid SL {sl:.2f} + TP {tp:.2f} on fill {fill:.2f}…",
                flush=True,
            )
            ok = self._force_protect(ticket, side, sl, tp)
        if not ok:
            print(
                "❌ SL/TP never confirmed on Lucid — flattening this entry (will not leave naked).",
                flush=True,
            )
            try:
                # Known entry side — close opposite without inventing size from history.
                # Position net: long = +qty (flatten sells), short = -qty (flatten buys).
                close_net = 1 if side.lower() in ("buy", "long") else -1
                rith_sym, exchange = self.rithmic._resolve_symbol("MNQ")  # noqa: SLF001
                self.rithmic._run_sync(  # noqa: SLF001
                    self.rithmic._async_submit_flatten_market_order(  # noqa: SLF001
                        symbol="MNQ",
                        broker_sym=rith_sym,
                        exchange=exchange,
                        net_qty=close_net,
                    ),
                    timeout=20,
                )
                self.rithmic.cancel_working_orders_after_exit(
                    "MNQ", ticket, reason="protect_failed", already_flat=True
                )
            except Exception as exc:
                print(f"  abort flatten: {redact_secrets(exc)}", flush=True)
            return False

        self.state.open = OpenPos(
            direction=sig["direction"],
            entry=fill,
            sl=sl,
            tp_far=tp,
            entry_ts=time.time(),
            ticket=ticket,
        )
        emoji = "📈" if sig["direction"] == "long" else "📉"
        print(
            f"{emoji} ENTRY {sig['direction'].upper()} @ {fill:.2f} | "
            f"SL {sl:.2f} | Lucid TP {tp:.2f} (${TP_USD:.0f}) | #{ticket}",
            flush=True,
        )
        self._chart_howto(sig["direction"], sl, tp)
        # Show what Lucid should list so Stop vs Limit is obvious.
        try:
            rows = self.rithmic.list_working_orders("MNQ") or []
            bot_rows = [
                r for r in rows
                if self.rithmic._is_bot_owned_listed_order(r)  # noqa: SLF001
            ]
            if bot_rows:
                print("  Lucid working bot legs:", flush=True)
                for r in bot_rows:
                    ptype = str(
                        r.get("price_type") or r.get("order_type") or "?"
                    ).upper()
                    tag = str(r.get("user_tag") or r.get("basket_id") or "")
                    role = "?"
                    if tag.endswith("_sl") or "STOP" in ptype:
                        role = "Stop Loss"
                    elif tag.endswith("_tp") or ptype == "LIMIT":
                        role = "Take Profit"
                    print(
                        f"    [{role}] {r.get('transaction_type') or '?'} "
                        f"{ptype} "
                        f"trigger={r.get('trigger_price') or '-'} "
                        f"limit={r.get('price') or '-'} "
                        f"tag={tag or '?'}",
                        flush=True,
                    )
        except Exception:
            pass
        self.log.record(
            "entry",
            title="MICRO SCALP ENTRY",
            lines=[
                f"{sig['direction']} @ {fill:.2f}",
                f"SL {sl:.2f} | Lucid TP {tp:.2f} (${TP_USD:.0f})",
                f"impulse={sig.get('impulse', 0):.2f}",
                f"Lucid {ticket}",
                "legs_confirmed=yes",
            ],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )
        return True

    def _close_bot_position(self, pos: OpenPos) -> bool:
        """Market-close this bot entry only (known side), then cancel its SL/TP legs."""
        if self.rithmic is None or not self.rithmic.connected:
            return False
        # Flatten helper: net>0 sells, net<0 buys. Long +1 → SELL; short -1 → BUY.
        close_net = 1 if pos.direction == "long" else -1
        try:
            rith_sym, exchange = self.rithmic._resolve_symbol("MNQ")  # noqa: SLF001
            ok = self.rithmic._run_sync(  # noqa: SLF001
                self.rithmic._async_submit_flatten_market_order(  # noqa: SLF001
                    symbol="MNQ",
                    broker_sym=rith_sym,
                    exchange=exchange,
                    net_qty=close_net,
                ),
                timeout=20,
            )
            if not ok:
                print("⚠️  Bot market close did not fill", flush=True)
        except Exception as exc:
            print(f"⚠️  Bot market close: {redact_secrets(exc)}", flush=True)
            ok = False
        cleaned = self._clear_chart(pos.ticket, reason=pos.exit_reason or "exit")
        return bool(ok) and cleaned

    def _request_exit(self, reason: str, px: float) -> None:
        pos = self.state.open
        if pos is None:
            return
        if not pos.exit_pending:
            pos.exit_pending = True
            pos.exit_reason = reason
            hold = time.time() - pos.entry_ts
            u = unrealized_usd(pos.direction, pos.entry, px)
            print(
                f"⏳ EXIT {reason} @ {px:.2f} | hold {hold:.1f}s | ~${u:.2f} "
                f"— close bot fill + cancel bot SL/TP…",
                flush=True,
            )
        # Lucid TP/SL may already have closed size — still cancel leftover legs.
        if reason in {"TP", "SL"}:
            cleaned = self._clear_chart(pos.ticket, reason=reason)
            # If legs still working, broker may not have filled yet — force close.
            left, _ = self._working_bot_count()
            if left and left > 0:
                cleaned = self._close_bot_position(pos) or cleaned
        else:
            cleaned = self._close_bot_position(pos)
        if not cleaned:
            print(
                "⚠️  Bot SL/TP still working — retrying (manual orders untouched).",
                flush=True,
            )
            return
        hold = time.time() - pos.entry_ts
        u = unrealized_usd(pos.direction, pos.entry, px)
        mark = "✅" if reason == "TP" else ("❌" if reason == "SL" else "🏁")
        print(
            f"{mark} EXIT {reason} CONFIRMED @ {px:.2f} | hold {hold:.1f}s | ~${u:.2f} "
            f"| bot legs cleared (from {pos.entry:.2f} {pos.direction})",
            flush=True,
        )
        self.log.record(
            "exit",
            title=f"EXIT {reason}",
            lines=[
                f"{pos.direction} @ {px:.2f}",
                f"hold={hold:.1f}s pnl~${u:.2f}",
                "bot_legs_cleared=yes",
            ],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )
        on_closed(self.state, reason)

    def _broker_leg_hit(self, pos: OpenPos, px: float) -> Optional[str]:
        if pos.direction == "long":
            if px <= pos.sl:
                return "SL"
            if px >= pos.tp_far:
                return "TP"
        else:
            if px >= pos.sl:
                return "SL"
            if px <= pos.tp_far:
                return "TP"
        return None

    def _heartbeat(self, px: Optional[float], note: str) -> None:
        now = time.time()
        if now - self._last_hb < HEARTBEAT_SEC:
            return
        self._last_hb = now
        ts = datetime.now(ET).strftime("%H:%M:%S ET")
        px_txt = f"{px:.2f}" if px is not None else "—"
        if self.state.open:
            pos = self.state.open
            hold = now - pos.entry_ts
            u = unrealized_usd(pos.direction, pos.entry, px or pos.entry)
            if pos.exit_pending:
                tp_ready = f"CLEARING {pos.exit_reason}"
            else:
                tp_ready = "TP-armed" if hold >= MIN_HOLD_SEC else f"min-hold {MIN_HOLD_SEC - hold:.1f}s"
            print(
                f"💓 {ts} MNQ {px_txt} | {pos.direction} @{pos.entry:.0f} "
                f"hold {hold:.1f}s ~${u:.2f} | {tp_ready} | LucidTP {pos.tp_far:.2f} | LucidSL {pos.sl:.2f}",
                flush=True,
            )
        else:
            impulse = self.state.buf.impulse_pts()
            fast, slow = self.state.buf.ema_pair()
            if fast is None or slow is None:
                ema_txt = "EMA warming"
            else:
                align = "↑" if fast > slow else ("↓" if fast < slow else "=")
                ema_txt = f"EMA{EMA_FAST}/{EMA_SLOW} {align}"
            need = IMPULSE_PTS
            print(
                f"💓 {ts} MNQ {px_txt} | flat | {note} | "
                f"impulse {impulse:+.1f}/{need:.1f}pts {LOOKBACK_SEC:.0f}s | {ema_txt}",
                flush=True,
            )
        write_status(
            {
                "bot_running": True,
                "paper_mode": True,
                "session": SESSION_NAME,
                "micro_scalp_20": True,
                "note": note,
                "open": self.state.open is not None,
                "exit_pending": bool(self.state.open and self.state.open.exit_pending),
                "last_price": px,
                "lucid_tp": None if not self.state.open else self.state.open.tp_far,
                "lucid_sl": None if not self.state.open else self.state.open.sl,
            }
        )

    def run(self, *, duration_minutes: int = 0) -> None:
        for line in recipe_banner_lines():
            print(line)
        print(f"Desk: http://127.0.0.1:{os.getenv('DASHBOARD_PORT', '5055')}")
        print("Leave this window open. Ctrl+C to stop.")
        print("Manual Lucid trades OK — bot only cancels its own bot_* orders.\n")
        self._connect()
        try:
            if self.rithmic and self.rithmic.connected:
                print("🧹 Startup: clear bot_* orders only (manual left alone)…", flush=True)
                self._clear_chart(reason="startup")
        except Exception as exc:
            print(f"⚠️  startup clear: {redact_secrets(exc)}", flush=True)
        deadline = time.time() + duration_minutes * 60 if duration_minutes > 0 else None
        try:
            while True:
                if deadline and time.time() >= deadline:
                    break
                now_et = datetime.now(ET)
                if rth_only_enabled() and not in_rth_et((now_et.hour, now_et.minute)):
                    self._heartbeat(self._quote(), "outside RTH")
                    time.sleep(SCAN_SEC)
                    continue
                if day_bot_owns_lucid(now_et):
                    self._heartbeat(self._quote(), "waiting — another bot owns Lucid")
                    time.sleep(SCAN_SEC)
                    continue
                px = self._quote()
                if px is None:
                    self._heartbeat(None, "no ticker yet")
                    time.sleep(SCAN_SEC)
                    continue
                self.state.buf.push(px)
                if self.state.open is not None:
                    pos = self.state.open
                    if pos.exit_pending:
                        self._request_exit(pos.exit_reason or "RETRY", px)
                    else:
                        self._guard_open_protection()
                        hit = check_exit(pos, px)
                        if not hit:
                            leg = self._broker_leg_hit(pos, px)
                            if leg:
                                hit = (leg, px)
                        if hit:
                            reason, exit_px = hit
                            self._request_exit(reason, exit_px)
                        else:
                            self._heartbeat(px, "in trade")
                else:
                    left, kinds = self._working_bot_count()
                    if left and left > 0:
                        print(
                            f"🧹 Flat but {left} bot order(s) remain {kinds} — "
                            f"clearing bot only (manual left alone)…",
                            flush=True,
                        )
                        self._clear_chart(reason="orphan_flat")
                        self._heartbeat(px, "clearing bot orders")
                    else:
                        sig = check_entry(self.state)
                        if sig:
                            self._place(sig, px)
                        self._heartbeat(px, "scanning impulse")
                time.sleep(SCAN_SEC)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping micro scalp $20")
        finally:
            if self.state.open is not None:
                px = self._quote() or self.state.open.entry
                self._request_exit("SHUTDOWN", px)
                for _ in range(5):
                    if self.state.open is None:
                        break
                    time.sleep(1.0)
                    self._request_exit("SHUTDOWN", self._quote() or px)
            if self.rithmic and self.rithmic.connected:
                try:
                    self._clear_chart(
                        self.state.open.ticket if self.state.open else "",
                        reason="shutdown",
                    )
                    self.rithmic.shutdown()
                except Exception:
                    traceback.print_exc()
            write_status({"bot_running": False, "session": SESSION_NAME})


def run_micro_scalp20_paper(*, duration_minutes: int = 0) -> None:
    MicroScalp20PaperEngine().run(duration_minutes=duration_minutes)
