"""MNQ S16 mid Lucid PAPER runner — HybridS15_IB on TEST sim.

Continuous RTH bot replacing day_hybrid for paper. live_mode=False always.
Session label: s16_mid (never day_hybrid). Funded live stays OFF.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from src.ai.action_log import ActionLog, write_status
from src.broker.rithmic_connector import (
    RithmicConnector,
    apply_lucid_plants_with_history,
)
from src.data.paper_csv_feed import PaperCsvFeed, default_1m_csv_path, paper_rithmic_brackets_enabled
from src.strategy.s16_mid import (
    SESSION_NAME,
    DayState,
    account_equity_guess,
    check_bar_exit,
    check_entry,
    explain_skip,
    is_rth_et,
    paper_size,
    recipe_banner_lines,
    record_day_pnl,
    should_flatten_rth,
    verify_lock,
)
from src.strategy.mnq_15m_ema_eod import overlay_ticker_last_on_1m, parse_mnq_1m_datetime
from src.strategy.overnight_research import day_bot_owns_lucid, skip_local_fake_fill
from src.utils.redact import redact_secrets

ET = pytz.timezone("US/Eastern")
SCAN_SEC = float(os.getenv("S16_MID_SCAN_SEC", "30"))
HEARTBEAT_IDLE_SEC = float(os.getenv("S16_MID_HEARTBEAT_SEC", "120"))


def _last_price(feed: PaperCsvFeed, df: Optional[pd.DataFrame]) -> Optional[float]:
    inner = getattr(feed, "rithmic", None)
    if inner is not None and getattr(inner, "connected", False):
        try:
            q = inner.get_latest_price("MNQ")
        except Exception:
            q = None
        if q:
            px = float(q.get("last") or q.get("bid") or 0)
            if px > 0:
                return px
    if df is not None and not df.empty:
        return float(df.iloc[-1]["close"])
    return None


def _brackets_from_pts(sig: Dict[str, Any], ref_px: float) -> Tuple[float, float]:
    pts = float(sig["sl_pts"])
    if sig["direction"] == "long":
        return ref_px - pts, ref_px + pts
    return ref_px + pts, ref_px - pts


def _trade_emoji(direction: str) -> str:
    return "" if direction == "long" else ""


def _exit_emoji(reason: str) -> str:
    r = reason.lower()
    if "tp" in r or "target" in r:
        return ""
    if "sl" in r or "stop" in r:
        return ""
    if "flat" in r or "eod" in r:
        return ""
    return ""


class S16MidPaperEngine:
    def __init__(self) -> None:
        self.params = verify_lock()
        self.feed = PaperCsvFeed(default_1m_csv_path(), refresh_sec=60)
        self.feed.rithmic = None
        self.rithmic: Optional[RithmicConnector] = None
        self.log = ActionLog()
        self.state = DayState()
        self.open_pos: Optional[Dict[str, Any]] = None
        self._last_heartbeat_ts = 0.0
        self._last_orphan_sweep_ts = 0.0
        self.account = float(account_equity_guess())
        self.size = int(paper_size())

    def _connect(self) -> None:
        os.environ["PAPER_USE_RITHMIC"] = "true"
        os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
        os.environ["TRADING_MODE"] = "paper"
        os.environ["ALLOW_LIVE"] = "false"
        try:
            self.feed.initialize()
        except Exception as exc:
            print(f"CSV/Databento seed: {redact_secrets(exc)}")
        if day_bot_owns_lucid():
            print("  Stop start_mnq_live.bat first — one Lucid session only.")
        if not paper_rithmic_brackets_enabled():
            print("Set PAPER_RITHMIC_BRACKETS=true for Lucid sim orders")
            return
        try:
            apply_lucid_plants_with_history()
            raw = RithmicConnector(live_mode=False, quotes_only=False)
            raw._symbols_to_watch = ["MNQ"]
            raw._disable_yahoo_fallback = True
            raw.initialize()
            if raw.connected:
                self.rithmic = raw
                self.feed.rithmic = raw
                print(" Lucid sim connected (TEST + history 1m). Close R|Trader if logged in elsewhere.")
                try:
                    # Only wipe leftovers when broker is flat — never strip a live short's SL
                    n = int(
                        raw.clear_chart_stops(
                            "MNQ",
                            include_limits=True,
                            rounds=4,
                            only_if_flat=True,
                        )
                        or 0
                    )
                    if n:
                        print(f" startup: cleared {n} leftover bot stop/limit(s) (flat)", flush=True)
                except Exception as exc:
                    print(f"startup bracket check: {redact_secrets(exc)}", flush=True)
            else:
                err = redact_secrets(getattr(raw, "_last_connect_error", "") or "not connected")
                print(f" Lucid connect failed ({err})")
                try:
                    raw.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            print(f" Rithmic error: {redact_secrets(exc)}")

    def _merge_rithmic_1m(self, seed: pd.DataFrame) -> pd.DataFrame:
        """Append fresh Rithmic 1m bars so 15m/5m stack is not stuck on stale CSV."""
        if self.rithmic is None or not self.rithmic.connected:
            return seed
        try:
            live = self.rithmic.get_candles("MNQ", timeframe_minutes=1, num_candles=600)
        except Exception as exc:
            print(f"Rithmic 1m history fetch failed: {redact_secrets(exc)}")
            return seed
        if live is None or live.empty:
            return seed
        live = live.copy()
        live["datetime"] = parse_mnq_1m_datetime(live["datetime"])
        live = live.dropna(subset=["datetime"]).sort_values("datetime")
        if seed is None or seed.empty:
            return live.reset_index(drop=True)
        seed = seed.copy()
        seed["datetime"] = parse_mnq_1m_datetime(seed["datetime"])
        seed = seed.dropna(subset=["datetime"])
        last_seed = seed["datetime"].max()
        add = live[live["datetime"] > last_seed]
        if add.empty:
            # Overlap: prefer live for the shared window so today's bars win
            cutoff = live["datetime"].min()
            seed = seed[seed["datetime"] < cutoff]
            out = pd.concat([seed, live], ignore_index=True)
            added = len(live)
        else:
            out = pd.concat([seed, add], ignore_index=True)
            added = len(add)
        out = out.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
        out = out.reset_index(drop=True)
        if not getattr(self, "_rithmic_1m_logged", False):
            self._rithmic_1m_logged = True
            last = pd.Timestamp(out["datetime"].iloc[-1])
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            last_et = last.tz_convert(ET).strftime("%Y-%m-%d %H:%M ET")
            print(
                f" Rithmic 1m merged (+{added} bars) → {len(out)} total, last {last_et}",
                flush=True,
            )
        return out

    def _get_df(self) -> Optional[pd.DataFrame]:
        self.feed.refresh(force=False)
        df = self.feed._ensure_1m()
        if df is None or df.empty:
            return None
        df = self._merge_rithmic_1m(df)
        px = _last_price(self.feed, df)
        if px is not None:
            df = overlay_ticker_last_on_1m(df, px, now=datetime.now(pytz.UTC))
        return df

    def _place(self, sig: Dict[str, Any]) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print(" Lucid brackets required — set PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true")
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("⏳ No Lucid order plant — skip entry")
            return False
        ref = _last_price(self.feed, None) or float(sig["entry"])
        sl, tp = float(sig["sl"]), float(sig["tp"])
        side = "buy"
        emoji = _trade_emoji(sig["direction"])
        try:
            result = self.rithmic.place_order(
                symbol="MNQ",
                order_type=side,
                size=int(sig.get("size") or getattr(self, "size", 1)),
                entry_price=ref,
                stop_loss=sl,
                take_profit=tp,
            )
        except Exception as exc:
            print(f" place_order failed: {redact_secrets(exc)}")
            return False
        if not result or not (result.get("ticket") or result.get("order_id")):
            print(" Lucid order not booked")
            return False
        fill = float(result.get("entry_price") or result.get("price_ref") or ref)
        sl = float(result.get("stop_loss") or sl)
        tp = float(result.get("take_profit") or tp)
        sig = dict(sig)
        sig["entry"] = fill
        sig["sl"] = sl
        sig["tp"] = tp
        sig["lucid_ticket"] = str(result.get("ticket") or result.get("order_id"))
        sig["entry_time"] = datetime.now(pytz.UTC).isoformat()
        bracket = result.get("bracket_mode", "?")
        sl_ok = "" if result.get("native_stop_attached") or result.get("protective_sl_order_id") else ""
        tp_ok = "" if result.get("native_target_attached") or result.get("protective_tp_order_id") else ""
        self.open_pos = {
            "direction": "long",
            "leg": sig.get("leg", "orb"),
            "entry": fill,
            "sl": sl,
            "tp": tp,
            "size": int(sig.get("size") or getattr(self, "size", 1)),
            "lucid_ticket": str(result.get("ticket") or result.get("order_id")),
            "entry_time": datetime.now(pytz.UTC).isoformat(),
            "signal_bar": sig.get("signal_bar"),
            "session": SESSION_NAME,
            "strategy": "HybridS15_IB",
        }
        if sig.get("leg") == "orb":
            self.state.orb_taken = True
        else:
            self.state.ib_taken = True
        rsi_note = f" | RSI {sig['rsi']:.0f}" if sig.get("rsi") is not None else ""
        print(
            f"{emoji} ENTRY {sig['leg'].upper()} {sig['direction'].upper()} @ {fill:.2f} "
            f"| SL {sl:.2f} {sl_ok} | TP {tp:.2f} {tp_ok}{rsi_note} | #{sig['lucid_ticket']}",
            flush=True,
        )
        self.log.record(
            "entry",
            title="TRADE PLACED",
            lines=[
                f"{sig['leg'].upper()} {sig['direction']} @ {fill:.2f}",
                f"SL {sl:.2f} TP {tp:.2f}"
                + (f" RSI {sig['rsi']:.0f}" if sig.get("rsi") is not None else ""),
                f"Lucid {sig['lucid_ticket']}",
            ],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )
        self._last_heartbeat_ts = 0.0
        return True

    def _open_tickets(self) -> set:
        return {
            str(p.get("lucid_ticket") or "")
            for p in ([self.open_pos] if self.open_pos else [])
            if p.get("lucid_ticket")
        }

    def _clear_chart(self, *, reason: str, exited_ticket: str = "") -> int:
        """On close: cancel that ticket's leftover legs, then wipe orphans not in live book.

        When local book is empty, wipe ALL bot stops/limits (full chart clear).
        """
        if self.rithmic is None or not self.rithmic.connected:
            return 0
        n = 0
        if exited_ticket:
            try:
                n += int(
                    self.rithmic.cancel_entry_bracket_legs(
                        exited_ticket, "MNQ", reason=reason, force=True,
                    )
                    or 0
                )
            except Exception as exc:
                print(f"  cancel ticket {exited_ticket}: {redact_secrets(exc)}", flush=True)
        keep = self._open_tickets()
        try:
            if keep:
                n += int(
                    self.rithmic.clear_chart_stops(
                        "MNQ",
                        include_limits=True,
                        rounds=4,
                        keep_entry_ids=keep,
                        only_if_flat=False,
                    )
                    or 0
                )
            else:
                # Fully flat locally — nuke every bot stop/limit on the chart
                n += int(
                    self.rithmic.clear_chart_stops(
                        "MNQ",
                        include_limits=True,
                        rounds=5,
                        keep_entry_ids=set(),
                        only_if_flat=False,
                    )
                    or 0
                )
                n += int(self.rithmic.cancel_all_bot_orders("MNQ") or 0)
        except Exception as exc:
            print(f"  clear_chart_stops: {redact_secrets(exc)}", flush=True)
        print(
            f" CHART CLEARED ({reason}): cancelled {n} bot stop/limit(s) "
            f"(kept {len(keep)} live ticket(s))",
            flush=True,
        )
        return n

    def _reprotect_open(self) -> None:
        """After a chart wipe, re-attach SL/TP for lots still in our book."""
        if self.rithmic is None or not self.rithmic.connected:
            return
        for pos in ([self.open_pos] if self.open_pos else []):
            ticket = str(pos.get("lucid_ticket") or "")
            if not ticket:
                continue
            side = "buy" if pos.get("direction") == "long" else "sell"
            try:
                ok = self.rithmic.ensure_protective_orders(
                    ticket,
                    "MNQ",
                    side,
                    1,
                    float(pos["sl"]),
                    float(pos["tp"]),
                )
                print(
                    f"  re-protect #{ticket} {pos['direction']} "
                    f"SL {pos['sl']:.2f} TP {pos['tp']:.2f} → {'ok' if ok else 'FAIL'}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  re-protect #{ticket}: {redact_secrets(exc)}", flush=True)

    def _working_leg_flags(self, ticket: str) -> Tuple[Optional[bool], Optional[bool]]:
        """(sl_working, tp_working); (None, None) if list failed."""
        if not ticket or self.rithmic is None or not self.rithmic.connected:
            return None, None
        try:
            rows = self.rithmic.list_working_orders("MNQ") or []
        except Exception:
            return None, None
        sl_w = tp_w = False
        prefix = str(ticket)
        for row in rows:
            tag = str(row.get("user_tag") or row.get("order_id") or "")
            if not tag.startswith(prefix):
                continue
            if tag.endswith("_sl") or f"{prefix}_sl" in tag:
                sl_w = True
            if tag.endswith("_tp") or tag.endswith("_tp_lit") or f"{prefix}_tp" in tag:
                tp_w = True
        return sl_w, tp_w

    def _sweep_orphan_brackets(self) -> None:
        """If local flat: full chart wipe. Else: cancel orphans not in live book."""
        if self.rithmic is None or not self.rithmic.connected:
            return
        now = time.time()
        if now - self._last_orphan_sweep_ts < 10.0:
            return
        self._last_orphan_sweep_ts = now
        live = self._open_tickets()
        if not live:
            self._clear_chart(reason="flat_sweep")
            return
        try:
            n = int(
                self.rithmic.cancel_stale_bot_brackets("MNQ", live_entry_ids=live)
                or 0
            )
            if n:
                print(
                    f" orphan sweep cancelled {n} leftover stop/limit(s) "
                    f"(live={len(live)})",
                    flush=True,
                )
                self._reprotect_open()
        except Exception as exc:
            print(f"  orphan sweep: {redact_secrets(exc)}", flush=True)

    def _broker_exit_hint(self, pos: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """If Lucid already filled one protective leg, treat as exit and cancel sibling."""
        ticket = str(pos.get("lucid_ticket") or "")
        if not ticket:
            return None
        try:
            age = time.time() - datetime.fromisoformat(
                str(pos.get("entry_time") or "").replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            age = 999.0
        if age < 12.0:
            return None
        sl_w, tp_w = self._working_leg_flags(ticket)
        if sl_w is None:
            return None
        if sl_w and tp_w:
            return None
        if not sl_w and not tp_w:
            return "BROKER", float(pos.get("entry") or 0)
        if sl_w and not tp_w:
            return "TP", float(pos.get("tp") or pos.get("entry") or 0)
        if tp_w and not sl_w:
            return "SL", float(pos.get("sl") or pos.get("entry") or 0)
        return None

    def _market_close_one(self, direction: str) -> bool:
        """Close one MNQ lot at market (long→sell, short→buy)."""
        if self.rithmic is None or not self.rithmic.connected:
            return False
        close_net = 1 if direction == "long" else -1
        try:
            rith_sym, exchange = self.rithmic._resolve_symbol("MNQ")  # noqa: SLF001
            return bool(
                self.rithmic._run_sync(  # noqa: SLF001
                    self.rithmic._async_submit_flatten_market_order(  # noqa: SLF001
                        symbol="MNQ",
                        broker_sym=rith_sym,
                        exchange=exchange,
                        net_qty=close_net,
                    ),
                    timeout=20,
                )
            )
        except Exception as exc:
            print(f"  market close 1: {redact_secrets(exc)}", flush=True)
            return False

    def _manage_leg(self, positions: List[Dict], leg: str, bar_hi: float, bar_lo: float, bar_cl: float, flat_eod: bool) -> None:
        still: List[Dict] = []
        exited: List[Tuple[Dict, str, float]] = []
        for pos in positions:
            hit = self._broker_exit_hint(pos)
            if not hit:
                hit = check_bar_exit(pos, bar_hi, bar_lo, bar_cl, flat_eod=flat_eod)
            if hit:
                reason, px = hit
                exited.append((pos, reason, px))
            else:
                still.append(pos)

        # Drop closed tickets from the book FIRST so keep-set is correct for multi-exit
        self.open_pos = still[0] if still else None

        for pos, reason, px in exited:
            emoji = _exit_emoji(reason)
            print(
                f"{emoji} EXIT {leg.upper()} {reason.upper()} @ {px:.2f} "
                f"({pos['direction']} from {pos['entry']:.2f})",
                flush=True,
            )
            ticket = str(pos.get("lucid_ticket") or "")
            if ticket:
                sl_w, tp_w = self._working_leg_flags(ticket)
                if (sl_w and tp_w) or reason == "EOD":
                    self._market_close_one(str(pos.get("direction") or leg))
                    time.sleep(0.35)
            self._clear_chart(reason=f"exit_{reason.lower()}", exited_ticket=ticket)
            # S16 mid: record day pnl for halt
            try:
                pts = float(px) - float(pos.get("entry") or px)
                pnl = pts * 2.0 * int(pos.get("size") or 1)
                halt = float(self.params.get("daily_loss_halt_pct", 2.0))
                self.state = record_day_pnl(self.state, pnl, self.account, halt)
            except Exception:
                pass
            self.log.record(
                "exit",
                title=f"EXIT {reason}",
                lines=[f"{leg} {pos['direction']} @ {px:.2f}", f"Lucid {ticket}"],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )

        if flat_eod and exited and self.rithmic and self.rithmic.connected:
            try:
                self.rithmic.flatten_symbol("MNQ", max_attempts=3, verify_delay=0.5)
            except Exception as exc:
                print(f"  EOD flatten: {redact_secrets(exc)}", flush=True)

        if exited and not self.open_pos:
            # Both books empty — one more full wipe so no Buy Stop ghosts remain
            time.sleep(0.4)
            self._clear_chart(reason="post_exit_flat")
        elif exited and self.open_pos:
            time.sleep(0.4)
            self._reprotect_open()

    def _maybe_enter(self, df: pd.DataFrame, now: datetime) -> None:
        if self.open_pos is not None or self.state.halted:
            return
        sig, self.state = check_entry(
            df, self.state, open_count=0, now=now, params=self.params,
        )
        if sig:
            self._place(sig)

    def _status(self, df: Optional[pd.DataFrame], note: str) -> None:
        write_status(
            {
                "bot_running": True,
                "paper_mode": True,
                "session": SESSION_NAME,
                "day_hybrid": False, "s16_mid": True,
                "note": note,
                "open": 1 if self.open_pos else 0,
                "last_price": _last_price(self.feed, df),
            }
        )

    def _heartbeat(self, df: Optional[pd.DataFrame], note: str) -> None:
        open_count = 1 if self.open_pos else 0
        now_ts = time.time()
        if open_count == 0 and now_ts - self._last_heartbeat_ts < HEARTBEAT_IDLE_SEC:
            return
        self._last_heartbeat_ts = now_ts
        px = _last_price(self.feed, df)
        px_txt = f"{px:.2f}" if px is not None else "—"
        ts = datetime.now(ET).strftime("%H:%M ET")
        if open_count:
            parts = []
            for pos in ([self.open_pos] if self.open_pos else []):
                parts.append(
                    f"{_trade_emoji(pos['direction'])} {pos['direction'][0].upper()} "
                    f"{pos['entry']:.0f} SL{pos['sl']:.0f} TP{pos['tp']:.0f}"
                )
            detail = " | ".join(parts)
            print(f" {ts} MNQ {px_txt} | open {open_count}: {detail}", flush=True)
        else:
            print(f" {ts} MNQ {px_txt} | flat | {note}", flush=True)
        self._status(df, note)

    def run(self, *, duration_minutes: int = 0) -> None:
        for line in recipe_banner_lines(self.params):
            print(line)
        print(f"Desk: http://127.0.0.1:{os.getenv('DASHBOARD_PORT', '5055')}")
        print("Leave this window open. Ctrl+C to stop.\n")
        self._connect()
        deadline = None
        if duration_minutes and duration_minutes > 0:
            deadline = time.time() + duration_minutes * 60
        try:
            while True:
                if deadline and time.time() >= deadline:
                    break
                now = datetime.now(ET)
                df = self._get_df()
                flat_eod = should_flatten_rth(now)
                if df is not None and not df.empty:
                    row = df.iloc[-1]
                    hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
                    if self.open_pos is not None:
                        self._manage_leg([self.open_pos], "long", hi, lo, cl, flat_eod)
                        # _manage_leg mutates list copies; sync back
                        pass
                self._sweep_orphan_brackets()
                if is_rth_et(now) and df is not None:
                    if day_bot_owns_lucid(now):
                        self._heartbeat(df, "waiting — day ema15 owns Lucid")
                    else:
                        self._maybe_enter(df, now)
                        skip = explain_skip(
                            self.state,
                            open_count=1 if self.open_pos else 0,
                            now=now,
                        )
                        self._heartbeat(
                            df,
                            f"RTH | {skip}",
                        )
                else:
                    self._heartbeat(df, "CME closed")
                time.sleep(SCAN_SEC)
        except KeyboardInterrupt:
            print("\n  Stopping S16 mid paper")
        finally:
            if self.rithmic and self.rithmic.connected:
                try:
                    self.rithmic.flatten_symbol("MNQ", max_attempts=6, verify_delay=1.0)
                    self.rithmic.shutdown()
                except Exception:
                    pass
            write_status({"bot_running": False, "session": SESSION_NAME})


def run_s16_mid_paper(*, duration_minutes: int = 0) -> None:
    S16MidPaperEngine().run(duration_minutes=duration_minutes)
