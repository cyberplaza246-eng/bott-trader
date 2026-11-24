"""Micro-scalp paper runner — Lucid sim, $20+ TP after 10s hold."""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import pytz

from src.ai.action_log import ActionLog, write_status
from src.broker.rithmic_connector import RithmicConnector, apply_lucid_ticker_order_plants
from src.data.paper_csv_feed import PaperCsvFeed, default_1m_csv_path, paper_rithmic_brackets_enabled
from src.strategy.micro_scalp import (
    SESSION_NAME,
    check_entry,
    check_quote_exit,
    is_rth_scan_et,
    on_closed,
    recipe_banner_lines,
    should_flatten_rth,
    MicroScalpState,
)
from src.strategy.mnq_15m_ema_eod import overlay_ticker_last_on_1m
from src.strategy.overnight_research import day_bot_owns_lucid, skip_local_fake_fill
from src.utils.redact import redact_secrets

ET = pytz.timezone("US/Eastern")
SCAN_SEC = float(os.getenv("MICRO_SCALP_SCAN_SEC", "1"))
HEARTBEAT_IDLE_SEC = float(os.getenv("MICRO_SCALP_HEARTBEAT_SEC", "60"))


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


class MicroScalpPaperEngine:
    def __init__(self) -> None:
        self.feed = PaperCsvFeed(default_1m_csv_path(), refresh_sec=60)
        self.feed.rithmic = None
        self.rithmic: Optional[RithmicConnector] = None
        self.log = ActionLog()
        self.state = MicroScalpState()
        self.open_pos: Optional[Dict[str, Any]] = None
        self._last_heartbeat_ts = 0.0

    def _connect(self) -> None:
        os.environ["PAPER_USE_RITHMIC"] = "true"
        os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
        try:
            self.feed.initialize()
        except Exception as exc:
            print(f"CSV/Databento seed: {redact_secrets(exc)}")
        if day_bot_owns_lucid():
            print("⚠️  Stop start_mnq_live.bat / other Lucid bots first — one session only.")
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
                self.feed.rithmic = raw
                print("🔗 Lucid sim connected (TEST011). Close R|Trader if logged in elsewhere.")
            else:
                err = redact_secrets(getattr(raw, "_last_connect_error", "") or "not connected")
                print(f"❌ Lucid connect failed ({err})")
                try:
                    raw.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            print(f"❌ Rithmic error: {redact_secrets(exc)}")

    def _get_df(self) -> Optional[pd.DataFrame]:
        self.feed.refresh(force=False)
        df = self.feed._ensure_1m()
        if df is None or df.empty:
            return None
        px = _last_price(self.feed, df)
        if px is not None:
            df = overlay_ticker_last_on_1m(df, px, now=datetime.now(pytz.UTC))
        return df

    def _place(self, sig: Dict[str, Any]) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print("❌ Lucid brackets required — set PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true")
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("⏳ No Lucid order plant — skip entry")
            return False
        ref = _last_price(self.feed, None) or float(sig["entry"])
        sl_pts = float(sig["sl_pts"])
        tp_pts = float(sig["tp_pts"])
        if sig["direction"] == "long":
            sl = ref - sl_pts
            tp = ref + tp_pts
        else:
            sl = ref + sl_pts
            tp = ref - tp_pts
        side = "buy" if sig["direction"] == "long" else "sell"
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
        sl = float(result.get("stop_loss") or sl)
        tp = float(result.get("take_profit") or tp)
        pos = dict(sig)
        pos["entry"] = fill
        pos["sl"] = sl
        pos["tp"] = tp
        pos["lucid_ticket"] = str(result.get("ticket") or result.get("order_id"))
        pos["entry_ts"] = time.time()
        pos["entry_time"] = datetime.now(pytz.UTC).isoformat()
        self.open_pos = pos
        self.state.open_side = int(sig["side"])
        print(
            f"{'📈' if sig['direction'] == 'long' else '📉'} ENTRY {sig['direction'].upper()} "
            f"@ {fill:.2f} | SL {sl:.2f} | TP ${pos['tp_dollars']:.0f} after "
            f"{pos['min_hold_sec']:.0f}s | #{pos['lucid_ticket']}",
            flush=True,
        )
        self.log.record(
            "entry",
            title="MICRO SCALP ENTRY",
            lines=[
                f"{sig['direction']} @ {fill:.2f}",
                f"SL {sl:.2f} | ${pos['tp_dollars']:.0f} TP after {pos['min_hold_sec']:.0f}s",
                f"Lucid {pos['lucid_ticket']}",
            ],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )
        self._last_heartbeat_ts = 0.0
        return True

    def _exit(self, reason: str, px: float) -> None:
        pos = self.open_pos
        if not pos:
            return
        if self.rithmic and self.rithmic.connected:
            try:
                self.rithmic.flatten_symbol("MNQ", max_attempts=4, verify_delay=0.8)
            except Exception as exc:
                print(f"⚠️  flatten: {redact_secrets(exc)}")
        entry = float(pos["entry"])
        pts = (px - entry) if pos["direction"] == "long" else (entry - px)
        usd = pts * 2.0
        print(
            f"{'✅' if usd >= 0 else '❌'} EXIT {reason.upper()} @ {px:.2f} "
            f"({pos['direction']} from {entry:.2f}) {usd:+.2f}$",
            flush=True,
        )
        self.log.record(
            "exit",
            title=f"EXIT {reason}",
            lines=[f"{pos['direction']} @ {px:.2f} {usd:+.2f}$"],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )
        self.open_pos = None
        self.state = on_closed(self.state)

    def _manage(self, last: Optional[float], flatten_eod: bool) -> None:
        if not self.open_pos or last is None:
            return
        held = time.time() - float(self.open_pos["entry_ts"])
        hit = check_quote_exit(
            self.open_pos, last, held_sec=held, flatten_eod=flatten_eod,
        )
        if hit:
            reason, px = hit
            self._exit(reason, px)

    def _maybe_enter(self, df: pd.DataFrame, now: datetime) -> None:
        if self.open_pos:
            return
        sig, self.state = check_entry(df, self.state, now=now, open_count=0)
        if sig:
            self._place(sig)

    def _status(self, df: Optional[pd.DataFrame], note: str) -> None:
        write_status(
            {
                "bot_running": True,
                "paper_mode": True,
                "session": SESSION_NAME,
                "micro_scalp": True,
                "note": note,
                "open": 1 if self.open_pos else 0,
                "trades_today": self.state.trades_today,
                "last_price": _last_price(self.feed, df),
            }
        )

    def _heartbeat(self, df: Optional[pd.DataFrame], note: str) -> None:
        now_ts = time.time()
        if self.open_pos is None and now_ts - self._last_heartbeat_ts < HEARTBEAT_IDLE_SEC:
            return
        self._last_heartbeat_ts = now_ts
        px = _last_price(self.feed, df)
        px_txt = f"{px:.2f}" if px is not None else "—"
        ts = datetime.now(ET).strftime("%H:%M:%S ET")
        if self.open_pos:
            held = now_ts - float(self.open_pos["entry_ts"])
            armed = "ARMED" if held >= float(self.open_pos["min_hold_sec"]) else f"wait {self.open_pos['min_hold_sec']-held:.0f}s"
            print(
                f"💓 {ts} MNQ {px_txt} | {self.open_pos['direction']} "
                f"@{self.open_pos['entry']:.0f} held {held:.0f}s TP {armed}",
                flush=True,
            )
        else:
            print(f"💓 {ts} MNQ {px_txt} | flat | day {self.state.trades_today} | {note}", flush=True)
        self._status(df, note)

    def run(self, *, duration_minutes: int = 0) -> None:
        for line in recipe_banner_lines():
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
                last = _last_price(self.feed, df)
                flatten = should_flatten_rth(now)
                self._manage(last, flatten)
                if is_rth_scan_et(now) and df is not None:
                    if day_bot_owns_lucid(now):
                        self._heartbeat(df, "waiting — another day bot owns Lucid")
                    else:
                        self._maybe_enter(df, now)
                        self._heartbeat(df, "RTH scan")
                else:
                    self._heartbeat(df, "outside RTH / CME closed")
                time.sleep(SCAN_SEC)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping micro scalp")
        except Exception:
            print(traceback.format_exc())
        finally:
            if self.open_pos and self.rithmic and self.rithmic.connected:
                try:
                    self.rithmic.flatten_symbol("MNQ", max_attempts=6, verify_delay=1.0)
                except Exception:
                    pass
            if self.rithmic and self.rithmic.connected:
                try:
                    self.rithmic.shutdown()
                except Exception:
                    pass
            write_status({"bot_running": False, "session": SESSION_NAME})


def run_micro_scalp_paper(*, duration_minutes: int = 0) -> None:
    MicroScalpPaperEngine().run(duration_minutes=duration_minutes)
