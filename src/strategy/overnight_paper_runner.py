"""
Overnight PAPER RESEARCH loop.

Real MNQ last print from Rithmic ticker when connected (PAPER_USE_RITHMIC).
1m bars come from Databento/CSV seed — never Rithmic history (1011 denied).
Default: Lucid TEST/sim orders (ticker+order, live_mode=False). Not local fake fills.
PAPER_TEST_FILL is opt-in only. History/pnl plants skipped (1011).
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import pytz

from src.ai.action_log import (
    ActionLog,
    MONITOR_INTERVAL_SEC,
    format_human_event,
    write_status,
)
from src.ai.llm_advisor import LLMTradeAdvisor
from src.ai.overnight_journal import (
    JOURNAL_PATH,
    SUGGESTIONS_PATH,
    ZONES_PATH,
    OvernightPaperJournal,
    attach_mae_mfe,
    read_zones,
    write_zones,
)
from src.data.paper_csv_feed import (
    PaperCsvFeed,
    default_1m_csv_path,
    paper_rithmic_brackets_enabled,
)
from src.strategy.mnq_15m_ema_eod import load_1m_seed_csv, merge_1m_history
from src.strategy.overnight_research import (
    SESSION_NAME,
    QTY,
    allow_new_overnight_entry,
    check_paper_exit,
    compute_zones,
    day_bot_owns_lucid,
    enable_overnight_lucid_sim_orders,
    evaluate_overnight_cue,
    increment_zone_touch,
    in_rth_hard_idle,
    is_overnight_research_hours,
    overnight_entry_blocked_reason,
    paper_pnl_usd,
    paper_test_fill_from_env,
    paper_test_fill_signal,
    should_flatten_before_rth,
    skip_local_fake_fill,
    update_mae_mfe,
)
from src.utils.redact import redact_secrets
from src.utils.trading_session import is_globex_session_et, seconds_until_session_open_et

ET = pytz.timezone("US/Eastern")
BLOCKED_ORDER_METHODS = frozenset({
    "place_order",
    "close_position",
    "modify_position",
    "flatten_symbol",
    "cancel_order",
    "cancel_all",
    "modify_stop",
})


class QuotesOnlyBroker:
    """Rithmic market data only. Used when Lucid sim orders are explicitly off."""

    def __init__(self, inner: Any):
        self._inner = inner
        self._history_skip_logged = False

    def __getattr__(self, name: str):
        if name in BLOCKED_ORDER_METHODS:
            raise RuntimeError(
                "Overnight paper research never sends Rithmic orders "
                f"(blocked {name})"
            )
        return getattr(self._inner, name)

    def _skip_history(self, *_a, **_k):
        if not self._history_skip_logged:
            self._history_skip_logged = True
            print("history plant denied — using CSV/Databento 1m + ticker last")
        return None

    def get_candles(self, *a, **k):
        return self._skip_history(*a, **k)

    def get_candles_seconds(self, *a, **k):
        return self._skip_history(*a, **k)

    def get_candles_deep(self, *a, **k):
        return self._skip_history(*a, **k)

    def fetch_history_chunked(self, *a, **k):
        return self._skip_history(*a, **k)

    @property
    def connected(self) -> bool:
        return bool(getattr(self._inner, "connected", False))

    def shutdown(self) -> None:
        inner = self._inner
        if inner is not None and hasattr(inner, "shutdown"):
            inner.shutdown()


class OvernightQuoteFeed:
    """CSV/Databento history + optional Rithmic last print. No Yahoo."""

    def __init__(self):
        self.csv = PaperCsvFeed(default_1m_csv_path(), refresh_sec=60)
        self.rithmic: Optional[Any] = None
        self.source = "databento"
        self.brackets = False
        self.connect_error = ""
        self._ticker_retry_at = 0.0
        self._ticker_backoff = 5.0
        self._logout_hinted = False
        self._had_quote = False
        self._awaiting_reconnect = False
        self._parked_rth = False
        self._connect_retry_at = 0.0
        self.on_event = None

    def _notify(self, kind: str, **fields: Any) -> None:
        cb = self.on_event
        if cb is None:
            return
        try:
            cb(kind, fields)
        except Exception:
            pass

    def initialize(self) -> None:
        os.environ["PAPER_USE_RITHMIC"] = "true"
        os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
        try:
            self.csv.initialize()
        except Exception as exc:
            print(f"CSV/Databento seed skipped: {redact_secrets(exc)}")
        if day_bot_owns_lucid():
            self._parked_rth = True
            self.source = "parked_rth"
            print(
                "Day session — not taking the Lucid ticker (09:20–18:00 ET). "
                "Ctrl+C the day bot first. Overnight connects after 18:00 ET Globex. "
                "No entries 9:30–16:00 ET."
            )
            return
        self._connect_rithmic()

    def _connect_rithmic(self) -> None:
        try:
            from src.broker.rithmic_connector import (
                RithmicConnector,
                apply_lucid_ticker_order_plants,
            )

            self.brackets = paper_rithmic_brackets_enabled()
            if self.brackets:
                apply_lucid_ticker_order_plants()
                raw = RithmicConnector(live_mode=False, quotes_only=False)
            else:
                from src.broker.rithmic_connector import prefer_live_mnq_contract

                prefer_live_mnq_contract()
                os.environ["RITHMIC_QUOTES_ONLY"] = "true"
                raw = RithmicConnector(live_mode=False, quotes_only=True)
            raw._symbols_to_watch = ["MNQ"]
            raw._disable_yahoo_fallback = True
            raw.initialize()
            if raw.connected:
                if self.brackets:
                    self.rithmic = raw
                    self.source = "rithmic"
                    print(
                        "Rithmic ticker+order connected — Lucid TEST/sim orders "
                        "(live_mode=False, no history/pnl). Fills must show on Lucid."
                    )
                else:
                    self.rithmic = QuotesOnlyBroker(raw)
                    self.source = "rithmic"
                    print("Rithmic quotes connected — PAPER_RITHMIC_BRACKETS is off")
                print("history plant denied — using CSV/Databento 1m + ticker last")
                print(
                    "If Lucid / R|Trader desktop is logged in on this account, "
                    "close it — Rithmic often allows only one session."
                )
            else:
                err = redact_secrets(getattr(raw, "_last_connect_error", "") or "not connected")
                self.connect_error = str(err)
                print(f"Rithmic quotes unavailable ({err}). Using Databento 1m.")
                print(
                    "If Lucid desktop is open, close it and restart to free the ticker session."
                )
                try:
                    raw.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            self.connect_error = redact_secrets(exc)
            print(f"Rithmic quotes unavailable ({self.connect_error}). Using Databento 1m.")
            print(
                "If Lucid desktop is open, close it and restart to free the ticker session."
            )

    def shutdown(self) -> None:
        if self.rithmic is not None:
            try:
                self.rithmic.shutdown()
            except Exception:
                pass
            self.rithmic = None
        self.brackets = False
        self.csv.shutdown()

    def _disconnect_rithmic(self) -> None:
        if self.rithmic is not None:
            try:
                self.rithmic.shutdown()
            except Exception:
                pass
            self.rithmic = None
        self.brackets = False

    def maybe_align_lucid_session(self) -> None:
        """One Lucid login: park 09:20–18:00 ET, connect after Globex open."""
        if day_bot_owns_lucid():
            if self.rithmic is not None:
                print(
                    "Releasing Lucid session for the day bot (09:20–18:00 ET). "
                    "Stop this window ~09:20, then start_mnq_live.bat."
                )
                self._disconnect_rithmic()
            self._parked_rth = True
            self.source = "parked_rth"
            return
        if self.rithmic is not None and self.rithmic.connected:
            self._parked_rth = False
            return
        now = time.time()
        if now < getattr(self, "_connect_retry_at", 0.0):
            return
        if self._parked_rth or self.rithmic is None:
            self._parked_rth = False
            print("Globex window — connecting Lucid ticker+order (TEST/sim)")
            self._connect_rithmic()
            if self.rithmic is None or not self.rithmic.connected:
                self._connect_retry_at = now + max(15.0, self._ticker_backoff)
                self._ticker_backoff = min(60.0, self._ticker_backoff * 1.5)
            else:
                self._ticker_backoff = 5.0
                self._connect_retry_at = 0.0

    def get_1m(self) -> Optional[pd.DataFrame]:
        try:
            self.csv.refresh(force=False)
        except Exception as exc:
            print(f"CSV/Databento refresh skipped: {redact_secrets(exc)}")
        try:
            seed = load_1m_seed_csv(self.csv.csv_path)
        except Exception:
            seed = None
        # Never call Rithmic history for overnight paper (1011 permission denied).
        df = merge_1m_history(seed, None)
        quote = self.get_latest_price()
        if df is None or df.empty or not quote:
            return df
        last = float(quote.get("last") or quote.get("bid") or 0)
        if last <= 0:
            return df
        row = df.iloc[-1].copy()
        ts = pd.to_datetime(row["datetime"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        # Overlay last print onto the current minute; do not invent a future bar.
        if now - ts < pd.Timedelta(minutes=2):
            df = df.copy()
            df.loc[df.index[-1], "close"] = last
            df.loc[df.index[-1], "high"] = max(float(row["high"]), last)
            df.loc[df.index[-1], "low"] = min(float(row["low"]), last)
        return df

    def maybe_resubscribe_ticker(self) -> None:
        """Re-subscribe THIS ticker plant after ForcedLogout. Never open a second plant."""
        if self.rithmic is None:
            return
        inner = getattr(self.rithmic, "_inner", None) or self.rithmic
        q = None
        try:
            q = inner.get_latest_price("MNQ")
        except Exception:
            q = None
        if q and float(q.get("last") or q.get("bid") or 0) > 0:
            self._ticker_backoff = 5.0
            if self._awaiting_reconnect:
                self._awaiting_reconnect = False
                self._notify(
                    "connection",
                    title="CONNECTION",
                    lines=["Reconnected successfully"],
                    layer="human",
                )
            self._had_quote = True
            return
        now = time.time()
        if now < self._ticker_retry_at:
            return
        self._ticker_retry_at = now + self._ticker_backoff
        self._ticker_backoff = min(60.0, self._ticker_backoff * 1.5)
        if self._had_quote and not self._awaiting_reconnect:
            self._awaiting_reconnect = True
            self._notify(
                "connection",
                title="CONNECTION",
                lines=["Reconnecting after ForcedLogout"],
                layer="human",
            )
        if not self._logout_hinted:
            self._logout_hinted = True
            msg = (
                "Ticker quote missing (ForcedLogout?) — re-subscribing this plant, "
                "not opening a second session. Close Lucid AND extra python "
                "start_live processes."
            )
            print(msg)
            self._notify("dev", reason=msg, layer="dev")
        try:
            inner._market_data_ready = False
            inner._ensure_market_data()
        except Exception as exc:
            err = f"Ticker resubscribe skipped: {redact_secrets(exc)}"
            print(err)
            self._notify("dev", reason=err, layer="dev")

    def get_latest_price(self) -> Optional[Dict[str, float]]:
        if self.rithmic is not None and self.rithmic.connected:
            try:
                q = self.rithmic.get_latest_price("MNQ")
                if q and float(q.get("last") or q.get("bid") or 0) > 0:
                    return q
            except Exception:
                pass
        return self.csv.get_latest_price("MNQ")

    def quote_age_seconds(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        if self.rithmic is not None and self.rithmic.connected:
            q = self.get_latest_price()
            if q:
                return 0.0
        if df is None or df.empty:
            return None
        ts = pd.to_datetime(df.iloc[-1]["datetime"], utc=True)
        return max(0.0, (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds())


def _last_price(feed: OvernightQuoteFeed, df: Optional[pd.DataFrame]) -> Optional[float]:
    q = feed.get_latest_price()
    if q:
        px = float(q.get("last") or q.get("bid") or 0)
        if px > 0:
            return px
    if df is not None and not df.empty:
        return float(df.iloc[-1]["close"])
    return None


class OvernightPaperEngine:
    def __init__(self):
        self.feed = OvernightQuoteFeed()
        self.journal = OvernightPaperJournal(JOURNAL_PATH)
        self.actions = ActionLog()
        self.llm = LLMTradeAdvisor()
        self.position: Optional[Dict[str, Any]] = None
        self.used_cues: set = set()
        self.filled_session: Optional[str] = None
        self.daily_pnl = 0.0
        self.fills_tonight = 0
        self.last_fill: Optional[Dict[str, Any]] = None
        self.zones: Dict[str, Any] = read_zones()
        self._session_key = ""
        self._last_alive_print = 0.0
        self.feed.on_event = self._on_feed_event

    def _on_feed_event(self, kind: str, fields: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(fields or {})
        if str(kind) == "dev" or str(payload.get("layer") or "") == "dev":
            self._emit_dev(str(payload.get("reason") or ""), already_printed=True)
            return
        self._emit(str(kind), **payload)

    def _emit(self, kind: str, *, console: bool = True, **fields: Any) -> Optional[Dict[str, Any]]:
        try:
            rec = self.actions.record(kind, **fields)
        except Exception:
            rec = {"kind": kind, **fields}
        if console:
            try:
                print(format_human_event(rec))
            except Exception:
                pass
        return rec

    def _emit_dev(self, message: str, *, already_printed: bool = False) -> None:
        msg = redact_secrets(message)
        if not already_printed:
            print(msg)
        try:
            self.actions.record("dev", reason=msg, layer="dev")
        except Exception:
            pass

    def _emit_throttled(
        self,
        kind: str,
        key: str,
        interval_sec: float,
        **fields: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            rec = self.actions.record_throttled(
                kind, key=key, interval_sec=interval_sec, **fields,
            )
        except Exception:
            rec = None
        if rec:
            try:
                print(format_human_event(rec))
            except Exception:
                pass
        return rec

    def _reset_session_if_needed(self, now) -> None:
        from src.strategy.overnight_research import globex_session_start_et

        key = globex_session_start_et(now).isoformat()
        if key != self._session_key:
            self._session_key = key
            self.used_cues = set()
            self.filled_session = None
            self.fills_tonight = 0
            self.daily_pnl = 0.0

    def _count_tonight(self) -> int:
        from src.strategy.overnight_research import globex_session_start_et

        start = pd.Timestamp(globex_session_start_et()).tz_convert("UTC")
        n = 0
        for row in self.journal.closed_trades():
            ts = row.get("exit_time") or row.get("timestamp")
            if not ts:
                continue
            try:
                t = pd.Timestamp(ts)
                if t.tzinfo is None:
                    t = t.tz_localize("UTC")
                if t >= start:
                    n += 1
            except (TypeError, ValueError):
                continue
        return n

    def _heartbeat(self, note: str = "", df: Optional[pd.DataFrame] = None) -> None:
        px = _last_price(self.feed, df)
        age = self.feed.quote_age_seconds(df)
        et = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        llm = "off"
        if getattr(self.llm, "enabled", False):
            llm = f"{self.llm.provider}:{self.llm.model}"
        open_pos = []
        open_label = "flat"
        if self.position:
            open_label = str(self.position.get("side") or "open").upper()
            open_pos = [{
                "symbol": "MNQ",
                "direction": str(self.position.get("side") or "").upper(),
                "entry": self.position.get("entry_price"),
                "sl": self.position.get("stop"),
                "tp": self.position.get("target"),
                "atr_stop_pts": self.position.get("atr_stop_pts"),
                "flatten_et": self.position.get("flatten_et") or "09:25",
                "size": QTY,
                "cue": self.position.get("cue"),
                "zone_name": self.position.get("zone_name"),
            }]
        try:
            write_status({
                "running": True,
                "strategy": SESSION_NAME,
                "paper_mode": True,
                "mode": "paper",
                "overnight_research": True,
                "session": SESSION_NAME,
                "daily_pnl": round(self.daily_pnl, 2),
                "open": open_label,
                "open_positions": open_pos,
                "live_trades_today": int(self.fills_tonight) + (1 if self.position else 0),
                "last_price": px,
                "last_quote_ts_et": et,
                "quote_age_seconds": age,
                "quote_source": self.feed.source,
                "llm": llm,
                "llm_mode": os.getenv("LLM_MODE", "advisory"),
                "symbols": ["MNQ"],
                "last_note": note,
                "journal_path": self.journal.path,
                "suggestions_path": SUGGESTIONS_PATH,
                "zones_path": ZONES_PATH,
                "zones": (self.zones or {}).get("zones") or [],
                "last_paper_fill": self.last_fill,
                "suggestions_ready": False,
            })
        except Exception:
            pass

    def _gemini_note(self, rec: Dict[str, Any]) -> str:
        if not getattr(self.llm, "enabled", False):
            return ""
        try:
            return self.llm.overnight_close_note(rec)
        except Exception:
            try:
                return self.llm.post_mortem_line(rec)
            except Exception:
                return ""

    def _close_position(self, exit_price: float, reason: str, df: Optional[pd.DataFrame]) -> None:
        pos = self.position
        if not pos:
            return
        now = datetime.now(timezone.utc)
        pts, usd = paper_pnl_usd(pos["side"], float(pos["entry_price"]), float(exit_price), QTY)
        rec = {
            "trade_id": pos["trade_id"],
            "symbol": "MNQ",
            "side": pos["side"],
            "direction": pos["side"],
            "qty": QTY,
            "entry_price": pos["entry_price"],
            "exit_price": round(float(exit_price), 2),
            "stop": pos.get("stop"),
            "target": pos.get("target"),
            "pts": pts,
            "pnl_usd": usd,
            "hold_minutes": round((now - pos["entry_time"]).total_seconds() / 60.0, 2),
            "exit_reason": reason,
            "entry_time": pos["entry_time"].isoformat(),
            "exit_time": now.isoformat(),
            "entry_ts_et": pos.get("entry_ts_et"),
            "exit_ts_et": now.astimezone(ET).isoformat(),
            "atr_stop_pts": pos.get("atr_stop_pts"),
            "cue": pos.get("cue"),
            "zone_name": pos.get("zone_name"),
            "zone_price": pos.get("zone_price"),
            "mae_pts": pos.get("mae_pts"),
            "mfe_pts": pos.get("mfe_pts"),
            "why": {
                "primary_reason": pos.get("why") or reason,
                "facts": [
                    f"cue={pos.get('cue')}",
                    f"zone={pos.get('zone_name')} @{pos.get('zone_price')}",
                    f"exit={reason}",
                ],
            },
        }
        rec = attach_mae_mfe(
            rec, df, pos["entry_time"], now, pos["side"], float(pos["entry_price"]),
        )
        if rec.get("mae_pts") is None:
            rec["mae_pts"] = pos.get("mae_pts")
        if rec.get("mfe_pts") is None:
            rec["mfe_pts"] = pos.get("mfe_pts")
        note = self._gemini_note(rec)
        rec = self.journal.log_close(rec, gemini_note=note)
        self.journal.refresh_suggestions()
        self.daily_pnl = round(self.daily_pnl + usd, 2)
        self.fills_tonight = self._count_tonight()
        self.last_fill = {
            "trade_id": rec.get("trade_id"),
            "side": rec.get("side"),
            "cue": rec.get("cue"),
            "zone_name": rec.get("zone_name"),
            "entry_price": rec.get("entry_price"),
            "exit_price": rec.get("exit_price"),
            "pts": rec.get("pts"),
            "pnl_usd": rec.get("pnl_usd"),
            "exit_reason": rec.get("exit_reason"),
            "gemini_note": rec.get("gemini_note") or "",
            "session": SESSION_NAME,
        }
        if pos.get("lucid_ticket") and self._lucid_brackets():
            try:
                self.feed.rithmic.flatten_symbol("MNQ", max_attempts=8, verify_delay=1.5)
            except Exception as exc:
                err = f"PAPER_RITHMIC_BRACKETS flatten failed: {redact_secrets(exc)}"
                print(err)
                self._emit_dev(err, already_printed=True)
        title = "PROFITABLE TRADE" if usd >= 0 else "LOSING TRADE"
        emoji = "💰" if usd >= 0 else "🔴"
        self._emit(
            "profit" if usd >= 0 else "loss",
            emoji=emoji,
            title=title,
            symbol="MNQ",
            direction=pos["side"],
            entry=pos.get("entry_price"),
            exit=round(float(exit_price), 2),
            pnl=usd,
            qty=QTY,
            paper=True,
            session=SESSION_NAME,
            reason=reason,
        )
        self.position = None

    def _lucid_brackets(self) -> bool:
        return bool(self.feed.brackets) and self.feed.rithmic is not None and not isinstance(
            self.feed.rithmic, QuotesOnlyBroker
        )

    def _open_paper_position(self, sig: Dict[str, Any]) -> None:
        blocked = overnight_entry_blocked_reason()
        if blocked:
            print(blocked)
            self._emit(
                "skip",
                title="TRADE SKIPPED",
                lines=[blocked],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        lucid_ticket = ""
        want_lucid = skip_local_fake_fill() or self._lucid_brackets()
        if want_lucid:
            if not self._lucid_brackets():
                msg = "Lucid order plant not connected — not booking a local fake fill"
                print(msg)
                self._emit_dev(msg, already_printed=True)
                self._emit(
                    "skip",
                    title="TRADE SKIPPED",
                    lines=["Waiting for Lucid order plant (no local fake fill)"],
                    symbol="MNQ",
                    paper=True,
                    session=SESSION_NAME,
                )
                return
            inner = self.feed.rithmic
            side = "buy" if str(sig.get("side") or "").lower() == "long" else "sell"
            try:
                result = inner.place_order(
                    symbol="MNQ",
                    order_type=side,
                    size=int(QTY),
                    entry_price=float(sig["entry_price"]),
                    stop_loss=float(sig.get("stop") or 0),
                    take_profit=float(sig.get("target") or 0),
                )
            except Exception as exc:
                err = f"PAPER_RITHMIC_BRACKETS place_order failed: {redact_secrets(exc)}"
                print(err)
                self._emit_dev(err, already_printed=True)
                self._emit(
                    "skip",
                    title="TRADE SKIPPED",
                    lines=["Lucid sim order failed"],
                    symbol="MNQ",
                    paper=True,
                    session=SESSION_NAME,
                )
                return
            if not result or not (result.get("ticket") or result.get("order_id")):
                msg = "PAPER_RITHMIC_BRACKETS: Lucid order not booked — skipping local fill"
                print(msg)
                self._emit_dev(msg, already_printed=True)
                self._emit(
                    "skip",
                    title="TRADE SKIPPED",
                    lines=["Lucid order not booked"],
                    symbol="MNQ",
                    paper=True,
                    session=SESSION_NAME,
                )
                return
            lucid_ticket = str(result.get("ticket") or result.get("order_id"))
            if not result.get("native_stop_attached"):
                msg = "PAPER_RITHMIC_BRACKETS fill has no broker stop — flattening"
                print(msg)
                self._emit_dev(msg, already_printed=True)
                try:
                    inner.flatten_symbol("MNQ", max_attempts=8, verify_delay=1.5)
                except Exception as exc:
                    err = f"flatten after naked fill failed: {redact_secrets(exc)}"
                    print(err)
                    self._emit_dev(err, already_printed=True)
                self._emit(
                    "skip",
                    title="TRADE SKIPPED",
                    lines=["Fill had no broker stop — flattened"],
                    symbol="MNQ",
                    paper=True,
                    session=SESSION_NAME,
                )
                return
        tid = lucid_ticket or f"ON_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.journal.log_entry(sig, trade_id=tid)
        self.used_cues.add(str(sig.get("cue") or ""))
        if str(sig.get("cue") or "") != "test_fill":
            self.filled_session = self._session_key
        increment_zone_touch(self.zones, str(sig.get("zone_name") or ""))
        write_zones(self.zones)
        self.position = {
            **sig,
            "trade_id": tid,
            "lucid_ticket": lucid_ticket,
            "entry_time": datetime.now(timezone.utc),
            "entry_ts_et": datetime.now(ET).isoformat(),
            "mae_pts": 0.0,
            "mfe_pts": 0.0,
        }
        self.last_fill = {
            "trade_id": tid,
            "side": sig.get("side"),
            "cue": sig.get("cue"),
            "zone_name": sig.get("zone_name"),
            "entry_price": sig.get("entry_price"),
            "exit_price": None,
            "pts": None,
            "pnl_usd": None,
            "exit_reason": "open",
            "session": SESSION_NAME,
        }
        dest = (
            f"Lucid sim ticket={lucid_ticket}"
            if lucid_ticket
            else "local journal only"
        )
        self._emit(
            "entry",
            symbol="MNQ",
            direction=sig["side"],
            reason=sig.get("why"),
            entry=sig["entry_price"],
            stop=sig.get("stop"),
            atr_stop_pts=sig.get("atr_stop_pts"),
            flatten_et=sig.get("flatten_et") or "09:25",
            qty=QTY,
            session=SESSION_NAME,
            paper=True,
            cue=sig.get("cue"),
            dest=dest,
        )

    def _ticker_last(self) -> Optional[float]:
        inner = getattr(self.feed.rithmic, "_inner", None) or self.feed.rithmic
        if inner is None:
            return None
        try:
            q = inner.get_latest_price("MNQ")
        except Exception:
            return None
        if not q:
            return None
        px = float(q.get("last") or q.get("bid") or 0)
        return px if px > 0 else None

    def _place_test_fill(self) -> None:
        blocked = overnight_entry_blocked_reason()
        if blocked:
            msg = f"PAPER TEST FILL skipped — {blocked}"
            print(msg)
            self._emit_dev(msg, already_printed=True)
            self._emit(
                "skip",
                title="TRADE SKIPPED",
                lines=[blocked],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        if self._lucid_brackets():
            print("PAPER TEST FILL — sending 1 MNQ to Lucid TEST simulator")
        else:
            print("PAPER TEST FILL — not sent to Lucid")
        df = None
        try:
            df = self.feed.get_1m()
        except Exception as exc:
            print(f"Test fill CSV bars skipped: {redact_secrets(exc)}")
        px = None
        for _ in range(60):
            try:
                self.feed.maybe_resubscribe_ticker()
            except Exception:
                pass
            px = self._ticker_last()
            if px:
                break
            time.sleep(0.5)
        if not px or float(px) <= 0:
            msg = "PAPER TEST FILL skipped — waiting for live ticker last (not stale CSV)"
            print(msg)
            self._emit_dev(msg, already_printed=True)
            self._emit(
                "skip",
                title="TRADE SKIPPED",
                lines=["Waiting for live ticker last (not stale CSV)"],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        now = datetime.now(ET)
        self._reset_session_if_needed(now)
        sig = paper_test_fill_signal(df, float(px))
        self._open_paper_position(sig)
        sent = "Lucid sim" if (self.position or {}).get("lucid_ticket") else "not sent to Lucid"
        note = (
            f"PAPER TEST FILL · MNQ {str(sig['side']).upper()} @ {sig['entry_price']} "
            f"SL {sig['stop']} · {sent}"
        )
        self._heartbeat(note, df)

    def _maybe_enter(self, df: pd.DataFrame, now) -> None:
        if self.position is not None:
            return
        already = self.filled_session == self._session_key
        sig = evaluate_overnight_cue(
            df, now, already_filled=already, used_cues=self.used_cues,
        )
        if not sig:
            if already:
                line = "Already filled this Globex session"
            elif not allow_new_overnight_entry(now):
                line = "Overnight session closed — no new entries"
            else:
                line = "Overnight cue not met"
            self._emit_throttled(
                "idle",
                "overnight-idle",
                MONITOR_INTERVAL_SEC,
                title="NO TRADE",
                lines=[line],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        px = _last_price(self.feed, df)
        if px:
            sig["entry_price"] = round(float(px), 2)
            sl = float(sig["atr_stop_pts"])
            if sig["side"] == "short":
                sig["stop"] = round(sig["entry_price"] + sl, 2)
                sig["target"] = round(sig["entry_price"] - sl, 2)
            else:
                sig["stop"] = round(sig["entry_price"] - sl, 2)
                sig["target"] = round(sig["entry_price"] + sl, 2)
        self._open_paper_position(sig)

    def _manage_open(self, df: pd.DataFrame, now) -> None:
        if not self.position:
            return
        px = _last_price(self.feed, df)
        if px is None:
            return
        high = px
        low = px
        # Databento/CSV seed is often RTH; do not SL/TP a live ticker fill off an 8h-old bar.
        if df is not None and not df.empty:
            try:
                ts = pd.to_datetime(df.iloc[-1]["datetime"], utc=True)
                age = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds()
            except Exception:
                age = 10_000.0
            if age < 120:
                high = max(float(df.iloc[-1]["high"]), px)
                low = min(float(df.iloc[-1]["low"]), px)
        if str(self.position.get("cue") or "") == "test_fill":
            if should_flatten_before_rth(now):
                self._close_position(px, "flatten_before_rth", df)
            return
        self.position["mae_pts"], self.position["mfe_pts"] = update_mae_mfe(
            self.position["side"],
            float(self.position["entry_price"]),
            high, low,
            float(self.position.get("mae_pts") or 0),
            float(self.position.get("mfe_pts") or 0),
        )
        hit = check_paper_exit(self.position, high=high, low=low, last=px, now=now)
        if hit:
            self._close_position(float(hit["exit_price"]), str(hit["reason"]), df)

    def _start_desk(self) -> None:
        if os.getenv("DASHBOARD", "true").lower() not in ("1", "true", "yes"):
            return
        port = int(os.getenv("DASHBOARD_PORT", "5055"))
        from src.dashboard.mtf_actions import start_mtf_dashboard
        threading.Thread(target=start_mtf_dashboard, kwargs={"port": port}, daemon=True).start()
        print(f"Desk: http://127.0.0.1:{port}")

    def run(self, duration_minutes: int = 0, test_fill: bool = False) -> None:
        enable_overnight_lucid_sim_orders(keep_test_fill=bool(test_fill))
        print("=" * 70)
        print("  PAPER OVERNIGHT — break_settled_onh_onl only")
        print("  Lucid TEST/sim orders — not funded live. Day ema15 stays RTH.")
        print("  Quotes: Rithmic ticker. History plant skipped (1011).")
        print("  PAPER_RITHMIC_BRACKETS — fills must appear on Lucid (no local fake fill)")
        print("  Cue: first 1m close beyond settled ONH/ONL (NO fade)")
        print("  1 MNQ · ATR×1.5 stop (12–40) · 1R TP · flatten 09:25 ET")
        print("  HARD idle 9:30–16:00 ET. Stop overnight ~09:20 before the day bot.")
        print("=" * 70)
        print(
            "If Lucid / R|Trader desktop is logged in, close it so this bot "
            "can keep the ticker session. One Lucid login only — Ctrl+C the day bot first."
        )
        want_test = bool(test_fill) or paper_test_fill_from_env()
        try:
            self.feed.initialize()
        except Exception as exc:
            print(f"Feed init error (continuing with whatever connected): {redact_secrets(exc)}")
            if self.feed.rithmic is None and not day_bot_owns_lucid():
                try:
                    self.feed._connect_rithmic()
                except Exception as exc2:
                    print(f"Rithmic retry failed: {redact_secrets(exc2)}")
        self._start_desk()
        self.fills_tonight = self._count_tonight()
        deadline = None
        if duration_minutes and duration_minutes > 0:
            deadline = time.time() + duration_minutes * 60
        self._heartbeat("overnight paper starting")
        self._print_alive()
        self._emit(
            "data",
            title="MARKET",
            lines=[
                "Overnight paper starting",
                f"Quotes from {self.feed.source}",
                "Desk http://127.0.0.1:5055",
            ],
            paper=True,
            session=SESSION_NAME,
        )
        if want_test:
            self._place_test_fill()
        try:
            while True:
                if deadline and time.time() >= deadline:
                    break
                now = datetime.now(ET)
                self._reset_session_if_needed(now)
                try:
                    self.feed.maybe_align_lucid_session()
                    self.feed.maybe_resubscribe_ticker()
                    df = self.feed.get_1m()
                    px = _last_price(self.feed, df)
                    if df is not None and not df.empty:
                        self.zones = compute_zones(
                            df, now, last_price=px,
                            prev_zones=(self.zones or {}).get("zones"),
                        )
                        self.zones["source"] = self.feed.source
                        write_zones(self.zones)
                    if self.position is not None:
                        self._manage_open(df if df is not None else pd.DataFrame(), now)
                    if should_flatten_before_rth(now) and self.position is not None:
                        px = _last_price(self.feed, df) or float(self.position["entry_price"])
                        self._close_position(px, "flatten_before_rth", df)
                    if allow_new_overnight_entry(now) and df is not None:
                        self._maybe_enter(df, now)
                    elif not is_overnight_research_hours(now):
                        blocked = overnight_entry_blocked_reason(now)
                        if in_rth_hard_idle(now):
                            note = (
                                "RTH — overnight paper idle "
                                "(locked live recipe owns 9:30–16:00; no entries)"
                            )
                        elif day_bot_owns_lucid(now):
                            note = (
                                "Pre-Globex idle — waiting for 18:00 ET "
                                "(one Lucid session; no entries yet)"
                            )
                        else:
                            note = blocked or "Globex closed — overnight paper waiting"
                        self._heartbeat(note, df)
                        self._print_alive(df)
                        self._emit_throttled(
                            "idle",
                            "session-idle",
                            MONITOR_INTERVAL_SEC,
                            title="NO TRADE",
                            lines=[note],
                            symbol="MNQ",
                            paper=True,
                            session=SESSION_NAME,
                        )
                        sleep_s = 30.0
                        if not is_globex_session_et(now):
                            sleep_s = min(seconds_until_session_open_et(now, "extended"), 1800)
                        time.sleep(max(10.0, min(sleep_s, 60.0)))
                        continue
                    src = self.feed.source
                    if self.position and str(self.position.get("cue") or "") == "test_fill":
                        sent = (
                            f"Lucid {self.position.get('lucid_ticket')}"
                            if self.position.get("lucid_ticket")
                            else "not sent to Lucid"
                        )
                        note = (
                            f"PAPER TEST FILL · MNQ {str(self.position.get('side') or '').upper()} "
                            f"@ {self.position.get('entry_price')} SL {self.position.get('stop')} "
                            f"· {sent}"
                        )
                    else:
                        note = (
                            f"PAPER OVERNIGHT RESEARCH · {src} quotes · "
                            f"{int(self.fills_tonight) + (1 if self.position else 0)} "
                            f"paper fills this Globex session"
                        )
                    self._heartbeat(note, df)
                    self._print_alive(df)
                    time.sleep(10 if self.position is None else 5)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    err = f"Overnight paper scan error: {redact_secrets(exc)}"
                    print(err)
                    tb = redact_secrets(traceback.format_exc())
                    print(tb)
                    self._emit(
                        "warning",
                        title="PROBLEM",
                        lines=["Scan error — still running", str(redact_secrets(exc))],
                        paper=True,
                        session=SESSION_NAME,
                    )
                    self._emit_dev(tb, already_printed=True)
                    self._heartbeat(f"scan error (still running): {redact_secrets(exc)}")
                    time.sleep(15)
        except KeyboardInterrupt:
            print("\nOvernight paper research stopped")
            if self.position is not None:
                df = self.feed.get_1m()
                px = _last_price(self.feed, df) or float(self.position["entry_price"])
                self._close_position(px, "shutdown", df)
        finally:
            try:
                write_status({
                    "running": False,
                    "strategy": SESSION_NAME,
                    "paper_mode": True,
                    "mode": "paper",
                    "overnight_research": True,
                    "session": SESSION_NAME,
                    "last_note": "overnight paper stopped",
                    "live_trades_today": int(self.fills_tonight) + (1 if self.position else 0),
                    "zones": (self.zones or {}).get("zones") or [],
                    "last_paper_fill": self.last_fill,
                    "journal_path": self.journal.path,
                })
            except Exception:
                pass
            self.feed.shutdown()

    def _print_alive(self, df: Optional[pd.DataFrame] = None) -> None:
        px = _last_price(self.feed, df)
        px_s = f"{px:.2f}" if px else "none"
        sl_s = "flat"
        if self.position:
            try:
                sl_s = f"{float(self.position.get('stop')):.2f}"
            except (TypeError, ValueError):
                sl_s = "—"
        print(
            f"PAPER OVERNIGHT alive SL={sl_s} TP=1R flatten 09:25 last={px_s}",
            flush=True,
        )
        if not self.position:
            return
        side = str(self.position.get("side") or "trade").capitalize()
        self._emit_throttled(
            "monitoring",
            "open-monitor",
            MONITOR_INTERVAL_SEC,
            title="MONITORING",
            lines=[
                f"{side} still open · MNQ {px_s}",
                f"Stop {sl_s} · 1R TP · Flatten 09:25 ET",
            ],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )


def run_overnight_paper(*, duration_minutes: int = 0, test_fill: bool = False) -> None:
    OvernightPaperEngine().run(duration_minutes=duration_minutes, test_fill=test_fill)
