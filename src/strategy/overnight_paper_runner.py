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
from typing import Any, Dict, List, Optional, Sequence

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
    load_last_open_overnight,
    load_open_overnight_positions,
    read_zones,
    write_zones,
)
from src.data.paper_csv_feed import (
    PaperCsvFeed,
    default_1m_csv_path,
    paper_rithmic_brackets_enabled,
)
from src.strategy.mnq_15m_ema_eod import (
    load_1m_seed_csv,
    merge_1m_history,
    overlay_ticker_last_on_1m,
)
from src.strategy.overnight_research import (
    SESSION_NAME,
    QTY,
    CUE_BREAK_ONH,
    CUE_BREAK_ONL,
    allow_new_overnight_entry,
    allow_overnight_addon,
    allow_overnight_entry_confidence,
    allow_overnight_macd_entry,
    explain_overnight_wait,
    fill_window_start_et,
    globex_session_start_et,
    overnight_cap_reached,
    overnight_addon_confidence_min,
    overnight_min_confidence,
    overnight_tp_r,
    overnight_tp_text,
    overnight_entry_dedupe_sec,
    overnight_entry_fingerprint,
    overnight_fill_limit,
    overnight_max_open,
    overnight_open_slots_full,
    overnight_recent_entry_blocked,
    overnight_same_cue_open,
    paper_max_fills_base,
    recipe_session_stats,
    recipe_cooldown_until,
    in_overnight_cooldown,
    overnight_cooldown_override_message,
    overnight_completed_bars,
    settled_overnight_range,
    last_completed_bar,
    overnight_htf_sides,
    macd_trend_15m_on_1m,
    overnight_macd_enabled,
    overnight_macd_require,
    overnight_stack_gate,
    should_defer_entry_for_stack,
    overnight_stack_why_suffix,
    entry_confidence_pct,
    format_trade_placed_banner,
    format_entry_confidence_line,
    recipe_historical_win_pct,
    check_paper_exit,
    compute_zones,
    day_bot_owns_lucid,
    enable_overnight_lucid_sim_orders,
    evaluate_overnight_cue,
    increment_zone_touch,
    is_overnight_research_hours,
    overnight_entry_blocked_reason,
    paper_pnl_usd,
    paper_test_fill_from_env,
    paper_test_fill_signal,
    skip_local_fake_fill,
    update_mae_mfe,
)
from src.utils.redact import redact_secrets
from src.utils.trade_console import (
    bracket_ok_mark,
    exit_emoji,
    format_entry_line,
    format_exit_line,
    format_heartbeat_line,
    trade_emoji,
)
from src.utils.trading_session import is_globex_session_et, seconds_until_session_open_et

ET = pytz.timezone("US/Eastern")
LIVE_1M_PATH = "data/mnq_overnight_live_1m.csv"
BROKER_FLAT_SYNC_SEC = float(os.getenv("BROKER_FLAT_SYNC_SEC", "30"))
BROKER_FLAT_SYNC_SCANS = int(os.getenv("BROKER_FLAT_SYNC_SCANS", "2"))
LEFTOVER_SWEEP_SEC = float(os.getenv("OVERNIGHT_LEFTOVER_SWEEP_SEC", "10"))


def _pos_qty(obj: Any) -> int:
    try:
        n = int((obj or {}).get("qty") or QTY)
        return n if n >= 1 else QTY
    except (TypeError, ValueError):
        return QTY


def _session_live_1m(df: Optional[pd.DataFrame], now=None) -> Optional[pd.DataFrame]:
    if df is None or getattr(df, "empty", True):
        return None
    out = df.copy()
    ts = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    start = pd.Timestamp(globex_session_start_et(now)).tz_convert("UTC")
    out = out.loc[ts >= start]
    if out.empty:
        return None
    return out.reset_index(drop=True)


def load_persisted_live_1m(path: str = LIVE_1M_PATH, now=None) -> Optional[pd.DataFrame]:
    try:
        raw = load_1m_seed_csv(path)
    except Exception:
        raw = None
    return _session_live_1m(raw, now)


def save_persisted_live_1m(df: Optional[pd.DataFrame], path: str = LIVE_1M_PATH, now=None) -> None:
    keep = _session_live_1m(df, now)
    if keep is None:
        return
    cols = [c for c in ("datetime", "open", "high", "low", "close", "volume") if c in keep.columns]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keep[cols].to_csv(path, index=False)
BLOCKED_ORDER_METHODS = frozenset({
    "place_order",
    "close_position",
    "modify_position",
    "flatten_symbol",
    "cancel_order",
    "cancel_all",
    "cancel_working_orders_after_exit",
    "modify_stop",
})


def should_restore_open(row: Optional[Dict[str, Any]]) -> bool:
    """Do not revive a paper test fill after flatten. Only a real break cue."""
    if not row:
        return False
    return str(row.get("cue") or "") != "test_fill"


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
        self._live_1m: Optional[pd.DataFrame] = None
        self._live_saved_at = 0.0
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
                "Day ema15 bot is using Lucid — this research runner will not take "
                "the ticker. Ctrl+C start_mnq_live.bat first (one Lucid login)."
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
        """One Lucid login: park only if the ema15 day bot process is running."""
        if day_bot_owns_lucid():
            if self.rithmic is not None:
                print(
                    "Releasing Lucid session for the running day bot. "
                    "Ctrl+C this window if you want start_mnq_live.bat instead."
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
            print("Connecting Lucid ticker+order (TEST/sim) — research runner keeps RTH")
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
        last = 0.0
        if quote:
            last = float(quote.get("last") or quote.get("bid") or 0)
        now = datetime.now(timezone.utc)
        if self._live_1m is None or getattr(self._live_1m, "empty", True):
            self._live_1m = load_persisted_live_1m(LIVE_1M_PATH, now)
        age = None
        if quote and quote.get("updated_at"):
            age = time.time() - float(quote["updated_at"])
        if last > 0 and (age is None or age < 20.0):
            if self._live_1m is None or getattr(self._live_1m, "empty", True):
                minute = pd.Timestamp(now).tz_convert("UTC").floor("min")
                self._live_1m = pd.DataFrame([{
                    "datetime": minute,
                    "open": last,
                    "high": last,
                    "low": last,
                    "close": last,
                    "volume": 0,
                }])
            else:
                self._live_1m = overlay_ticker_last_on_1m(self._live_1m, last, now=now)
            df = merge_1m_history(df, self._live_1m)
            if time.time() - float(self._live_saved_at or 0) >= 30.0:
                try:
                    save_persisted_live_1m(self._live_1m, LIVE_1M_PATH, now)
                    self._live_saved_at = time.time()
                except Exception:
                    pass
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
        if self.rithmic is not None:
            try:
                q = self.rithmic.get_latest_price("MNQ")
                if q and float(q.get("last") or q.get("bid") or 0) > 0:
                    return q
            except Exception:
                pass
            return None
        return self.csv.get_latest_price("MNQ")

    def quote_age_seconds(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        if self.rithmic is not None:
            try:
                q = self.rithmic.get_latest_price("MNQ")
            except Exception:
                q = None
            if q and float(q.get("last") or q.get("bid") or 0) > 0:
                stamp = q.get("updated_at")
                if stamp:
                    return max(0.0, time.time() - float(stamp))
                return 0.0
        if df is None or df.empty:
            return None
        ts = pd.to_datetime(df.iloc[-1]["datetime"], utc=True)
        return max(0.0, (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds())


LIVE_QUOTE_MAX_AGE_SEC = 120.0


def _bar_age_seconds(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    try:
        ts = pd.to_datetime(df.iloc[-1]["datetime"], utc=True)
        return max(0.0, (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds())
    except Exception:
        return None


def _rithmic_last(feed: OvernightQuoteFeed) -> Optional[float]:
    inner = getattr(feed, "rithmic", None)
    inner = getattr(inner, "_inner", None) or inner
    if inner is None:
        return None
    try:
        q = inner.get_latest_price("MNQ")
    except Exception:
        return None
    if not q:
        return None
    px = float(q.get("last") or q.get("bid") or 0)
    if px <= 0:
        return None
    stamp = q.get("updated_at")
    if stamp:
        age = time.time() - float(stamp)
        if age > LIVE_QUOTE_MAX_AGE_SEC:
            return None
    return px


def _last_price(feed: OvernightQuoteFeed, df: Optional[pd.DataFrame]) -> Optional[float]:
    """Live ticker when a Rithmic plant exists. Never leftover RTH CSV for last."""
    if getattr(feed, "rithmic", None) is not None:
        return _rithmic_last(feed)
    age = _bar_age_seconds(df)
    if age is not None and age <= LIVE_QUOTE_MAX_AGE_SEC:
        return float(df.iloc[-1]["close"])
    q = feed.get_latest_price()
    if q:
        px = float(q.get("last") or q.get("bid") or 0)
        if px > 0 and age is not None and age <= LIVE_QUOTE_MAX_AGE_SEC:
            return px
    return None


class OvernightPaperEngine:
    def __init__(self):
        self.feed = OvernightQuoteFeed()
        self.journal = OvernightPaperJournal(JOURNAL_PATH)
        self.actions = ActionLog()
        self.llm = LLMTradeAdvisor()
        self.positions: List[Dict[str, Any]] = []
        self.used_cues: set = set()
        self.filled_session: Optional[str] = None
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.fills_tonight = 0
        self.cooldown_until = None
        self.last_fill: Optional[Dict[str, Any]] = None
        self.zones: Dict[str, Any] = read_zones()
        self._session_key = ""
        self._last_alive_print = 0.0
        self._last_leftover_sweep = 0.0
        self._broker_flat_since: Optional[datetime] = None
        self._broker_flat_scans = 0
        self._last_entry_fingerprint = ""
        self._last_entry_at = 0.0
        self._entry_placed_this_loop = False
        self.feed.on_event = self._on_feed_event

    @property
    def position(self) -> Optional[Dict[str, Any]]:
        """First open trade — backward compat for single-position callers/tests."""
        return self.positions[0] if self.positions else None

    @position.setter
    def position(self, value: Optional[Dict[str, Any]]) -> None:
        if value is None:
            self.positions = []
        else:
            self.positions = [value]

    def _open_count(self) -> int:
        return len(self.positions)

    def _has_open(self) -> bool:
        return bool(self.positions)

    def _open_tickets(self) -> set:
        tickets: set = set()
        for pos in self.positions:
            tid = str(pos.get("lucid_ticket") or pos.get("trade_id") or "")
            if tid:
                tickets.add(tid)
        return tickets

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
        key = fill_window_start_et(now).date().isoformat()
        if key != self._session_key:
            self._session_key = key
            self.used_cues = set()
            self.filled_session = None
            self.session_pnl = 0.0
            self.daily_pnl = 0.0
            self.fills_tonight = self._count_tonight()

    def _restore_open_position(self) -> None:
        if self._has_open():
            return
        rows = load_open_overnight_positions()
        rows = [r for r in rows if should_restore_open(r)]
        if not rows:
            stale = load_last_open_overnight()
            if stale and not should_restore_open(stale):
                print("Skip restore: last journal open was a test_fill, not the break recipe")
            return
        cap = overnight_max_open()
        if len(rows) > cap:
            rows = rows[-cap:]
        if self._lucid_brackets():
            try:
                net = self.feed.rithmic.get_symbol_list_positions_net("MNQ")
            except Exception:
                net = None
            if net is None or int(net) == 0:
                print(
                    "Skip restore: Lucid MNQ is not an open position "
                    f"(list_positions net={net}); leftover sweep will run"
                )
                return
        self._reset_session_if_needed(datetime.now(ET))
        restored: List[Dict[str, Any]] = []
        for row in rows:
            raw_ts = row.get("entry_time") or row.get("timestamp")
            try:
                et = pd.Timestamp(raw_ts)
                if et.tzinfo is None:
                    et = et.tz_localize("UTC")
                entry_dt = et.tz_convert("UTC").to_pydatetime()
            except Exception:
                entry_dt = datetime.now(timezone.utc)
            tid = str(row.get("trade_id") or "")
            restored.append({
                **row,
                "side": row.get("side") or row.get("direction"),
                "entry_price": row.get("entry_price") or row.get("entry"),
                "entry_time": entry_dt,
                "qty": int(row.get("qty") or QTY),
                "lucid_ticket": tid if tid.startswith("bot_") else "",
                "mae_pts": 0.0,
                "mfe_pts": 0.0,
            })
            self.used_cues.add(str(row.get("cue") or ""))
        self.positions = restored
        self.filled_session = self._session_key
        if restored:
            last = restored[-1]
            self._last_entry_fingerprint = overnight_entry_fingerprint(last)
            try:
                et = last.get("entry_time") or last.get("timestamp")
                self._last_entry_at = pd.Timestamp(et).timestamp()
            except Exception:
                self._last_entry_at = time.time()
        for pos in restored:
            print(
                f"Restored open overnight {pos.get('side')} "
                f"@ {pos.get('entry_price')} ticket={pos.get('lucid_ticket') or pos.get('trade_id')}"
            )

    @staticmethod
    def _working_order_kinds(rows: Sequence[Any]) -> str:
        kinds = set()
        for row in rows or []:
            if not isinstance(row, dict):
                kinds.add("UNKNOWN")
                continue
            kind = str(row.get("kind") or "").strip()
            if kind:
                kinds.add(kind)
                continue
            side = str(
                row.get("transaction_type") or row.get("transactionType") or ""
            ).upper().strip()
            ptype = str(
                row.get("price_type")
                or row.get("priceType")
                or row.get("status")
                or "UNKNOWN"
            ).upper().strip()
            kinds.add(f"{side} {ptype}".strip())
        return ",".join(sorted(kinds)) or "?"

    @staticmethod
    def _is_bot_owned_working_row(row: Any) -> bool:
        """True for bot_* / overnight-tagged tickets only — never manual SL/TP."""
        if not isinstance(row, dict):
            return False
        tag = str(row.get("user_tag") or row.get("userTag") or "").strip()
        basket = str(row.get("basket_id") or row.get("basketId") or "").strip()
        if tag.startswith("bot_") or basket.startswith("bot_"):
            return True
        return False

    @classmethod
    def _bot_owned_working(cls, rows: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        return [
            row for row in (rows or [])
            if isinstance(row, dict) and cls._is_bot_owned_working_row(row)
        ]

    def _list_mnq_working(self, *, fail_closed: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Listed MNQ working tickets, or None when the scan must fail closed."""
        if not self._lucid_brackets():
            return []
        try:
            rows = self.feed.rithmic.list_working_orders("MNQ")
        except Exception as exc:
            err = f"MNQ working-order scan failed: {redact_secrets(exc)}"
            print(err)
            self._emit_dev(err, already_printed=True)
            return None if fail_closed else []
        if not isinstance(rows, (list, tuple)):
            return None if fail_closed else []
        return list(rows)

    def _log_manual_working_left_alone(self, rows: Optional[Sequence[Any]]) -> int:
        manual_n = len(rows or []) - len(self._bot_owned_working(rows))
        if manual_n > 0:
            msg = f"leaving {manual_n} manual working orders alone"
            print(msg)
            self._emit_dev(msg, already_printed=True)
        return manual_n

    def _aggressive_remnant_wipe(self, reason: str) -> int:
        """Cancel leftover bot-owned MNQ tickets when paper-flat. Never flatten size."""
        if self._has_open() or not self._lucid_brackets():
            return 0
        remaining = self._list_mnq_working()
        if not remaining:
            return 0
        bot_rows = self._bot_owned_working(remaining)
        self._log_manual_working_left_alone(remaining)
        if not bot_rows:
            return 0
        n = self._cancel_leftover_working_if_flat(reason)
        remaining = self._list_mnq_working()
        bot_left = self._bot_owned_working(remaining)
        if bot_left:
            kinds = self._working_order_kinds(bot_left)
            msg = (
                f"{len(bot_left)} bot-owned MNQ working order(s) still listed after wipe "
                f"({reason}) kinds={kinds}"
            )
            print(msg)
            self._emit_dev(msg, already_printed=True)
        elif remaining:
            self._log_manual_working_left_alone(remaining)
        return n

    def _cancel_leftover_working_if_flat(self, reason: str) -> int:
        """Cancel leftover bot-owned Lucid MNQ orders when paper-flat. No flatten."""
        if self._has_open() or not self._lucid_brackets():
            return 0
        try:
            n = int(
                self.feed.rithmic.cancel_working_orders_after_exit(
                    "MNQ", reason=reason, already_flat=True,
                )
                or 0
            )
        except Exception as exc:
            err = f"Leftover Lucid order cancel ({reason}) failed: {redact_secrets(exc)}"
            print(err)
            self._emit_dev(err, already_printed=True)
            return 0
        if n:
            msg = f"Cancelled {n} leftover Lucid working order(s) ({reason})"
            print(msg)
            self._emit_dev(msg, already_printed=True)
        return n

    def _ensure_working_book_clear(self, reason: str) -> bool:
        """True when safe to place a new entry.

        Flat: bot-owned working book must be empty (orphan sweep). Manual
        user SL/TP are ignored. With open bot trades, live SL/TP brackets
        are expected — never cancel them for an add-on.
        """
        if not self._lucid_brackets():
            return True
        open_count = self._open_count()
        rows = self._list_mnq_working(fail_closed=True)
        if rows is None:
            msg = f"Blocking new overnight entry — MNQ working-order scan failed ({reason})"
            print(msg)
            self._emit_dev(msg, already_printed=True)
            return False
        self._log_manual_working_left_alone(rows)
        bot_rows = self._bot_owned_working(rows)
        if open_count > 0:
            # Up to ~3 listed legs per open trade (entry + SL + TP / native bracket).
            cap = open_count * 3
            if len(bot_rows) > cap:
                kinds = self._working_order_kinds(bot_rows)
                msg = (
                    f"{len(bot_rows)} bot-owned MNQ working order(s) with {open_count} "
                    f"open trade(s) (>{cap} expected) kinds={kinds} — add-on blocked ({reason})"
                )
                print(msg)
                self._emit_dev(msg, already_printed=True)
                return False
            return True
        if bot_rows:
            kinds = self._working_order_kinds(bot_rows)
            msg = (
                f"{len(bot_rows)} bot-owned MNQ working order(s) still listed ({reason}) "
                f"kinds={kinds} — sweeping before any new entry"
            )
            print(msg)
            self._emit_dev(msg, already_printed=True)
            self._cancel_leftover_working_if_flat(reason)
            rows = self._list_mnq_working(fail_closed=True)
            if rows is None:
                msg = (
                    f"Blocking new overnight entry — MNQ working-order rescan "
                    f"failed after sweep ({reason})"
                )
                print(msg)
                self._emit_dev(msg, already_printed=True)
                return False
            self._log_manual_working_left_alone(rows)
            bot_rows = self._bot_owned_working(rows)
        if bot_rows:
            kinds = self._working_order_kinds(bot_rows)
            msg = (
                f"Blocking new overnight entry — {len(bot_rows)} bot-owned MNQ "
                f"working order(s) remain kinds={kinds} ({reason})"
            )
            print(msg)
            self._emit_dev(msg, already_printed=True)
            self._emit_throttled(
                "warning",
                "overnight-working-book",
                15.0,
                title="WAITING",
                lines=[
                    "New trade blocked until bot-owned MNQ working book is empty",
                    f"{len(bot_rows)} leftover: {kinds}",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return False
        return True

    def _lucid_order_id(self, pos: Optional[Dict[str, Any]] = None) -> str:
        row = pos or self.position or {}
        return str(row.get("lucid_ticket") or row.get("trade_id") or "")

    def _reset_broker_flat_tracking(self) -> None:
        self._broker_flat_since = None
        self._broker_flat_scans = 0

    def _broker_flat_duration_sec(self) -> float:
        if self._broker_flat_since is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._broker_flat_since).total_seconds()

    def _should_force_sync_ghost(self) -> bool:
        if self._broker_flat_since is None:
            return False
        return (
            self._broker_flat_duration_sec() >= BROKER_FLAT_SYNC_SEC
            or self._broker_flat_scans >= BROKER_FLAT_SYNC_SCANS
        )

    def _apply_lucid_bracket_close(
        self,
        fill_info: Dict[str, Any],
        df: Optional[pd.DataFrame],
        pos: Dict[str, Any],
    ) -> None:
        stop = float(pos.get("stop") or 0)
        target = float(pos.get("target") or 0)
        entry = float(pos.get("entry_price") or 0)
        order_id = self._lucid_order_id(pos)
        exit_price = float(fill_info.get("exit_price") or 0)
        if exit_price <= 0:
            leg_hint = str(fill_info.get("leg") or "")
            if leg_hint == "TP":
                exit_price = target
            elif leg_hint == "SL":
                exit_price = stop
            else:
                exit_price = _last_price(self.feed, df) or entry
        leg = str(fill_info.get("leg") or "")
        if leg in ("SL", "TP"):
            reason = leg
        elif fill_info.get("inferred"):
            reason = str(fill_info.get("sync_reason") or "BROKER_SYNC")
        else:
            reason = "BROKER_BRACKET"
        print(
            f"Lucid bracket exit: {leg or reason} @ {exit_price:.2f} "
            f"(order {order_id}, inferred={bool(fill_info.get('inferred'))})"
        )
        try:
            self.feed.rithmic.acknowledge_flat_position(order_id, "MNQ")
        except Exception as exc:
            self._emit_dev(
                f"acknowledge_flat_position failed: {redact_secrets(exc)}",
            )
        self._reset_broker_flat_tracking()
        self._close_position(exit_price, reason, df, pos=pos)

    def _sync_lucid_bracket_exit(
        self,
        df: Optional[pd.DataFrame],
        pos: Dict[str, Any],
    ) -> bool:
        """Detect Lucid SL/TP fill for one tracked open trade."""
        if not self._lucid_brackets():
            return False
        order_id = self._lucid_order_id(pos)
        if not order_id:
            return False
        rithmic = self.feed.rithmic
        stop = float(pos.get("stop") or 0)
        target = float(pos.get("target") or 0)
        entry = float(pos.get("entry_price") or 0)
        fill_info = rithmic.confirm_bracket_exit_fill(
            order_id,
            "MNQ",
            stop,
            target,
            entry_price=entry,
        )
        if fill_info and fill_info.get("confirmed"):
            self._apply_lucid_bracket_close(fill_info, df, pos)
            return True

        # Ghost-flat sync only when this is the sole tracked trade.
        if self._open_count() != 1:
            return False

        pos_open = rithmic._broker_symbol_has_position("MNQ")
        if pos_open is None:
            return False
        if pos_open:
            self._reset_broker_flat_tracking()
            return False

        now = datetime.now(timezone.utc)
        if self._broker_flat_since is None:
            self._broker_flat_since = now
        self._broker_flat_scans += 1
        force_sync = self._should_force_sync_ghost()
        fill_info = rithmic.confirm_bracket_exit_fill(
            order_id,
            "MNQ",
            stop,
            target,
            force_sync=force_sync,
            flat_duration_sec=self._broker_flat_duration_sec(),
            entry_price=entry,
        )
        if fill_info and fill_info.get("confirmed"):
            self._apply_lucid_bracket_close(fill_info, df, pos)
            return True
        if force_sync:
            self._emit_dev(
                f"Broker flat {self._broker_flat_duration_sec():.0f}s "
                f"but Lucid bracket fill not confirmed ({order_id})",
            )
        return False

    def _reconcile_restored_if_broker_flat(self) -> None:
        """If journal restored open trades but Lucid is already flat, drop them."""
        if not self._has_open() or not self._lucid_brackets():
            return
        try:
            net = self.feed.rithmic.get_symbol_list_positions_net("MNQ")
        except Exception:
            net = None
        if net is None:
            return
        if int(net) != 0:
            return
        print("Drop restored overnight: Lucid MNQ is already flat (no phantom long)")
        self.positions = []
        self._cancel_leftover_working_if_flat("overnight_restore_broker_flat")

    def _maybe_sweep_leftover_working(self) -> None:
        if self._has_open():
            return
        now = time.time()
        if now - float(self._last_leftover_sweep or 0) < LEFTOVER_SWEEP_SEC:
            return
        self._last_leftover_sweep = now
        self._aggressive_remnant_wipe("overnight_periodic_flat")

    def _count_tonight(self) -> int:
        start = pd.Timestamp(fill_window_start_et()).tz_convert("UTC")
        rows = self.journal.closed_trades()
        n, pnl = recipe_session_stats(rows, start)
        self.session_pnl = pnl
        self.cooldown_until = recipe_cooldown_until(rows, start, datetime.now(ET))
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
        if self.positions:
            sides = [str(p.get("side") or "open").upper() for p in self.positions]
            open_label = sides[0] if len(sides) == 1 else f"{len(sides)} open"
            for pos in self.positions:
                open_pos.append({
                    "symbol": "MNQ",
                    "direction": str(pos.get("side") or "").upper(),
                    "entry": pos.get("entry_price"),
                    "sl": pos.get("stop"),
                    "tp": pos.get("target"),
                    "atr_stop_pts": pos.get("atr_stop_pts"),
                    "flatten_et": pos.get("flatten_et") or "",
                    "size": _pos_qty(pos),
                    "cue": pos.get("cue"),
                    "zone_name": pos.get("zone_name"),
                    "trade_id": pos.get("trade_id") or pos.get("lucid_ticket"),
                    "entry_confidence_pct": pos.get("entry_confidence_pct"),
                    "entry_confidence": pos.get("entry_confidence"),
                })
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
                "overnight_max_open": overnight_max_open(),
                "overnight_min_confidence": overnight_min_confidence(),
                "overnight_addon_confidence_min": overnight_addon_confidence_min(),
                "live_trades_today": int(self.fills_tonight) + self._open_count(),
                "fills_tonight": int(self.fills_tonight),
                "session_recipe_pnl": round(float(self.session_pnl), 2),
                "overnight_fill_limit": overnight_fill_limit(self.fills_tonight, self.session_pnl),
                "cooldown_until_et": (
                    self.cooldown_until.isoformat()
                    if self.cooldown_until is not None else None
                ),
                "last_price": px,
                "last_quote_ts_et": et,
                "quote_age_seconds": age,
                "quote_source": (
                    "waiting_ticker"
                    if self.feed.rithmic is not None and px is None
                    else self.feed.source
                ),
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

    def _close_position(
        self,
        exit_price: float,
        reason: str,
        df: Optional[pd.DataFrame],
        *,
        pos: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pos is None:
            pos = self.position
        if not pos:
            return
        trade_id = str(pos.get("trade_id") or "")
        now = datetime.now(timezone.utc)
        qty = _pos_qty(pos)
        raw_entry = pos.get("entry_time") or pos.get("timestamp") or now
        try:
            entry_dt = raw_entry if isinstance(raw_entry, datetime) else pd.Timestamp(raw_entry).to_pydatetime()
            if getattr(entry_dt, "tzinfo", None) is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        except Exception:
            entry_dt = now
        pts, usd = paper_pnl_usd(pos["side"], float(pos["entry_price"]), float(exit_price), qty)
        rec = {
            "trade_id": pos["trade_id"],
            "symbol": "MNQ",
            "side": pos["side"],
            "direction": pos["side"],
            "qty": qty,
            "entry_price": pos["entry_price"],
            "exit_price": round(float(exit_price), 2),
            "stop": pos.get("stop"),
            "target": pos.get("target"),
            "pts": pts,
            "pnl_usd": usd,
            "hold_minutes": round((now - entry_dt).total_seconds() / 60.0, 2),
            "exit_reason": reason,
            "entry_time": entry_dt.isoformat(),
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
            rec, df, entry_dt, now, pos["side"], float(pos["entry_price"]),
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
        if self._lucid_brackets():
            if str(reason) == "shutdown" and self._open_count() <= 1:
                try:
                    self.feed.rithmic.flatten_symbol("MNQ", max_attempts=8, verify_delay=1.5)
                except Exception as exc:
                    err = f"PAPER_RITHMIC_BRACKETS flatten failed: {redact_secrets(exc)}"
                    print(err)
                    self._emit_dev(err, already_printed=True)
            try:
                cancelled = self.feed.rithmic.cancel_working_orders_after_exit(
                    "MNQ",
                    str(pos.get("lucid_ticket") or ""),
                    reason=f"overnight_{str(reason).lower()}_fill",
                    already_flat=True,
                )
                if cancelled:
                    self._emit_dev(
                        f"Cancelled {cancelled} leftover Lucid order(s) after {reason}",
                        already_printed=False,
                    )
            except Exception as exc:
                err = (
                    f"PAPER_RITHMIC_BRACKETS leftover cancel after {reason} failed: "
                    f"{redact_secrets(exc)}"
                )
                print(err)
                self._emit_dev(err, already_printed=True)
            self._last_leftover_sweep = 0.0
        if self._lucid_brackets():
            try:
                self.feed.maybe_resubscribe_ticker()
            except Exception:
                pass
        title = "PROFITABLE TRADE" if usd >= 0 else "LOSING TRADE"
        emoji = exit_emoji(reason)
        print(
            format_exit_line(
                pos["side"],
                reason,
                float(exit_price),
                float(pos["entry_price"]),
            ),
            flush=True,
        )
        self._emit(
            "profit" if usd >= 0 else "loss",
            emoji=emoji,
            title=title,
            symbol="MNQ",
            direction=pos["side"],
            entry=pos.get("entry_price"),
            exit=round(float(exit_price), 2),
            pnl=usd,
            qty=qty,
            paper=True,
            session=SESSION_NAME,
            reason=reason,
            console=False,
        )
        self.positions = [
            p for p in self.positions
            if str(p.get("trade_id") or "") != trade_id
        ]
        if not self.positions:
            self._reset_broker_flat_tracking()
        if self._lucid_brackets() and not self._has_open():
            self._aggressive_remnant_wipe(f"overnight_{str(reason).lower()}_fill")
            remaining = self._list_mnq_working()
            bot_left = self._bot_owned_working(remaining)
            if bot_left:
                kinds = self._working_order_kinds(bot_left)
                err = (
                    f"{len(bot_left)} leftover bot-owned Lucid order(s) after {reason} "
                    f"kinds={kinds} — next entry blocked until bot working book is empty"
                )
                print(err)
                self._emit_dev(err, already_printed=True)
            elif remaining:
                self._log_manual_working_left_alone(remaining)

    def _lucid_brackets(self) -> bool:
        return (
            bool(getattr(self.feed, "brackets", False))
            and self.feed.rithmic is not None
            and not isinstance(self.feed.rithmic, QuotesOnlyBroker)
        )

    def _block_max_open_entry(self, reason: str) -> None:
        cap = overnight_max_open()
        open_count = self._open_count()
        msg = f"Max open {cap} reached ({open_count} open) — {reason}"
        print(msg)
        self._emit_throttled(
            "skip",
            "overnight-max-open",
            MONITOR_INTERVAL_SEC,
            title="TRADE SKIPPED",
            lines=[f"Max open {cap} reached — no new entry"],
            symbol="MNQ",
            paper=True,
            session=SESSION_NAME,
        )

    def _open_paper_position(self, sig: Dict[str, Any]) -> None:
        if overnight_open_slots_full(self._open_count()):
            self._block_max_open_entry("entry blocked before place")
            return
        if getattr(self, "_entry_placed_this_loop", False):
            self._block_max_open_entry("one entry per loop")
            return
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
        sl_attached = False
        tp_attached = False
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
            if not self._ensure_working_book_clear("overnight_pre_entry"):
                return
            inner = self.feed.rithmic
            side = "buy" if str(sig.get("side") or "").lower() == "long" else "sell"
            try:
                result = inner.place_order(
                    symbol="MNQ",
                    order_type=side,
                    size=_pos_qty(sig),
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
            fill = float(
                result.get("entry_price") or result.get("price_ref") or sig.get("entry_price") or 0
            )
            if fill > 0:
                sig["entry_price"] = fill
            if result.get("stop_loss"):
                sig["stop"] = float(result["stop_loss"])
            if result.get("take_profit"):
                sig["target"] = float(result["take_profit"])
            sig["lucid_ticket"] = lucid_ticket
            sl_attached = bool(
                result.get("native_stop_attached") or result.get("protective_sl_order_id")
            )
            tp_attached = bool(
                result.get("native_target_attached") or result.get("protective_tp_order_id")
            )
            if overnight_open_slots_full(self._open_count()):
                msg = (
                    f"Max open {overnight_max_open()} reached after Lucid fill "
                    f"({self._open_count()} tracked) — flattening orphan ticket={lucid_ticket}"
                )
                print(msg)
                self._emit_dev(msg, already_printed=True)
                try:
                    inner.flatten_symbol("MNQ", max_attempts=8, verify_delay=1.5)
                except Exception as exc:
                    err = f"flatten orphan add-on failed: {redact_secrets(exc)}"
                    print(err)
                    self._emit_dev(err, already_printed=True)
                self._emit(
                    "skip",
                    title="TRADE SKIPPED",
                    lines=[f"Max open {overnight_max_open()} reached — orphan flattened"],
                    symbol="MNQ",
                    paper=True,
                    session=SESSION_NAME,
                )
                return
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
        if str(sig.get("cue") or "") != "test_fill":
            self.used_cues.add(str(sig.get("cue") or ""))
            if overnight_cap_reached(int(self.fills_tonight) + 1, self.session_pnl):
                self.filled_session = self._session_key
        increment_zone_touch(self.zones, str(sig.get("zone_name") or ""))
        write_zones(self.zones)
        self.positions.append({
            **sig,
            "trade_id": tid,
            "lucid_ticket": lucid_ticket,
            "entry_time": datetime.now(timezone.utc),
            "entry_ts_et": datetime.now(ET).isoformat(),
            "mae_pts": 0.0,
            "mfe_pts": 0.0,
        })
        self._last_entry_fingerprint = overnight_entry_fingerprint(sig)
        self._last_entry_at = time.time()
        self._entry_placed_this_loop = True
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
        try:
            banner = format_trade_placed_banner(
                sig,
                sl_ok=bracket_ok_mark(sl_attached) if want_lucid else "",
                tp_ok=bracket_ok_mark(tp_attached) if want_lucid else "",
            )
            try:
                print(banner, flush=True)
            except UnicodeEncodeError:
                print(banner.encode("ascii", errors="replace").decode("ascii"), flush=True)
        except Exception:
            pct = sig.get("entry_confidence_pct")
            side_u = str(sig.get("side") or "trade").upper()
            print(
                f"TRADE PLACED — {side_u} {_pos_qty(sig)} MNQ @ {sig.get('entry_price')} "
                f"| Win confidence: {pct}%",
                flush=True,
            )
        self._emit(
            "entry",
            symbol="MNQ",
            direction=sig["side"],
            reason=sig.get("why"),
            entry=sig["entry_price"],
            stop=sig.get("stop"),
            atr_stop_pts=sig.get("atr_stop_pts"),
            flatten_et=sig.get("flatten_et") or "",
            qty=_pos_qty(sig),
            session=SESSION_NAME,
            paper=True,
            cue=sig.get("cue"),
            dest=dest,
            entry_confidence_pct=sig.get("entry_confidence_pct"),
            entry_confidence=sig.get("entry_confidence"),
            console=False,
        )

    def _ticker_last(self) -> Optional[float]:
        return _rithmic_last(self.feed)

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
        open_count = self._open_count()
        if overnight_open_slots_full(open_count):
            cap = overnight_max_open()
            self._emit_throttled(
                "idle",
                "overnight-max-open",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Max open {cap} reached ({open_count} open) — no new entry",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        already = overnight_cap_reached(self.fills_tonight, self.session_pnl)
        cooling = in_overnight_cooldown(self.cooldown_until, now)
        if cooling:
            line = explain_overnight_wait(
                now=now, fills_tonight=int(self.fills_tonight),
                session_pnl=self.session_pnl, cooldown_until=self.cooldown_until,
            )
            self._emit_throttled(
                "idle",
                "overnight-idle",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[line],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        px = _last_price(self.feed, df)
        if px is None and self.feed.rithmic is not None:
            self._emit_throttled(
                "idle",
                "overnight-idle",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=["Waiting for live ticker last (not leftover RTH CSV)"],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        sig = evaluate_overnight_cue(
            df, now, already_filled=already, last_price=px,
        )
        if not sig:
            if already:
                line = explain_overnight_wait(
                    already_filled=True, now=now,
                    fills_tonight=int(self.fills_tonight), session_pnl=self.session_pnl,
                    cooldown_until=self.cooldown_until,
                )
            elif not allow_new_overnight_entry(now):
                line = "Overnight session closed — no new entries"
            else:
                rng = settled_overnight_range(overnight_completed_bars(df, now), now)
                line = explain_overnight_wait(
                    bars=int(rng.get("bars") or 0),
                    onh=rng.get("onh"),
                    onl=rng.get("onl"),
                    onh_settled=bool(rng.get("onh_settled")),
                    onl_settled=bool(rng.get("onl_settled")),
                    onh_age=rng.get("onh_age_bars"),
                    onl_age=rng.get("onl_age_bars"),
                    last=_last_price(self.feed, df),
                    now=now,
                    fills_tonight=int(self.fills_tonight),
                    session_pnl=self.session_pnl,
                    cooldown_until=self.cooldown_until,
                )
            self._emit_throttled(
                "idle",
                "overnight-idle",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
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
            tp_r = overnight_tp_r()
            if sig["side"] == "short":
                sig["stop"] = round(sig["entry_price"] + sl, 2)
                sig["target"] = round(sig["entry_price"] - sl * tp_r, 2)
            else:
                sig["stop"] = round(sig["entry_price"] - sl, 2)
                sig["target"] = round(sig["entry_price"] + sl * tp_r, 2)
        px = _last_price(self.feed, df) or float(sig["entry_price"])
        on_bars = overnight_completed_bars(df, now)
        bar = last_completed_bar(on_bars)
        close = float(bar["close"]) if bar is not None else float(px)
        level = float(sig.get("zone_price") or 0)
        t15, t60 = overnight_htf_sides(df)
        macd_t = macd_trend_15m_on_1m(df) if overnight_macd_enabled() else 0
        if should_defer_entry_for_stack(
            str(sig.get("side") or "long"), float(px), close, level, t15, t60,
        ):
            self._emit_throttled(
                "idle",
                "overnight-stack-wait",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Stack setup — last+HTF agree, waiting 1m close through {level:.2f}",
                    f"15m EMA trend={t15} · 60m EMA trend={t60} · MACD={macd_t}",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        gate = overnight_stack_gate(
            str(sig.get("side") or "long"), float(px), close, level, t15, t60, macd_t,
        )
        sig["qty"] = int(gate.get("qty") or QTY)
        sig["macd_trend"] = int(macd_t)
        suffix = overnight_stack_why_suffix(gate)
        if suffix and suffix not in str(sig.get("why") or ""):
            sig["why"] = str(sig.get("why") or "") + suffix
        hist = recipe_historical_win_pct(
            self.journal.closed_trades(), str(sig.get("cue") or ""),
        )
        conf = entry_confidence_pct(str(sig.get("side") or "long"), gate, **hist)
        sig["entry_confidence_pct"] = conf["pct"]
        sig["entry_confidence"] = conf
        if not allow_overnight_macd_entry(str(sig.get("side") or "long"), macd_t):
            side_u = str(sig.get("side") or "").upper()
            self._emit_throttled(
                "idle",
                "overnight-macd-opposed",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"MACD opposed to {side_u} break — skip "
                    f"(OVERNIGHT_MACD_REQUIRE, macd={macd_t})",
                    f"{conf.get('summary') or 'break'}",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        if not allow_overnight_entry_confidence(conf["pct"]):
            need = overnight_min_confidence()
            summary = conf.get("summary") or "break"
            self._emit_throttled(
                "idle",
                "overnight-min-confidence",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Confidence {conf['pct']}% < min {need}% "
                    f"(OVERNIGHT_MIN_CONFIDENCE) — skip entry",
                    f"{summary}",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        if open_count > 0 and overnight_same_cue_open(self.positions, sig.get("cue")):
            self._emit_throttled(
                "idle",
                "overnight-same-cue-addon",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Same cue {sig.get('cue')} already open — no duplicate add-on",
                    f"Max open {overnight_max_open()} · confidence {conf['pct']}% ignored",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        if overnight_recent_entry_blocked(
            sig,
            last_fingerprint=self._last_entry_fingerprint,
            last_entry_at=self._last_entry_at,
        ):
            wait = overnight_entry_dedupe_sec()
            self._emit_throttled(
                "idle",
                "overnight-entry-dedupe",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Duplicate {sig.get('cue')} @ {sig.get('zone_price')} — "
                    f"dedupe {wait}s",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        if open_count > 0 and not allow_overnight_addon(open_count, conf["pct"]):
            need = overnight_addon_confidence_min()
            self._emit_throttled(
                "idle",
                "overnight-addon-confidence",
                MONITOR_INTERVAL_SEC,
                title="WAITING",
                lines=[
                    f"Trade open — add-on needs confidence >= {need}%",
                    f"Signal {conf['pct']}% ({conf.get('summary') or 'break'})",
                ],
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )
            return
        self._open_paper_position(sig)

    def _manage_open(self, df: pd.DataFrame, now) -> None:
        if not self.positions:
            return
        closed_any = False
        for pos in list(self.positions):
            if self._lucid_brackets() and self._sync_lucid_bracket_exit(df, pos):
                closed_any = True
                continue
            px = _rithmic_last(self.feed)
            if px is None:
                if not closed_any:
                    self._emit_throttled(
                        "warning",
                        "stale-quote-hold",
                        60.0,
                        title="WAITING",
                        lines=[
                            "Open trade held — no live ticker last",
                            "Will not SL/TP on leftover RTH CSV",
                        ],
                        symbol="MNQ",
                        paper=True,
                        session=SESSION_NAME,
                    )
                continue
            high = px
            low = px
            if df is not None and not df.empty:
                try:
                    ts = pd.to_datetime(df.iloc[-1]["datetime"], utc=True)
                    age = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds()
                except Exception:
                    age = 10_000.0
                if age < 120:
                    high = max(float(df.iloc[-1]["high"]), px)
                    low = min(float(df.iloc[-1]["low"]), px)
            pos["mae_pts"], pos["mfe_pts"] = update_mae_mfe(
                pos["side"],
                float(pos["entry_price"]),
                high, low,
                float(pos.get("mae_pts") or 0),
                float(pos.get("mfe_pts") or 0),
            )
            hit = check_paper_exit(pos, high=high, low=low, last=px, now=now)
            if hit:
                self._close_position(float(hit["exit_price"]), str(hit["reason"]), df, pos=pos)
                closed_any = True

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
        print("  PAPER RESEARCH — last 20m high/low break; keeps running through RTH once started")
        print("  Lucid TEST/sim orders — not funded live. Do not also run start_mnq_live.bat.")
        print("  Quotes: Rithmic ticker. History plant skipped (1011).")
        print("  PAPER_RITHMIC_BRACKETS — fills must appear on Lucid (no local fake fill)")
        print("  Cue: last-through last 20 1m high/low (NO fade)")
        print("  Qty 1, or qty 2 if last+close through AND 15m+60m EMA8/21 agree")
        cap = paper_max_fills_base()
        if cap is None:
            print("  No ET-day fill cap (PAPER_MAX_FILLS=0). Set PAPER_MAX_FILLS>0 to cap.")
        else:
            print(
                f"  {cap} fills per ET calendar date (+5 if those {cap} net green). "
                "Resets 00:00 ET."
            )
        print(
            f"  5 losers in a row → 30m cooldown. ATR×1.5 stop (12–40) · "
            f"{overnight_tp_text()} · no 09:25 flatten"
        )
        if overnight_macd_enabled():
            req = "skip if opposed" if overnight_macd_require() else "score only"
            print(f"  MACD 15m filter ON ({req})")
        override_msg = overnight_cooldown_override_message()
        if override_msg:
            print(f"  *** {override_msg} ***")
        print(
            f"  Confidence floor: "
            f"{'OFF' if overnight_min_confidence() <= 0 else f'>={overnight_min_confidence()}%'} "
            f"(OVERNIGHT_MIN_CONFIDENCE); add-on >= {overnight_addon_confidence_min()}% "
            f"(OVERNIGHT_ADDON_CONFIDENCE_MIN); max {overnight_max_open()} open"
        )
        print("  CME closed Fri 17:00–Sun 18:00 ET (halt 17:00–18:00 Mon–Thu). Ctrl+C to flatten.")
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
        self._session_key = fill_window_start_et(datetime.now(ET)).date().isoformat()
        self.fills_tonight = self._count_tonight()
        deadline = None
        if duration_minutes and duration_minutes > 0:
            deadline = time.time() + duration_minutes * 60
        self._restore_open_position()
        self._reconcile_restored_if_broker_flat()
        self._cancel_leftover_working_if_flat("overnight_startup_flat")
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
        if want_test and not self._has_open():
            self._place_test_fill()
        try:
            while True:
                if deadline and time.time() >= deadline:
                    break
                now = datetime.now(ET)
                self._entry_placed_this_loop = False
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
                    if self._has_open():
                        self._manage_open(df if df is not None else pd.DataFrame(), now)
                    else:
                        self._maybe_sweep_leftover_working()
                    if allow_new_overnight_entry(now) and df is not None:
                        self._maybe_enter(df, now)
                    elif not is_overnight_research_hours(now):
                        blocked = overnight_entry_blocked_reason(now)
                        if day_bot_owns_lucid(now):
                            note = (
                                "Day ema15 bot owns Lucid — this runner will not enter "
                                "(one Lucid session)"
                            )
                        else:
                            note = blocked or "CME closed — overnight paper waiting"
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
                        fill_n = int(self.fills_tonight) + self._open_count()
                        limit = overnight_fill_limit(self.fills_tonight, self.session_pnl)
                        if limit is None:
                            fill_txt = f"{fill_n} paper fills this ET calendar date (no cap)"
                        else:
                            fill_txt = f"{fill_n} of {limit} paper fills this ET calendar date"
                        open_note = ""
                        if self._open_count() == 1:
                            open_note = f" · qty {_pos_qty(self.position)}"
                        elif self._open_count() > 1:
                            open_note = f" · {self._open_count()} open"
                        note = (
                            f"PAPER OVERNIGHT RESEARCH · {src} quotes · {fill_txt}"
                            + open_note
                            + (
                                f" · cooling until {self.cooldown_until.strftime('%H:%M')} ET"
                                if in_overnight_cooldown(self.cooldown_until) else ""
                            )
                        )
                    self._heartbeat(note, df)
                    self._print_alive(df)
                    time.sleep(5 if self._has_open() else 10)
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
            if self._has_open():
                df = self.feed.get_1m()
                px = _last_price(self.feed, df)
                for pos in list(self.positions):
                    exit_px = px or float(pos["entry_price"])
                    self._close_position(exit_px, "shutdown", df, pos=pos)
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
                    "live_trades_today": int(self.fills_tonight) + self._open_count(),
                    "zones": (self.zones or {}).get("zones") or [],
                    "last_paper_fill": self.last_fill,
                    "journal_path": self.journal.path,
                })
            except Exception:
                pass
            self.feed.shutdown()

    def _print_alive(self, df: Optional[pd.DataFrame] = None) -> None:
        now_ts = time.time()
        idle_sec = float(os.getenv("OVERNIGHT_HEARTBEAT_SEC", "120"))
        if not self.positions and now_ts - self._last_alive_print < idle_sec:
            return
        self._last_alive_print = now_ts
        px = _last_price(self.feed, df)
        ts = datetime.now(ET).strftime("%H:%M ET")
        open_cards: List[Dict[str, Any]] = []
        for pos in self.positions:
            open_cards.append({
                "side": pos.get("side"),
                "entry_price": pos.get("entry_price"),
                "stop": pos.get("stop"),
                "target": pos.get("target"),
            })
        note = overnight_tp_text().replace(" TP", " target")
        if not self.positions:
            note = f"flat · {note}"
        print(format_heartbeat_line(ts, px, open_cards, note=note), flush=True)
        if not self.positions:
            return
        for pos in self.positions:
            sl_s = "—"
            tp_label = overnight_tp_text()
            tp_s = tp_label
            try:
                sl_s = f"{float(pos.get('stop')):.2f}"
            except (TypeError, ValueError):
                pass
            try:
                tp_s = f"{float(pos.get('target')):.2f}"
            except (TypeError, ValueError):
                pass
            px_s = f"{px:.2f}" if px else "waiting ticker"
            conf = pos.get("entry_confidence") or {}
            tid = pos.get("lucid_ticket") or pos.get("trade_id") or ""
            side = str(pos.get("side") or "trade").capitalize()
            r_tag = tp_label.replace(" TP", "")
            monitor_lines = [
                f"{trade_emoji(str(pos.get('side') or ''))} {side} still open · MNQ {px_s}",
                f"Stop {sl_s} · Target {tp_s} ({r_tag})",
            ]
            conf_line = format_entry_confidence_line(conf)
            if conf_line and "Confidence: 0%" not in conf_line:
                monitor_lines.append(conf_line.replace("Confidence:", "Win confidence:"))
            self._emit_throttled(
                "monitoring",
                f"open-monitor-{tid}",
                MONITOR_INTERVAL_SEC,
                title="MONITORING",
                lines=monitor_lines,
                symbol="MNQ",
                paper=True,
                session=SESSION_NAME,
            )


def run_overnight_paper(*, duration_minutes: int = 0, test_fill: bool = False) -> None:
    OvernightPaperEngine().run(duration_minutes=duration_minutes, test_fill=test_fill)
