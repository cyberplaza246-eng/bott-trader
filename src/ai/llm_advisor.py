"""
LLM trade advisor — DeepSeek / OpenAI-compatible API, Gemini via native generateContent.

Used as a context filter on top of rule-based MTF signals.
Does NOT replace risk management or entry rules.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.ai.mnq_context import build_mnq_context, compute_setup_score
from src.ai.economic_calendar import EconomicCalendar
from src.ai.action_log import OVERNIGHT_FLATTEN_ET, OVERNIGHT_TP_TEXT, is_overnight_rec, stop_pts_of
from src.strategy.mnq_15m_ema_eod import (
    ET,
    FLATTEN_ET,
    LOCKED_RULES_PARAGRAPH,
    SEP_MIN,
    TP_PTS,
    capture_market_snapshot,
    high_confidence,
    load_1m_seed_csv,
    next_window_info,
    window_requires_quality,
)
from src.utils.logger import bot_logger

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_OPENAI_COMPAT_SUFFIX = "/openai/chat/completions"
GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)
GEMINI_RETIRED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
}
MNQ_1M_CSV = "data/MNQ_1m.csv"
SNAPSHOT_STALE_SEC = 300
QUOTE_STALE_ASK_SEC = 900  # 15 min — Gemini/desk first line
DIRECTION_ASK_SYSTEM = (
    "You are a second-opinion advisor for locked MNQ ema15_eod. The recipe is still the boss. "
    "You are ADVISORY only: never place orders, never change stops, windows, or mnq_profit_config, "
    "and never say 'I bought', 'I sold', or 'order sent'.\n"
    "Python already computed the only legal numbers in context.market.levels. "
    "Copy those numbers. Do not invent prices. Do not answer with window/EMA filter prose only.\n"
    "First sentence MUST be exactly levels.first_line. If quote_age_sec > 900, that line is "
    "'WAIT — quote is X min old — last MNQ <price> as of <timestamp>'. "
    "Do not lead with historical trade counts or profit factor.\n"
    "Then EXIT: that stop, flatten 15:50 ET, do not trail. "
    "Label the answer 'second opinion — recipe is still the boss.'"
)
OVERNIGHT_ASK_SYSTEM = (
    "You are a second-opinion advisor for PAPER overnight MNQ research. "
    "You are ADVISORY only: never place orders, never change stops, and never say an order was sent.\n"
    "Python already computed the only legal numbers in context.market.levels from the LIVE ticker. "
    "Copy those numbers. Do not invent prices. Do not use a stale MNQ_1m.csv or Databento file print.\n"
    "First sentence MUST be exactly levels.first_line (open paper position, live last, stop, "
    "1R TP, Flatten 09:25 ET).\n"
    "Do not lead with RTH WAIT, next window 09:35, flatten 15:50, or a file price from a prior date.\n"
    "Label the answer 'second opinion — recipe is still the boss.'"
)
_CSV_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "df": None}


def invalidate_1m_cache() -> None:
    _CSV_CACHE["path"] = None
    _CSV_CACHE["mtime"] = None
    _CSV_CACHE["df"] = None

# Locked ema15_eod official recipe (docs/PROFITABLE_LIVE.md) — BACKTEST, not live.
_LOCKED_BACKTEST_DEFAULT = {
    "label": "Official (6 windows, noon+ quality, 2-lot)",
    "is_pf": 1.41,
    "is_pnl_usd": 5643.0,
    "oos_pf": 1.59,
    "oos_pnl_usd": 5371.0,
    "max_dd_usd": 2084.0,
    "source": "BACKTEST data/MNQ_1m.csv (not live P&L)",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sanitize_secret_text(text: str) -> str:
    raw = text or ""
    raw = re.sub(r"([?&]key=)[^&\s\"']+", r"\1[REDACTED]", raw, flags=re.I)
    raw = re.sub(r"(AIza[0-9A-Za-z_-]{10,}|AQ\.[0-9A-Za-z_-]{10,})", "[REDACTED]", raw)
    raw = re.sub(r"(Bearer )\S+", r"\1[REDACTED]", raw, flags=re.I)
    return raw


def _is_profit_question(question: str) -> bool:
    q = (question or "").lower()
    keys = ("profit", "profitable", "p&l", "pnl", "make money", "winning")
    return any(k in q for k in keys)


def _is_direction_question(question: str) -> bool:
    q = (question or "").lower()
    keys = (
        "buy", "sell", "enter", "entry", "exit", "flatten", "price",
        "go long", "go short", "long or short", "buy or sell",
        "should i long", "should i short", "where should", "at what",
    )
    return any(k in q for k in keys)


def _side_txt(v: Any) -> str:
    try:
        n = int(v or 0)
    except (TypeError, ValueError):
        return "none"
    if n > 0:
        return "long"
    if n < 0:
        return "short"
    return "none"


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_recipe_levels(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic ema15_eod entry/stop from a snapshot. Gemini must not invent these."""
    px = _as_float(packet.get("price_1m"))
    sl = _as_float(packet.get("atr_stop_pts"))
    t15 = int(packet.get("trend_15m") or 0)
    daily = int(packet.get("daily_permission") or 0)
    t60 = int(packet.get("trend_60m") or 0)
    sep_f = _as_float(packet.get("sep15"))
    sep_v = float("nan") if sep_f is None else sep_f
    in_window = bool(packet.get("in_window"))
    win_idx = packet.get("window_index")
    try:
        win_idx_i = None if win_idx is None else int(win_idx)
    except (TypeError, ValueError):
        win_idx_i = None
    session = packet.get("session")
    stale = bool(packet.get("price_stale"))
    refresh_ok = packet.get("refresh_ok")
    if refresh_ok is False:
        px = None
    ts = packet.get("last_bar_ts_et") or packet.get("now_et") or "unknown time"
    age_sec = packet.get("last_bar_age_sec")
    if age_sec is None:
        age_sec = packet.get("quote_age_sec")
    nxt = packet.get("next_window_name") or "09:35"
    missing: List[str] = []
    if stale:
        missing.append("quote is not a live print (file timestamp)")
    if session not in ("RTH",):
        missing.append(f"session is {session or 'closed'} (RTH only)")
    if not in_window:
        missing.append(f"not in a locked window (next {nxt} ET)")
    aligned = 0
    if t15 > 0 and daily > 0 and t60 > 0:
        aligned = 1
    elif t15 < 0 and daily < 0 and t60 < 0:
        aligned = -1
    else:
        last = f"{px:.2f}" if px is not None else "n/a"
        ema8 = packet.get("ema15_fast")
        ema21 = packet.get("ema15_slow")
        ema_bit = ""
        if ema8 is not None and ema21 is not None:
            ema_bit = f" (EMA8 {ema8} vs EMA21 {ema21})"
        if t15 < 0:
            missing.append(f"15m still bearish; last {last}; need EMA8>EMA21{ema_bit}")
        elif t15 > 0 and daily != t15:
            missing.append(
                f"15m is long but daily permission is {_side_txt(daily)} "
                f"(prior RTH {packet.get('prior_rth_close')} vs EMA20 {packet.get('daily_ema20')})"
            )
        elif t15 > 0:
            missing.append(f"15m long; last {last}; daily filter not aligned")
        elif t15 == 0:
            missing.append(f"15m flat; last {last}; need EMA8>EMA21 or EMA8<EMA21{ema_bit}")
        if t15 < 0 and daily != t15:
            missing.append(
                f"daily permission is {_side_txt(daily)} "
                f"(prior RTH {packet.get('prior_rth_close')} vs EMA20 {packet.get('daily_ema20')})"
            )
        if t15 != 0 and t60 != t15:
            missing.append(
                f"60m is {_side_txt(t60)} (need 60m EMA8/21 same direction as 15m)"
            )
    needs_q = window_requires_quality(win_idx_i)
    confident = high_confidence(aligned or t15, t60, sep_v, SEP_MIN)
    if in_window and needs_q and not confident:
        missing.append(
            f"noon+ window still needs sep>= {SEP_MIN} "
            f"(60m {_side_txt(t60)}, sep {packet.get('sep15')})"
        )
    valid_now = (
        px is not None
        and sl is not None
        and aligned != 0
        and in_window
        and session == "RTH"
        and not stale
        and (not needs_q or confident)
    )
    hypo = aligned if aligned else t15
    entry = stop = tp = None
    side_name = None
    if px is not None and sl is not None and hypo != 0:
        side_name = "LONG" if hypo > 0 else "SHORT"
        entry = round(px, 2)
        stop = round(px - sl, 2) if hypo > 0 else round(px + sl, 2)
        tp = round(px + float(TP_PTS), 2) if hypo > 0 else round(px - float(TP_PTS), 2)
    elif px is not None and sl is not None:
        side_name = "LONG"
        entry = round(px, 2)
        stop = round(px - sl, 2)
        tp = round(px + float(TP_PTS), 2)
    callout = "WAIT"
    if valid_now and aligned > 0:
        callout = "BUY"
    elif valid_now and aligned < 0:
        callout = "SELL"
    source = "live" if not stale else "file, not live"
    levels: Dict[str, Any] = {
        "callout": callout,
        "valid_now": valid_now,
        "waiting": not valid_now,
        "missing": missing,
        "last_price": None if px is None else round(px, 2),
        "last_bar_ts_et": ts,
        "quote_age_sec": None if age_sec is None else int(round(float(age_sec))),
        "quote_age_min": None if age_sec is None else round(float(age_sec) / 60.0, 1),
        "refresh_ok": True if refresh_ok is None else bool(refresh_ok),
        "refresh_error": packet.get("refresh_error"),
        "source": source,
        "next_window": nxt,
        "in_window": in_window,
        "stop_pts": None if sl is None else round(sl, 2),
        "atr15": packet.get("atr15"),
        "side": side_name,
        "entry": entry,
        "stop": stop,
        "tp_cap": tp,
        "flatten_et": FLATTEN_ET,
        "ema15_fast": packet.get("ema15_fast"),
        "ema15_slow": packet.get("ema15_slow"),
        "daily_permission_txt": packet.get("daily_permission_txt"),
    }
    levels["if_valid_line"] = _if_valid_line(levels)
    levels["first_line"] = format_levels_line(levels)
    return levels


def _if_valid_line(levels: Dict[str, Any]) -> str:
    entry = levels.get("entry")
    stop = levels.get("stop")
    sl = levels.get("stop_pts")
    flatten = levels.get("flatten_et") or FLATTEN_ET
    if entry is None or stop is None or sl is None:
        return ""
    side = str(levels.get("side") or "LONG")
    act = "BUY" if side == "LONG" else "SELL"
    line = (
        f"If still aligned {side.lower()} then: {act} ~{float(entry):.2f}, "
        f"stop ~{float(stop):.2f} (ATR {float(sl):.0f} pts), flatten {flatten} ET"
    )
    tp = levels.get("tp_cap")
    if tp is not None:
        line += f", TP cap {float(tp):.0f}"
    return line


def format_levels_line(levels: Dict[str, Any]) -> str:
    """First sentence with real MNQ numbers. Never filter-prose only."""
    px = levels.get("last_price")
    ts = levels.get("last_bar_ts_et") or "unknown time"
    src = levels.get("source") or "file, not live"
    nxt = levels.get("next_window") or "09:35"
    callout = str(levels.get("callout") or "WAIT").upper()
    if levels.get("refresh_ok") is False:
        err = levels.get("refresh_error") or "market download failed"
        return f"WAIT — {err}. Last file print is not current."
    age = levels.get("quote_age_sec")
    last_bit = (
        f"last MNQ {float(px):.2f} as of {ts} ({src})"
        if px is not None
        else "last MNQ unavailable"
    )
    if age is not None and float(age) > QUOTE_STALE_ASK_SEC:
        mins = max(1, int(round(float(age) / 60.0)))
        stale_bit = (
            f"last MNQ {float(px):.2f} as of {ts}"
            if px is not None
            else "last MNQ unavailable"
        )
        return f"WAIT — quote is {mins} min old — {stale_bit}. Next window {nxt} ET."
    if_valid = levels.get("if_valid_line") or _if_valid_line(levels)
    if callout in ("BUY", "SELL") and levels.get("valid_now") and levels.get("entry") is not None:
        return (
            f"{callout} — {last_bit}. {levels.get('side')} ~{float(levels['entry']):.2f}, "
            f"stop ~{float(levels['stop']):.2f} (ATR {float(levels['stop_pts']):.0f} pts), "
            f"flatten {levels.get('flatten_et') or FLATTEN_ET} ET."
        )
    miss = ""
    for m in levels.get("missing") or []:
        if str(m).startswith("quote is not"):
            continue
        if m.startswith("15m") or m.startswith("daily") or "EMA8" in m:
            miss = f" {m}."
            break
    if not miss:
        for m in levels.get("missing") or []:
            if str(m).startswith("quote is not"):
                continue
            miss = f" Missing: {m}."
            break
    hypo = f" {if_valid}." if if_valid else ""
    return f"WAIT — {last_bit}. Next window {nxt} ET.{hypo}{miss}"


def ensure_levels_in_answer(text: str, levels: Optional[Dict[str, Any]]) -> str:
    """Guarantee the Python last-price (and first_line) appear in Gemini's reply."""
    levels = levels or {}
    first = str(levels.get("first_line") or "").strip()
    body = (text or "").strip()
    px = levels.get("last_price")
    if first and (
        not body
        or body.lower().startswith("gemini/llm is off")
        or body.lower().startswith("gemini unavailable")
    ):
        return f"{first}\n\n{body}".strip() if body else first
    if px is None:
        return body or first
    token = f"{float(px):.2f}"
    token_i = str(int(float(px)))
    has_px = token in body or token_i in body
    starts = bool(re.match(r"^(BUY|SELL|WAIT)\b", body, re.I))
    if has_px and starts:
        return body
    if first:
        return f"{first}\n\n{body}".strip() if body else first
    return body


def extract_direction_callout(answer: str) -> Optional[str]:
    """First-line BUY / SELL / WAIT from a Gemini direction answer."""
    text = (answer or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip().lstrip("*#_ ").strip()
    first = re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0].strip()
    m = re.match(r"^(BUY|SELL|WAIT)\b(?:\s*[—\-:]\s*|\s+)?(.*)$", first, re.I)
    if not m:
        return None
    word = m.group(1).upper()
    rest = (m.group(2) or "").strip(" —-:")
    if rest:
        return f"{word} — {rest}"[:320]
    return word


def _load_cached_1m_csv(path: Optional[str] = None):
    csv_path = Path(path) if path else (_repo_root() / MNQ_1M_CSV)
    if not csv_path.is_file():
        return None
    mtime = csv_path.stat().st_mtime
    key = str(csv_path.resolve())
    if (
        _CSV_CACHE.get("path") == key
        and _CSV_CACHE.get("mtime") == mtime
        and _CSV_CACHE.get("df") is not None
    ):
        return _CSV_CACHE["df"]
    df = load_1m_seed_csv(str(csv_path))
    _CSV_CACHE["path"] = key
    _CSV_CACHE["mtime"] = mtime
    _CSV_CACHE["df"] = df
    return df


def _last_closed_why(recent_trades: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not recent_trades:
        return None
    t = recent_trades[-1]
    if not isinstance(t, dict):
        return None
    why = t.get("why") if isinstance(t.get("why"), dict) else {}
    primary = t.get("primary_reason") or why.get("primary_reason")
    if not primary and t.get("pnl_usd") is None and t.get("desk_exit_reason") is None:
        return None
    return {
        "side": t.get("direction") or t.get("side"),
        "pnl_usd": t.get("pnl_usd"),
        "exit_reason": t.get("desk_exit_reason") or t.get("exit_reason"),
        "primary_reason": primary,
        "facts": (why.get("facts") or [])[:4],
        "window_name": t.get("window_name"),
    }


def _position_from_status(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status = status or {}
    overnight = is_overnight_rec(status)
    flatten = OVERNIGHT_FLATTEN_ET if overnight else FLATTEN_ET
    opens = status.get("open_positions") or []
    if not isinstance(opens, list):
        opens = []
    live = [p for p in opens if isinstance(p, dict)]
    if not live:
        return {
            "state": "flat",
            "side": None,
            "entry": None,
            "stop": None,
            "flatten_et": flatten,
            "take_profit": OVERNIGHT_TP_TEXT if overnight else f"Flatten {flatten} ET",
        }
    p = live[0]
    return {
        "state": "open",
        "symbol": p.get("symbol"),
        "side": p.get("direction") or p.get("side"),
        "entry": p.get("entry") or p.get("entry_price"),
        "stop": p.get("sl") or p.get("stop"),
        "flatten_et": p.get("flatten_et") or flatten,
        "take_profit": OVERNIGHT_TP_TEXT if overnight else f"Flatten {flatten} ET",
        "open_count": len(live),
    }


def overlay_overnight_levels(
    rth: Dict[str, Any],
    status: Optional[Dict[str, Any]],
    *,
    price: Any = None,
    source: str = "rithmic",
) -> Dict[str, Any]:
    """Replace RTH WAIT with the live overnight paper position / ticker."""
    levels = dict(rth or {})
    status = status or {}
    pos_list = status.get("open_positions") or []
    pos = pos_list[0] if isinstance(pos_list, list) and pos_list and isinstance(pos_list[0], dict) else {}
    px = _as_float(price)
    if px is None:
        px = _as_float(status.get("last_price") or status.get("price"))
    src = str(source or status.get("quote_source") or "rithmic")
    levels["refresh_ok"] = True
    levels["price_stale"] = False
    levels["source"] = src
    levels["last_price"] = None if px is None else round(px, 2)
    levels["overnight"] = True
    levels["flatten_et"] = OVERNIGHT_FLATTEN_ET
    levels["take_profit"] = OVERNIGHT_TP_TEXT
    last_bit = f"{px:.2f}" if px is not None else "unavailable"
    if pos:
        side = str(pos.get("direction") or pos.get("side") or "LONG").upper()
        entry = _as_float(pos.get("entry") or pos.get("entry_price"))
        stop = _as_float(pos.get("sl") or pos.get("stop"))
        pts = stop_pts_of({
            **pos,
            "entry": entry,
            "stop": stop,
            "atr_stop_pts": pos.get("atr_stop_pts"),
        })
        levels["callout"] = f"PAPER {side}"
        levels["waiting"] = False
        levels["side"] = side
        levels["entry"] = entry
        levels["stop"] = stop
        levels["stop_pts"] = pts
        stop_bit = f"{stop:.2f}" if stop is not None else "—"
        pts_bit = f" ({pts:.0f} pts)" if pts is not None else ""
        entry_bit = f"{entry:.2f}" if entry is not None else "—"
        levels["first_line"] = (
            f"PAPER {side} — last MNQ {last_bit} ({src}). "
            f"Entry {entry_bit}, stop {stop_bit}{pts_bit}. {OVERNIGHT_TP_TEXT}."
        )
    else:
        levels["callout"] = "PAPER"
        levels["waiting"] = True
        levels["first_line"] = (
            f"PAPER OVERNIGHT — flat. Last MNQ {last_bit} ({src}). "
            f"{OVERNIGHT_TP_TEXT}. Locked RTH recipe starts 09:35."
        )
    return levels


def apply_overnight_live_quote(packet: Dict[str, Any], status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Force Gemini/desk onto the live ticker when overnight paper is running."""
    if not is_overnight_rec(status or {}):
        return packet
    out = dict(packet or {})
    live_px = _as_float((status or {}).get("last_price") or (status or {}).get("price"))
    src = str((status or {}).get("quote_source") or "rithmic")
    if live_px is not None:
        out["price_1m"] = live_px
        out["price_stale"] = False
        out["refresh_ok"] = True
        out["refresh_error"] = None
        out["data_source"] = src
        if (status or {}).get("quote_age_seconds") is not None:
            try:
                age = float(status["quote_age_seconds"])
                out["last_bar_age_sec"] = round(age)
                out["quote_age_sec"] = round(age)
            except (TypeError, ValueError):
                pass
        if (status or {}).get("last_quote_ts_et"):
            out["last_bar_ts_et"] = str(status["last_quote_ts_et"])
        out["stale_note"] = None
        out["warning"] = None
    rth = out.get("levels") or compute_recipe_levels(out)
    out["rth_levels"] = dict(rth)
    out["levels"] = overlay_overnight_levels(rth, status, price=out.get("price_1m"), source=src)
    out["position"] = _position_from_status(status)
    return out


def load_desk_quote_meta() -> Dict[str, Any]:
    path = _repo_root() / "data" / "mnq_desk_quote.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_advisory_market_packet(
    *,
    status: Optional[Dict[str, Any]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
    df_1m=None,
    csv_path: Optional[str] = None,
    stale_sec: int = SNAPSHOT_STALE_SEC,
) -> Dict[str, Any]:
    """Desk/Gemini snapshot from MNQ_1m.csv + locked helpers when the live bot is off."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_et = now.astimezone(ET)
    et_mins = int(now_et.hour * 60 + now_et.minute)
    win = next_window_info(et_mins)
    if df_1m is None:
        df_1m = _load_cached_1m_csv(csv_path)
    last_bar_ts_et = None
    last_age_sec: Optional[float] = None
    stale = True
    snap: Dict[str, Any] = {}
    source = str(csv_path or MNQ_1M_CSV)
    if df_1m is not None and not getattr(df_1m, "empty", True):
        try:
            snap = capture_market_snapshot(df_1m, now=now)
        except Exception as e:
            snap = {"error": str(e)}
        row = df_1m.iloc[-1]
        ts = row["datetime"] if "datetime" in df_1m.columns else None
        if ts is not None:
            t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if getattr(t, "tzinfo", None) is None:
                t = t.replace(tzinfo=timezone.utc)
            last_bar_ts_et = t.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
            last_age_sec = (now - t.astimezone(timezone.utc)).total_seconds()
            stale = last_age_sec > float(stale_sec)
    live_px = (status or {}).get("last_price") or (status or {}).get("price")
    overnight = is_overnight_rec(status or {})
    status_fresh = False
    updated = (status or {}).get("updated_at")
    if updated:
        try:
            ut = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
            if ut.tzinfo is None:
                ut = ut.replace(tzinfo=timezone.utc)
            status_fresh = (now - ut.astimezone(timezone.utc)).total_seconds() <= 30
        except (TypeError, ValueError):
            status_fresh = False
    use_live = live_px is not None and (status_fresh or overnight)
    if use_live:
        price = live_px
        stale = False
        source = str((status or {}).get("quote_source") or "live status")
        if (status or {}).get("quote_age_seconds") is not None:
            try:
                last_age_sec = float(status["quote_age_seconds"])
            except (TypeError, ValueError):
                pass
        if (status or {}).get("last_quote_ts_et"):
            last_bar_ts_et = str(status["last_quote_ts_et"])
    else:
        price = live_px if live_px is not None else snap.get("price_1m")
    warning = None
    if stale:
        if last_bar_ts_et:
            warning = (
                f"STALE DATA: last 1m bar is {last_bar_ts_et} (file, not a live quote)."
            )
        else:
            warning = "STALE DATA: no 1m bar available; say the timestamp is stale."
    out = {
        "now_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "session": snap.get("session"),
        "in_window": bool(win["in_window"]),
        "window_index": win.get("window_index"),
        "window_name": win["window_name"],
        "next_window_name": win["next_window_name"],
        "minutes_until_next": win["minutes_until_next"],
        "next_is_tomorrow": win["next_is_tomorrow"],
        "daily_permission": snap.get("daily_permission"),
        "daily_permission_txt": _side_txt(snap.get("daily_permission")),
        "prior_rth_close": snap.get("prior_rth_close"),
        "daily_ema20": snap.get("daily_ema20"),
        "ema15_fast": snap.get("ema15_fast"),
        "ema15_slow": snap.get("ema15_slow"),
        "trend_15m": snap.get("trend_15m"),
        "trend_15m_txt": _side_txt(snap.get("trend_15m")),
        "atr15": snap.get("atr15"),
        "atr_stop_pts": snap.get("atr_stop_pts"),
        "ema60_fast": snap.get("ema60_fast"),
        "ema60_slow": snap.get("ema60_slow"),
        "trend_60m": snap.get("trend_60m"),
        "trend_60m_txt": _side_txt(snap.get("trend_60m")),
        "sep15": snap.get("sep15"),
        "price_1m": price,
        "last_bar_ts_et": last_bar_ts_et,
        "last_bar_age_sec": None if last_age_sec is None else round(last_age_sec),
        "quote_age_sec": None if last_age_sec is None else round(last_age_sec),
        "price_stale": stale,
        "refresh_ok": True,
        "refresh_error": None,
        "stale_note": warning,
        "warning": warning,
        "data_source": source,
        "position": _position_from_status(status),
        "locked_rules": LOCKED_RULES_PARAGRAPH,
        "last_closed_why": _last_closed_why(recent_trades),
        "advisory_only": True,
    }
    meta = load_desk_quote_meta()
    live_ok = overnight and live_px is not None
    if meta and not live_ok:
        out["refresh_ok"] = bool(meta.get("ok", True))
        out["refresh_error"] = meta.get("error")
        if meta.get("ok") is False:
            out["price_1m"] = None
            out["price_stale"] = True
            fail = meta.get("error") or "Databento download failed"
            out["stale_note"] = fail
            out["warning"] = fail
            out["data_source"] = "download failed"
    rth = compute_recipe_levels(out)
    out["rth_levels"] = rth
    if overnight:
        out = apply_overnight_live_quote({**out, "levels": rth}, status)
    else:
        out["levels"] = rth
    return out


def load_locked_backtest_facts() -> Dict[str, Any]:
    """Prefer on-disk official facts; fall back to the locked recipe numbers."""
    facts = dict(_LOCKED_BACKTEST_DEFAULT)
    root = _repo_root()
    for rel in (
        "data/window_edge_analysis.json",
        "data/ema15_official_backtest.json",
    ):
        path = root / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        nested = data.get("official") if isinstance(data.get("official"), dict) else data
        for src, dest in (
            ("is_pf", "is_pf"),
            ("is_pnl", "is_pnl_usd"),
            ("is_pnl_usd", "is_pnl_usd"),
            ("oos_pf", "oos_pf"),
            ("oos_pnl", "oos_pnl_usd"),
            ("oos_pnl_usd", "oos_pnl_usd"),
            ("max_dd", "max_dd_usd"),
            ("max_dd_usd", "max_dd_usd"),
        ):
            if nested.get(src) is not None:
                facts[dest] = nested[src]
        facts["source"] = f"BACKTEST {rel} (not live P&L)"
        return facts

    md = root / "docs" / "PROFITABLE_LIVE.md"
    if md.is_file():
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
            m = re.search(
                r"Official \(6 windows[^\n]*\|\s*\**([0-9.]+)\s*/\s*\$([0-9,]+)\**"
                r"\s*\|\s*\**([0-9.]+)\s*/\s*\$([0-9,]+)\**"
                r"\s*\|\s*\**\$?([0-9,]+)",
                text,
            )
            if m:
                facts["is_pf"] = float(m.group(1))
                facts["is_pnl_usd"] = float(m.group(2).replace(",", ""))
                facts["oos_pf"] = float(m.group(3))
                facts["oos_pnl_usd"] = float(m.group(4).replace(",", ""))
                facts["max_dd_usd"] = float(m.group(5).replace(",", ""))
                facts["source"] = "BACKTEST docs/PROFITABLE_LIVE.md (not live P&L)"
        except OSError:
            pass
    return facts


def format_locked_backtest_answer(facts: Optional[Dict[str, Any]] = None) -> str:
    f = facts or load_locked_backtest_facts()
    is_pnl = float(f["is_pnl_usd"])
    oos_pnl = float(f["oos_pnl_usd"])
    dd = float(f.get("max_dd_usd") or 0)
    return (
        f"{f.get('source', 'BACKTEST (not live P&L)')}. "
        f"Locked ema15_eod {f.get('label', 'official')}: "
        f"in-sample PF {float(f['is_pf']):.2f} / ${is_pnl:,.0f}; "
        f"out-of-sample PF {float(f['oos_pf']):.2f} / ${oos_pnl:,.0f}"
        f"{f' (max DD ${dd:,.0f})' if dd else ''}. "
        "This is historical backtest, not live trading P&L."
    )


class LLMTradeAdvisor:
    """Ask an LLM whether a proposed MTF trade aligns with macro/context."""

    def __init__(self):
        self.enabled = os.getenv("LLM_ENABLED", "false").lower() == "true"
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        self.min_confidence = float(os.getenv("LLM_MIN_CONFIDENCE", "0.60"))
        self.min_setup_score = float(os.getenv("LLM_MIN_SETUP_SCORE", "75"))
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "12"))
        self.cache_ttl_sec = int(os.getenv("LLM_CACHE_TTL_SEC", "900"))  # 15 min
        self.model = (os.getenv("LLM_MODEL") or self._default_model()).strip()
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._working_gemini_model: Optional[str] = None

        if self.enabled and not self.api_key:
            bot_logger.warning("LLM advisor enabled but no API key — advisor disabled")
            self.enabled = False
        elif self.enabled:
            bot_logger.info(
                f"LLM advisor active: provider={self.provider} model={self.model} "
                f"min_confidence={self.min_confidence}"
            )

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4o-mini"
        if self.provider in ("gemini", "google"):
            return "gemini-3.5-flash"
        return "deepseek-chat"

    def _resolve_api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "").strip()
        if self.provider in ("gemini", "google"):
            return (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
                or os.getenv("LLM_API_KEY", "").strip()
            )
        return os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")).strip()

    def _resolve_base_url(self) -> str:
        explicit = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        if self.provider in ("gemini", "google"):
            base = explicit or DEFAULT_GEMINI_BASE
            if "/openai" in base:
                base = base.split("/openai")[0].rstrip("/")
            return base or DEFAULT_GEMINI_BASE
        if explicit:
            return explicit
        if self.provider == "openai":
            return DEFAULT_OPENAI_BASE
        return DEFAULT_DEEPSEEK_BASE

    def _cache_key(self, symbol: str, direction: str) -> str:
        hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        return f"{symbol}:{direction}:{hour_bucket}"

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _cache_key_historical(self, symbol: str, direction: str, dt_iso: str) -> str:
        return f"{symbol}:{direction}:{dt_iso}"

    def _build_prompt(self, signal: Dict[str, Any], context: Dict[str, Any]) -> str:
        direction = signal.get("direction", "").upper()
        symbol = signal.get("symbol", "MNQ")
        return f"""You are an expert MNQ micro futures scalping advisor.

A rule-based MTF system proposes {direction} on {symbol}.
Review the setup using session awareness, market structure, VWAP, volume, MTF alignment, and news risk.

Respond ONLY with JSON:
{{
  "action": "allow" or "skip",
  "bias": "bullish" or "bearish" or "neutral",
  "confidence": 0.0 to 1.0,
  "setup_score": 0 to 100,
  "market_type": "trend" or "range" or "chop",
  "position_size_pct": 0 or 50 or 100,
  "reason": "one short sentence",
  "warnings": ["optional list"]
}}

Guidelines:
- SKIP midday chop (11am-2pm ET) unless setup_score would be 90+.
- SKIP within 15-30 min of high-impact US news (CPI, NFP, FOMC).
- SKIP if 1m direction fights 5m/15m trend.
- SKIP chop/volatile_chop unless exceptional confluence.
- ALLOW ny_open and power_hour when trend-aligned, above/below VWAP, strong volume.
- setup_score: 95-100 full size, 80-94 normal, 60-79 half, below 60 skip.
- confidence = certainty in allow/skip decision (not just bullish/bearish).

Context:
{json.dumps(context, default=str, indent=2)}
"""

    def _chat_completions_url(self) -> str:
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1") or "/v1" in base:
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _is_gemini(self) -> bool:
        return self.provider in ("gemini", "google")

    def _gemini_native_base(self) -> str:
        base = (self.base_url or DEFAULT_GEMINI_BASE).rstrip("/")
        if "/openai" in base:
            base = base.split("/openai")[0].rstrip("/")
        return base or DEFAULT_GEMINI_BASE

    def _gemini_model_id(self, name: Optional[str] = None) -> str:
        mid = (name or self.model or "gemini-flash-latest").strip()
        if mid.startswith("models/"):
            mid = mid[len("models/") :]
        return mid

    def _gemini_generate_url(self, model: Optional[str] = None) -> str:
        return f"{self._gemini_native_base()}/models/{self._gemini_model_id(model)}:generateContent"

    def _gemini_model_candidates(self) -> List[str]:
        ordered: List[str] = []
        if self._working_gemini_model:
            ordered.append(self._gemini_model_id(self._working_gemini_model))
        configured = self._gemini_model_id(self.model)
        preferred = (
            tuple(GEMINI_FALLBACK_MODELS) + (configured,)
            if configured in GEMINI_RETIRED_MODELS
            else (configured, *GEMINI_FALLBACK_MODELS)
        )
        for mid in preferred:
            mid = self._gemini_model_id(mid)
            if mid not in ordered:
                ordered.append(mid)
        return ordered

    def _extract_gemini_text(self, data: Dict[str, Any]) -> Optional[str]:
        cands = data.get("candidates") or []
        if not cands:
            return None
        parts = ((cands[0].get("content") or {}).get("parts") or [])
        texts = [
            str(p.get("text")).strip()
            for p in parts
            if isinstance(p, dict) and p.get("text")
        ]
        joined = "\n".join(t for t in texts if t)
        return joined or None

    def _public_http_error(self, status: int, reason: str, url: str, body: str) -> str:
        display = (url or "").split("?")[0]
        if GEMINI_OPENAI_COMPAT_SUFFIX in display:
            display = display.replace(GEMINI_OPENAI_COMPAT_SUFFIX, "/models/{model}:generateContent")
        snippet = _sanitize_secret_text((body or "").replace("\n", " "))[:240]
        return f"{status} {reason} for {display}: {snippet}"

    def _public_error(self, exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            resp = exc.response
            return self._public_http_error(
                resp.status_code, resp.reason or "", (resp.url or "").split("?")[0], resp.text or ""
            )
        return _sanitize_secret_text(str(exc))

    def _call_gemini_native(
        self, prompt: str, *, json_mode: bool = True, system: Optional[str] = None
    ) -> Optional[str]:
        """POST v1beta/models/{model}:generateContent. Never uses openai/chat/completions."""
        if not system:
            system = (
                "You output strict JSON only for trade risk decisions."
                if json_mode
                else "You are a concise MNQ futures desk assistant."
            )

        def _payload(include_thinking: bool) -> Dict[str, Any]:
            gen_cfg: Dict[str, Any] = {
                "temperature": 0.2 if json_mode else 0.4,
                "maxOutputTokens": 1024 if json_mode else 4096,
            }
            if include_thinking:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
            if json_mode:
                gen_cfg["responseMimeType"] = "application/json"
            return {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        timeout = max(float(self.timeout_sec), 20.0)
        last_err = "Gemini generateContent failed"
        for model in self._gemini_model_candidates():
            url = self._gemini_generate_url(model)
            payload = _payload(True)
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as e:
                last_err = self._public_error(e)
                continue
            if resp.status_code == 400:
                try:
                    resp = requests.post(
                        url, headers=headers, json=_payload(False), timeout=timeout
                    )
                except requests.RequestException as e:
                    last_err = self._public_error(e)
                    continue
            if resp.status_code in (401, 403):
                try:
                    resp = requests.post(
                        url,
                        params={"key": self.api_key},
                        headers={"Content-Type": "application/json"},
                        json=_payload(False),
                        timeout=timeout,
                    )
                except requests.RequestException as e:
                    last_err = self._public_error(e)
                    continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    text = self._extract_gemini_text(data)
                except ValueError:
                    data = {}
                    text = None
                finish = ""
                cands = data.get("candidates") if isinstance(data, dict) else None
                if cands:
                    finish = str((cands[0] or {}).get("finishReason") or "")
                truncated = (
                    not json_mode
                    and text
                    and finish == "MAX_TOKENS"
                    and len(text) < 120
                )
                if text and not truncated:
                    self._working_gemini_model = model
                    return text
                last_err = f"200 empty/truncated body for {url} finish={finish or 'n/a'}"
                continue
            last_err = self._public_http_error(
                resp.status_code, resp.reason or "", url, resp.text or ""
            )
            if resp.status_code in (400, 404, 429, 503):
                bot_logger.warning(f"Gemini {model} {resp.status_code}; trying next model")
                continue
            break
        raise RuntimeError(last_err)

    def _call_api(
        self, prompt: str, *, json_mode: bool = True, system: Optional[str] = None
    ) -> Optional[str]:
        if self._is_gemini():
            return self._call_gemini_native(prompt, json_mode=json_mode, system=system)

        url = self._chat_completions_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if not system:
            system = (
                "You output strict JSON only for trade risk decisions."
                if json_mode
                else "You are a concise MNQ futures desk assistant."
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2 if json_mode else 0.4,
            "max_tokens": 200 if json_mode else 400,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        if resp.status_code >= 400:
            raise RuntimeError(
                self._public_http_error(
                    resp.status_code, resp.reason or "", url, resp.text or ""
                )
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _decide_from_parsed(self, parsed: Dict[str, Any], direction: str) -> Dict[str, Any]:
        action = str(parsed.get("action", "allow")).lower()
        bias = str(parsed.get("bias", "neutral")).lower()
        confidence = float(parsed.get("confidence", 0.5))
        setup_score = float(parsed.get("setup_score", confidence * 100))
        market_type = str(parsed.get("market_type", "range"))
        size_pct = int(parsed.get("position_size_pct", 100))
        reason = str(parsed.get("reason", "LLM response"))
        warnings = parsed.get("warnings") or []

        wants_long = direction.lower() in ("long", "buy")
        bias_conflicts = (wants_long and bias == "bearish") or (
            (not wants_long) and bias == "bullish"
        )

        allowed = (
            action == "allow"
            and confidence >= self.min_confidence
            and setup_score >= self.min_setup_score
            and not bias_conflicts
        )
        if action == "skip" or setup_score < self.min_setup_score:
            allowed = False

        return {
            "allowed": allowed,
            "action": action,
            "bias": bias,
            "confidence": confidence,
            "setup_score": setup_score,
            "market_type": market_type,
            "position_size_pct": size_pct if allowed else 0,
            "reason": reason,
            "warnings": warnings,
            "source": "llm",
        }

    def evaluate_trade(
        self,
        signal: Dict[str, Any],
        ctx_5m: Dict[str, Any],
        row_1m: Optional[Any] = None,
        rich_context: Optional[Dict[str, Any]] = None,
        df_1m=None,
        df_5m=None,
        df_15m=None,
        dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Review proposed trade; on API error returns allow (rules-only fallback)."""
        if not self.enabled:
            return {
                "allowed": True,
                "action": "allow",
                "confidence": 1.0,
                "setup_score": 100,
                "reason": "LLM disabled",
                "source": "fallback",
                "position_size_pct": 100,
            }

        symbol = signal.get("symbol", "MNQ")
        direction = signal.get("direction", "long")
        when = dt or datetime.now(timezone.utc)
        dt_iso = when.strftime("%Y%m%d%H%M")
        cache_key = self._cache_key_historical(symbol, direction, dt_iso)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        if rich_context is None and row_1m is not None and dt is not None:
            check_dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            nb, ne = EconomicCalendar().is_event_blocked("USD/JPY", check_dt)
            rich_context = build_mnq_context(
                dt, row_1m, ctx_5m, df_1m, df_5m, df_15m,
                news_blocked=nb, news_event=ne,
            )
            rich_context["proposed_direction"] = direction
            rich_context["entry"] = signal.get("entry")
            rich_context["stop_loss"] = signal.get("sl")
            rich_context["take_profit"] = signal.get("tp")
            rich_context["rule_setup_score"] = compute_setup_score(direction, rich_context)["setup_score"]

        context = rich_context or {
            "utc_time": when.isoformat(),
            "symbol": symbol,
            "proposed_direction": direction,
            "entry": signal.get("entry"),
            "stop_loss": signal.get("sl"),
            "take_profit": signal.get("tp"),
            "trend_5m": ctx_5m.get("trend"),
            "adx_5m": ctx_5m.get("adx"),
        }
        if row_1m is not None and "rule_setup_score" not in context:
            context["rsi_1m"] = float(row_1m.get("rsi", 50))
            context["volume_ratio_1m"] = float(row_1m.get("volume_ratio", 1))

        try:
            raw = self._call_api(self._build_prompt(signal, context))
            parsed = self._parse_json_response(raw) or {}
            result = self._decide_from_parsed(parsed, direction)
            self._cache[cache_key] = (now, result)
            bot_logger.info(
                f"🤖 LLM {symbol} {direction}: "
                f"{'ALLOW' if result['allowed'] else 'SKIP'} "
                f"conf={result['confidence']:.0%} score={result.get('setup_score', 0):.0f} — "
                f"{result['reason']}"
            )
            return result
        except Exception as e:
            bot_logger.warning(f"LLM advisor error (rules-only fallback): {self._public_error(e)}")
            return {
                "allowed": True,
                "action": "allow",
                "confidence": 0.0,
                "setup_score": 0,
                "reason": f"LLM unavailable: {self._public_error(e)}",
                "source": "fallback",
                "position_size_pct": 100,
            }

    def evaluate_ema15(
        self,
        signal: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Advisor for the locked 15m+daily recipe. Does not replace the rules."""
        meta = meta or signal.get("entry_meta") or {}
        context = {
            "strategy": "ema15_eod",
            "symbol": signal.get("symbol", "MNQ"),
            "direction": signal.get("direction"),
            "entry": signal.get("entry"),
            "stop_loss": signal.get("sl"),
            "take_profit": signal.get("tp"),
            "trend_15m": meta.get("trend_15m"),
            "trend_60m": meta.get("trend_60m"),
            "daily_trend": meta.get("daily_trend"),
            "sep15": meta.get("sep15"),
            "high_confidence": meta.get("high_confidence"),
            "add_on": meta.get("add_on"),
            "sl_pts": meta.get("sl_pts"),
            "window": meta.get("window"),
            "note": "Rules already require 15m EMA + yesterday RTH daily agreement. "
                    "You are a second opinion: flag CPI/FOMC/news risk or chop. "
                    "Do not skip just because win rate is low — this strategy holds winners to 15:50 ET.",
        }
        return self.evaluate_trade(signal, {"trend": None, "adx": 0}, rich_context=context)

    def ask(self, question: str, extra_context: Optional[Dict[str, Any]] = None) -> str:
        """Free-form question for the dashboard chat box."""
        facts = load_locked_backtest_facts()
        local = format_locked_backtest_answer(facts)
        ctx_obj = dict(extra_context or {})
        direction = _is_direction_question(question)
        if "market" not in ctx_obj:
            ctx_obj["market"] = build_advisory_market_packet(
                status=ctx_obj.get("status"),
                recent_trades=ctx_obj.get("recent_trades"),
            )
        market = ctx_obj.get("market") or {}
        if is_overnight_rec(ctx_obj.get("status") or {}):
            market = apply_overnight_live_quote(market, ctx_obj.get("status"))
        levels = market.get("levels") or compute_recipe_levels(market)
        market["levels"] = levels
        ctx_obj["market"] = market
        first = str(levels.get("first_line") or "")
        overnight = bool(levels.get("overnight") or is_overnight_rec(ctx_obj.get("status") or {}))
        if not self.enabled:
            if _is_profit_question(question):
                return local
            if direction and first:
                return (
                    f"{first}\n\nGemini is off. Second opinion — recipe is still the boss. "
                    "These numbers come from the locked ema15_eod snapshot, not an order."
                )
            return "Gemini/LLM is off. Set LLM_ENABLED=true and GEMINI_API_KEY in .env."
        ctx_obj["locked_backtest"] = facts
        ctx_obj["advisory_only"] = True
        raw_ctx = json.dumps(ctx_obj, default=str)
        if direction:
            ctx = raw_ctx[:14000]
            nums = json.dumps(levels, default=str)
            exit_bit = (
                "Then EXIT using those numbers only. 1R TP, then flatten 09:25 ET. "
                "Do not lead with RTH WAIT or 09:35."
                if overnight
                else "Then ENTRY/EXIT using those numbers only. Flatten 15:50 ET. Do not trail."
            )
            prompt = (
                f"QUOTE THESE PYTHON NUMBERS (do not invent others):\n{nums}\n\n"
                f"Start with this exact first sentence:\n{first}\n\n"
                f"Context JSON:\n{ctx}\n\nOperator question: {question}\n\n"
                f"{exit_bit} "
                "Label second opinion — recipe is still the boss. Never say you placed an order."
            )
            system = OVERNIGHT_ASK_SYSTEM if overnight else DIRECTION_ASK_SYSTEM
        else:
            ctx = raw_ctx[:4000]
            rules = (
                "Overnight paper is open. Lead with the live ticker and the open paper SL / "
                "1R TP, flatten 09:25 ET. Do not cite a stale MNQ_1m.csv print."
                if overnight
                else
                "Rules: enter only when daily+15m+60m agree (15m EMA8/21, 60m EMA8/21, "
                "yesterday RTH daily EMA20), ATR×2 stop (20-60 pts), "
                "windows 9:35/10:15/11:00/12:00/13:30/14:30 ET "
                "(noon+ still needs sep>=0.45), flatten 15:50, "
                "optional 2nd lot if sep>=0.45. Overnight is off."
            )
            prompt = (
                "You are the MNQ desk assistant. Be concise. "
                "You are ADVISORY only — never place orders or say 'order sent'. "
                "Lead with the live snapshot in market.levels.first_line. "
                "Do not lead with historical trade counts or profit factor unless asked "
                "if the strategy is profitable (then cite locked_backtest and say BACKTEST not live). "
                f"{rules}\n\n"
                f"Live first sentence:\n{first}\n\n"
                f"Context JSON:\n{ctx}\n\nOperator question: {question}"
            )
            system = OVERNIGHT_ASK_SYSTEM if overnight else None
        try:
            raw = self._call_api(prompt, json_mode=False, system=system)
            text = (raw or "").strip() or "No reply."
            text = ensure_levels_in_answer(text, levels)
            if _is_profit_question(question) and ("1.41" not in text or len(text) < 120):
                text = f"{text}\n\n{local}"
            return text
        except Exception as e:
            err = self._public_error(e)
            bot_logger.warning(f"Gemini ask failed: {err}")
            if direction and first:
                return (
                    f"{first}\n\nGemini unavailable: {err}\n\n"
                    "Second opinion — recipe is still the boss."
                )
            if _is_profit_question(question):
                return f"Gemini unavailable: {err}\n\n{local}"
            return f"Gemini unavailable: {err}"

    def overnight_close_note(self, record: Dict[str, Any]) -> str:
        """Short advisory note for an overnight paper close. Never mutates live rules."""
        if not self.enabled:
            return ""
        compact = {
            "session": "overnight_research",
            "symbol": record.get("symbol"),
            "side": record.get("side") or record.get("direction"),
            "cue": record.get("cue"),
            "zone_name": record.get("zone_name"),
            "zone_price": record.get("zone_price"),
            "entry_price": record.get("entry_price"),
            "exit_price": record.get("exit_price"),
            "pts": record.get("pts"),
            "pnl_usd": record.get("pnl_usd"),
            "mae_pts": record.get("mae_pts"),
            "mfe_pts": record.get("mfe_pts"),
            "exit_reason": record.get("exit_reason"),
            "why": record.get("why"),
        }
        prompt = (
            "One or two sentences on this OVERNIGHT PAPER research trade. "
            "Advisory only. Do not recommend changing locked RTH ema15 windows, "
            "stops, flatten 15:50, or mnq_profit_config. Do not say you placed an order.\n"
            f"{json.dumps(compact, default=str)}"
        )
        try:
            raw = self._call_api(
                prompt,
                json_mode=False,
                system=(
                    "You review overnight Globex paper trades. Advisory only. "
                    "Never change live RTH rules."
                ),
            )
            line = (raw or "").strip().replace("\n", " ")
            return line[:400]
        except Exception:
            return ""

    def post_mortem_line(self, record: Dict[str, Any]) -> str:
        """One-line commentary after a close. Never changes orders or params."""
        if not self.enabled:
            return ""
        why = record.get("why") or {}
        compact = {
            "symbol": record.get("symbol"),
            "side": record.get("side") or record.get("direction"),
            "pts": record.get("pts"),
            "pnl_usd": record.get("pnl_usd"),
            "exit_reason": record.get("exit_reason"),
            "mae_pts": record.get("mae_pts"),
            "mfe_pts": record.get("mfe_pts"),
            "primary_reason": why.get("primary_reason"),
            "facts": (why.get("facts") or [])[:5],
        }
        prompt = (
            "One sentence on this closed MNQ ema15_eod trade. "
            "Do not recommend changing stops, windows, or live rules. "
            "Do not invent a take-profit. Comment on the measured facts only.\n"
            f"{json.dumps(compact, default=str)}"
        )
        try:
            raw = self._call_api(prompt, json_mode=False)
            line = (raw or "").strip().replace("\n", " ")
            return line[:280]
        except Exception:
            return ""
