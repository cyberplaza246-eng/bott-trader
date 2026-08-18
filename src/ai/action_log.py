"""Append-only action feed for the MTF / ema15 live dashboard."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pytz
    ET = pytz.timezone("US/Eastern")
except Exception:  # pragma: no cover
    ET = timezone.utc

ACTION_LOG_PATH = "data/bot_actions.jsonl"
STATUS_PATH = "data/bot_status.json"
MAX_ACTIONS = 400

MONITOR_INTERVAL_SEC = 90.0
DATA_INTERVAL_SEC = 90.0

DEV_KINDS = frozenset({
    "dev", "debug", "rithmic", "http", "traceback", "werkzeug", "gemini",
})
HUMAN_KINDS = frozenset({
    "entry", "exit", "close", "flatten", "rejected", "skip", "idle", "no_trade",
    "warning", "error", "reconnect", "connection", "monitoring", "test",
    "data", "market", "paper_trade", "profit", "loss",
})
HUMAN_SKIP_KINDS = frozenset({"gemini_chat"})
DEV_HINTS = (
    "forcedlogout",
    "rpcode",
    "get /api/desk",
    "traceback (most recent",
    "traceback:",
    "async_rithmic",
    "websocket",
    "1011",
    "permission denied",
    "heartbeat_interval",
    "werkzeug",
)


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def format_clock(ts: Any) -> str:
    t = _parse_ts(ts)
    if t is None:
        return ""
    local = t.astimezone(ET)
    stamp = local.strftime("%I:%M %p")
    return stamp.lstrip("0") if stamp.startswith("0") else stamp


def fmt_px(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{n:,.2f}"


def fmt_usd(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    sign = "+" if n >= 0 else "-"
    return f"{sign}${abs(n):,.2f}"


OVERNIGHT_FLATTEN_ET = "09:25"
OVERNIGHT_TP_TEXT = "1R TP, flatten 09:25 ET"


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_overnight_rec(rec: Optional[Dict[str, Any]]) -> bool:
    rec = rec or {}
    if rec.get("overnight_research") is True:
        return True
    session = str(rec.get("session") or rec.get("strategy") or "").lower()
    return session == "overnight_research"


def stop_pts_of(rec: Optional[Dict[str, Any]]) -> Optional[float]:
    rec = rec or {}
    pts = _num(rec.get("atr_stop_pts") or rec.get("stop_pts") or rec.get("sl_pts"))
    if pts is not None:
        return abs(pts)
    entry = _num(rec.get("entry") or rec.get("entry_price"))
    stop = _num(_stop_px(rec))
    if entry is not None and stop is not None:
        return abs(entry - stop)
    return None


def format_stop_line(rec: Optional[Dict[str, Any]]) -> str:
    stop = fmt_px(_stop_px(rec or {}))
    if not stop:
        return ""
    pts = stop_pts_of(rec)
    if pts is not None:
        return f"Stop {stop} ({pts:.0f} pts)"
    return f"Stop {stop}"


def take_profit_text(rec: Optional[Dict[str, Any]]) -> str:
    rec = rec or {}
    if is_overnight_rec(rec):
        flatten = str(rec.get("flatten_et") or OVERNIGHT_FLATTEN_ET)
        if flatten.endswith(" ET"):
            flatten = flatten.replace(" ET", "")
        return f"1R TP, flatten {flatten} ET"
    flatten = rec.get("flatten_et")
    cap_px = fmt_px(rec.get("tp_cap") or rec.get("tp") or rec.get("take_profit"))
    if flatten:
        line = f"Flatten {flatten} ET (no scalp TP)"
        if cap_px:
            line += f" · TP cap {cap_px}"
        return line
    return ""


def format_tp_line(rec: Optional[Dict[str, Any]]) -> str:
    rec = rec or {}
    if is_overnight_rec(rec):
        text = take_profit_text(rec)
        if not text:
            return ""
        if text.lower().startswith("target "):
            return text
        return f"Target {text}"
    live = format_live_entry_brackets(rec)
    if live:
        return live
    text = take_profit_text(rec)
    if not text:
        return ""
    if text.lower().startswith("tp ") or text.lower().startswith("target "):
        return text
    return f"TP {text}"


def format_live_entry_brackets(rec: Optional[Dict[str, Any]]) -> str:
    """Day (ema15_eod) entry line: SL placed @ … / TP cap @ …"""
    rec = rec or {}
    if is_overnight_rec(rec):
        return ""
    sl = fmt_px(_stop_px(rec))
    cap = rec.get("tp_cap") or rec.get("tp") or rec.get("take_profit")
    tp = fmt_px(cap)
    parts = []
    if sl:
        parts.append(f"SL placed @ {sl}")
    if tp:
        parts.append(f"TP cap @ {tp}")
    return " / ".join(parts)


def _merge_sl_tp_lines(lines: Sequence[str], rec: Dict[str, Any]) -> List[str]:
    out = [str(x).strip() for x in (lines or []) if str(x).strip()]
    overnight = is_overnight_rec(rec)
    stop_line = format_stop_line(rec) if overnight else ""
    tp_line = format_tp_line(rec)
    live = format_live_entry_brackets(rec) if not overnight else ""
    if live:
        replaced = False
        for i, ln in enumerate(out):
            if "sl placed" in ln.lower() or ln.lower().startswith("sl "):
                out[i] = live
                replaced = True
                break
        if not replaced:
            insert_at = 1 if out else 0
            out.insert(insert_at, live)
        return out
    if stop_line:
        replaced = False
        for i, ln in enumerate(out):
            if ln.lower().startswith("stop "):
                out[i] = stop_line
                replaced = True
                break
        if not replaced:
            insert_at = 1 if out else 0
            out.insert(insert_at, stop_line)
    if tp_line and not any(
        "flatten" in ln.lower()
        or ln.lower().startswith("tp ")
        or ln.lower().startswith("target ")
        or "take profit" in ln.lower()
        or "sl placed" in ln.lower()
        for ln in out
    ):
        out.append(tp_line)
    return out


def _qty(rec: Dict[str, Any]) -> int:
    for key in ("qty", "size", "contracts"):
        try:
            n = int(rec.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return 1


def _side(rec: Dict[str, Any]) -> str:
    return str(rec.get("direction") or rec.get("side") or "").strip().lower()


def _is_paper(rec: Dict[str, Any]) -> bool:
    if rec.get("paper") is True:
        return True
    session = str(rec.get("session") or "").lower()
    if session in ("overnight_research", "paper"):
        return True
    blob = " ".join(
        str(rec.get(k) or "") for k in ("reason", "title", "cue", "note")
    ).lower()
    return "paper" in blob or "test_fill" in blob or "not sent" in blob


def _stop_px(rec: Dict[str, Any]) -> Any:
    for key in ("stop", "sl", "stop_loss"):
        if rec.get(key) not in (None, ""):
            return rec.get(key)
    return None


def _blob(rec: Dict[str, Any]) -> str:
    parts = [
        str(rec.get("kind") or ""),
        str(rec.get("reason") or ""),
        str(rec.get("title") or ""),
        str(rec.get("detail") or ""),
        " ".join(str(x) for x in (rec.get("lines") or [])),
    ]
    return " ".join(parts).lower()


def looks_like_dev(rec: Dict[str, Any]) -> bool:
    layer = str(rec.get("layer") or "").lower()
    if layer == "human":
        return False
    if layer == "dev":
        return True
    kind = str(rec.get("kind") or "").lower()
    if kind in HUMAN_KINDS:
        return False
    if kind in DEV_KINDS:
        return True
    if kind in HUMAN_SKIP_KINDS:
        return True
    return any(h in _blob(rec) for h in DEV_HINTS)


def _default_emoji_title(kind: str, rec: Dict[str, Any]) -> Tuple[str, str]:
    kind = str(kind or "").lower()
    side = _side(rec).upper() or "TRADE"
    paper = _is_paper(rec)
    pnl = rec.get("pnl")
    if pnl is None:
        pnl = rec.get("pnl_usd")
    try:
        pnl_n = float(pnl) if pnl is not None else None
    except (TypeError, ValueError):
        pnl_n = None

    if kind in ("entry", "paper_trade"):
        title = f"PAPER {side}" if paper else side
        if rec.get("cue") == "test_fill" or "test_fill" in str(rec.get("reason") or "").lower():
            title = f"PAPER {side}"
        return "🟢", title or "TRADE PLACED"
    if kind in ("profit",) or (kind in ("exit", "close", "flatten") and pnl_n is not None and pnl_n >= 0):
        return "💰", "PROFITABLE TRADE" if kind in ("exit", "close", "flatten", "profit") else "PROFIT"
    if kind in ("loss",) or (kind in ("exit", "close", "flatten") and pnl_n is not None and pnl_n < 0):
        return "🔴", "LOSING TRADE"
    if kind in ("exit", "close", "flatten"):
        return "💰", "TRADE CLOSED"
    if kind in ("skip", "rejected"):
        return "❌", "TRADE SKIPPED"
    if kind in ("idle", "no_trade"):
        return "⏸️", "NO TRADE"
    if kind in ("warning", "error"):
        return "⚠️", "PROBLEM"
    if kind in ("reconnect", "connection"):
        return "🔄", "CONNECTION"
    if kind in ("monitoring",):
        return "🟡", "MONITORING"
    if kind in ("test",):
        return "🧪", "TEST"
    if kind in ("data", "market"):
        return "📊", "MARKET"
    return "🟡", str(kind or "UPDATE").upper().replace("_", " ")


def _entry_lines(rec: Dict[str, Any]) -> List[str]:
    symbol = str(rec.get("symbol") or "MNQ").upper()
    qty = _qty(rec)
    px = fmt_px(rec.get("entry") or rec.get("entry_price"))
    lines = []
    if px:
        lines.append(f"{symbol} ×{qty} @ {px}")
    else:
        lines.append(f"{symbol} ×{qty}")
    if is_overnight_rec(rec):
        stop_line = format_stop_line(rec)
        if stop_line:
            lines.append(stop_line)
        tp_line = format_tp_line(rec)
        if tp_line:
            lines.append(tp_line)
    else:
        live = format_live_entry_brackets(rec)
        if live:
            lines.append(live)
        else:
            stop_line = format_stop_line(rec)
            if stop_line:
                lines.append(stop_line)
            tp_line = format_tp_line(rec)
            if tp_line:
                lines.append(tp_line)
    if _is_paper(rec) or rec.get("cue") == "test_fill":
        dest = str(rec.get("dest") or "")
        if "lucid" in dest.lower() and "ticket" in dest.lower():
            lines.append(f"🧪 Paper · {dest}")
        else:
            lines.append("🧪 Paper only")
    why = str(rec.get("reason") or "").strip()
    if why and "test_fill" not in why.lower() and why.lower() not in ("ema15_eod",):
        if why not in lines:
            lines.append(why)
    return lines


def _close_lines(rec: Dict[str, Any]) -> List[str]:
    symbol = str(rec.get("symbol") or "MNQ").upper()
    side = (_side(rec) or "trade").capitalize()
    pnl = rec.get("pnl")
    if pnl is None:
        pnl = rec.get("pnl_usd")
    money = fmt_usd(pnl) if pnl is not None else ""
    lines = [f"{symbol} {side}" + (f" · {money}" if money else "")]
    entry = fmt_px(rec.get("entry") or rec.get("entry_price"))
    exit_px = fmt_px(rec.get("exit") or rec.get("exit_price"))
    if entry:
        lines.append(f"Entry {entry}")
    if exit_px:
        lines.append(f"Exit {exit_px}")
    why = str(rec.get("reason") or rec.get("exit_reason") or "").strip()
    if why:
        lines.append(why)
    return [x for x in lines if x]


def _generic_lines(rec: Dict[str, Any]) -> List[str]:
    if rec.get("lines"):
        out = [str(x).strip() for x in rec["lines"] if str(x).strip()]
        if out:
            return out
    human = rec.get("human")
    if isinstance(human, list):
        return [str(x).strip() for x in human if str(x).strip()]
    if isinstance(human, str) and human.strip():
        return [human.strip()]
    reason = str(rec.get("reason") or rec.get("detail") or "").strip()
    return [reason] if reason else []


def humanize_action(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Add emoji, title, lines, clock, layer. Safe on old jsonl rows."""
    out = dict(rec or {})
    kind = str(out.get("kind") or "")
    emoji, title = _default_emoji_title(kind, out)
    if not out.get("emoji"):
        out["emoji"] = emoji
    if not out.get("title"):
        out["title"] = title
    if not out.get("clock"):
        out["clock"] = format_clock(out.get("ts"))
    if looks_like_dev(out) and str(out.get("layer") or "") != "human":
        out["layer"] = "dev"
    else:
        out["layer"] = str(out.get("layer") or "human")
        if out["layer"] != "dev":
            out["layer"] = "human"
    if not out.get("lines"):
        k = kind.lower()
        if k in ("entry", "paper_trade"):
            out["lines"] = _entry_lines(out)
        elif k in ("exit", "close", "flatten", "profit", "loss"):
            out["lines"] = _close_lines(out)
        else:
            out["lines"] = _generic_lines(out)
    else:
        out["lines"] = [str(x) for x in out["lines"] if str(x).strip()]
    if str(kind or "").lower() in ("entry", "paper_trade"):
        out["lines"] = _merge_sl_tp_lines(out["lines"], out)
    return out


def format_human_event(rec: Dict[str, Any]) -> str:
    ev = humanize_action(rec)
    clock = ev.get("clock") or format_clock(ev.get("ts"))
    head = f"{ev.get('emoji') or '🟡'} {clock}  {ev.get('title') or ''}".rstrip()
    body = "\n".join(f"   {line}" for line in (ev.get("lines") or []) if line)
    return head + ("\n" + body if body else "")


def open_position_card(status: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    status = status or {}
    positions = status.get("open_positions") or []
    if not positions:
        return None
    pos = positions[0] if isinstance(positions, list) else positions
    if not isinstance(pos, dict):
        return None
    side = str(pos.get("direction") or pos.get("side") or "").upper() or "TRADE"
    symbol = str(pos.get("symbol") or "MNQ").upper()
    try:
        qty = int(pos.get("size") or pos.get("qty") or 1)
    except (TypeError, ValueError):
        qty = 1
    entry = fmt_px(pos.get("entry") or pos.get("entry_price"))
    paper = bool(status.get("paper_mode") or status.get("overnight_research"))
    overnight = is_overnight_rec(status) or is_overnight_rec(pos)
    cue = str(pos.get("cue") or status.get("last_paper_fill", {}).get("cue") or "")
    card_rec = {
        **pos,
        "session": "overnight_research" if overnight else pos.get("session") or status.get("session"),
        "overnight_research": overnight,
        "entry": pos.get("entry") or pos.get("entry_price"),
        "stop": pos.get("sl") or pos.get("stop"),
        "atr_stop_pts": pos.get("atr_stop_pts") or status.get("atr_stop_pts"),
        "flatten_et": pos.get("flatten_et") or (OVERNIGHT_FLATTEN_ET if overnight else None),
        "paper": True,
    }
    if overnight:
        level_lines = [format_stop_line(card_rec), format_tp_line(card_rec)]
    else:
        live = format_live_entry_brackets(card_rec)
        level_lines = [live] if live else [format_stop_line(card_rec), format_tp_line(card_rec)]
    lines = [
        f"{side} / {symbol} · {qty}",
        f"Entry {entry}" if entry else "",
        *level_lines,
        "Status Open",
    ]
    if paper or cue == "test_fill":
        lines.append("Test: Paper only — not sent to Rithmic")
    return {
        "kind": "paper_trade",
        "layer": "human",
        "emoji": "🟢",
        "title": f"PAPER TRADE — {side}",
        "clock": format_clock(status.get("updated_at")),
        "lines": [x for x in lines if x],
        "symbol": symbol,
        "direction": side.lower(),
        "entry": pos.get("entry") or pos.get("entry_price"),
        "stop": pos.get("sl") or pos.get("stop"),
        "qty": qty,
        "paper": True,
        "live": True,
        "session": card_rec.get("session"),
        "flatten_et": card_rec.get("flatten_et"),
        "ts": status.get("updated_at"),
    }


def _is_today(rec: Dict[str, Any], today) -> bool:
    t = _parse_ts(rec.get("ts"))
    if t is None:
        return True
    return t.astimezone(ET).date() == today


def build_today_activity(
    actions: Sequence[Dict[str, Any]],
    status: Optional[Dict[str, Any]] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Split jsonl + live status into human activity vs raw developer lines."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    today = now.astimezone(ET).date()
    status = status or {}
    human: List[Dict[str, Any]] = []
    developer: List[Dict[str, Any]] = []
    for raw in actions:
        kind = str(raw.get("kind") or "")
        ev = humanize_action(raw)
        if kind in HUMAN_SKIP_KINDS:
            continue
        if ev.get("layer") == "dev" or looks_like_dev(raw):
            developer.append(ev)
            continue
        if not _is_today(ev, today):
            continue
        if str(ev.get("kind") or "").lower() in ("entry", "paper_trade") and status.get("open_positions"):
            pos = status["open_positions"][0] if isinstance(status["open_positions"], list) else {}
            if isinstance(pos, dict):
                ev = dict(ev)
                ev.setdefault("entry", pos.get("entry") or pos.get("entry_price"))
                ev.setdefault("stop", pos.get("sl") or pos.get("stop"))
                ev.setdefault("atr_stop_pts", pos.get("atr_stop_pts"))
                if is_overnight_rec(status) or is_overnight_rec(ev):
                    ev.setdefault("flatten_et", pos.get("flatten_et") or OVERNIGHT_FLATTEN_ET)
                    ev.setdefault("session", status.get("session") or "overnight_research")
                ev["lines"] = _merge_sl_tp_lines(ev.get("lines") or [], ev)
        human.append(ev)

    card = open_position_card(status)
    last_px = status.get("last_price")
    open_label = str(status.get("open") or "").strip()
    if card and last_px not in (None, ""):
        side_word = (_side(card) or open_label or "trade").capitalize()
        pos = (status.get("open_positions") or [None])[0] or {}
        overnight = is_overnight_rec(status) or is_overnight_rec(pos)
        extra = []
        if overnight:
            extra.append(format_stop_line({
                **pos,
                "session": "overnight_research",
                "entry": pos.get("entry") or pos.get("entry_price"),
                "stop": pos.get("sl") or pos.get("stop"),
                "atr_stop_pts": pos.get("atr_stop_pts"),
            }))
            extra.append(format_tp_line({**pos, "session": "overnight_research"}))
        else:
            extra.append(format_live_entry_brackets({
                **pos,
                "stop": pos.get("sl") or pos.get("stop"),
                "tp": pos.get("tp") or pos.get("tp_cap"),
            }))
        monitor = {
            "kind": "monitoring",
            "layer": "human",
            "emoji": "🟡",
            "title": "MONITORING",
            "clock": format_clock(status.get("updated_at") or now),
            "ts": status.get("updated_at") or now.isoformat(),
            "lines": [x for x in [f"{side_word} still open · MNQ {fmt_px(last_px)}", *extra] if x],
            "live": True,
        }
        if not human or human[-1].get("kind") != "monitoring":
            human.append(monitor)

    return {
        "card": card,
        "events": human,
        "developer": developer[-120:],
    }


class ActionLog:
    def __init__(self, path: str = ACTION_LOG_PATH, max_rows: int = MAX_ACTIONS):
        self.path = path
        self.max_rows = max_rows
        self._lock = threading.RLock()
        self._last_emit: Dict[str, float] = {}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def record(self, kind: str, **fields: Any) -> Dict[str, Any]:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        try:
            rec = humanize_action(rec)
        except Exception:
            pass
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return rec

    def record_throttled(
        self,
        kind: str,
        *,
        key: str,
        interval_sec: float = MONITOR_INTERVAL_SEC,
        **fields: Any,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        stamp_key = f"{kind}:{key}"
        with self._lock:
            last = self._last_emit.get(stamp_key, 0.0)
            if now - last < float(interval_sec):
                return None
            self._last_emit[stamp_key] = now
        return self.record(kind, **fields)

    def recent(self, n: int = 80) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-n:]


def write_status(payload: Dict[str, Any], path: str = STATUS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def read_status(path: str = STATUS_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def read_journal(path: str = "data/trade_journal.jsonl", n: int = 40) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-n:]
