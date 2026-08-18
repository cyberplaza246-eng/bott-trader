"""
Overnight paper journal, zone file, and advisory suggestions.

Never writes mnq_profit_config, ENTRY_WINDOWS, stops, or live defaults.
Gemini notes are stored on the close record. Suggestions are never auto-applied.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ai.trade_review import (
    compute_mae_mfe_pts,
    profit_factor,
    ts_et_iso,
)
from src.strategy.overnight_research import SESSION_NAME

JOURNAL_PATH = "data/paper_trade_journal.jsonl"
ZONES_PATH = "data/overnight_zones.json"
SUGGESTIONS_PATH = "data/overnight_suggestions.json"
MNQ_PROFIT_CONFIG = "data/mnq_profit_config.json"

SUGGEST_MIN_OVERNIGHT = 20


def overnight_suggest_min() -> int:
    return max(5, int(os.getenv("OVERNIGHT_SUGGEST_MIN_TRADES", str(SUGGEST_MIN_OVERNIGHT))))


def is_overnight_research_row(row: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get("session") or "").lower() == SESSION_NAME


def load_jsonl(path: str) -> List[Dict[str, Any]]:
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
    return rows


def load_overnight_closes(path: str = JOURNAL_PATH) -> List[Dict[str, Any]]:
    return [
        r for r in load_jsonl(path)
        if is_overnight_research_row(r) and str(r.get("event") or "close") == "close"
    ]


def read_zones(path: str = ZONES_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_zones(payload: Dict[str, Any], path: str = ZONES_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(payload)
    body.setdefault("session", SESSION_NAME)
    body["writes_live_config"] = False
    body["auto_apply"] = False
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def read_overnight_suggestions(path: str = SUGGESTIONS_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_overnight_suggestions(payload: Dict[str, Any], path: str = SUGGESTIONS_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(payload)
    body["advisory_only"] = True
    body["auto_apply"] = False
    body["writes_live_config"] = False
    body["writes_mnq_profit_config"] = False
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def build_overnight_suggestions(
    closes: List[Dict[str, Any]],
    *,
    min_trades: Optional[int] = None,
) -> Dict[str, Any]:
    min_trades = overnight_suggest_min() if min_trades is None else int(min_trades)
    rows = [r for r in closes if is_overnight_research_row(r)]
    n = len(rows)
    pnls = [float(r.get("pnl_usd") or 0) for r in rows]
    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_at_et": ts_et_iso(),
        "session": SESSION_NAME,
        "n_closes": n,
        "overall_pf": profit_factor(pnls) if pnls else None,
        "overall_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
        "min_trades_required": min_trades,
        "advisory_only": True,
        "auto_apply": False,
        "writes_live_config": False,
        "writes_mnq_profit_config": False,
        "suggestions": [],
        "ready": False,
        "note": "",
    }
    if n < min_trades:
        payload["note"] = (
            f"Need {min_trades} overnight paper closes before suggestions "
            f"({n}/{min_trades}). Never auto-applied. Live RTH rules unchanged."
        )
        payload["need_trades"] = max(0, min_trades - n)
        return payload

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('cue') or 'unknown'}|{row.get('zone_name') or 'unknown'}"
        buckets.setdefault(key, []).append(row)
    scored: List[Dict[str, Any]] = []
    overall_pf = payload["overall_pf"]
    for key, group in buckets.items():
        if len(group) < 3:
            continue
        g_pnls = [float(r.get("pnl_usd") or 0) for r in group]
        cue, zone = key.split("|", 1)
        pf = profit_factor(g_pnls)
        scored.append({
            "cluster": "cue_zone",
            "cue": cue,
            "zone_name": zone,
            "n": len(group),
            "pf": pf,
            "pnl_usd": round(sum(g_pnls), 2),
            "tweak": (
                f"{cue} at {zone}: {len(group)} paper trades, PF "
                f"{'n/a' if pf is None else f'{pf:.2f}'} — review by hand; do not change live windows/stops"
            ),
            "user_applies": True,
        })
    scored.sort(key=lambda s: abs(float(s.get("pnl_usd") or 0)), reverse=True)
    payload["ready"] = True
    payload["suggestions"] = [{**s, "rank": i + 1} for i, s in enumerate(scored[:8])]
    payload["note"] = (
        "Advisory only — apply any tweak yourself. Bot will not write mnq_profit_config "
        "or change locked RTH ENTRY_WINDOWS / stops."
    )
    payload["overall_pf"] = overall_pf
    return payload


class OvernightPaperJournal:
    """Append-only paper JSONL tagged session=overnight_research."""

    def __init__(self, path: str = JOURNAL_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def _append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec["session"] = SESSION_NAME
        rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        rec.setdefault("timestamp_et", ts_et_iso(rec["timestamp"]))
        rec["writes_live_config"] = False
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return rec

    def log_entry(self, signal: Dict[str, Any], *, trade_id: str) -> Dict[str, Any]:
        return self._append({
            "event": "entry",
            "trade_id": trade_id,
            "symbol": signal.get("symbol", "MNQ"),
            "side": signal.get("side"),
            "direction": signal.get("direction") or signal.get("side"),
            "qty": int(signal.get("qty") or 1),
            "entry_price": signal.get("entry_price"),
            "stop": signal.get("stop"),
            "target": signal.get("target"),
            "atr_stop_pts": signal.get("atr_stop_pts"),
            "cue": signal.get("cue"),
            "zone_name": signal.get("zone_name"),
            "zone_price": signal.get("zone_price"),
            "why": signal.get("why"),
            "flatten_et": signal.get("flatten_et") or "09:25",
            "paper": True,
        })

    def log_close(self, record: Dict[str, Any], *, gemini_note: str = "") -> Dict[str, Any]:
        rec = dict(record)
        rec["event"] = "close"
        rec["session"] = SESSION_NAME
        rec["paper"] = True
        why = rec.get("why")
        if not isinstance(why, dict):
            why = {"primary_reason": str(why or rec.get("exit_reason") or ""), "facts": []}
        if gemini_note:
            why["gemini_note"] = gemini_note[:400]
        rec["why"] = why
        rec["gemini_note"] = (gemini_note or "")[:400]
        rec.setdefault("primary_reason", why.get("primary_reason"))
        return self._append(rec)

    def closed_trades(self) -> List[Dict[str, Any]]:
        return load_overnight_closes(self.path)

    def refresh_suggestions(self) -> Dict[str, Any]:
        payload = build_overnight_suggestions(self.closed_trades())
        write_overnight_suggestions(payload)
        return payload


def attach_mae_mfe(
    rec: Dict[str, Any],
    df_1m,
    entry_ts,
    exit_ts,
    side: str,
    entry_price: float,
) -> Dict[str, Any]:
    mae, mfe = compute_mae_mfe_pts(df_1m, entry_ts, exit_ts, side, entry_price)
    rec["mae_pts"] = mae
    rec["mfe_pts"] = mfe
    return rec
