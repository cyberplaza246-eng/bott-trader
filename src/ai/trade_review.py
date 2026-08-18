"""
Append-only trade journal + deterministic WHY + advisory suggestions.

Never writes strategy params, mnq_profit_config, or live defaults.
Suggestions are for the operator to apply by hand.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.strategy.mnq_15m_ema_eod import (
    ET,
    capture_market_snapshot,
    window_label,
)

JOURNAL_LIVE = "data/trade_journal.jsonl"
JOURNAL_PAPER = "data/paper_trade_journal.jsonl"
SUGGESTIONS_PATH = "data/trade_suggestions.json"

EXIT_SL = "SL"
EXIT_EOD = "EOD 15:50"
EXIT_TP = "TP cap"
EXIT_REJECTED = "rejected"
EXIT_OTHER = "other"

# Desk labels for every completed trade (Stop / EOD flatten / TP cap).
DESK_EXIT_STOP = "Stop"
DESK_EXIT_EOD = "EOD flatten"
DESK_EXIT_TP = "TP cap"

_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def journal_path(paper_mode: bool = False) -> str:
    return JOURNAL_PAPER if paper_mode else JOURNAL_LIVE


def suggest_min_trades() -> int:
    return max(1, int(os.getenv("SUGGEST_MIN_TRADES", "30")))


def suggest_min_losers() -> int:
    return max(1, int(os.getenv("SUGGEST_MIN_LOSERS", "10")))


def suggest_min_cluster() -> int:
    return max(3, int(os.getenv("SUGGEST_MIN_CLUSTER", "5")))


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def ts_et_iso(ts=None) -> str:
    t = _as_utc(ts or datetime.now(timezone.utc))
    return t.tz_convert(ET).isoformat()


def side_int(side: str) -> int:
    s = (side or "").lower()
    if s in ("long", "buy"):
        return 1
    if s in ("short", "sell"):
        return -1
    return 0


def normalize_ema15_exit_reason(reason: str) -> str:
    """Canonical exit: SL / EOD 15:50 / TP cap / rejected / other."""
    r = (reason or "").strip().upper().replace(" ", "_")
    if r in ("REJECTED", "REJECT", "ORDER_REJECTED"):
        return EXIT_REJECTED
    if r in ("SL", "MAX_LOSS", "STOP", "STOP_LOSS", "ATR_STOP"):
        return EXIT_SL
    if r in ("EOD_FLATTEN", "EOD_1550", "EOD", "FLAT_1550", "EMA15_EOD", "FLAT") or "1550" in r or "15:50" in r:
        return EXIT_EOD
    if r in ("SESSION_END",) or r.startswith("SESSION"):
        return EXIT_EOD
    if r in ("TP", "TAKE_PROFIT", "TP_CAP", "CAP"):
        return EXIT_TP
    if r in ("BE",):
        return EXIT_OTHER
    if not r:
        return EXIT_OTHER
    return EXIT_OTHER


def desk_exit_reason(reason: str) -> str:
    """Operator-facing exit: Stop / EOD flatten / TP cap."""
    r = str(reason or "").strip()
    if r in (DESK_EXIT_STOP, DESK_EXIT_EOD, DESK_EXIT_TP, EXIT_REJECTED, EXIT_OTHER):
        return r
    n = normalize_ema15_exit_reason(r)
    if n == EXIT_SL:
        return DESK_EXIT_STOP
    if n == EXIT_EOD:
        return DESK_EXIT_EOD
    if n == EXIT_TP:
        return DESK_EXIT_TP
    return n


def compute_mae_mfe_pts(
    df_1m: Optional[pd.DataFrame],
    entry_ts,
    exit_ts,
    side: str,
    entry_price: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Max adverse / max favorable excursion in points from 1m bars in [entry, exit]."""
    if df_1m is None or getattr(df_1m, "empty", True) or not entry_price:
        return None, None
    tmp = df_1m
    if "datetime" in tmp.columns:
        dt = pd.to_datetime(tmp["datetime"], utc=True)
    elif isinstance(tmp.index, pd.DatetimeIndex):
        idx = tmp.index
        dt = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    else:
        return None, None
    t0 = _as_utc(entry_ts)
    t1 = _as_utc(exit_ts)
    if t1 < t0:
        t0, t1 = t1, t0
    mask = (dt >= t0) & (dt <= t1)
    if not mask.any():
        return None, None
    bars = tmp.loc[mask.values if hasattr(mask, "values") else mask]
    if bars.empty or "high" not in bars.columns or "low" not in bars.columns:
        return None, None
    hi = float(bars["high"].max())
    lo = float(bars["low"].min())
    si = side_int(side)
    if si >= 0:
        mae = max(0.0, float(entry_price) - lo)
        mfe = max(0.0, hi - float(entry_price))
    else:
        mae = max(0.0, hi - float(entry_price))
        mfe = max(0.0, float(entry_price) - lo)
    return round(mae, 2), round(mfe, 2)


def _num(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None or val == "":
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if pd.isna(f):
        return default
    return f


def classify_why(record: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic post-mortem from snapshots + MAE/MFE. No LLM."""
    pnl = _num(record.get("pnl_usd"), 0.0) or 0.0
    pts = _num(record.get("pts"), 0.0) or 0.0
    if pnl > 0 or pts > 0:
        outcome = "win"
    elif pnl < 0 or pts < 0:
        outcome = "lose"
    else:
        outcome = "flat"

    exit_reason = normalize_ema15_exit_reason(str(record.get("exit_reason") or ""))
    if record.get("exit_reason") in (EXIT_SL, EXIT_EOD, EXIT_TP, EXIT_REJECTED, EXIT_OTHER):
        exit_reason = str(record.get("exit_reason"))

    entry_snap = record.get("entry_snapshot") or {}
    exit_snap = record.get("exit_snapshot") or {}
    side = side_int(str(record.get("side") or record.get("direction") or ""))
    mae = _num(record.get("mae_pts"))
    mfe = _num(record.get("mfe_pts"))
    sl_pts = (
        _num(record.get("atr_stop_pts"))
        or _num(record.get("sl_pts"))
        or _num(entry_snap.get("atr_stop_pts"))
        or 0.0
    )
    sl_pts = float(sl_pts or 0.0)
    mins_into = entry_snap.get("minutes_into_window")
    try:
        mins_into = int(mins_into) if mins_into is not None else None
    except (TypeError, ValueError):
        mins_into = None

    trend15_exit = int(exit_snap.get("trend_15m") or 0)
    daily_agree = bool(entry_snap.get("daily_agree"))
    tf15_exit_agree = bool(side and trend15_exit == side)
    tf60_agree = bool(entry_snap.get("tf60_agree"))
    prior_close = _num(entry_snap.get("prior_rth_close"))
    daily_ema = _num(entry_snap.get("daily_ema20"))
    atr15 = _num(entry_snap.get("atr15")) or _num(exit_snap.get("atr15"))
    daily_dist = None
    if prior_close is not None and daily_ema is not None:
        daily_dist = abs(prior_close - daily_ema)
    daily_weak = (not daily_agree) or (
        daily_dist is not None and atr15 and atr15 > 0 and (daily_dist / atr15) < 0.20
    )

    facts: List[str] = []
    facts.append(f"exit={exit_reason} pts={pts:+.2f} ${pnl:+.2f}")
    if sl_pts:
        facts.append(f"ATR stop {sl_pts:.1f} pts")
    if mae is not None:
        facts.append(f"MAE {mae:.1f} pts")
    if mfe is not None:
        facts.append(f"MFE {mfe:.1f} pts")
    r_mult = _num(record.get("r_multiple"))
    if r_mult is not None:
        facts.append(f"R-multiple {r_mult:+.2f}")
    if entry_snap.get("window_name"):
        into = f", {mins_into} min into window" if mins_into is not None else ""
        facts.append(f"window {entry_snap.get('window_name')}{into}")
    facts.append(
        f"entry align daily={bool(entry_snap.get('daily_agree'))} "
        f"15m={bool(entry_snap.get('tf15_agree'))} "
        f"60m={bool(entry_snap.get('tf60_agree'))} "
        f"2-lot={bool(entry_snap.get('high_confidence'))}"
    )
    if daily_dist is not None:
        facts.append(
            f"prior RTH close {prior_close:.2f} vs daily EMA20 {daily_ema:.2f} "
            f"(gap {daily_dist:.1f} pts)"
        )
    if exit_snap:
        facts.append(
            f"exit 15m trend={exit_snap.get('trend_15m')} "
            f"60m={exit_snap.get('trend_60m')} "
            f"daily={exit_snap.get('daily_permission')}"
        )
    hold = _num(record.get("hold_minutes"))
    if hold is not None:
        facts.append(f"held {hold:.1f} min")

    primary = f"closed via {exit_reason}"
    if exit_reason == EXIT_REJECTED:
        primary = "order rejected before fill"
    elif outcome == "lose" and trend15_exit and side and trend15_exit != side:
        primary = "15m flipped after entry"
        facts.append(
            f"15m EMA8/21 at exit was {trend15_exit:+d} vs trade side {side:+d}"
        )
    elif (
        exit_reason == EXIT_SL
        and sl_pts > 0
        and mae is not None
        and mae >= sl_pts * 0.85
        and mfe is not None
        and mfe >= sl_pts * 0.35
    ):
        primary = "MAE hit ATR stop then reversed"
        facts.append(
            f"MFE {mfe:.1f} then MAE {mae:.1f} reached {sl_pts:.1f}-pt ATR stop"
        )
    elif exit_reason == EXIT_SL:
        primary = "stopped before trend continued"
        if tf15_exit_agree:
            facts.append("15m still agreed at exit; stop hit first")
        if mfe is not None and sl_pts and mfe < sl_pts * 0.35:
            facts.append(f"MFE only {mfe:.1f} vs {sl_pts:.1f}-pt stop")
    elif exit_reason == EXIT_TP:
        primary = "TP cap hit"
        facts.append("500-pt TP cap filled (rare; recipe is hold-to-EOD)")
    elif exit_reason == EXIT_EOD and outcome == "win":
        primary = "EOD flattened in profit"
        facts.append("flattened at 15:50 ET with positive P&L (TP cap unused)")
    elif exit_reason == EXIT_EOD and outcome == "lose":
        primary = "EOD flattened in loss"
        facts.append("flattened at 15:50 ET still underwater")
    elif outcome == "lose" and daily_weak:
        primary = "daily permission weak"
        if not daily_agree:
            facts.append("daily permission did not agree with trade side at entry")
        elif daily_dist is not None and atr15:
            facts.append(
                f"daily gap {daily_dist:.1f} pts is < 0.20× ATR15 {atr15:.1f}"
            )
    elif (
        outcome != "win"
        and mins_into is not None
        and mins_into >= 15
        and sl_pts > 0
        and mae is not None
        and mfe is not None
        and mae < sl_pts * 0.60
        and mfe < sl_pts * 0.60
    ):
        primary = "entered late in window / chop"
        facts.append(
            f"entered {mins_into} min into window; MAE {mae:.1f} and MFE {mfe:.1f} "
            f"both < 0.60× stop"
        )
    elif outcome == "lose" and not tf60_agree:
        primary = "60m disagreed at entry"
        facts.append("60m EMA8/21 did not agree with trade side")
    elif outcome == "flat":
        primary = f"flat close via {exit_reason}"

    return {
        "outcome": outcome,
        "primary_reason": primary,
        "facts": facts,
        "exit_reason": exit_reason,
    }


def _pnl_list(rows: Iterable[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = _num(r.get("pnl_usd"), r.get("pnl"))
        if v is None:
            continue
        out.append(float(v))
    return out


def profit_factor(pnls: Iterable[float]) -> Optional[float]:
    vals = list(pnls)
    wins = sum(p for p in vals if p > 0)
    losses = sum(-p for p in vals if p < 0)
    if losses <= 0:
        return 99.0 if wins > 0 else None
    return round(wins / losses, 3)


def _mae_bucket(mae: Optional[float]) -> str:
    if mae is None:
        return "mae_unknown"
    if mae < 10:
        return "mae_0_10"
    if mae < 20:
        return "mae_10_20"
    if mae < 40:
        return "mae_20_40"
    return "mae_40_plus"


def _hold_bucket(minutes: Optional[float]) -> str:
    if minutes is None:
        return "hold_unknown"
    if minutes < 30:
        return "hold_0_30m"
    if minutes < 120:
        return "hold_30_120m"
    if minutes < 300:
        return "hold_2_5h"
    return "hold_5h_plus"


def _entry_hour_et(row: Dict[str, Any]) -> Optional[int]:
    snap = row.get("entry_snapshot") or {}
    ts = snap.get("ts_et") or row.get("entry_ts_et") or row.get("timestamp_et")
    if not ts:
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(ET)
        return int(t.tz_convert(ET).hour)
    except Exception:
        return None


def _dow_et(row: Dict[str, Any]) -> Optional[str]:
    snap = row.get("entry_snapshot") or {}
    ts = snap.get("ts_et") or row.get("entry_ts_et") or row.get("timestamp_et")
    if not ts:
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(ET)
        return _DOW[int(t.tz_convert(ET).weekday())]
    except Exception:
        return None


def _cluster_key_value(row: Dict[str, Any], kind: str) -> Optional[str]:
    snap = row.get("entry_snapshot") or {}
    if kind == "window":
        return str(snap.get("window_name") or row.get("window_name") or "none")
    if kind == "hour":
        h = _entry_hour_et(row)
        return None if h is None else f"{h:02d}:00"
    if kind == "exit_reason":
        return normalize_ema15_exit_reason(str(row.get("exit_reason") or ""))
    if kind == "mae_bucket":
        return _mae_bucket(_num(row.get("mae_pts")))
    if kind == "tf60_disagree":
        return "60m_disagree" if not snap.get("tf60_agree") else "60m_agree"
    if kind == "hold_time":
        return _hold_bucket(_num(row.get("hold_minutes")))
    if kind == "day_of_week":
        return _dow_et(row)
    return None


def _tweak_for(kind: str, bucket: str, n: int, pf: Optional[float], overall: Optional[float]) -> str:
    pf_s = "n/a" if pf is None else f"{pf:.2f}"
    ov_s = "n/a" if overall is None else f"{overall:.2f}"
    worse = pf is not None and overall is not None and pf < overall
    if kind == "window":
        if worse:
            return f"skip {bucket} window: {n} trades PF {pf_s} vs overall {ov_s}"
        return f"keep {bucket} window: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "hour":
        if worse:
            return f"avoid new entries near {bucket} ET: {n} trades PF {pf_s} vs overall {ov_s}"
        return f"{bucket} ET holds up: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "exit_reason":
        if bucket == EXIT_SL and worse:
            return (
                f"SL exits are dragging PF ({n} trades PF {pf_s} vs overall {ov_s}) — "
                "consider a wider ATR multiple only if you choose to change it"
            )
        if bucket == EXIT_EOD:
            return f"EOD 15:50 exits: {n} trades PF {pf_s} vs overall {ov_s} (hold-to-close is the recipe)"
        if bucket == EXIT_TP:
            return f"TP cap exits: {n} trades PF {pf_s} vs overall {ov_s} (cap only — do not tighten TP)"
        return f"{bucket} exits: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "mae_bucket":
        if worse:
            return (
                f"trades with {bucket.replace('_', ' ')}: {n} PF {pf_s} vs overall {ov_s} — "
                "review stop distance vs typical MAE"
            )
        return f"{bucket.replace('_', ' ')}: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "tf60_disagree":
        if bucket == "60m_disagree" and worse:
            return (
                f"skip when 60m disagrees: {n} trades PF {pf_s} vs overall {ov_s} "
                "(2-lot already requires 60m agree; you could require it for lot 1)"
            )
        return f"{bucket.replace('_', ' ')}: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "hold_time":
        if worse:
            return f"weak {bucket.replace('_', ' ')} holds: {n} trades PF {pf_s} vs overall {ov_s}"
        return f"{bucket.replace('_', ' ')}: {n} trades PF {pf_s} vs overall {ov_s}"
    if kind == "day_of_week":
        if worse:
            return f"skip {bucket}: {n} trades PF {pf_s} vs overall {ov_s}"
        return f"{bucket} is fine: {n} trades PF {pf_s} vs overall {ov_s}"
    return f"{kind}={bucket}: {n} trades PF {pf_s} vs overall {ov_s}"


def build_suggestions(
    closes: List[Dict[str, Any]],
    *,
    min_trades: Optional[int] = None,
    min_losers: Optional[int] = None,
    min_cluster: Optional[int] = None,
) -> Dict[str, Any]:
    """Cluster closed trades. Never mutates live config. Operator applies tweaks."""
    min_trades = suggest_min_trades() if min_trades is None else int(min_trades)
    min_losers = suggest_min_losers() if min_losers is None else int(min_losers)
    min_cluster = suggest_min_cluster() if min_cluster is None else int(min_cluster)
    now = datetime.now(timezone.utc)
    rows = [r for r in closes if str(r.get("event") or "close") == "close"]
    rows = [r for r in rows if normalize_ema15_exit_reason(str(r.get("exit_reason") or "")) != EXIT_REJECTED]
    pnls = _pnl_list(rows)
    n = len(rows)
    n_losers = sum(1 for p in pnls if p < 0)
    overall_pf = profit_factor(pnls)
    overall_pnl = round(sum(pnls), 2) if pnls else 0.0
    payload: Dict[str, Any] = {
        "generated_at": now.isoformat(),
        "generated_at_et": ts_et_iso(now),
        "n_closes": n,
        "n_losers": n_losers,
        "overall_pf": overall_pf,
        "overall_pnl_usd": overall_pnl,
        "min_trades_required": min_trades,
        "min_losers_required": min_losers,
        "advisory_only": True,
        "auto_apply": False,
        "writes_live_config": False,
        "suggestions": [],
        "ready": False,
        "note": "",
    }
    if n < min_trades or n_losers < min_losers:
        need_t = max(0, min_trades - n)
        need_l = max(0, min_losers - n_losers)
        payload["note"] = (
            f"Need {min_trades} closed trades and {min_losers} losers before suggestions "
            f"({n}/{min_trades} closes, {n_losers}/{min_losers} losers). "
            "Bot will not auto-tweak."
        )
        payload["need_trades"] = need_t
        payload["need_losers"] = need_l
        return payload

    payload["ready"] = True
    payload["note"] = "Advisory only — apply any tweak yourself. Bot will not auto-tweak."

    kinds = (
        "window",
        "hour",
        "exit_reason",
        "mae_bucket",
        "tf60_disagree",
        "hold_time",
        "day_of_week",
    )
    scored: List[Dict[str, Any]] = []
    for kind in kinds:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = _cluster_key_value(row, kind)
            if not key:
                continue
            buckets.setdefault(key, []).append(row)
        for bucket, group in buckets.items():
            if len(group) < min_cluster:
                continue
            g_pnls = _pnl_list(group)
            pf = profit_factor(g_pnls)
            wins = sum(1 for p in g_pnls if p > 0)
            losses = sum(1 for p in g_pnls if p < 0)
            delta = 0.0
            if pf is not None and overall_pf is not None:
                delta = overall_pf - pf
            scored.append({
                "cluster": kind,
                "bucket": bucket,
                "n": len(group),
                "wins": wins,
                "losses": losses,
                "pf": pf,
                "overall_pf": overall_pf,
                "pnl_usd": round(sum(g_pnls), 2),
                "delta_pf": round(delta, 3),
                "tweak": _tweak_for(kind, bucket, len(group), pf, overall_pf),
                "user_applies": True,
            })

    scored.sort(key=lambda s: (abs(float(s.get("delta_pf") or 0)), int(s["n"])), reverse=True)
    # Prefer weaker-than-overall buckets first, then strong keeps.
    weak = [s for s in scored if (s.get("pf") is not None and overall_pf is not None and s["pf"] < overall_pf)]
    strong = [s for s in scored if s not in weak]
    ordered = weak + strong
    seen = set()
    picked: List[Dict[str, Any]] = []
    for s in ordered:
        key = (s["cluster"], s["bucket"])
        if key in seen:
            continue
        seen.add(key)
        s = dict(s)
        s["rank"] = len(picked) + 1
        picked.append(s)
        if len(picked) >= 8:
            break
    payload["suggestions"] = picked[:8]
    if len(payload["suggestions"]) < 3 and scored:
        payload["suggestions"] = [{**s, "rank": i + 1} for i, s in enumerate(ordered[:8])]
    return payload


def write_suggestions(payload: Dict[str, Any], path: str = SUGGESTIONS_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def read_suggestions(path: str = SUGGESTIONS_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_journal_rows(path: str) -> List[Dict[str, Any]]:
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


def load_closes(path: str) -> List[Dict[str, Any]]:
    """ema15 closes only — overnight_research rows stay out of RTH stats."""
    return [
        r for r in load_journal_rows(path)
        if str(r.get("event") or "close") == "close"
        and str(r.get("session") or "").lower() != "overnight_research"
    ]


class TradeReviewJournal:
    """Append-only JSONL for ema15 entries/closes. Suggestion-only — no live writes."""

    def __init__(self, paper_mode: bool = False, path: Optional[str] = None):
        self.paper_mode = paper_mode
        self.path = path or journal_path(paper_mode)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def _append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        rec.setdefault("timestamp_et", ts_et_iso(rec["timestamp"]))
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return rec

    def log_entry(
        self,
        *,
        trade_id: str,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        snapshot: Dict[str, Any],
        sl_pts: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        win = snapshot.get("window")
        return self._append({
            "event": "entry",
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "direction": side,
            "qty": qty,
            "window": win,
            "window_name": snapshot.get("window_name") or window_label(win),
            "entry_price": entry_price,
            "atr_stop_pts": sl_pts or snapshot.get("atr_stop_pts"),
            "entry_snapshot": snapshot,
            "alignment": {
                "daily_agree": bool(snapshot.get("daily_agree")),
                "tf15_agree": bool(snapshot.get("tf15_agree")),
                "tf60_agree": bool(snapshot.get("tf60_agree")),
                "high_confidence": bool(snapshot.get("high_confidence")),
            },
        })

    def log_close(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec["event"] = "close"
        rec["exit_reason"] = normalize_ema15_exit_reason(str(rec.get("exit_reason") or ""))
        why = classify_why(rec)
        rec["why"] = why
        rec["outcome"] = why["outcome"]
        rec["win"] = why["outcome"] == "win"
        rec["primary_reason"] = why["primary_reason"]
        rec["reason"] = why["primary_reason"]
        rec["pnl"] = rec.get("pnl_usd")
        try:
            from src.ai.window_edge import attach_desk_fields
            rec = attach_desk_fields(rec)
        except Exception:
            rec["desk_exit_reason"] = desk_exit_reason(rec["exit_reason"])
        return self._append(rec)

    def log_rejected(
        self,
        *,
        trade_id: str,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        snapshot: Optional[Dict[str, Any]] = None,
        detail: str = "",
    ) -> Dict[str, Any]:
        snap = snapshot or {}
        rec = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "direction": side,
            "qty": qty,
            "window_name": snap.get("window_name"),
            "entry_price": entry_price,
            "exit_price": entry_price,
            "pts": 0.0,
            "pnl_usd": 0.0,
            "hold_minutes": 0.0,
            "exit_reason": EXIT_REJECTED,
            "entry_snapshot": snap,
            "exit_snapshot": snap,
            "mae_pts": None,
            "mfe_pts": None,
            "reject_detail": detail,
        }
        return self.log_close(rec)

    def closed_trades(self) -> List[Dict[str, Any]]:
        return load_closes(self.path)

    def refresh_suggestions(self, *, force: bool = False) -> Dict[str, Any]:
        closes = self.closed_trades()
        n = len(closes)
        if not force and n < suggest_min_trades():
            payload = build_suggestions(closes)
            write_suggestions(payload)
            return payload
        payload = build_suggestions(closes)
        write_suggestions(payload)
        return payload


def snapshot_from_df(
    df_1m: Optional[pd.DataFrame],
    *,
    side: str,
    atr_stop_pts: Optional[float] = None,
    window: Optional[int] = None,
    now=None,
) -> Dict[str, Any]:
    return capture_market_snapshot(
        df_1m if df_1m is not None else pd.DataFrame(),
        now=now,
        side=side_int(side),
        atr_stop_pts=atr_stop_pts,
        window=window,
    )
