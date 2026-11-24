"""
Closed-trade desk schema + window/side/alignment expectancy.

Never writes mnq_profit_config, ENTRY_WINDOWS, stops, or MAX_TRADES.
Locked 3-window recipe (9:35 / 11:00 / 13:30, max 3, no afternoon quality)
is simulated here so the desk can show backtest truth even if live research
is still trying extra slots.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.ai.trade_review import (
    EXIT_EOD,
    EXIT_SL,
    EXIT_TP,
    _num,
    compute_mae_mfe_pts,
    desk_exit_reason,
    profit_factor,
    side_int,
    ts_et_iso,
)
from src.strategy.mnq_15m_ema_eod import (
    ET,
    FLAT_MINUTE,
    SEP_MIN,
    atr15_on_1m,
    daily_trend_on_1m,
    high_confidence,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_label,
)

WINDOW_EDGE_PATH = "data/window_edge_analysis.json"
POINT_VALUE = 2.0
COMMISSION = 1.24
OOS_START = pd.Timestamp("2026-06-01", tz="UTC")

# Systems-review locked recipe. Do not assign these onto live ENTRY_WINDOWS.
LOCKED_WINDOWS: Tuple[Tuple[int, int], ...] = (
    (9 * 60 + 35, 10 * 60),
    (11 * 60, 11 * 60 + 30),
    (13 * 60 + 30, 14 * 60),
)
LOCKED_MAX_TRADES = 3
CANON_WINDOWS = ("09:35", "11:00", "13:30")

MFE_BUCKETS = (
    (0, 15, "0-15"),
    (15, 20, "15-20"),
    (20, 40, "20-40"),
    (40, 80, "40-80"),
    (80, 120, "80-120"),
    (120, 200, "120-200"),
    (200, None, "200+"),
)

TREND_TXT = {1: "up", -1: "down", 0: "flat"}


def locked_window_index(et_minutes: int) -> Optional[int]:
    for i, (start, end) in enumerate(LOCKED_WINDOWS):
        if start <= et_minutes < end:
            return i
    return None


def locked_window_name(win: Optional[int]) -> str:
    if win is None or win < 0 or win >= len(CANON_WINDOWS):
        return "none"
    return CANON_WINDOWS[win]


def _median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return round(float(np.median(np.asarray(vals, dtype=float))), 2)


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return round(float(np.mean(np.asarray(vals, dtype=float))), 2)


def mfe_bucket(mfe: Optional[float]) -> str:
    if mfe is None:
        return "unknown"
    x = float(mfe)
    for lo, hi, name in MFE_BUCKETS:
        if hi is None:
            if x >= lo:
                return name
        elif lo <= x < hi:
            return name
    return "unknown"


def _trend_int(val: Any) -> int:
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return 0


def attach_desk_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten journal/backtest close into the desk per-trade schema."""
    rec = dict(record)
    snap = rec.get("entry_snapshot") or {}
    align = rec.get("alignment") or {}
    side = str(rec.get("direction") or rec.get("side") or "")
    si = side_int(side)
    trend15 = _trend_int(rec.get("trend_15m") if rec.get("trend_15m") is not None else snap.get("trend_15m"))
    trend60 = _trend_int(rec.get("trend_60m") if rec.get("trend_60m") is not None else snap.get("trend_60m"))
    daily = _trend_int(
        rec.get("daily_trend")
        if rec.get("daily_trend") is not None
        else snap.get("daily_permission")
    )
    stop = (
        _num(rec.get("stop_distance"))
        or _num(rec.get("atr_stop_pts"))
        or _num(rec.get("sl_pts"))
        or _num(snap.get("atr_stop_pts"))
    )
    atr = _num(rec.get("atr_at_entry")) or _num(snap.get("atr15")) or _num(rec.get("atr"))
    win_name = rec.get("window_name") or snap.get("window_name")
    if not win_name or win_name == "none":
        win_name = window_label(rec.get("window") if rec.get("window") is not None else snap.get("window"))
    daily_agree = bool(align.get("daily_agree", snap.get("daily_agree", si and daily == si)))
    tf15_agree = bool(align.get("tf15_agree", snap.get("tf15_agree", si and trend15 == si)))
    tf60_agree = bool(align.get("tf60_agree", snap.get("tf60_agree", si and trend60 == si)))
    aligned = bool(si and trend15 == si and trend60 == si and daily == si)
    rec["direction"] = side.lower() if side else ""
    rec["side"] = rec["direction"]
    rec["entry_ts_et"] = rec.get("entry_ts_et") or snap.get("ts_et")
    rec["exit_ts_et"] = rec.get("exit_ts_et") or (rec.get("exit_snapshot") or {}).get("ts_et")
    rec["stop_distance"] = None if stop is None else round(float(stop), 2)
    rec["atr_stop_pts"] = rec.get("atr_stop_pts") or rec["stop_distance"]
    rec["atr_at_entry"] = None if atr is None else round(float(atr), 2)
    rec["trend_15m"] = trend15
    rec["trend_60m"] = trend60
    rec["daily_trend"] = daily
    rec["trend_15m_txt"] = TREND_TXT.get(trend15, "flat")
    rec["trend_60m_txt"] = TREND_TXT.get(trend60, "flat")
    rec["daily_trend_txt"] = TREND_TXT.get(daily, "flat")
    rec["window_name"] = win_name or "none"
    rec["aligned"] = aligned
    rec["alignment"] = {
        "daily_agree": daily_agree,
        "tf15_agree": tf15_agree,
        "tf60_agree": tf60_agree,
        "high_confidence": bool(align.get("high_confidence", snap.get("high_confidence"))),
        "aligned": aligned,
    }
    rec["desk_exit_reason"] = desk_exit_reason(str(rec.get("exit_reason") or rec.get("desk_exit_reason") or ""))
    rec["mae_pts"] = _num(rec.get("mae_pts"))
    rec["mfe_pts"] = _num(rec.get("mfe_pts"))
    rec["pnl_usd"] = _num(rec.get("pnl_usd"), rec.get("pnl"))
    rec["pts"] = _num(rec.get("pts"))
    rec["entry_price"] = _num(rec.get("entry_price"))
    rec["exit_price"] = _num(rec.get("exit_price"))
    return rec


def closed_trade_view(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = attach_desk_fields(record)
    return {
        "event": rec.get("event") or "close",
        "entry_ts_et": rec.get("entry_ts_et"),
        "exit_ts_et": rec.get("exit_ts_et"),
        "entry_time": rec.get("entry_time") or rec.get("entry_ts_et"),
        "exit_time": rec.get("exit_time") or rec.get("exit_ts_et"),
        "direction": rec.get("direction"),
        "entry_price": rec.get("entry_price"),
        "exit_price": rec.get("exit_price"),
        "stop_distance": rec.get("stop_distance"),
        "mae_pts": rec.get("mae_pts"),
        "mfe_pts": rec.get("mfe_pts"),
        "desk_exit_reason": rec.get("desk_exit_reason"),
        "exit_reason": rec.get("exit_reason"),
        "trend_15m": rec.get("trend_15m"),
        "trend_60m": rec.get("trend_60m"),
        "daily_trend": rec.get("daily_trend"),
        "trend_15m_txt": rec.get("trend_15m_txt"),
        "trend_60m_txt": rec.get("trend_60m_txt"),
        "daily_trend_txt": rec.get("daily_trend_txt"),
        "atr_at_entry": rec.get("atr_at_entry"),
        "window_name": rec.get("window_name"),
        "aligned": rec.get("aligned"),
        "pnl_usd": rec.get("pnl_usd"),
        "pts": rec.get("pts"),
        "primary_reason": rec.get("primary_reason") or (rec.get("why") or {}).get("primary_reason"),
        "outcome": rec.get("outcome"),
    }


def group_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [attach_desk_fields(r) for r in rows]
    pnls = [float(p) for p in (_num(r.get("pnl_usd"), r.get("pnl")) for r in rows) if p is not None]
    n = len(rows)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    mae = [float(v) for v in (_num(r.get("mae_pts")) for r in rows) if v is not None]
    mfe = [float(v) for v in (_num(r.get("mfe_pts")) for r in rows) if v is not None]
    win_mfe = [
        float(v)
        for r in rows
        if (_num(r.get("pnl_usd"), r.get("pnl")) or 0) > 0
        for v in (_num(r.get("mfe_pts")),)
        if v is not None
    ]
    lose_mae = [
        float(v)
        for r in rows
        if (_num(r.get("pnl_usd"), r.get("pnl")) or 0) < 0
        for v in (_num(r.get("mae_pts")),)
        if v is not None
    ]
    pf = profit_factor(pnls)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": None if not n else round(100.0 * wins / n, 1),
        "pf": pf,
        "expectancy_usd": None if not pnls else round(float(np.mean(pnls)), 2),
        "total_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
        "avg_mae": _mean(mae),
        "avg_mfe": _mean(mfe),
        "median_mfe_winners": _median(win_mfe),
        "median_mae_losers": _median(lose_mae),
    }


def _group_map(
    rows: Sequence[Dict[str, Any]],
    key_fn,
    ordered: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ordered}
    extra: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = key_fn(row) or "none"
        if key in buckets:
            buckets[key].append(row)
        else:
            extra.setdefault(key, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for key in list(ordered) + list(extra.keys()):
        group = buckets.get(key) or extra.get(key) or []
        stats = group_stats(group)
        stats["label"] = key
        out[key] = stats
    return out


def mfe_mae_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [attach_desk_fields(r) for r in rows]
    winners = [r for r in rows if (_num(r.get("pnl_usd"), r.get("pnl")) or 0) > 0]
    losers = [r for r in rows if (_num(r.get("pnl_usd"), r.get("pnl")) or 0) < 0]
    win_mfe = [float(v) for r in winners for v in (_num(r.get("mfe_pts")),) if v is not None]
    lose_mfe = [float(v) for r in losers for v in (_num(r.get("mfe_pts")),) if v is not None]
    win_mae = [float(v) for r in winners for v in (_num(r.get("mae_pts")),) if v is not None]
    lose_mae = [float(v) for r in losers for v in (_num(r.get("mae_pts")),) if v is not None]

    def _bucket_counts(vals: Sequence[float]) -> Dict[str, int]:
        counts = {name: 0 for *_, name in MFE_BUCKETS}
        counts["unknown"] = 0
        for v in vals:
            counts[mfe_bucket(v)] = counts.get(mfe_bucket(v), 0) + 1
        return counts

    n_w = len(winners)
    pct_tiny = round(100.0 * sum(1 for v in win_mfe if v <= 20) / n_w, 1) if n_w else None
    pct_mid = round(100.0 * sum(1 for v in win_mfe if 20 < v < 80) / n_w, 1) if n_w else None
    pct_big = round(100.0 * sum(1 for v in win_mfe if v >= 80) / n_w, 1) if n_w else None
    med_mfe_w = _median(win_mfe)
    med_mae_l = _median(lose_mae)
    eod_wins = sum(1 for r in winners if desk_exit_reason(str(r.get("exit_reason") or "")) == "EOD flatten")
    stop_wins = sum(1 for r in winners if desk_exit_reason(str(r.get("exit_reason") or "")) == "Stop")
    tp_wins = sum(1 for r in winners if desk_exit_reason(str(r.get("exit_reason") or "")) == "TP cap")

    behavior = "insufficient"
    if n_w >= 5 and win_mfe:
        if (pct_big or 0) >= 40 and (med_mfe_w or 0) >= 80:
            behavior = "trend-follow"
        elif (pct_tiny or 0) >= 50 and (med_mae_l or 0) > (med_mfe_w or 0):
            behavior = "stop-sizing mismatch"
        elif (med_mfe_w or 0) >= 60 and eod_wins >= stop_wins:
            behavior = "trend-follow"
        else:
            behavior = "mixed"

    return {
        "n_winners": n_w,
        "n_losers": len(losers),
        "median_mfe_winners": med_mfe_w,
        "median_mae_winners": _median(win_mae),
        "median_mfe_losers": _median(lose_mfe),
        "median_mae_losers": med_mae_l,
        "pct_winners_mfe_le_20": pct_tiny,
        "pct_winners_mfe_20_80": pct_mid,
        "pct_winners_mfe_ge_80": pct_big,
        "winner_mfe_buckets": _bucket_counts(win_mfe),
        "loser_mfe_buckets": _bucket_counts(lose_mfe),
        "winner_exits": {"EOD flatten": eod_wins, "Stop": stop_wins, "TP cap": tp_wins},
        "behavior": behavior,
    }


def expectancy_tables(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [attach_desk_fields(r) for r in rows]
    by_window = _group_map(rows, lambda r: r.get("window_name") or "none", CANON_WINDOWS)
    by_side = _group_map(
        rows,
        lambda r: "long" if str(r.get("direction") or "").lower() == "long" else (
            "short" if str(r.get("direction") or "").lower() == "short" else "none"
        ),
        ("long", "short"),
    )
    by_alignment = _group_map(
        rows,
        lambda r: "aligned" if r.get("aligned") else "not aligned",
        ("aligned", "not aligned"),
    )
    dist = mfe_mae_distribution(rows)
    overall = group_stats(rows)
    return {
        "overall": overall,
        "by_window": by_window,
        "by_side": by_side,
        "by_alignment": by_alignment,
        "mfe_mae": dist,
        "advisory": advisory_text(by_window, by_side, by_alignment, dist, overall),
        "advisory_only": True,
        "writes_live_config": False,
    }


def advisory_text(
    by_window: Dict[str, Dict[str, Any]],
    by_side: Dict[str, Dict[str, Any]],
    by_alignment: Dict[str, Dict[str, Any]],
    dist: Dict[str, Any],
    overall: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    windows = [(k, v) for k, v in by_window.items() if (v.get("n") or 0) > 0]
    windows.sort(key=lambda kv: (kv[1].get("pf") is None, -(kv[1].get("pf") or 0)))
    if len(windows) >= 2:
        best, worst = windows[0], windows[-1]
        bp, wp = best[1].get("pf"), worst[1].get("pf")
        if bp is not None and wp is not None:
            lines.append(
                f"{best[0]} window PF {bp:.2f} (n={best[1]['n']}) vs "
                f"{worst[0]} window PF {wp:.2f} (n={worst[1]['n']})"
            )
        carrying = [w for w in windows if (w[1].get("pf") or 0) >= 1.15 and (w[1].get("total_pnl_usd") or 0) > 0]
        if len(carrying) == 1:
            lines.append(f"{carrying[0][0]} is carrying most of the edge; other windows are not.")
        elif carrying:
            names = ", ".join(c[0] for c in carrying)
            lines.append(f"Edge is split across {names} — not a single window.")
    long_s, short_s = by_side.get("long") or {}, by_side.get("short") or {}
    if long_s.get("n") and short_s.get("n") and long_s.get("pf") is not None and short_s.get("pf") is not None:
        lines.append(
            f"Longs PF {long_s['pf']:.2f} (n={long_s['n']}) vs shorts PF {short_s['pf']:.2f} (n={short_s['n']})"
        )
    al, nal = by_alignment.get("aligned") or {}, by_alignment.get("not aligned") or {}
    if al.get("n") and nal.get("n") and al.get("pf") is not None and nal.get("pf") is not None:
        lines.append(
            f"Trend aligned (15m+60m+daily) PF {al['pf']:.2f} (n={al['n']}) vs "
            f"not aligned PF {nal['pf']:.2f} (n={nal['n']})"
        )
    beh = dist.get("behavior")
    med_mfe = dist.get("median_mfe_winners")
    med_mae = dist.get("median_mae_losers")
    pct_big = dist.get("pct_winners_mfe_ge_80")
    pct_tiny = dist.get("pct_winners_mfe_le_20")
    if med_mfe is not None:
        lines.append(
            f"Winners median MFE {med_mfe:.0f} pts "
            f"({pct_tiny or 0:.0f}% <=20, {pct_big or 0:.0f}% >=80); "
            f"losers median MAE {med_mae if med_mae is not None else '—'} pts"
        )
    if beh == "trend-follow":
        lines.append(
            "Looks like trend-follow: winners typically run far (then EOD flatten), not +15/+20 then die."
        )
    elif beh == "stop-sizing mismatch":
        lines.append(
            "Looks like stop-sizing mismatch: winners rarely extend while losers print large MAE. "
            "Do not auto-widen stops from this panel."
        )
    ov = overall.get("pf")
    n = overall.get("n") or 0
    if ov is not None:
        lines.append(f"Overall n={n} PF {ov:.2f}. Advisory only — do not auto-change ENTRY_WINDOWS, stops, or MAX_TRADES.")
    else:
        lines.append("Not enough closed trades for expectancy. Bot will not auto-tweak.")
    return lines


def split_is_oos_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ins, oos = [], []
    for r in rows:
        ts = r.get("entry_time") or r.get("entry_ts_et")
        if not ts:
            ins.append(r)
            continue
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        (oos if t >= OOS_START else ins).append(r)
    return ins, oos


def _et_iso(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).isoformat()


def simulate_locked_ema15(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Official locked 3-window path with MAE/MFE + trend state on every close."""
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    trades: List[Dict[str, Any]] = []
    opens: List[Dict[str, Any]] = []
    fired: Dict[Any, bool] = {}
    counts: Dict[Any, int] = {}

    def _close(p: Dict[str, Any], ts, exit_px: float, reason: str) -> None:
        pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
        sl_pts = float(p["sl_pts"])
        rec = {
            "event": "close",
            "symbol": "MNQ",
            "direction": "long" if p["dir"] == 1 else "short",
            "side": "long" if p["dir"] == 1 else "short",
            "entry_time": p["ts"].isoformat(),
            "exit_time": pd.Timestamp(ts).isoformat(),
            "entry_ts_et": _et_iso(p["ts"]),
            "exit_ts_et": _et_iso(ts),
            "entry_price": round(float(p["entry"]), 2),
            "exit_price": round(float(exit_px), 2),
            "stop_distance": round(sl_pts, 2),
            "atr_stop_pts": round(sl_pts, 2),
            "sl_pts": round(sl_pts, 2),
            "mae_pts": round(float(p["mae"]), 2),
            "mfe_pts": round(float(p["mfe"]), 2),
            "exit_reason": reason,
            "desk_exit_reason": desk_exit_reason(reason),
            "trend_15m": int(p["t15"]),
            "trend_60m": int(p["t60"]),
            "daily_trend": int(p["daily"]),
            "atr_at_entry": None if p["atr"] is None or (isinstance(p["atr"], float) and np.isnan(p["atr"])) else round(float(p["atr"]), 2),
            "window": p["window"],
            "window_name": locked_window_name(p["window"]),
            "aligned": bool(p["dir"] and p["t15"] == p["dir"] and p["t60"] == p["dir"] and p["daily"] == p["dir"]),
            "pts": round(float(pts), 2),
            "pnl_usd": round(float(pts) * POINT_VALUE - COMMISSION, 2),
            "hold_minutes": round((pd.Timestamp(ts) - p["ts"]).total_seconds() / 60.0, 2),
            "r_multiple": None if sl_pts <= 0 else round(float(pts) / sl_pts, 3),
            "source": "locked_ema15_backtest",
        }
        rec["entry_snapshot"] = {
            "ts_et": rec["entry_ts_et"],
            "window_name": rec["window_name"],
            "window": rec["window"],
            "trend_15m": rec["trend_15m"],
            "trend_60m": rec["trend_60m"],
            "daily_permission": rec["daily_trend"],
            "atr15": rec["atr_at_entry"],
            "atr_stop_pts": rec["stop_distance"],
            "daily_agree": rec["daily_trend"] == p["dir"],
            "tf15_agree": rec["trend_15m"] == p["dir"],
            "tf60_agree": rec["trend_60m"] == p["dir"],
            "high_confidence": bool(p.get("conf")),
        }
        trades.append(attach_desk_fields(rec))

    for i in range(80, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            if p["dir"] == 1:
                p["mae"] = max(p["mae"], p["entry"] - lo[i])
                p["mfe"] = max(p["mfe"], hi[i] - p["entry"])
            else:
                p["mae"] = max(p["mae"], hi[i] - p["entry"])
                p["mfe"] = max(p["mfe"], p["entry"] - lo[i])
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = EXIT_EOD, float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = EXIT_SL, p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = EXIT_TP, p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = EXIT_SL, p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = EXIT_TP, p["tp"]
            if reason:
                _close(p, ts, float(exit_px), reason)
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        win = locked_window_index(m)
        if win is None:
            continue
        side = int(t15[i])
        if side == 0 or side != int(td[i]):
            continue
        key = (day, win)
        if key in fired or counts.get(day, 0) >= LOCKED_MAX_TRADES:
            continue
        conf = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        if len(opens) >= 2:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue
        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp = entry + 500 if side == 1 else entry - 500
        atr_i = float(atr[i])
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_pts": use,
            "atr": None if np.isnan(atr_i) else atr_i,
            "window": win,
            "t15": int(t15[i]),
            "t60": int(t60[i]),
            "daily": int(td[i]),
            "conf": conf,
            "mae": 0.0,
            "mfe": 0.0,
        })
        fired[key] = True
        counts[day] = counts.get(day, 0) + 1
    return trades


def enrich_mae_mfe_from_path(
    df_1m: pd.DataFrame,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fill MAE/MFE from 1m bars between entry and exit when missing."""
    out = []
    for row in rows:
        rec = dict(row)
        if rec.get("mae_pts") is None or rec.get("mfe_pts") is None:
            mae, mfe = compute_mae_mfe_pts(
                df_1m,
                rec.get("entry_time") or rec.get("entry_ts_et"),
                rec.get("exit_time") or rec.get("exit_ts_et"),
                str(rec.get("direction") or rec.get("side") or ""),
                float(rec.get("entry_price") or 0),
            )
            if rec.get("mae_pts") is None:
                rec["mae_pts"] = mae
            if rec.get("mfe_pts") is None:
                rec["mfe_pts"] = mfe
        out.append(attach_desk_fields(rec))
    return out


def build_snapshot(
    rows: Sequence[Dict[str, Any]],
    *,
    csv: str = "data/MNQ_1m.csv",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ins, oos = split_is_oos_rows(rows)
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "generated_at": now.isoformat(),
        "generated_at_et": ts_et_iso(now),
        "source": "locked ema15_eod 9:35/11:00/13:30 max 3, ATR×2, flatten 15:50",
        "csv": csv,
        "advisory_only": True,
        "auto_apply": False,
        "writes_live_config": False,
        "n_all": len(rows),
        "n_is": len(ins),
        "n_oos": len(oos),
        "all": expectancy_tables(rows),
        "is": expectancy_tables(ins),
        "oos": expectancy_tables(oos),
        "windows_analyzed": list(CANON_WINDOWS),
        "note": (
            "Snapshot of the locked 3-window recipe. Does not write mnq_profit_config "
            "or change live ENTRY_WINDOWS / stops / MAX_TRADES."
        ),
    }
    if extra:
        payload.update(extra)
    all_adv = payload["all"].get("advisory") or []
    oos_adv = payload["oos"].get("advisory") or []
    payload["answer"] = {
        "n_all": len(rows),
        "n_oos": len(oos),
        "all_advisory": all_adv,
        "oos_advisory": oos_adv,
        "behavior_all": (payload["all"].get("mfe_mae") or {}).get("behavior"),
        "behavior_oos": (payload["oos"].get("mfe_mae") or {}).get("behavior"),
        "by_window_all": payload["all"].get("by_window"),
        "by_window_oos": payload["oos"].get("by_window"),
    }
    return payload


def write_snapshot(payload: Dict[str, Any], path: str = WINDOW_EDGE_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def read_snapshot(path: str = WINDOW_EDGE_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def live_expectancy(closes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in closes if str(r.get("event") or "close") == "close"]
    rows = [r for r in rows if desk_exit_reason(str(r.get("exit_reason") or "")) != "rejected"]
    ema15 = [
        r for r in rows
        if r.get("entry_snapshot") or r.get("window_name") or r.get("atr_stop_pts") or r.get("source")
    ]
    use = ema15 if ema15 else []
    tables = expectancy_tables(use)
    tables["n_closes"] = len(use)
    tables["empty"] = len(use) == 0
    if not use:
        tables["advisory"] = []
    return tables
