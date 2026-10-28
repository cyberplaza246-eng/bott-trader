#!/usr/bin/env python3
"""
MNQ ema15_eod timeframe-stack experiment (structure only — no parameter tuning).

Compare five trend-filter stacks on identical rules:
  A: daily + 15m + 60m          (current live)
  B: daily + 15m + 5m
  C: daily + 5m + 1m
  D: daily + 15m + 5m + 1m
  E: daily + 15m + 1m

Shared: 6 gapped windows, quiet 1m body≤4, sep/ATR≥0.45 noon+, ATR×2 SL 20-60,
1R TP, flatten 15:50, max 20/day, window re-entry, 2-lot cap, $2/pt, $1.24 RT.
IS/OOS split: pre-Jun 2026 vs Jun 2026+.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    EMA_FAST,
    EMA_SLOW,
    ENTRY_MINUTE,
    ENTRY_WINDOWS,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    QUALITY_AFTER_MINUTE,
    SEP_MIN,
    TP_R,
    _as_utc_frame,
    _ohlc_resample,
    atr15_on_1m,
    daily_trend_on_1m,
    ema_series,
    high_confidence,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_index,
    window_label,
    window_requires_quality,
)

ET = pytz.timezone("US/Eastern")
WARMUP = 80
PT, COMM = official.PT, official.COMM
WINDOW_NAMES = tuple(window_label(i) for i in range(len(ENTRY_WINDOWS)))
MORNING_END = 12 * 60  # before 12:00 ET

VARIANTS = {
    "A": "daily+15m+60m",
    "B": "daily+15m+5m",
    "C": "daily+5m+1m",
    "D": "daily+15m+5m+1m",
    "E": "daily+15m+1m",
}


@dataclass
class TradeRec:
    trade: scalp_bt.ScalpTrade
    window: str
    sl_pts: float
    variant: str


def trend_tf_on_1m(df_1m: pd.DataFrame, rule: str, *, completed_only: bool = True) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    bars = _ohlc_resample(tmp, rule)
    fast = ema_series(bars["close"], EMA_FAST)
    slow = ema_series(bars["close"], EMA_SLOW)
    trend = pd.Series(0, index=bars.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    if completed_only:
        trend = trend.shift(1)
    return trend.reindex(tmp.index, method="ffill").fillna(0).astype(int)


def trend_5m_on_1m(df_1m: pd.DataFrame, *, completed_only: bool = True) -> pd.Series:
    return trend_tf_on_1m(df_1m, "5min", completed_only=completed_only)


def trend_1m_completed(df_1m: pd.DataFrame) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    fast = ema_series(tmp["close"], EMA_FAST)
    slow = ema_series(tmp["close"], EMA_SLOW)
    trend = pd.Series(0, index=tmp.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    return trend.shift(1).fillna(0).astype(int)


def high_confidence_5m(side: int, trend5: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    if side == 0 or trend5 != side:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def high_confidence_sep(side: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    """Sep-only gate when 15m direction already in stack (variant E)."""
    if side == 0:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def stack_ok(variant: str, side: int, td: int, t15: int, t60: int, t5: int, t1: int) -> bool:
    if side == 0:
        return False
    if variant == "A":
        return side == td and side == t15 and side == t60
    if variant == "B":
        return side == td and side == t15 and side == t5
    if variant == "C":
        return side == td and side == t5 and side == t1
    if variant == "D":
        return side == td and side == t15 and side == t5 and side == t1
    if variant == "E":
        return side == td and side == t15 and side == t1
    raise ValueError(f"unknown variant {variant}")


def primary_side(variant: str, t15: int, t5: int) -> int:
    """Direction candidate before stack agreement check."""
    if variant == "C":
        return int(t5)
    return int(t15)


def confidence_ok(variant: str, side: int, t60: int, t5: int, sep: float) -> bool:
    if variant == "A":
        return high_confidence(side, int(t60), sep, SEP_MIN)
    if variant in ("B", "C", "D"):
        return high_confidence_5m(side, int(t5), sep, SEP_MIN)
    return high_confidence_sep(side, sep, SEP_MIN)


def max_consecutive_losses(trades: List[scalp_bt.ScalpTrade]) -> int:
    best = cur = 0
    for t in trades:
        if t.pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def group_metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_consec_losses": 0,
        }
    s = edge.summarize(trades)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades": s["trades"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "expectancy": s["expectancy"],
        "total_pnl": s["total_pnl"],
        "max_drawdown": s["max_drawdown"],
        "avg_winner": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loser": round(float(np.mean(losses)), 2) if losses else 0.0,
        "max_consec_losses": max_consecutive_losses(trades),
    }


def avg_stop_pts(recs: List[TradeRec]) -> float:
    if not recs:
        return 0.0
    return round(float(np.mean([r.sl_pts for r in recs])), 1)


def session_bucket(entry_ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(entry_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    m = ts.tz_convert(ET).hour * 60 + ts.tz_convert(ET).minute
    return "morning" if m < MORNING_END else "afternoon"


def simulate_variant(df: pd.DataFrame, *, variant: str) -> List[TradeRec]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    t5 = trend_5m_on_1m(work).to_numpy(dtype=int)
    t1 = trend_1m_completed(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    op = work["open"].to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    recs: List[TradeRec] = []
    opens: List[Dict] = []
    fired_windows: Set[int] = set()
    counts: Dict = {}

    for i in range(WARMUP, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                tr = gapped._close_position(p, ts, float(exit_px), reason)
                tr.version = variant
                recs.append(TradeRec(tr, p["win_name"], p["sl_pts"], variant))
                if p.get("win") is not None:
                    fired_windows.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue

        win = window_index(m)
        if win is None or win in fired_windows:
            continue

        side = primary_side(variant, int(t15[i]), int(t5[i]))
        if side == 0:
            continue
        if not stack_ok(variant, side, int(td[i]), int(t15[i]), int(t60[i]), int(t5[i]), int(t1[i])):
            continue
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        conf = confidence_ok(variant, side, int(t60[i]), int(t5[i]), float(sep[i]))
        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp_dist = use * TP_R
        tp = entry + tp_dist if side == 1 else entry - tp_dist
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": win,
            "win_name": window_label(win),
            "sl_pts": use,
            "variant": variant,
        })
        fired_windows.add(win)
        counts[day] = counts.get(day, 0) + 1

    return recs


def simulate_hypothetical(
    df: pd.DataFrame,
    start_i: int,
    side: int,
    entry: float,
    sl_pts: float,
    mins_arr: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    dt: pd.DatetimeIndex,
) -> Tuple[float, str, float]:
    """Return (pnl_usd, exit_reason, mfe_pts) for a single hypothetical entry."""
    sl = entry - sl_pts if side == 1 else entry + sl_pts
    tp = entry + sl_pts if side == 1 else entry - sl_pts
    mfe = 0.0
    for j in range(start_i + 1, len(df)):
        m = int(mins_arr[j])
        if m >= FLAT_MINUTE:
            exit_px = float(cl[j])
            pts = (exit_px - entry) if side == 1 else (entry - exit_px)
            pnl = pts * PT - COMM
            return pnl, "EOD", mfe
        if side == 1:
            mfe = max(mfe, float(hi[j]) - entry)
            if lo[j] <= sl:
                return -sl_pts * PT - COMM, "SL", mfe
            if hi[j] >= tp:
                return sl_pts * PT - COMM, "TP", mfe
        else:
            mfe = max(mfe, entry - float(lo[j]))
            if hi[j] >= sl:
                return -sl_pts * PT - COMM, "SL", mfe
            if lo[j] <= tp:
                return sl_pts * PT - COMM, "TP", mfe
    exit_px = float(cl[-1])
    pts = (exit_px - entry) if side == 1 else (entry - exit_px)
    return pts * PT - COMM, "END", mfe


def analyze_60m_blocks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Signals where daily+15m agree + quiet 1m, but 60m blocks (variant A stack fail).
    Check whether 5m/1m agreed and quantify hypothetical P&L / MFE.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    t5 = trend_5m_on_1m(work).to_numpy(dtype=int)
    t1 = trend_1m_completed(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    op = work["open"].to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    blocked: List[Dict[str, Any]] = []
    fired: Dict = {}  # (day, win) -> True — one shot per window like live

    for i in range(WARMUP, len(work)):
        m = int(mins[i])
        day = dates[i]
        if dow[i] >= 5 or m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        win = window_index(m)
        if win is None:
            continue
        key = (day, win)
        if fired.get(key):
            continue

        side = int(t15[i])
        if side == 0 or side != int(td[i]):
            continue
        if side == int(t60[i]):
            continue  # not blocked by 60m
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        conf60 = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        if window_requires_quality(win) and conf60:
            continue  # would fail sep anyway, not purely 60m

        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        pnl, reason, mfe = simulate_hypothetical(
            work, i, side, entry, use, mins, hi, lo, cl, dt,
        )
        t5_ok = int(t5[i]) == side
        t1_ok = int(t1[i]) == side
        fired[key] = True
        blocked.append({
            "day": str(day),
            "window": window_label(win),
            "side": "long" if side == 1 else "short",
            "t5_agrees": t5_ok,
            "t1_agrees": t1_ok,
            "both_5m_1m": t5_ok and t1_ok,
            "hypo_pnl": round(pnl, 2),
            "hypo_reason": reason,
            "mfe_pts": round(mfe, 1),
            "sl_pts": round(use, 1),
            "would_win": pnl > 0,
        })

    n = len(blocked)
    if n == 0:
        return {"n_blocked": 0}

    hypo_pnls = [b["hypo_pnl"] for b in blocked]
    winners = [b for b in blocked if b["would_win"]]
    both = [b for b in blocked if b["both_5m_1m"]]
    both_win = [b for b in both if b["would_win"]]

    return {
        "n_blocked": n,
        "pct_5m_agrees": round(100 * sum(1 for b in blocked if b["t5_agrees"]) / n, 1),
        "pct_1m_agrees": round(100 * sum(1 for b in blocked if b["t1_agrees"]) / n, 1),
        "pct_both_5m_1m": round(100 * len(both) / n, 1),
        "hypo_win_rate": round(100 * len(winners) / n, 1),
        "hypo_total_pnl": round(sum(hypo_pnls), 2),
        "hypo_avg_pnl": round(float(np.mean(hypo_pnls)), 2),
        "both_agree_n": len(both),
        "both_agree_hypo_pnl": round(sum(b["hypo_pnl"] for b in both), 2),
        "both_agree_win_rate": round(100 * len(both_win) / len(both), 1) if both else 0.0,
        "both_agree_avg_mfe": round(float(np.mean([b["mfe_pts"] for b in both])), 1) if both else 0.0,
        "avg_mfe_all": round(float(np.mean([b["mfe_pts"] for b in blocked])), 1),
        "avg_mfe_winners": round(
            float(np.mean([b["mfe_pts"] for b in winners])), 1,
        ) if winners else 0.0,
        "missed_tp_count": sum(1 for b in blocked if b["hypo_reason"] == "TP"),
        "missed_sl_count": sum(1 for b in blocked if b["hypo_reason"] == "SL"),
    }


def full_report(variant: str, recs: List[TradeRec]) -> Dict[str, Any]:
    trades = [r.trade for r in recs]
    ins, oos = edge.split_is_oos(trades)
    ins_recs = [r for r in recs if r.trade in ins]
    oos_recs = [r for r in recs if r.trade in oos]

    overall = group_metrics(trades)
    is_m = group_metrics(ins)
    oos_m = group_metrics(oos)

    by_side: Dict[str, Dict] = {}
    for label in ("long", "short"):
        subset = [r.trade for r in recs if r.trade.direction == label]
        by_side[label] = group_metrics(subset)

    by_session: Dict[str, Dict] = {}
    for label in ("morning", "afternoon"):
        subset = [r.trade for r in recs if session_bucket(r.trade.entry_time) == label]
        by_session[label] = group_metrics(subset)

    by_window: Dict[str, Dict] = {}
    for wn in WINDOW_NAMES:
        subset = [r.trade for r in recs if r.window == wn]
        by_window[wn] = group_metrics(subset)

    return {
        "variant": variant,
        "stack": VARIANTS[variant],
        "avg_stop_pts": avg_stop_pts(recs),
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_side": by_side,
        "by_session": by_session,
        "by_window": by_window,
    }


def print_variant_report(r: Dict[str, Any]) -> None:
    v = r["variant"]
    o = r["overall"]
    print(f"\n{'=' * 72}")
    print(f"VARIANT {v}: {r['stack']}")
    print(f"{'=' * 72}")
    print(
        f"  Trades={o['trades']}  WR={o['win_rate']:.1f}%  PF={o['profit_factor']:.2f}  "
        f"Net=${o['total_pnl']:.1f}  Exp=${o['expectancy']:.2f}/tr  DD=${o['max_drawdown']:.1f}  "
        f"AvgStop={r['avg_stop_pts']:.1f}pt  MaxConsecL={o['max_consec_losses']}"
    )
    print(f"  AvgWinner=${o['avg_winner']:.2f}  AvgLoser=${o['avg_loser']:.2f}")
    print(
        f"  IS:  n={r['is']['trades']} WR={r['is']['win_rate']:.1f}% PF={r['is']['profit_factor']:.2f} "
        f"${r['is']['total_pnl']:.1f} Exp=${r['is']['expectancy']:.2f} DD=${r['is']['max_drawdown']:.1f}"
    )
    print(
        f"  OOS: n={r['oos']['trades']} WR={r['oos']['win_rate']:.1f}% PF={r['oos']['profit_factor']:.2f} "
        f"${r['oos']['total_pnl']:.1f} Exp=${r['oos']['expectancy']:.2f} DD=${r['oos']['max_drawdown']:.1f}"
    )
    print("  Long / Short:")
    for side in ("long", "short"):
        s = r["by_side"][side]
        print(
            f"    {side:<6} n={s['trades']:3d} WR={s['win_rate']:5.1f}% PF={s['profit_factor']:5.2f} "
            f"${s['total_pnl']:8.1f} Exp=${s['expectancy']:6.2f}"
        )
    print("  Morning / Afternoon:")
    for sess in ("morning", "afternoon"):
        s = r["by_session"][sess]
        print(
            f"    {sess:<10} n={s['trades']:3d} WR={s['win_rate']:5.1f}% PF={s['profit_factor']:5.2f} "
            f"${s['total_pnl']:8.1f} Exp=${s['expectancy']:6.2f}"
        )
    print("  By window:")
    for wn in WINDOW_NAMES:
        s = r["by_window"][wn]
        if s["trades"] == 0:
            print(f"    {wn}  n=  0")
        else:
            print(
                f"    {wn}  n={s['trades']:3d} WR={s['win_rate']:5.1f}% PF={s['profit_factor']:5.2f} "
                f"${s['total_pnl']:7.1f} Exp=${s['expectancy']:6.2f}"
            )


def rank_variants(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OOS expectancy first, then PF, then lower drawdown."""
    return sorted(
        reports,
        key=lambda r: (
            r["oos"]["expectancy"],
            r["oos"]["profit_factor"],
            -r["oos"]["max_drawdown"],
        ),
        reverse=True,
    )


def print_ranked_table(reports: List[Dict[str, Any]]) -> None:
    ranked = rank_variants(reports)
    print("\n" + "=" * 110)
    print("RANKED TABLE (OOS expectancy -> PF -> DD)  |  ALL-period metrics")
    print("=" * 110)
    hdr = (
        f"{'Rank':<5} {'Var':<4} {'Stack':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} "
        f"{'Net$':>9} {'Exp$':>7} {'DD$':>8} {'Stop':>5} | "
        f"{'OOS n':>5} {'OOS PF':>6} {'OOS Exp$':>8} {'OOS DD$':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for rank, r in enumerate(ranked, 1):
        o = r["overall"]
        oo = r["oos"]
        print(
            f"{rank:<5} {r['variant']:<4} {r['stack']:<22} {o['trades']:6d} {o['win_rate']:6.1f} "
            f"{o['profit_factor']:6.2f} {o['total_pnl']:9.1f} {o['expectancy']:7.2f} "
            f"{o['max_drawdown']:8.1f} {r['avg_stop_pts']:5.1f} | "
            f"{oo['trades']:5d} {oo['profit_factor']:6.2f} {oo['expectancy']:8.2f} {oo['max_drawdown']:8.1f}"
        )
    return ranked


def print_expectancy_trend(reports: List[Dict[str, Any]]) -> None:
    order = ["A", "B", "E", "D", "C"]  # slow → fast conceptual order
    print("\nEXPECTANCY TREND (slow -> fast stacks)")
    print(f"{'Variant':<6} {'Stack':<22} {'ALL Exp$':>9} {'OOS Exp$':>9} {'OOS PF':>7}")
    print("-" * 58)
    for v in order:
        r = next(x for x in reports if x["variant"] == v)
        print(
            f"{v:<6} {r['stack']:<22} {r['overall']['expectancy']:9.2f} "
            f"{r['oos']['expectancy']:9.2f} {r['oos']['profit_factor']:7.2f}"
        )


def recommend(ranked: List[Dict[str, Any]], baseline: Dict[str, Any]) -> str:
    winner = ranked[0]
    wv = winner["variant"]
    if wv == "A":
        return (
            f"Keep variant A (daily+15m+60m). Despite slower filters, it ranks #1 on OOS "
            f"expectancy (${winner['oos']['expectancy']:.2f}/tr), PF {winner['oos']['profit_factor']:.2f}, "
            f"DD ${winner['oos']['max_drawdown']:.0f}."
        )
    beat_exp = winner["oos"]["expectancy"] > baseline["oos"]["expectancy"]
    beat_pf = winner["oos"]["profit_factor"] >= baseline["oos"]["profit_factor"]
    beat_dd = winner["oos"]["max_drawdown"] <= baseline["oos"]["max_drawdown"]
    lines = [
        f"OOS leader: variant {wv} ({winner['stack']}) - "
        f"Exp ${winner['oos']['expectancy']:.2f}/tr vs A ${baseline['oos']['expectancy']:.2f}, "
        f"PF {winner['oos']['profit_factor']:.2f} vs {baseline['oos']['profit_factor']:.2f}, "
        f"DD ${winner['oos']['max_drawdown']:.0f} vs ${baseline['oos']['max_drawdown']:.0f}."
    ]
    if beat_exp and beat_pf:
        lines.append(f"Recommend paper-testing variant {wv} - beats baseline on OOS expectancy and PF.")
    elif beat_exp and beat_dd:
        lines.append(f"Recommend cautious paper test of {wv} — better OOS expectancy and DD, mixed PF.")
    else:
        lines.append(f"Variant {wv} leads ranking but does not clearly dominate A on all OOS metrics.")
    return " ".join(lines)


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out_json = os.path.join("data", "ema15_timeframe_stacks.json")

    print("MNQ ema15_eod - TIMEFRAME STACK EXPERIMENT")
    print(f"Data: {csv}")
    print(
        "Rules: 6 gapped windows, quiet 1m, sep/ATR>=0.45 noon+, ATRx2 SL 20-60, 1R TP, "
        f"flatten 15:50, {MAX_TRADES_DAY}/day, re-entry, max {DEFAULT_MAX_OPEN_LOTS} lots, "
        f"${PT}/pt ${COMM} RT"
    )
    print(f"Variants: {', '.join(f'{k}={v}' for k, v in VARIANTS.items())}")
    print(f"IS/OOS split: before Jun 2026 / Jun 2026+")

    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    reports: List[Dict[str, Any]] = []
    for v in ("A", "B", "C", "D", "E"):
        recs = simulate_variant(df, variant=v)
        rep = full_report(v, recs)
        reports.append(rep)
        print_variant_report(rep)

    ranked = print_ranked_table(reports)
    print_expectancy_trend(reports)

    block = analyze_60m_blocks(df)
    print("\n" + "=" * 72)
    print("60m FILTER BLOCK ANALYSIS (daily+15m agree, 60m disagrees, quiet 1m ok)")
    print("=" * 72)
    if block.get("n_blocked", 0) == 0:
        print("  No blocked signals found.")
    else:
        print(f"  Blocked window slots: {block['n_blocked']}")
        print(f"  5m agreed with direction: {block['pct_5m_agrees']:.1f}%")
        print(f"  1m agreed with direction: {block['pct_1m_agrees']:.1f}%")
        print(f"  Both 5m+1m agreed: {block['pct_both_5m_1m']:.1f}% ({block['both_agree_n']} slots)")
        print(
            f"  Hypothetical if entered anyway: WR={block['hypo_win_rate']:.1f}%  "
            f"total=${block['hypo_total_pnl']:.1f}  avg=${block['hypo_avg_pnl']:.2f}/tr"
        )
        print(
            f"  When 5m+1m both agreed: n={block['both_agree_n']} WR={block['both_agree_win_rate']:.1f}%  "
            f"hypo P&L=${block['both_agree_hypo_pnl']:.1f}  avg MFE={block['both_agree_avg_mfe']:.1f}pt"
        )
        print(
            f"  MFE on blocked signals: avg={block['avg_mfe_all']:.1f}pt  "
            f"winners avg={block['avg_mfe_winners']:.1f}pt"
        )
        print(
            f"  Would-have exits: TP={block['missed_tp_count']} SL={block['missed_sl_count']}"
        )
        if block["both_agree_hypo_pnl"] > 0 and block["both_agree_win_rate"] >= 50:
            print("  -> 60m filter IS delaying entries and missing profitable 5m+1m-aligned moves.")
        elif block["hypo_total_pnl"] <= 0:
            print("  -> 60m filter is protective - blocked signals were net losers hypothetically.")
        else:
            print("  -> Mixed: some missed movement but overall blocked-set P&L is ambiguous.")

    baseline = next(r for r in reports if r["variant"] == "A")
    print("\nRECOMMENDATION")
    print(f"  {recommend(ranked, baseline)}")

    payload = {
        "csv": csv,
        "variants": reports,
        "ranking": [r["variant"] for r in ranked],
        "block_analysis": block,
        "recommendation": recommend(ranked, baseline),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nJSON saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
