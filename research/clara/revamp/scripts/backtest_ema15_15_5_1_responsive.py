#!/usr/bin/env python3
"""
MNQ responsive 15→5→1 experiment: trend stack vs 1m entry trigger + stop-model grid.

Conceptual strategy:
  15m: primary direction (EMA8 vs EMA21 on completed bar)
  5m:  confirms immediate trend (aligned with 15m)
  1m:  entry/timing ONLY (quiet body≤4, close with trend — NOT in trend stack)
  No 60m.

Variants (compare OOS):
  A: daily + 15m + 5m agree → quiet 1m entry
  B: 15m + 5m agree (NO daily) → quiet 1m entry

Stop models (per variant):
  1. Baseline reference: 15m ATR×2 clip 20-60
  2. 5m ATR×mult clip min-max (1.5×/2.0×, 8-30 / 12-40)
  3. 1m ATR×mult clip min-max (1.5×/2.0×, 6-25 / 8-30)
  4. Fixed 15/20/25 pt with 1R TP

Shared: 6 gapped windows, sep/ATR≥0.45 noon+ on 15m sep, 1R TP, flatten 15:50,
max 20/day, window re-entry, 2-lot cap, $2/pt + $1.24 RT, IS/OOS split Jun 2026.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
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
    SEP_MIN,
    TP_R,
    _as_utc_frame,
    _atr,
    _ohlc_resample,
    atr15_on_1m,
    daily_trend_on_1m,
    ema_series,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    window_index,
    window_label,
    window_requires_quality,
)

ET = pytz.timezone("US/Eastern")
WARMUP = 80
PT, COMM = official.PT, official.COMM
WINDOW_NAMES = tuple(window_label(i) for i in range(len(ENTRY_WINDOWS)))
MORNING_END = 12 * 60

VARIANTS = {
    "A": "daily+15m+5m -> quiet 1m entry",
    "B": "15m+5m (no daily) -> quiet 1m entry",
}

STOP_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "15m_atr2_20_60",
        "label": "Baseline: 15m ATR×2 clip 20-60",
        "kind": "atr",
        "tf": "15m",
        "mult": 2.0,
        "min": 20.0,
        "max": 60.0,
    },
    {
        "id": "5m_atr1.5_8_30",
        "label": "5m ATR×1.5 clip 8-30",
        "kind": "atr",
        "tf": "5m",
        "mult": 1.5,
        "min": 8.0,
        "max": 30.0,
    },
    {
        "id": "5m_atr1.5_12_40",
        "label": "5m ATR×1.5 clip 12-40",
        "kind": "atr",
        "tf": "5m",
        "mult": 1.5,
        "min": 12.0,
        "max": 40.0,
    },
    {
        "id": "5m_atr2.0_8_30",
        "label": "5m ATR×2.0 clip 8-30",
        "kind": "atr",
        "tf": "5m",
        "mult": 2.0,
        "min": 8.0,
        "max": 30.0,
    },
    {
        "id": "5m_atr2.0_12_40",
        "label": "5m ATR×2.0 clip 12-40",
        "kind": "atr",
        "tf": "5m",
        "mult": 2.0,
        "min": 12.0,
        "max": 40.0,
    },
    {
        "id": "1m_atr1.5_6_25",
        "label": "1m ATR×1.5 clip 6-25",
        "kind": "atr",
        "tf": "1m",
        "mult": 1.5,
        "min": 6.0,
        "max": 25.0,
    },
    {
        "id": "1m_atr1.5_8_30",
        "label": "1m ATR×1.5 clip 8-30",
        "kind": "atr",
        "tf": "1m",
        "mult": 1.5,
        "min": 8.0,
        "max": 30.0,
    },
    {
        "id": "1m_atr2.0_6_25",
        "label": "1m ATR×2.0 clip 6-25",
        "kind": "atr",
        "tf": "1m",
        "mult": 2.0,
        "min": 6.0,
        "max": 25.0,
    },
    {
        "id": "1m_atr2.0_8_30",
        "label": "1m ATR×2.0 clip 8-30",
        "kind": "atr",
        "tf": "1m",
        "mult": 2.0,
        "min": 8.0,
        "max": 30.0,
    },
    {
        "id": "fixed_15",
        "label": "Fixed 15pt stop, 1R TP",
        "kind": "fixed",
        "pts": 15.0,
    },
    {
        "id": "fixed_20",
        "label": "Fixed 20pt stop, 1R TP",
        "kind": "fixed",
        "pts": 20.0,
    },
    {
        "id": "fixed_25",
        "label": "Fixed 25pt stop, 1R TP",
        "kind": "fixed",
        "pts": 25.0,
    },
]


@dataclass
class TradeRec:
    trade: scalp_bt.ScalpTrade
    window: str
    sl_pts: float
    variant: str
    stop_id: str


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


def atr_tf_on_1m(df_1m: pd.DataFrame, rule: str) -> pd.Series:
    """Completed-bar ATR14 on resampled TF, ffilled to 1m."""
    tmp = _as_utc_frame(df_1m)
    bars = _ohlc_resample(tmp, rule)
    atr = _atr(bars, 14).shift(1)
    return atr.reindex(tmp.index, method="ffill")


def atr5_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    return atr_tf_on_1m(df_1m, "5min")


def atr1_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    return atr_tf_on_1m(df_1m, "1min")


def high_confidence_5m(side: int, trend5: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    """Afternoon / 2nd-lot gate: 5m trend agrees + 15m sep/ATR (same as live stack B)."""
    if side == 0 or trend5 != side:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def stack_ok(variant: str, side: int, td: int, t15: int, t5: int) -> bool:
    if side == 0 or side != t15 or side != t5:
        return False
    if variant == "A":
        return side == td
    if variant == "B":
        return True
    raise ValueError(f"unknown variant {variant}")


def stop_pts(cfg: Dict[str, Any], atr15: float, atr5: float, atr1: float) -> float:
    if cfg["kind"] == "fixed":
        return float(cfg["pts"])
    tf = cfg["tf"]
    mult = float(cfg["mult"])
    lo = float(cfg["min"])
    hi = float(cfg["max"])
    if tf == "15m":
        raw = atr15
    elif tf == "5m":
        raw = atr5
    elif tf == "1m":
        raw = atr1
    else:
        raise ValueError(f"unknown tf {tf}")
    return sl_pts_from_atr(raw, mult=mult, sl_min=lo, sl_max=hi)


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


def simulate(
    df: pd.DataFrame,
    *,
    variant: str,
    stop_cfg: Dict[str, Any],
) -> List[TradeRec]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t5 = trend_5m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr15 = atr15_on_1m(work).to_numpy(dtype=float)
    atr5 = atr5_on_1m(work).to_numpy(dtype=float)
    atr1 = atr1_on_1m(work).to_numpy(dtype=float)
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
                tr.version = f"{variant}_{stop_cfg['id']}"
                recs.append(
                    TradeRec(tr, p["win_name"], p["sl_pts"], variant, stop_cfg["id"])
                )
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

        side = int(t15[i])
        if not stack_ok(variant, side, int(td[i]), int(t15[i]), int(t5[i])):
            continue
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        conf = high_confidence_5m(side, int(t5[i]), float(sep[i]), SEP_MIN)
        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = stop_pts(stop_cfg, float(atr15[i]), float(atr5[i]), float(atr1[i]))
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
            "stop_id": stop_cfg["id"],
        })
        fired_windows.add(win)
        counts[day] = counts.get(day, 0) + 1

    return recs


def full_report(
    variant: str,
    stop_cfg: Dict[str, Any],
    recs: List[TradeRec],
) -> Dict[str, Any]:
    trades = [r.trade for r in recs]
    ins, oos = edge.split_is_oos(trades)

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
        "stop_id": stop_cfg["id"],
        "stop_label": stop_cfg["label"],
        "avg_stop_pts": avg_stop_pts(recs),
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_side": by_side,
        "by_session": by_session,
        "by_window": by_window,
    }


def rank_reports(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        reports,
        key=lambda r: (
            r["oos"]["expectancy"],
            r["oos"]["profit_factor"],
            -r["oos"]["max_drawdown"],
        ),
        reverse=True,
    )


def print_report(r: Dict[str, Any]) -> None:
    o = r["overall"]
    print(
        f"  {r['variant']}/{r['stop_id']:<18} stop={r['avg_stop_pts']:5.1f}pt  "
        f"n={o['trades']:3d} WR={o['win_rate']:5.1f}% PF={o['profit_factor']:5.2f} "
        f"Exp=${o['expectancy']:7.2f} Net=${o['total_pnl']:8.1f} DD=${o['max_drawdown']:7.1f} | "
        f"OOS n={r['oos']['trades']:3d} Exp=${r['oos']['expectancy']:7.2f} "
        f"PF={r['oos']['profit_factor']:5.2f} DD=${r['oos']['max_drawdown']:7.1f}"
    )


def print_full_variant_report(r: Dict[str, Any]) -> None:
    o = r["overall"]
    print(f"\n{'=' * 78}")
    print(f"{r['variant']} | {r['stop_label']} ({r['stop_id']})")
    print(f"{'=' * 78}")
    print(
        f"  Trades={o['trades']} WR={o['win_rate']:.1f}% PF={o['profit_factor']:.2f} "
        f"Net=${o['total_pnl']:.1f} Exp=${o['expectancy']:.2f}/tr DD=${o['max_drawdown']:.1f} "
        f"AvgStop={r['avg_stop_pts']:.1f}pt MaxConsecL={o['max_consec_losses']}"
    )
    print(f"  AvgWinner=${o['avg_winner']:.2f} AvgLoser=${o['avg_loser']:.2f}")
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


def load_prior_reference(path: str) -> Dict[str, Any]:
    """Load prior timeframe_stacks A and B (15m ATR stop) for comparison."""
    out: Dict[str, Any] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key, vid in (("prior_A_daily_15m_60m", "A"), ("prior_B_daily_15m_5m", "B")):
        for v in data.get("variants", []):
            if v.get("variant") == vid:
                out[key] = {
                    "source": "ema15_timeframe_stacks.json",
                    "variant": vid,
                    "stack": v.get("stack"),
                    "stop": "15m ATR×2 clip 20-60",
                    "avg_stop_pts": v.get("avg_stop_pts"),
                    "overall": v.get("overall"),
                    "is": v.get("is"),
                    "oos": v.get("oos"),
                    "by_side": v.get("by_side"),
                    "by_session": v.get("by_session"),
                    "by_window": v.get("by_window"),
                }
                break
    return out


def recommend(
    best_a: Dict[str, Any],
    best_b: Dict[str, Any],
    prior_a: Optional[Dict[str, Any]],
    prior_b: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(
        f"Best responsive A: {best_a['stop_id']} — OOS Exp ${best_a['oos']['expectancy']:.2f}/tr, "
        f"PF {best_a['oos']['profit_factor']:.2f}, DD ${best_a['oos']['max_drawdown']:.0f}."
    )
    lines.append(
        f"Best responsive B: {best_b['stop_id']} — OOS Exp ${best_b['oos']['expectancy']:.2f}/tr, "
        f"PF {best_b['oos']['profit_factor']:.2f}, DD ${best_b['oos']['max_drawdown']:.0f}."
    )
    winner = best_a if best_a["oos"]["expectancy"] >= best_b["oos"]["expectancy"] else best_b
    wv = winner["variant"]
    lines.append(f"Responsive winner (OOS expectancy): variant {wv} with {winner['stop_id']}.")

    if prior_a:
        pa = prior_a["oos"]
        lines.append(
            f"Prior live A (daily+15m+60m, 15m stop): OOS Exp ${pa['expectancy']:.2f}, "
            f"PF {pa['profit_factor']:.2f}, DD ${pa['max_drawdown']:.0f}."
        )
        if winner["oos"]["expectancy"] > pa["expectancy"]:
            lines.append(
                f"Responsive {wv} beats prior live A on OOS expectancy "
                f"(${winner['oos']['expectancy']:.2f} vs ${pa['expectancy']:.2f})."
            )
        else:
            lines.append(
                f"Prior live A still leads OOS expectancy "
                f"(${pa['expectancy']:.2f} vs best responsive ${winner['oos']['expectancy']:.2f})."
            )

    if prior_b:
        pb = prior_b["oos"]
        lines.append(
            f"Prior stack B (daily+15m+5m, 15m stop): OOS Exp ${pb['expectancy']:.2f}, "
            f"PF {pb['profit_factor']:.2f}, DD ${pb['max_drawdown']:.0f}."
        )
        best_a_oos = best_a["oos"]["expectancy"]
        if best_a_oos > pb["expectancy"]:
            lines.append(
                f"Responsive A (daily+15m+5m entry) with tighter stop beats prior B on OOS expectancy."
            )

    return " ".join(lines)


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out_json = os.path.join("data", "ema15_15_5_1_responsive.json")
    prior_path = os.path.join("data", "ema15_timeframe_stacks.json")

    print("MNQ responsive 15->5->1 + STOP MODEL GRID")
    print(f"Data: {csv}")
    print(
        "Trend: 15m direction + 5m confirm; 1m quiet entry only (body<=4). "
        "Sep gate: 15m sep/ATR>=0.45 noon+ with 5m trend for confidence."
    )
    print(
        f"Exits: 1R TP, flatten 15:50, {MAX_TRADES_DAY}/day, re-entry, max {DEFAULT_MAX_OPEN_LOTS} lots, "
        f"${PT}/pt ${COMM} RT"
    )
    print(f"Variants: {', '.join(f'{k}={v}' for k, v in VARIANTS.items())}")
    print(f"Stop configs: {len(STOP_CONFIGS)} per variant")
    print("IS/OOS split: before Jun 2026 / Jun 2026+")

    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    all_reports: List[Dict[str, Any]] = []
    by_variant: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}

    for variant in ("A", "B"):
        print(f"\n--- Variant {variant}: {VARIANTS[variant]} ---")
        for stop_cfg in STOP_CONFIGS:
            recs = simulate(df, variant=variant, stop_cfg=stop_cfg)
            rep = full_report(variant, stop_cfg, recs)
            all_reports.append(rep)
            by_variant[variant].append(rep)
            print_report(rep)

    ranked_a = rank_reports(by_variant["A"])
    ranked_b = rank_reports(by_variant["B"])
    best_a = ranked_a[0]
    best_b = ranked_b[0]

    print("\n" + "=" * 110)
    print("RANKED GRID - Variant A (daily+15m+5m -> quiet 1m)")
    print("=" * 110)
    for i, r in enumerate(ranked_a, 1):
        print(f"  #{i} {r['stop_id']:<18} OOS Exp=${r['oos']['expectancy']:7.2f} PF={r['oos']['profit_factor']:5.2f} DD=${r['oos']['max_drawdown']:7.1f}")

    print("\n" + "=" * 110)
    print("RANKED GRID - Variant B (15m+5m no daily -> quiet 1m)")
    print("=" * 110)
    for i, r in enumerate(ranked_b, 1):
        print(f"  #{i} {r['stop_id']:<18} OOS Exp=${r['oos']['expectancy']:7.2f} PF={r['oos']['profit_factor']:5.2f} DD=${r['oos']['max_drawdown']:7.1f}")

    print("\n" + "=" * 110)
    print("FULL REPORT - Best A vs Best B")
    print("=" * 110)
    print_full_variant_report(best_a)
    print_full_variant_report(best_b)

    prior = load_prior_reference(prior_path)
    prior_a = prior.get("prior_A_daily_15m_60m")
    prior_b = prior.get("prior_B_daily_15m_5m")

    if prior_a or prior_b:
        print("\n" + "=" * 110)
        print("PRIOR TIMEFRAME_STACKS REFERENCE (15m ATRx2 stop)")
        print("=" * 110)
        if prior_a:
            o = prior_a["oos"]
            print(
                f"  prior A daily+15m+60m: OOS n={o['trades']} Exp=${o['expectancy']:.2f} "
                f"PF={o['profit_factor']:.2f} DD=${o['max_drawdown']:.1f}"
            )
        if prior_b:
            o = prior_b["oos"]
            print(
                f"  prior B daily+15m+5m:  OOS n={o['trades']} Exp=${o['expectancy']:.2f} "
                f"PF={o['profit_factor']:.2f} DD=${o['max_drawdown']:.1f}"
            )

    rec = recommend(best_a, best_b, prior_a, prior_b)
    print("\nRECOMMENDATION")
    print(f"  {rec}")

    payload = {
        "csv": csv,
        "methodology": {
            "trend_stack": "15m primary + 5m confirm; 1m quiet entry only (NOT in stack)",
            "entry": "quiet 1m body≤4, close with trend",
            "sep_gate": "15m sep/ATR≥0.45 from 12:00 ET; confidence uses 5m trend + 15m sep",
            "tp": f"{TP_R}R",
            "flatten": "15:50 ET",
            "costs": f"${PT}/pt + ${COMM} RT",
            "is_oos_split": "Jun 2026",
        },
        "variants": VARIANTS,
        "stop_configs": STOP_CONFIGS,
        "grid": all_reports,
        "ranked_A": [r["stop_id"] for r in ranked_a],
        "ranked_B": [r["stop_id"] for r in ranked_b],
        "best_A": best_a,
        "best_B": best_b,
        "top3_A": ranked_a[:3],
        "top3_B": ranked_b[:3],
        "prior_reference": prior,
        "recommendation": rec,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nJSON saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
