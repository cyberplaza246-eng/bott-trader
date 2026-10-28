#!/usr/bin/env python3
"""
Validation phase: frozen combined spec — no optimization.

Frozen spec:
  LONG: 15/5/1 bullish, quiet 1m, ATR×1.5 [6-25], 1R, windows 9:35 + 11:00
  SHORT: 15/5/1 bearish, LH_LL, 1m pullback, same stop, 1R, all six windows
  One position, no VWAP, no August filters

Outputs:
  - Full-sample and extended-data metrics
  - Monthly P&L breakdown
  - Rolling walk-forward: 60 trading-day warmup → 20-day OOS folds (no tuning)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt

ET = combo.ET
FROZEN_SPEC = {
    "long": combo.LONG_RECIPE_W2,
    "short": combo.SHORT_RECIPE,
    "combined": "one_position long_w2_open short_w6_all",
    "frozen": True,
    "no_optimization": True,
}

TRAIN_DAYS = 60
OOS_DAYS = 20
SUPP_TRAIN_DAYS = 40
SUPP_OOS_DAYS = 15


def trade_entry_date(t: scalp_bt.ScalpTrade) -> pd.Timestamp:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ET).normalize()


def trading_dates_from_df(df: pd.DataFrame) -> List[pd.Timestamp]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True)).tz_convert(ET)
    days = sorted({ts.normalize() for ts in dt})
    # weekdays only
    return [d for d in days if d.weekday() < 5]


def metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    if not trades:
        z = resp.group_metrics([])
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2),
        "total_pnl": round(m["total_pnl"], 2),
        "expectancy": round(m["expectancy"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
    }


def leg_metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Dict]:
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    return {"long": metrics(longs), "short": metrics(shorts)}


def monthly_breakdown(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[scalp_bt.ScalpTrade]] = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        buckets.setdefault(key, []).append(t)
    return {k: {**metrics(v), **leg_metrics(v)} for k, v in sorted(buckets.items())}


def walk_forward_folds(
    trades: List[scalp_bt.ScalpTrade],
    trading_dates: List[pd.Timestamp],
    train_days: int = TRAIN_DAYS,
    oos_days: int = OOS_DAYS,
) -> List[Dict[str, Any]]:
    folds: List[Dict[str, Any]] = []
    idx = train_days
    while idx + oos_days <= len(trading_dates):
        oos_start = trading_dates[idx]
        oos_end = trading_dates[idx + oos_days - 1]
        oos_trades = [
            t for t in trades
            if oos_start <= trade_entry_date(t) <= oos_end
        ]
        fold = {
            "fold": len(folds) + 1,
            "train_days": train_days,
            "oos_trading_days": oos_days,
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "combined": metrics(oos_trades),
            "legs": leg_metrics(oos_trades),
        }
        folds.append(fold)
        idx += oos_days
    return folds


def summarize_folds(folds: List[Dict]) -> Dict[str, Any]:
    if not folds:
        return {}
    nets = [f["combined"]["net_pnl"] for f in folds]
    pfs = [f["combined"]["profit_factor"] for f in folds if f["combined"]["trades"] > 0]
    positive = sum(1 for n in nets if n > 0)
    return {
        "fold_count": len(folds),
        "positive_fold_count": positive,
        "positive_fold_pct": round(100.0 * positive / len(folds), 1),
        "total_oos_net": round(sum(nets), 2),
        "mean_fold_net": round(float(np.mean(nets)), 2),
        "median_fold_net": round(float(np.median(nets)), 2),
        "mean_pf": round(float(np.mean(pfs)), 2) if pfs else 0.0,
    }


def run_validation(csv_path: str, label: str) -> Dict[str, Any]:
    df = official.load_1m(csv_path)
    a = combo.prepare(df)
    trades = combo.simulate_combined_recommended(a)
    trading_dates = trading_dates_from_df(df)

    ins, oos = edge.split_is_oos(trades)
    fresh = combo.filter_trades(trades, combo.FRESH_START)

    folds = walk_forward_folds(trades, trading_dates)
    folds_supp = walk_forward_folds(
        trades, trading_dates, train_days=SUPP_TRAIN_DAYS, oos_days=SUPP_OOS_DAYS,
    )
    monthly = monthly_breakdown(trades)
    monthly_oos = monthly_breakdown(oos)

    return {
        "label": label,
        "csv": csv_path,
        "bars": len(df),
        "range": f"{df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}",
        "trading_days": len(trading_dates),
        "overall": metrics(trades),
        "is": metrics(ins),
        "oos_standard": metrics(oos),
        "fresh_aug_plus": metrics(fresh),
        "legs_overall": leg_metrics(trades),
        "legs_oos": leg_metrics(oos),
        "pass_standard_oos": hunt.is_pass(metrics(ins), metrics(oos)),
        "monthly_all": monthly,
        "monthly_oos": monthly_oos,
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "oos_days": OOS_DAYS,
            "folds": folds,
            "summary": summarize_folds(folds),
            "note": (
                f"Only {len(folds)} fold(s) at 60/20 — dataset has {len(trading_dates)} trading days."
                if len(folds) < 3
                else None
            ),
        },
        "walk_forward_supplementary": {
            "train_days": SUPP_TRAIN_DAYS,
            "oos_days": SUPP_OOS_DAYS,
            "folds": folds_supp,
            "summary": summarize_folds(folds_supp),
            "note": "Supplementary smaller windows for stability view — not primary spec.",
        },
    }


def print_monthly(title: str, monthly: Dict[str, Dict]) -> None:
    print(f"\n{title}")
    for month, m in monthly.items():
        sign = "+" if m["net_pnl"] > 0 else "−"
        print(
            f"  {month}: n={m['trades']:3d} PF={m['profit_factor']:.2f} "
            f"${m['net_pnl']:7.0f} ({sign})  "
            f"L=${m['long']['net_pnl']:6.0f} S=${m['short']['net_pnl']:6.0f}"
        )


def print_report(rep: Dict[str, Any]) -> None:
    print(f"\n{'=' * 90}")
    print(f"{rep['label']}  |  {rep['range']}")
    print(f"{'=' * 90}")
    for key, title in (
        ("overall", "ALL"),
        ("is", "IS (<Jun 2026)"),
        ("oos_standard", "OOS (>=Jun 2026)"),
        ("fresh_aug_plus", "Fresh (>=Aug 2026)"),
    ):
        m = rep[key]
        print(
            f"  {title:18s} n={m['trades']:3d} WR={m['win_rate']:5.1f}% "
            f"PF={m['profit_factor']:5.2f} Net=${m['net_pnl']:8.1f} DD=${m['max_drawdown']:7.1f}"
        )
    lo, so = rep["legs_oos"]["long"], rep["legs_oos"]["short"]
    print(f"  OOS legs: long PF={lo['profit_factor']:.2f} ${lo['net_pnl']:.0f} | short PF={so['profit_factor']:.2f} ${so['net_pnl']:.0f}")
    flag = "PASS" if rep["pass_standard_oos"] else "FAIL"
    print(f"  Standard OOS gate: [{flag}]")

    print_monthly("Monthly (all trades)", rep["monthly_all"])
    print_monthly("Monthly (OOS only)", rep["monthly_oos"])

    wf = rep["walk_forward"]
    print(f"\nWalk-forward ({wf['train_days']}d warmup → {wf['oos_days']}d OOS, frozen spec)")
    if wf.get("note"):
        print(f"  Note: {wf['note']}")
    s = wf["summary"]
    if s:
        print(
            f"  Folds: {s['fold_count']} | positive: {s['positive_fold_count']}/{s['fold_count']} "
            f"({s['positive_fold_pct']}%) | total OOS net ${s['total_oos_net']:.0f} | mean PF {s['mean_pf']:.2f}"
        )
    for f in wf["folds"]:
        c = f["combined"]
        sign = "+" if c["net_pnl"] > 0 else "−"
        print(
            f"  Fold {f['fold']:2d} {f['oos_start']}..{f['oos_end']}  "
            f"n={c['trades']:3d} PF={c['profit_factor']:.2f} ${c['net_pnl']:7.0f} ({sign})"
        )

    wfs = rep["walk_forward_supplementary"]
    print(f"\nSupplementary walk-forward ({wfs['train_days']}d → {wfs['oos_days']}d)")
    s2 = wfs["summary"]
    if s2:
        print(
            f"  Folds: {s2['fold_count']} | positive: {s2['positive_fold_count']}/{s2['fold_count']} "
            f"({s2['positive_fold_pct']}%) | total net ${s2['total_oos_net']:.0f}"
        )
    for f in wfs["folds"]:
        c = f["combined"]
        sign = "+" if c["net_pnl"] > 0 else "−"
        print(
            f"  Fold {f['fold']:2d} {f['oos_start']}..{f['oos_end']}  "
            f"n={c['trades']:3d} PF={c['profit_factor']:.2f} ${c['net_pnl']:7.0f} ({sign})"
        )


def interpret(reps: List[Dict[str, Any]]) -> str:
    rep = reps[-1]
    monthly = rep["monthly_oos"]
    wf = rep["walk_forward"]["summary"]
    aug = monthly.get("2026-08", {}).get("net_pnl", 0)
    post_aug = [m for k, m in monthly.items() if k > "2026-08"]
    post_neg = sum(1 for m in post_aug if m["net_pnl"] < 0)
    post_pos = sum(1 for m in post_aug if m["net_pnl"] > 0)

    parts = []
    if rep["pass_standard_oos"]:
        parts.append("Standard OOS gates PASS on frozen combined spec.")
    if aug < 0:
        parts.append(f"August OOS negative (${aug:.0f}).")
    if post_aug:
        if post_neg > post_pos:
            parts.append("Post-August months mostly negative — suggests degradation pattern.")
        elif post_pos > 0 and post_neg == 0:
            parts.append("Post-August months positive — August may be outlier.")
        else:
            parts.append("Post-August mixed — inconclusive; need more months.")
    if wf.get("positive_fold_pct", 0) >= 60:
        parts.append(f"Walk-forward: {wf['positive_fold_pct']}% folds positive.")
    elif wf.get("fold_count"):
        parts.append(f"Walk-forward: only {wf.get('positive_fold_pct', 0)}% folds positive — stability concern.")

    parts.append("MQL5: wait until extended data confirms stability.")
    return " ".join(parts)


def main() -> int:
    out = os.path.join("data", "frozen_combined_validation.json")
    csv = os.path.join("data", "MNQ_1m.csv")

    print("FROZEN SPEC VALIDATION — no optimization")
    print(json.dumps(FROZEN_SPEC, indent=2, default=str))
    print()

    reps = [run_validation(csv, "MNQ_1m.csv (current)")]

    for rep in reps:
        print_report(rep)

    verdict = interpret(reps)
    print(f"\n{'=' * 90}")
    print(f"Interpretation: {verdict}")

    payload = {
        "frozen_spec": FROZEN_SPEC,
        "validation": reps,
        "interpretation": verdict,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
