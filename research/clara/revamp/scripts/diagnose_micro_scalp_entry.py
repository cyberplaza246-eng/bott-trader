"""
Entry diagnostic on frozen micro-scalp signals.

Control baseline (do not overwrite lightly):
  OOS exp −$3.02 | OOS PF 0.71 | WR 47.5% | MFE med 14.8 | MAE med 16.0

Discovery on IS only (before 2026-06-01).
Validation: 2026-06-01 .. 2026-07-31.
Untouched OOS report: 2026-08-01 onward (never used to pick).

One condition at a time. No exit retune — path MFE/MAE + baseline $20 exit expectancy.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.mnq_context import compute_vwap
from src.strategy.day_hybrid import entry_pullback, prepare_arrays, stack_side_15_5_1
from src.strategy.mnq_15m_ema_eod import (
    ENTRY_MINUTE,
    FLAT_MINUTE,
    sl_pts_from_atr,
)

PT = 2.0
COMM = 1.24
SLIP = 0.25
HORIZON = 15
IS_END = date(2026, 6, 1)
VAL_END = date(2026, 8, 1)  # Jun–Jul = validation; Aug+ = untouched OOS
MFE_LEVELS = (4, 6, 8, 10, 12, 15, 20)

# Locked control metrics from exit-repair run (report comparison)
CONTROL = {
    "oos_exp": -3.02,
    "oos_pf": 0.71,
    "oos_wr": 47.5,
    "mfe_median": 14.8,
    "mae_median": 16.0,
    "source": "data/micro_scalp_exit_repair.json baseline OOS + path medians",
}


def collect_entries(a: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = len(a["cl"])
    out = []
    for i in range(2, n - 1):
        if int(a["dow"][i]) >= 5:
            continue
        m = int(a["mins"][i])
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        side = stack_side_15_5_1(i, a)
        if side == 0 or not entry_pullback(side, i, a):
            continue
        j = i + 1
        if j >= n or a["dates"][j] != a["dates"][i]:
            continue
        if int(a["mins"][j]) >= FLAT_MINUTE:
            continue
        atr = float(a["atr1"][i])
        atr_sl = (
            sl_pts_from_atr(atr, mult=1.0, sl_min=6.0, sl_max=12.0)
            if atr == atr and atr > 0
            else float("nan")
        )
        out.append({
            "sig": i,
            "fill_i": j,
            "side": int(side),
            "date": a["dates"][j],
            "mins": int(a["mins"][j]),
            "dow": int(a["dow"][j]),
            "raw_open": float(a["op"][j]),
            "atr": atr,
            "atr_sl": atr_sl,
            "op": float(a["op"][i]),
            "cl": float(a["cl"][i]),
            "cl_prev": float(a["cl"][i - 1]),
            "cl_prev2": float(a["cl"][i - 2]),
        })
    return out


def path_mfe_mae(a, e) -> Dict[str, Any]:
    j = e["fill_i"]
    side = e["side"]
    fill = e["raw_open"] + (SLIP if side > 0 else -SLIP)
    n = len(a["cl"])
    last = min(j + HORIZON, n)
    mfe = mae = 0.0
    t_mfe = t_mae = 0
    hit_p10_before_m10 = False
    seen_p10 = seen_m10 = False
    bars = 0
    for k in range(j, last):
        if a["dates"][k] != a["dates"][j] or int(a["mins"][k]) >= FLAT_MINUTE:
            break
        bars = k - j + 1
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            fav, adv = hi - fill, fill - lo
        else:
            fav, adv = fill - lo, hi - fill
        if fav > mfe:
            mfe, t_mfe = fav, bars
        if adv > mae:
            mae, t_mae = adv, bars
        if (not seen_p10) and fav >= 10:
            seen_p10 = True
            if not seen_m10:
                hit_p10_before_m10 = True
        if (not seen_m10) and adv >= 10:
            seen_m10 = True
    return {
        "mfe": float(mfe),
        "mae": float(mae),
        "t_mfe": int(t_mfe),
        "t_mae": int(t_mae),
        "p10_before_m10": hit_p10_before_m10,
        "bars": bars,
        "fill": fill,
    }


def baseline_exit_pnl(a, e) -> float:
    """Frozen $20 / ATR stop / 3-bar max; stop before target on same bar."""
    j = e["fill_i"]
    side = e["side"]
    sl = e["atr_sl"]
    if sl != sl or sl <= 0:
        return float("nan")
    fill = e["raw_open"] + (SLIP if side > 0 else -SLIP)
    stop = fill - sl if side > 0 else fill + sl
    tgt = fill + 10.0 if side > 0 else fill - 10.0
    n = len(a["cl"])
    last = min(j + 3, n)
    for k in range(j, last):
        if a["dates"][k] != a["dates"][j] or int(a["mins"][k]) >= FLAT_MINUTE:
            cl = float(a["cl"][max(j, k - 1)])
            px = cl - SLIP if side > 0 else cl + SLIP
            pts = (px - fill) if side > 0 else (fill - px)
            return pts * PT - COMM
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        hit_sl = (lo <= stop) if side > 0 else (hi >= stop)
        hit_tp = (hi >= tgt) if side > 0 else (lo <= tgt)
        if hit_sl:  # includes ambiguous
            pts = -sl - SLIP  # approx adverse fill
            # exact:
            px = stop - SLIP if side > 0 else stop + SLIP
            pts = (px - fill) if side > 0 else (fill - px)
            return pts * PT - COMM
        if hit_tp:
            px = tgt - SLIP if side > 0 else tgt + SLIP
            pts = (px - fill) if side > 0 else (fill - px)
            return pts * PT - COMM
    cl = float(a["cl"][last - 1])
    px = cl - SLIP if side > 0 else cl + SLIP
    pts = (px - fill) if side > 0 else (fill - px)
    return pts * PT - COMM


def enrich(a, entries, df_1m: pd.DataFrame) -> List[Dict[str, Any]]:
    work = df_1m.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True)
    if "volume" not in work.columns:
        work["volume"] = 1.0
    vwap_arr = compute_vwap(work).to_numpy(dtype=float)
    dt = pd.DatetimeIndex(work["datetime"])
    # session hi/lo running on RTH day
    session_hi = np.full(len(dt), np.nan)
    session_lo = np.full(len(dt), np.nan)
    et = dt.tz_convert("America/New_York")
    dates = np.array(et.date)
    mins = (et.hour * 60 + et.minute).to_numpy()
    cur_d = None
    shi = slo = None
    for i in range(len(dt)):
        if mins[i] < 9 * 60 + 30 or mins[i] >= 16 * 60:
            continue
        if dates[i] != cur_d:
            cur_d = dates[i]
            shi = float(a["hi"][i])
            slo = float(a["lo"][i])
        else:
            shi = max(shi, float(a["hi"][i]))
            slo = min(slo, float(a["lo"][i]))
        session_hi[i] = shi
        session_lo[i] = slo

    # volume regime: 1m vol vs 20-bar median
    vol = df_1m["volume"].to_numpy(dtype=float) if "volume" in df_1m.columns else np.ones(len(df_1m))
    vol_med = pd.Series(vol).rolling(20, min_periods=5).median().shift(1).to_numpy()

    # ATR percentile ranks on IS-style rolling — use expanding percentile of atr1
    atr = a["atr1"]
    atr_s = pd.Series(atr)
    atr_p50 = atr_s.rolling(500, min_periods=50).median().shift(1).to_numpy()

    t15 = a["t15"]
    rows = []
    for e in entries:
        i, j = e["sig"], e["fill_i"]
        path = path_mfe_mae(a, e)
        pnl = baseline_exit_pnl(a, e)
        px = e["raw_open"]
        vw = float(vwap_arr[i]) if vwap_arr[i] == vwap_arr[i] else float("nan")
        dist_vwap = (px - vw) if vw == vw else float("nan")
        shi = session_hi[j]
        slo = session_lo[j]
        rng = (shi - slo) if (shi == shi and slo == slo and shi > slo) else float("nan")
        pos_in_range = ((px - slo) / rng) if rng == rng and rng > 0 else float("nan")
        body = abs(e["cl"] - e["op"])
        prior_move = e["cl_prev"] - e["cl_prev2"]
        # structure vs 15m: side already aligned to stack; "with 15m" always true for this entry
        # counter would be empty — use 1m vs 15m of prior bar loosely: already same stack
        vm = vol_med[i]
        vol_ratio = (vol[i] / vm) if vm == vm and vm > 0 else float("nan")
        atr_med = atr_p50[i]
        atr_reg = "high" if (e["atr"] == e["atr"] and atr_med == atr_med and e["atr"] >= atr_med) else (
            "low" if e["atr"] == e["atr"] and atr_med == atr_med else "na"
        )
        m = e["mins"]
        if m < 10 * 60 + 30:
            tod = "open_0935_1030"
        elif m < 12 * 60:
            tod = "late_morn_1030_1200"
        elif m < 14 * 60:
            tod = "midday_1200_1400"
        else:
            tod = "afternoon_1400_1550"
        rows.append({
            **e,
            **path,
            "pnl": float(pnl) if pnl == pnl else 0.0,
            "long": e["side"] > 0,
            "tod": tod,
            "atr_reg": atr_reg,
            "dist_vwap": dist_vwap,
            "vwap_side": (
                "above" if dist_vwap == dist_vwap and dist_vwap > 0 else
                ("below" if dist_vwap == dist_vwap else "na")
            ),
            "range_pos": pos_in_range,
            "near_high": pos_in_range == pos_in_range and pos_in_range >= 0.8,
            "near_low": pos_in_range == pos_in_range and pos_in_range <= 0.2,
            "mid_range": pos_in_range == pos_in_range and 0.35 <= pos_in_range <= 0.65,
            "vol_high": vol_ratio == vol_ratio and vol_ratio >= 1.25,
            "vol_low": vol_ratio == vol_ratio and vol_ratio <= 0.75,
            "body_tiny": body <= 1.0,
            "body_quiet": body <= 2.0,
            "prior_against": (e["side"] > 0 and prior_move < 0) or (e["side"] < 0 and prior_move > 0),
            "prior_with": (e["side"] > 0 and prior_move > 0) or (e["side"] < 0 and prior_move < 0),
            "dow_name": ["Mon", "Tue", "Wed", "Thu", "Fri"][e["dow"]] if e["dow"] < 5 else "?",
            "t15": int(t15[i]),
        })
    return rows


def split(rows, start: Optional[date], end: Optional[date]):
    out = []
    for r in rows:
        d = r["date"]
        if start is not None and d < start:
            continue
        if end is not None and d >= end:
            continue
        out.append(r)
    return out


def bucket_stats(rows: List[Dict]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "mfe": None, "mae": None, "edge": None, "p10_b_m10": None, "exp": None, "pf": None, "wr": None}
    mfe = np.array([r["mfe"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    pnl = np.array([r["pnl"] for r in rows])
    w = pnl[pnl > 0].sum()
    l = pnl[pnl < 0].sum()
    pf = float(w / abs(l)) if l < 0 else (99.0 if w > 0 else 0.0)
    return {
        "n": len(rows),
        "mfe": round(float(np.median(mfe)), 2),
        "mae": round(float(np.median(mae)), 2),
        "edge": round(float(np.median(mfe) - np.median(mae)), 2),
        "p10_b_m10": round(100.0 * float(np.mean([r["p10_before_m10"] for r in rows])), 1),
        "exp": round(float(pnl.mean()), 2),
        "pf": round(pf, 3),
        "wr": round(100.0 * float((pnl > 0).mean()), 1),
        "net": round(float(pnl.sum()), 2),
    }


def fmt_row(name: str, st: Dict) -> str:
    if st["n"] == 0:
        return f"| {name:28} | {0:6} | {'—':>5} | {'—':>5} | {'—':>5} | {'—':>6} | {'—':>7} |"
    return (
        f"| {name:28} | {st['n']:6} | {st['mfe']:5.1f} | {st['mae']:5.1f} | "
        f"{st['edge']:+5.1f} | {st['p10_b_m10']:5.1f}% | {st['exp']:+7.2f} |"
    )


def main():
    df = pd.read_csv("data/MNQ_1m.csv")
    print(f"bars={len(df)}", flush=True)
    a = prepare_arrays(df)
    entries = collect_entries(a)
    print(f"frozen_entries={len(entries)}", flush=True)
    rows = enrich(a, entries, df)
    is_rows = split(rows, None, IS_END)
    val_rows = split(rows, IS_END, VAL_END)
    oos_rows = split(rows, VAL_END, None)  # Aug+ untouched
    print(f"IS={len(is_rows)} VAL(JunJul)={len(val_rows)} OOS(Aug+)={len(oos_rows)}", flush=True)

    conditions: List[Tuple[str, Callable]] = [
        ("All", lambda r: True),
        ("Long", lambda r: r["long"]),
        ("Short", lambda r: not r["long"]),
        ("ATR high", lambda r: r["atr_reg"] == "high"),
        ("ATR low", lambda r: r["atr_reg"] == "low"),
        ("Above VWAP", lambda r: r["vwap_side"] == "above"),
        ("Below VWAP", lambda r: r["vwap_side"] == "below"),
        ("Near session high", lambda r: r["near_high"]),
        ("Near session low", lambda r: r["near_low"]),
        ("Mid session range", lambda r: r["mid_range"]),
        ("Vol high", lambda r: r["vol_high"]),
        ("Vol low", lambda r: r["vol_low"]),
        ("Open 09:35-10:30", lambda r: r["tod"] == "open_0935_1030"),
        ("Late morn 10:30-12", lambda r: r["tod"] == "late_morn_1030_1200"),
        ("Midday 12-14", lambda r: r["tod"] == "midday_1200_1400"),
        ("Afternoon 14-15:50", lambda r: r["tod"] == "afternoon_1400_1550"),
        ("Body <=1", lambda r: r["body_tiny"]),
        ("Body <=2", lambda r: r["body_quiet"]),
        ("Prior bar against", lambda r: r["prior_against"]),
        ("Prior bar with", lambda r: r["prior_with"]),
        ("Mon", lambda r: r["dow_name"] == "Mon"),
        ("Tue", lambda r: r["dow_name"] == "Tue"),
        ("Wed", lambda r: r["dow_name"] == "Wed"),
        ("Thu", lambda r: r["dow_name"] == "Thu"),
        ("Fri", lambda r: r["dow_name"] == "Fri"),
        # single combined hypothesis candidates (still one concept each)
        ("Long+ATR high", lambda r: r["long"] and r["atr_reg"] == "high"),
        ("Short+ATR high", lambda r: (not r["long"]) and r["atr_reg"] == "high"),
        ("Open+ATR high", lambda r: r["tod"] == "open_0935_1030" and r["atr_reg"] == "high"),
        ("Long+above VWAP", lambda r: r["long"] and r["vwap_side"] == "above"),
        ("Short+below VWAP", lambda r: (not r["long"]) and r["vwap_side"] == "below"),
        ("Open+vol high", lambda r: r["tod"] == "open_0935_1030" and r["vol_high"]),
        ("Afternoon+ATR low", lambda r: r["tod"] == "afternoon_1400_1550" and r["atr_reg"] == "low"),
    ]

    header = (
        "| Condition                    | Trades |   MFE |   MAE | Edge | +10<-10 |   Exp $ |"
    )
    sep = "| ---------------------------- | -----: | ----: | ----: | ---: | ------: | ------: |"

    def table(label, subset):
        print(f"\n### {label}", flush=True)
        print(header, flush=True)
        print(sep, flush=True)
        out = []
        for name, fn in conditions:
            st = bucket_stats([r for r in subset if fn(r)])
            print(fmt_row(name, st), flush=True)
            out.append({"condition": name, **st})
        return out

    is_tab = table("IS (discovery — before 2026-06-01)", is_rows)
    val_tab = table("VALIDATION (Jun–Jul 2026)", val_rows)
    oos_tab = table("UNTOUCHED OOS (Aug 2026+ — report only)", oos_rows)

    # Find IS conditions where median MFE > MAE and n>=80 and edge>=2
    interesting = [
        r for r in is_tab
        if r["n"] >= 80 and r["edge"] is not None and r["edge"] >= 2.0
    ]
    interesting.sort(key=lambda r: r["edge"], reverse=True)

    print("\n=== IS candidates (MFE-MAE edge >= 2pts, n>=80) ===", flush=True)
    if not interesting:
        print("NONE", flush=True)
    for r in interesting:
        # check validation same condition
        val = next(x for x in val_tab if x["condition"] == r["condition"])
        oos = next(x for x in oos_tab if x["condition"] == r["condition"])
        print(
            f"{r['condition']}: IS edge={r['edge']} exp={r['exp']} n={r['n']} | "
            f"VAL edge={val['edge']} exp={val['exp']} n={val['n']} | "
            f"OOS(report) edge={oos['edge']} exp={oos['exp']} n={oos['n']}",
            flush=True,
        )

    # Soft: edge >= 1
    soft = [r for r in is_tab if r["n"] >= 80 and r["edge"] is not None and r["edge"] >= 1.0]
    soft.sort(key=lambda r: r["edge"], reverse=True)
    print("\n=== IS soft (edge >= 1pt, n>=80) ===", flush=True)
    for r in soft[:15]:
        val = next(x for x in val_tab if x["condition"] == r["condition"])
        print(
            f"{r['condition']}: IS edge={r['edge']} exp={r['exp']} | VAL edge={val['edge']} exp={val['exp']} n={val['n']}",
            flush=True,
        )

    # Best IS by expectancy (still discovery only)
    by_exp = sorted([r for r in is_tab if r["n"] >= 80], key=lambda r: r["exp"] or -999, reverse=True)
    print("\n=== Best IS expectancy (n>=80) — discovery only ===", flush=True)
    for r in by_exp[:8]:
        val = next(x for x in val_tab if x["condition"] == r["condition"])
        print(
            f"{r['condition']}: IS exp={r['exp']} pf={r['pf']} edge={r['edge']} | "
            f"VAL exp={val['exp']} pf={val['pf']} edge={val['edge']}",
            flush=True,
        )

    # Hypothesis gate: IS edge>=2 AND VAL edge>0 AND VAL exp > control path isn't enough —
    # require VAL exp > -1 at least improvement and edge>0
    survivors = []
    for r in interesting:
        val = next(x for x in val_tab if x["condition"] == r["condition"])
        if val["n"] >= 20 and val["edge"] is not None and val["edge"] > 0 and (val["exp"] or -999) > CONTROL["oos_exp"]:
            survivors.append({"is": r, "val": val})

    all_is = bucket_stats(is_rows)
    all_val = bucket_stats(val_rows)
    all_oos = bucket_stats(oos_rows)

    report = {
        "control_baseline": CONTROL,
        "sample": {
            "frozen_entries": len(entries),
            "is": len(is_rows),
            "validation_jun_jul": len(val_rows),
            "untouched_oos_aug_plus": len(oos_rows),
        },
        "all": {"is": all_is, "val": all_val, "oos_aug_plus": all_oos},
        "is_table": is_tab,
        "val_table": val_tab,
        "oos_table": oos_tab,
        "is_candidates_edge_ge_2": interesting,
        "validation_survivors": survivors,
        "verdict": (
            "CONDITIONAL_EDGE_CANDIDATES" if survivors else
            ("IS_EDGE_NO_VAL" if interesting else "NO_CONDITIONAL_EDGE")
        ),
        "note": (
            "Filters are diagnostic only. Untouched OOS must not choose the next hypothesis. "
            "One filter at a time for the next experiment."
        ),
    }
    path = "data/micro_scalp_entry_diagnostic.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    # Persist control baseline file
    with open("data/micro_scalp_control_baseline.json", "w", encoding="utf-8") as f:
        json.dump({
            **CONTROL,
            "locked": True,
            "beat_this_oos": True,
            "all_path_is": all_is,
            "all_path_val": all_val,
            "all_path_oos_aug_plus": all_oos,
        }, f, indent=2)
    print(f"\nVERDICT {report['verdict']}", flush=True)
    print(f"wrote {path}", flush=True)
    print("wrote data/micro_scalp_control_baseline.json", flush=True)


if __name__ == "__main__":
    main()
