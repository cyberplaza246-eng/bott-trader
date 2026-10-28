"""
EXIT-ONLY repair for frozen micro-scalp entries.

Entry frozen: 15m+5m+1m stack + quiet 1m pullback (same as micro_scalp).
Same-bar ambiguity: STOP before TARGET (conservative).
Sep never used to choose; OOS = Jun+; IS = before Jun 2026.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.day_hybrid import entry_pullback, prepare_arrays, stack_side_15_5_1
from src.strategy.mnq_15m_ema_eod import ENTRY_MINUTE, FLAT_MINUTE, sl_pts_from_atr

PT = 2.0
COMM_BASE = 1.24
SLIP_BASE = 0.25
OOS = date(2026, 6, 1)
AUG = date(2026, 8, 1)
SEP = date(2026, 9, 1)
YEAR_CUTS = [
    (date(2024, 9, 22), date(2025, 1, 1), "2024q4"),
    (date(2025, 1, 1), date(2025, 7, 1), "2025h1"),
    (date(2025, 7, 1), date(2026, 1, 1), "2025h2"),
    (date(2026, 1, 1), date(2026, 6, 1), "2026pre"),
    (date(2026, 6, 1), date(2026, 9, 22), "2026oos"),
]
# Horizon for MAE/MFE path (bars). 15 min covers longest time-stop tests.
HORIZON = 15
MFE_LEVELS = (4, 6, 8, 10, 12, 15, 20)


def collect_entries(a: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Frozen entry list. Fill = next open ± slip (slip applied later)."""
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
        out.append(
            {
                "sig": i,
                "fill_i": j,
                "side": int(side),
                "date": a["dates"][j],
                "raw_open": float(a["op"][j]),
                "atr": atr,
                "atr_sl": atr_sl,
            }
        )
    return out


def path_excursions(
    a: Dict[str, Any],
    e: Dict[str, Any],
    *,
    slip: float,
) -> Dict[str, Any]:
    """Walk up to HORIZON bars; stop only at EOD. Record MAE/MFE in points."""
    j = e["fill_i"]
    side = e["side"]
    fill = e["raw_open"] + (slip if side > 0 else -slip)
    n = len(a["cl"])
    last = min(j + HORIZON, n)
    mfe = 0.0
    mae = 0.0
    t_mfe = 0
    t_mae = 0
    first_hit = {lv: None for lv in MFE_LEVELS}
    bars = 0
    for k in range(j, last):
        if a["dates"][k] != a["dates"][j] or int(a["mins"][k]) >= FLAT_MINUTE:
            break
        bars = k - j + 1
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            fav = hi - fill
            adv = fill - lo
        else:
            fav = fill - lo
            adv = hi - fill
        if fav > mfe:
            mfe = fav
            t_mfe = bars
        if adv > mae:
            mae = adv
            t_mae = bars
        for lv in MFE_LEVELS:
            if first_hit[lv] is None and fav >= lv:
                first_hit[lv] = bars
    # close at last path bar for "eventual" reference close excursion
    k_end = j + bars - 1 if bars else j
    cl = float(a["cl"][k_end])
    if side > 0:
        close_pts = cl - fill - slip
    else:
        close_pts = fill - cl - slip
    return {
        "fill": fill,
        "mfe": float(mfe),
        "mae": float(mae),
        "t_mfe": int(t_mfe),
        "t_mae": int(t_mae),
        "first_hit": first_hit,
        "bars": int(bars),
        "close_pts": float(close_pts),
        "date": e["date"],
        "side": side,
        "atr_sl": e["atr_sl"],
        "atr": e["atr"],
        "fill_i": j,
        "sig": e["sig"],
    }


def simulate_exit(
    a: Dict[str, Any],
    e: Dict[str, Any],
    *,
    sl_pts: float,
    tp_pts: float,
    max_bars: int,
    slip: float,
    comm: float,
    be_at: Optional[float] = None,
    partial_r: Optional[float] = None,
    runner_r: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Conservative: on same bar, stop checked before target.
    partial_r: take half at partial_r * sl, rest to runner_r * sl.
    be_at: once MFE >= be_at pts, move stop to entry (± slip).
    """
    j = e["fill_i"]
    side = e["side"]
    fill = e["raw_open"] + (slip if side > 0 else -slip)
    stop = fill - sl_pts if side > 0 else fill + sl_pts
    target = fill + tp_pts if side > 0 else fill - tp_pts
    be_armed = False
    half_done = False
    pnl = 0.0
    reason = "time"
    n = len(a["cl"])
    last = min(j + max_bars, n)
    k_exit = j
    for k in range(j, last):
        if a["dates"][k] != a["dates"][j] or int(a["mins"][k]) >= FLAT_MINUTE:
            k_exit = max(j, k - 1)
            cl = float(a["cl"][k_exit])
            exit_px = cl - slip if side > 0 else cl + slip
            rem = 0.5 if half_done else 1.0
            pts = (exit_px - fill) if side > 0 else (fill - exit_px)
            pnl += pts * PT * rem
            reason = "eod"
            break
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            fav = hi - fill
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            fav = fill - lo
            hit_stop = hi >= stop
            hit_tgt = lo <= target

        if be_at is not None and (not be_armed) and fav >= be_at:
            be_armed = True
            stop = fill  # breakeven (fill already includes entry slip)

        # partial then runner
        if partial_r is not None and runner_r is not None and not half_done:
            part_px = fill + partial_r * sl_pts if side > 0 else fill - partial_r * sl_pts
            hit_part = (hi >= part_px) if side > 0 else (lo <= part_px)
            if hit_stop and hit_part:
                # adverse first
                exit_px = stop - slip if side > 0 else stop + slip
                pts = (exit_px - fill) if side > 0 else (fill - exit_px)
                pnl = pts * PT - comm
                reason = "stop"
                k_exit = k
                break
            if hit_part:
                half_done = True
                pts = partial_r * sl_pts
                pnl += pts * PT * 0.5
                target = fill + runner_r * sl_pts if side > 0 else fill - runner_r * sl_pts
                continue

        if hit_stop and hit_tgt:
            # CONSERVATIVE: stop first
            exit_px = stop - slip if side > 0 else stop + slip
            rem = 0.5 if half_done else 1.0
            pts = (exit_px - fill) if side > 0 else (fill - exit_px)
            pnl += pts * PT * rem
            pnl -= comm
            reason = "stop_ambig"
            k_exit = k
            break
        if hit_stop:
            exit_px = stop - slip if side > 0 else stop + slip
            rem = 0.5 if half_done else 1.0
            pts = (exit_px - fill) if side > 0 else (fill - exit_px)
            pnl += pts * PT * rem
            pnl -= comm
            reason = "stop"
            k_exit = k
            break
        if hit_tgt:
            exit_px = target - slip if side > 0 else target + slip
            rem = 0.5 if half_done else 1.0
            pts = (exit_px - fill) if side > 0 else (fill - exit_px)
            pnl += pts * PT * rem
            pnl -= comm
            reason = "target"
            k_exit = k
            break
    else:
        k_exit = last - 1
        cl = float(a["cl"][k_exit])
        exit_px = cl - slip if side > 0 else cl + slip
        rem = 0.5 if half_done else 1.0
        pts = (exit_px - fill) if side > 0 else (fill - exit_px)
        pnl += pts * PT * rem
        pnl -= comm
        reason = "time"

    return {
        "pnl": float(pnl),
        "reason": reason,
        "hold": int(k_exit - j + 1),
        "date": e["date"],
        "side": e["side"],
    }


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "pf": 0.0, "net": 0.0, "wr": 0.0, "dd": 0.0, "exp": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "med_hold": 0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wins = float(w.sum()) if len(w) else 0.0
    losses = float(l.sum()) if len(l) else 0.0
    pf = wins / abs(losses) if losses < 0 else (99.0 if wins > 0 else 0.0)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max()) if len(eq) else 0.0
    return {
        "n": int(len(pnl)),
        "pf": round(pf, 3),
        "net": round(float(pnl.sum()), 2),
        "wr": round(float((pnl > 0).mean() * 100), 1),
        "dd": round(dd, 2),
        "exp": round(float(pnl.mean()), 2),
        "avg_win": round(float(w.mean()), 2) if len(w) else 0.0,
        "avg_loss": round(float(l.mean()), 2) if len(l) else 0.0,
        "med_hold": int(np.median([t["hold"] for t in trades])),
    }


def year_nets(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    for start, end, label in YEAR_CUTS:
        sub = [t for t in trades if start <= t["date"] < end]
        out[label] = round(sum(t["pnl"] for t in sub), 2)
    return out


def walk_forward_t(trades: List[Dict[str, Any]], fold_days: int = 60) -> float:
    """Simple fold mean-exp / se across chronological folds on OOS+late IS."""
    if len(trades) < 40:
        return 0.0
    # group by date order
    by_day: Dict[date, List[float]] = defaultdict(list)
    for t in trades:
        by_day[t["date"]].append(t["pnl"])
    days = sorted(by_day.keys())
    if len(days) < fold_days * 2:
        # use equal thirds
        folds = np.array_split(days, 3)
    else:
        folds = []
        for i in range(0, len(days), fold_days):
            chunk = days[i : i + fold_days]
            if chunk:
                folds.append(chunk)
    exps = []
    for chunk in folds:
        pnls = []
        for d in chunk:
            pnls.extend(by_day[d])
        if pnls:
            exps.append(float(np.mean(pnls)))
    if len(exps) < 2:
        return 0.0
    arr = np.array(exps, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    if se <= 1e-12:
        return 0.0 if arr.mean() <= 0 else 99.0
    return float(arr.mean() / se)


def build_exit_grid() -> List[Dict[str, Any]]:
    grid = []
    # Named A–I style + neighbors for plateau
    fixed = [
        (5, 8, 5), (5, 10, 5), (6, 10, 5), (6, 12, 7), (7, 14, 7), (8, 16, 10),
        (4, 6, 3), (4, 8, 5), (5, 12, 7), (6, 8, 5), (6, 15, 7), (7, 10, 5),
        (7, 12, 7), (8, 12, 7), (8, 14, 10), (9, 14, 10), (9, 18, 10), (10, 15, 10),
        (10, 20, 10), (8, 10, 5), (6, 9, 5), (5, 7.5, 5),
    ]
    for sl, tp, mx in fixed:
        grid.append({"name": f"fix_sl{sl}_tp{tp}_m{mx}", "kind": "fixed",
                     "sl": float(sl), "tp": float(tp), "max_bars": int(mx)})
    for r, mx in [(1.0, 5), (1.25, 7), (1.5, 7), (1.5, 10), (1.75, 10), (2.0, 10)]:
        for sl_mode in ("atr", "atr15"):
            grid.append({
                "name": f"{sl_mode}_r{r}_m{mx}",
                "kind": sl_mode,
                "r": float(r),
                "max_bars": int(mx),
            })
    # breakeven variants on a few fixed
    for sl, tp, mx, be in [(6, 12, 7, 4), (6, 12, 7, 6), (8, 16, 10, 6), (5, 10, 5, 4)]:
        grid.append({"name": f"be{be}_sl{sl}_tp{tp}_m{mx}", "kind": "be",
                     "sl": float(sl), "tp": float(tp), "max_bars": int(mx), "be_at": float(be)})
    # partial 1R / runner
    for sl, mx, run in [(6, 7, 1.5), (6, 10, 2.0), (8, 10, 1.5), (8, 10, 2.0)]:
        grid.append({"name": f"part1_run{run}_sl{sl}_m{mx}", "kind": "partial",
                     "sl": float(sl), "max_bars": int(mx), "partial_r": 1.0, "runner_r": float(run)})
    # current baseline $20 / ATR stop / 3 min
    grid.append({"name": "BASELINE_usd20_atr_m3", "kind": "baseline", "max_bars": 3})
    return grid


def resolve_brackets(cfg: Dict[str, Any], e: Dict[str, Any]) -> Optional[Tuple[float, float, dict]]:
    kind = cfg["kind"]
    if kind == "baseline":
        sl = e["atr_sl"]
        if sl != sl or sl <= 0:
            return None
        return float(sl), 10.0, {"be_at": None, "partial_r": None, "runner_r": None}
    if kind == "fixed":
        return cfg["sl"], cfg["tp"], {"be_at": None, "partial_r": None, "runner_r": None}
    if kind == "be":
        return cfg["sl"], cfg["tp"], {"be_at": cfg["be_at"], "partial_r": None, "runner_r": None}
    if kind == "partial":
        sl = cfg["sl"]
        # tp unused; partial uses R multiples of sl
        return sl, sl * cfg["runner_r"], {
            "be_at": None, "partial_r": cfg["partial_r"], "runner_r": cfg["runner_r"],
        }
    if kind == "atr":
        sl = e["atr_sl"]
        if sl != sl or sl <= 0:
            return None
        return float(sl), float(sl) * cfg["r"], {"be_at": None, "partial_r": None, "runner_r": None}
    if kind == "atr15":
        atr = e["atr"]
        if atr != atr or atr <= 0:
            return None
        sl = sl_pts_from_atr(atr, mult=1.5, sl_min=6.0, sl_max=12.0)
        return float(sl), float(sl) * cfg["r"], {"be_at": None, "partial_r": None, "runner_r": None}
    return None


def run_config(
    a, entries, cfg, *, slip, comm,
) -> List[Dict[str, Any]]:
    trades = []
    next_free = 0
    for e in entries:
        if e["fill_i"] < next_free:
            continue
        br = resolve_brackets(cfg, e)
        if br is None:
            continue
        sl, tp, extra = br
        t = simulate_exit(
            a, e, sl_pts=sl, tp_pts=tp, max_bars=cfg["max_bars"],
            slip=slip, comm=comm,
            be_at=extra["be_at"],
            partial_r=extra["partial_r"],
            runner_r=extra["runner_r"],
        )
        trades.append(t)
        next_free = e["fill_i"] + t["hold"] + 1
    return trades


def plateau_neighbors(name: str, all_names: List[str]) -> List[str]:
    """Find names that share kind prefix and nearby numbers — simple: same kind token."""
    # For fixed_slX_tpY_mZ, neighbors with sl±1 or tp±2
    if not name.startswith("fix_"):
        base = name.split("_")[0]
        return [n for n in all_names if n != name and n.startswith(base)][:8]
    parts = name.replace("fix_", "").split("_")
    # sl6_tp12_m7
    try:
        sl = float(parts[0][2:])
        tp = float(parts[1][2:])
        mx = int(parts[2][1:])
    except Exception:
        return []
    neigh = []
    for n in all_names:
        if not n.startswith("fixed_") or n == name:
            continue
        p = n.replace("fixed_", "").split("_")
        try:
            s2 = float(p[0][2:])
            t2 = float(p[1][2:])
            m2 = int(p[2][1:])
        except Exception:
            continue
        if abs(s2 - sl) <= 1.01 and abs(t2 - tp) <= 2.01 and abs(m2 - mx) <= 2:
            neigh.append(n)
    return neigh


def main():
    df = pd.read_csv("data/MNQ_1m.csv")
    print(f"bars={len(df)}", flush=True)
    a = prepare_arrays(df)
    entries = collect_entries(a)
    print(f"frozen_entries={len(entries)}", flush=True)

    # --- MAE/MFE diagnosis (path, no exit) ---
    paths = [path_excursions(a, e, slip=SLIP_BASE) for e in entries]
    mfe = np.array([p["mfe"] for p in paths])
    mae = np.array([p["mae"] for p in paths])
    print("\n=== MAE/MFE DIAGNOSIS (15-bar path, entry frozen) ===", flush=True)
    for label, arr in (("MFE", mfe), ("MAE", mae)):
        print(
            f"{label}: median={np.median(arr):.2f} p25={np.percentile(arr,25):.2f} "
            f"p75={np.percentile(arr,75):.2f} p90={np.percentile(arr,90):.2f} mean={arr.mean():.2f}",
            flush=True,
        )
    # reach levels before MAE exceeds that level? use: hit +L while MAE so far < L
    # simpler report: % ever reaching +L within horizon
    print("\n% trades reaching +L pts (any time in first 15 bars):", flush=True)
    for lv in MFE_LEVELS:
        pct = 100.0 * np.mean([p["first_hit"][lv] is not None for p in paths])
        print(f"  +{lv}: {pct:.1f}%", flush=True)

    # Of trades where MAE eventually >= their atr_sl (proxy for "would stop"), how many hit +L first?
    # Path-based: for each path, did first_hit[L] occur with bars <= t when mae first exceeded L?
    # Approximate: mfe >= L and (t_mfe <= t_mae or mae < L at that point)
    print("\n% reaching +L BEFORE MAE reached +L (favorable first to that level):", flush=True)
    for lv in MFE_LEVELS:
        ok = 0
        for p in paths:
            if p["first_hit"][lv] is None:
                continue
            # reached +L; if mae < L overall or t_mfe for that level before mae hit L
            if p["mae"] < lv or (p["t_mfe"] > 0 and p["first_hit"][lv] <= p["t_mae"]):
                ok += 1
        print(f"  +{lv}: {100.0*ok/len(paths):.1f}% of all trades", flush=True)

    # Losers vs winners under BASELINE exit for labeling
    base_trades = run_config(a, entries, {"name": "BASELINE_usd20_atr_m3", "kind": "baseline", "max_bars": 3},
                             slip=SLIP_BASE, comm=COMM_BASE)
    base_by_fill = {}
    # map by chronological pairing — rebuild with fill index
    # rerun baseline tagging on paths via simulate storing fill_i
    tagged = []
    next_free = 0
    for e, p in zip(entries, paths):
        if e["fill_i"] < next_free:
            continue
        br = resolve_brackets({"kind": "baseline", "max_bars": 3}, e)
        if br is None:
            continue
        sl, tp, extra = br
        t = simulate_exit(a, e, sl_pts=sl, tp_pts=tp, max_bars=3, slip=SLIP_BASE, comm=COMM_BASE)
        tagged.append((p, t))
        next_free = e["fill_i"] + t["hold"] + 1

    winners = [p for p, t in tagged if t["reason"] == "target"]
    losers = [p for p, t in tagged if t["reason"] in ("stop", "stop_ambig")]
    print(f"\nBaseline tags: targets={len(winners)} stops={len(losers)} other={len(tagged)-len(winners)-len(losers)}", flush=True)
    print("% of EVENTUAL TARGET winners that reached each level (path MFE):", flush=True)
    for lv in MFE_LEVELS:
        if winners:
            print(f"  +{lv}: {100.0*np.mean([w['mfe']>=lv for w in winners]):.1f}%", flush=True)
    print("% of EVENTUAL STOP losers that reached each level BEFORE stopping (path MFE):", flush=True)
    for lv in MFE_LEVELS:
        if losers:
            print(f"  +{lv}: {100.0*np.mean([w['mfe']>=lv for w in losers]):.1f}%", flush=True)
    if winners:
        print(
            f"Winners MFE after would-be $20: median MFE={np.median([w['mfe'] for w in winners]):.2f} "
            f"(how far they went on path)",
            flush=True,
        )
    if losers:
        print(
            f"Losers: median MFE before stop path={np.median([w['mfe'] for w in losers]):.2f} "
            f"median MAE={np.median([w['mae'] for w in losers]):.2f}",
            flush=True,
        )

    # MFE by hold bar
    print("\nMedian MFE by bars held on path:", flush=True)
    for b in (1, 2, 3, 5, 7, 10):
        # mfe at end of bar b — approximate using first_hit / recompute light
        subset = [p["mfe"] for p in paths if p["bars"] >= b]
        if subset:
            print(f"  bar>={b}: median_path_mfe={np.median(subset):.2f} n={len(subset)}", flush=True)

    # --- Exit grid ---
    print("\n=== EXIT GRID (entry frozen, stop-before-target) ===", flush=True)
    grid = build_exit_grid()
    results = []
    for cfg in grid:
        trades = run_config(a, entries, cfg, slip=SLIP_BASE, comm=COMM_BASE)
        is_t = [t for t in trades if t["date"] < OOS]
        oos_t = [t for t in trades if t["date"] >= OOS]
        jj = [t for t in trades if OOS <= t["date"] < AUG]
        aug = [t for t in trades if AUG <= t["date"] < SEP]
        sep = [t for t in trades if t["date"] >= SEP]
        sm_is = summarize(is_t)
        sm_oos = summarize(oos_t)
        yn = year_nets(trades)
        pos_years = sum(1 for v in yn.values() if v > 0)
        total_pos = sum(v for v in yn.values() if v > 0) or 1.0
        max_year_share = max((v for v in yn.values() if v > 0), default=0) / total_pos
        wf_t = walk_forward_t(oos_t)
        # cost stress on OOS
        stress = {}
        for mult in (1, 2, 3):
            tr = run_config(a, entries, cfg, slip=SLIP_BASE * mult, comm=COMM_BASE * mult)
            o = [t for t in tr if t["date"] >= OOS]
            stress[f"{mult}x"] = summarize(o)["exp"]
        rec = {
            "name": cfg["name"],
            "is": sm_is,
            "oos": sm_oos,
            "jj": summarize(jj),
            "aug": summarize(aug),
            "sep": summarize(sep),  # report only
            "years": yn,
            "pos_years": pos_years,
            "max_year_share": round(max_year_share, 3),
            "wf_t": round(wf_t, 2),
            "stress_oos_exp": stress,
            "cfg": cfg,
        }
        results.append(rec)

    # Gates (Sep not used)
    def passes(r):
        o, i = r["oos"], r["is"]
        if o["n"] < 80 or o["exp"] <= 0 or o["pf"] < 1.20:
            return False
        if i["n"] < 200 or i["exp"] <= 0 or i["pf"] < 1.05:
            return False
        if r["wf_t"] < 2.0:
            return False
        if r["pos_years"] < 3:
            return False
        if r["max_year_share"] > 0.40:
            return False
        # cost stress: 2x still non-negative expectancy OOS preferred; require 1x>0 already
        if r["stress_oos_exp"]["2x"] <= 0:
            return False
        return True

    by_name = {r["name"]: r for r in results}
    names = list(by_name.keys())

    # Plateau: passer whose neighbors also OOS exp>0 and pf>=1.1
    passers = [r for r in results if passes(r)]
    plateau = []
    for r in passers:
        neigh = plateau_neighbors(r["name"], names)
        good_n = 0
        for nn in neigh:
            nr = by_name[nn]
            if nr["oos"]["exp"] > 0 and nr["oos"]["pf"] >= 1.1 and nr["is"]["exp"] > 0:
                good_n += 1
        # need at least 2 profitable neighbors OR (no neighbors and itself passes — atr family)
        if good_n >= 2 or (len(neigh) == 0 and passes(r)):
            r["neighbor_ok"] = good_n
            plateau.append(r)
        elif good_n >= 1 and r["oos"]["pf"] >= 1.25:
            r["neighbor_ok"] = good_n
            plateau.append(r)

    print(f"\nPASS_HARD {len(passers)} / {len(results)}", flush=True)
    print(f"PLATEAU {len(plateau)}", flush=True)

    # Rank plateau by OOS exp (not Sep)
    plateau.sort(key=lambda r: (r["oos"]["exp"], r["oos"]["pf"]), reverse=True)
    # Also show best OOS even if fail — for diagnosis
    ranked = sorted(results, key=lambda r: r["oos"]["exp"], reverse=True)

    print("\n--- TOP 8 by OOS expectancy (diagnostic, may FAIL gates) ---", flush=True)
    for r in ranked[:8]:
        print(
            f"{r['name']}: IS exp={r['is']['exp']} pf={r['is']['pf']} n={r['is']['n']} | "
            f"OOS exp={r['oos']['exp']} pf={r['oos']['pf']} n={r['oos']['n']} wr={r['oos']['wr']} "
            f"dd={r['oos']['dd']} wfT={r['wf_t']} years={r['years']} stress={r['stress_oos_exp']}",
            flush=True,
        )

    baseline = by_name.get("BASELINE_usd20_atr_m3")
    print("\n=== BASELINE ===", flush=True)
    if baseline:
        print(json.dumps({k: baseline[k] for k in ("is", "oos", "jj", "aug", "sep", "wf_t", "years", "stress_oos_exp")}, indent=2), flush=True)

    verdict = "FAIL"
    repaired = None
    if plateau:
        verdict = "PASS"
        repaired = plateau[0]
    elif passers:
        verdict = "FAIL_NO_PLATEAU"
        repaired = passers[0]

    # Entry invalid check: best OOS among all still exp<=0 or pf<1.1
    best_oos = ranked[0]
    entry_invalid = best_oos["oos"]["exp"] <= 0 and best_oos["is"]["exp"] <= 0

    report = {
        "verdict": verdict,
        "entry_invalid": entry_invalid,
        "baseline": {k: baseline[k] for k in ("is", "oos", "jj", "aug", "stress_oos_exp", "years")} if baseline else None,
        "repaired": None if repaired is None else {
            "name": repaired["name"],
            "cfg": repaired["cfg"],
            "is": repaired["is"],
            "oos": repaired["oos"],
            "jj": repaired["jj"],
            "aug": repaired["aug"],
            "sep": repaired["sep"],
            "wf_t": repaired["wf_t"],
            "years": repaired["years"],
            "max_year_share": repaired["max_year_share"],
            "stress_oos_exp": repaired["stress_oos_exp"],
            "neighbor_ok": repaired.get("neighbor_ok"),
        },
        "best_oos_any": {
            "name": best_oos["name"],
            "oos": best_oos["oos"],
            "is": best_oos["is"],
            "wf_t": best_oos["wf_t"],
        },
        "mfe_mae": {
            "mfe_median": round(float(np.median(mfe)), 2),
            "mae_median": round(float(np.median(mae)), 2),
            "reach_pct": {str(lv): round(100.0 * float(np.mean([p["first_hit"][lv] is not None for p in paths])), 1) for lv in MFE_LEVELS},
            "loser_reach_before_stop": {str(lv): round(100.0 * float(np.mean([w["mfe"] >= lv for w in losers])), 1) if losers else 0 for lv in MFE_LEVELS},
            "winner_mfe_median": round(float(np.median([w["mfe"] for w in winners])), 2) if winners else None,
            "loser_mfe_median": round(float(np.median([w["mfe"] for w in losers])), 2) if losers else None,
        },
        "grid_size": len(results),
        "pass_hard": len(passers),
        "plateau": len(plateau),
        "note": "Same-bar: stop before target. Sep report-only. Entry frozen.",
    }
    out_path = "data/micro_scalp_exit_repair.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nVERDICT {verdict}", flush=True)
    print(f"ENTRY_INVALID {entry_invalid}", flush=True)
    if repaired:
        print(f"REPAIRED {repaired['name']}", flush=True)
        print(json.dumps(report["repaired"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
