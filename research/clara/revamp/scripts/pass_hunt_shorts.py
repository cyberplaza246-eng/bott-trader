#!/usr/bin/env python3
"""
Short-side + both-direction hunt on lower-TF stack (15m+5m+1m only).

Goals:
  1) Short-only recipe that PASSes alone
  2) Dual recipe: locked long winner + short recipe -> combined PASS
  3) Single both-direction recipe that PASSes

PASS: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0
Short-only PASS uses lower n>=20 (THIN) flagged separately.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
from src.strategy.mnq_15m_ema_eod import (
    SEP_MIN,
    _as_utc_frame,
    window_index,
    window_label,
    window_requires_quality,
)
from src.strategy.session_levels import session_vwap_series

ET = resp.ET
WARMUP = base.WARMUP

WINSETS = {
    **base.WINSETS,
    "w2_pm": frozenset({4, 5}),           # 13:30, 14:30
    "w3_pm": frozenset({3, 4, 5}),        # 12:00, 13:30, 14:30
    "w1_1430": frozenset({5}),
    "w2_1200_1430": frozenset({3, 5}),
    "w6_all": frozenset({0, 1, 2, 3, 4, 5}),
}

LONG_WINNER = {
    "stack": "15_5_1",
    "loop": "window_loop",
    "entry": "quiet",
    "win": "w2_open",
    "stop": "atr1_1.5_6_25",
    "tp": "1R",
    "side": "long",
    "sep": False,
    "cooldown": 0,
    "max_day": 4,
    "max_lots": 1,
    "morning": False,
    "vwap": "none",
    "sep_short_min": None,
}


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    a = base.prepare(df)
    tmp = _as_utc_frame(df.copy())
    tmp["datetime"] = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    vwap = session_vwap_series(tmp).to_numpy(dtype=float)
    cl = tmp["close"].to_numpy(dtype=float)
    a["vwap"] = vwap
    a["above_vwap"] = (cl > vwap) & ~np.isnan(vwap)
    a["below_vwap"] = (cl < vwap) & ~np.isnan(vwap)
    return a


def is_pass(is_m: Dict, oos_m: Dict, *, min_oos: int = 30) -> bool:
    return (
        oos_m["trades"] >= min_oos
        and oos_m["profit_factor"] >= 1.15
        and oos_m["total_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["total_pnl"] > 0
    )


def vwap_ok(side: int, i: int, a: Dict, mode: str) -> bool:
    if mode == "none":
        return True
    if mode == "with_side":
        if side > 0:
            return bool(a["above_vwap"][i])
        if side < 0:
            return bool(a["below_vwap"][i])
    if mode == "strict":
        if side > 0:
            return bool(a["above_vwap"][i])
        return bool(a["below_vwap"][i])
    return True


def simulate_ext(
    a: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List:
    winset = WINSETS.get(cfg["win"]) if cfg.get("win") else None
    stop_cfg = base.STOPS[cfg["stop"]]
    tp_cfg = base.TPS[cfg["tp"]]
    tp_kind, tp_val = tp_cfg
    loop = cfg["loop"]
    side_mode = cfg["side"]
    sep_noon = cfg["sep"]
    cooldown_min = cfg["cooldown"]
    max_day = cfg["max_day"]
    max_lots = cfg["max_lots"]
    morning_only = cfg["morning"]
    vwap_mode = cfg.get("vwap", "none")
    sep_short_min = cfg.get("sep_short_min")

    trades = []
    opens: List[Dict] = []
    fired: Set[int] = set()
    counts: Dict = {}
    last_entry: Optional[Tuple] = None
    dt = a["dt"]
    n = len(a["cl"])

    for i in range(WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= base.FLAT_MINUTE:
                reason, exit_px = "EOD", float(a["cl"][i])
            elif p["dir"] == 1:
                if a["lo"][i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif a["hi"][i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if a["hi"][i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif a["lo"][i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                trades.append(gapped._close_position(p, ts, float(exit_px), reason))
                if loop == "window_loop" and p.get("win") is not None:
                    fired.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if a["dow"][i] >= 5:
            continue
        if m < base.ENTRY_MINUTE or m >= base.FLAT_MINUTE:
            continue
        if morning_only and m >= base.MORNING_END:
            continue
        if counts.get(day, 0) >= max_day:
            continue

        win: Optional[int] = None
        if loop in ("window_once", "window_loop"):
            win = window_index(m)
            if win is None or win not in (winset or set()):
                continue
            if loop == "window_once" and win in fired:
                continue
            if loop == "window_loop" and win in fired:
                continue
        elif loop in ("rth_loop", "morning_loop"):
            if winset is not None and base.in_winset(m, winset):
                pass
            elif winset is not None:
                continue
            if cooldown_min > 0 and last_entry is not None:
                ld, lm = last_entry
                if ld == day and m - lm < cooldown_min:
                    continue
        else:
            raise ValueError(loop)

        side = base.stack_side(cfg["stack"], i, a)
        if side == 0:
            continue
        if side_mode == "long" and side != 1:
            continue
        if side_mode == "short" and side != -1:
            continue
        if not base.entry_ok(cfg["entry"], side, i, a):
            continue
        if not vwap_ok(side, i, a, vwap_mode):
            continue

        sep = float(a["sep"][i])
        sep_min = float(sep_short_min) if (side < 0 and sep_short_min is not None) else SEP_MIN
        conf = (sep == sep) and sep >= sep_min
        if sep_noon:
            wcheck = win if win is not None else window_index(m)
            if wcheck is not None and window_requires_quality(wcheck) and not conf:
                continue
        if len(opens) >= max_lots:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = base.stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
        if use != use or use <= 0:
            continue
        entry_px = float(a["cl"][i])
        sl = entry_px - use if side == 1 else entry_px + use
        if tp_kind == "r":
            dist = use * tp_val
            tp_px = entry_px + dist if side == 1 else entry_px - dist
        else:
            tp_px = entry_px + tp_val if side == 1 else entry_px - tp_val

        widx = win if win is not None else window_index(m)
        opens.append({
            "ts": ts, "dir": side, "entry": entry_px, "sl": sl, "tp": tp_px,
            "sl_pts": use, "win": widx,
            "win_name": window_label(widx) if widx is not None else "loop",
        })
        if loop in ("window_once", "window_loop") and widx is not None:
            fired.add(int(widx))
        counts[day] = counts.get(day, 0) + 1
        last_entry = (day, m)
    return trades


def cfg_name(c: Dict) -> str:
    sm = c.get("sep_short_min")
    return (
        f"{c['stack']}|{c['loop']}|{c['entry']}|{c['win']}|{c['stop']}|{c['tp']}|"
        f"{c['side']}|vwap={c.get('vwap','none')}|sepS={sm}|cd{c['cooldown']}|md{c['max_day']}"
    )


def score_row(cfg: Dict, trades: List) -> Dict:
    overall = resp.group_metrics(trades)
    ins, oos = edge.split_is_oos(trades)
    is_m = resp.group_metrics(ins)
    oos_m = resp.group_metrics(oos)
    shorts = [t for t in trades if t.direction == "short"]
    longs = [t for t in trades if t.direction == "long"]
    short_oos = resp.group_metrics([t for t in edge.split_is_oos(trades)[1] if t.direction == "short"])
    return {
        "name": cfg_name(cfg),
        **cfg,
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "short_oos": short_oos,
        "n_short_all": len(shorts),
        "n_long_all": len(longs),
    }


def build_short_grid() -> List[List[Dict]]:
    """Phased grids: promising short configs first."""
    def row(**kw):
        base_kw = {
            "stack": "15_5_1", "loop": "window_loop", "entry": "quiet",
            "side": "short", "sep": False, "cooldown": 0, "max_day": 4,
            "max_lots": 1, "morning": False, "vwap": "none", "sep_short_min": None,
        }
        base_kw.update(kw)
        return base_kw

    p0 = []
    # Mirror long winner + PM / VWAP variants
    for win in ("w2_open", "w2_pm", "w3_pm", "w1_1430", "w2_1200_1430", "w3_morning"):
        for stop in ("atr1_1.5_6_25", "fix15", "fix20", "fix25"):
            for tp in ("1R", "1.5R", "2R", "3R"):
                for vwap in ("none", "with_side"):
                    for entry in ("quiet", "quiet2", "1m_pullback"):
                        p0.append(row(win=win, stop=stop, tp=tp, vwap=vwap, entry=entry))
    for win in ("w2_pm", "w3_pm"):
        for ss in (0.45, 0.55, 0.65):
            for stop in ("atr1_1.5_6_25", "fix20"):
                for tp in ("1R", "1.5R", "2R"):
                    p0.append(row(
                        win=win, stop=stop, tp=tp, vwap="with_side",
                        entry="quiet", sep=True, sep_short_min=ss,
                    ))

    p1 = []
    for stack in ("15_5_1", "15_5"):
        for loop in ("window_loop", "window_once"):
            for win in ("w2_pm", "w3_pm", "w2_open"):
                for entry in ("quiet", "1m_pullback"):
                    for stop in ("atr1_1.5_6_25", "fix20", "atr1_2_8_30"):
                        for tp in ("1R", "2R"):
                            for vwap in ("with_side", "none"):
                                p1.append(row(
                                    stack=stack, loop=loop, win=win, entry=entry,
                                    stop=stop, tp=tp, vwap=vwap,
                                ))

    p2 = []
    for stack in ("15_5_1", "15_5"):
        for win in ("w2_open", "w3_morning", "w6_all"):
            for stop in ("atr1_1.5_6_25", "fix20"):
                for tp in ("1R", "1.5R"):
                    p2.append({
                        "stack": stack, "loop": "window_loop", "entry": "quiet",
                        "win": win, "stop": stop, "tp": tp, "side": "both",
                        "sep": False, "cooldown": 0, "max_day": 6, "max_lots": 1,
                        "morning": False, "vwap": "with_side", "sep_short_min": None,
                    })

    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            k = tuple(sorted((k, str(v)) for k, v in r.items()))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    return [dedupe(p0), dedupe(p1), dedupe(p2)]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stop_on_first = "--all" not in sys.argv
    csv = args[0] if args else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "pass_hunt_shorts.json")

    print("SHORT + BOTH hunt (15m+5m+1m only)")
    print("Also tests DUAL: locked long winner + short recipe combined")
    print("Gates: OOS n>=30 PF>=1.15 (short-only n>=20 flagged THIN_PASS)\n")

    t0 = time.time()
    df = official.load_1m(csv)
    a = prepare(df)
    print(f"bars={len(df):,}\n")

    long_trades = simulate_ext(a, LONG_WINNER)
    long_row = score_row(LONG_WINNER, long_trades)
    print(
        f"Locked LONG winner OOS: PF={long_row['oos']['profit_factor']:.2f} "
        f"${long_row['oos']['total_pnl']:.0f} n={long_row['oos']['trades']}"
    )

    grid_phases = build_short_grid()
    total = sum(len(p) for p in grid_phases)
    print(f"Short grid: {total} candidates in {len(grid_phases)} phases\n")

    tested = 0
    short_pass: List[Dict] = []
    both_pass: List[Dict] = []
    dual_pass: List[Dict] = []
    weak: List[Dict] = []
    best_short_fail: Optional[Dict] = None

    for pi, grid in enumerate(grid_phases, 1):
        print(f"=== PHASE {pi}: {len(grid)} candidates ===", flush=True)
        for cfg in grid:
            trades = simulate_ext(a, cfg)
            row = score_row(cfg, trades)
            row["grade"] = "FAIL"
            tested += 1

            if cfg["side"] == "short":
                if is_pass(row["is"], row["oos"], min_oos=30):
                    row["grade"] = "PASS"
                    short_pass.append(row)
                    print(f"*** SHORT PASS *** {row['name']}", flush=True)
                    print(
                        f"  OOS n={row['oos']['trades']} PF={row['oos']['profit_factor']:.2f} "
                        f"${row['oos']['total_pnl']:.0f}",
                        flush=True,
                    )
                elif is_pass(row["is"], row["oos"], min_oos=20):
                    row["grade"] = "THIN_PASS"
                    short_pass.append(row)
                    print(f"*** SHORT THIN_PASS *** {row['name']}", flush=True)
                elif (
                    row["oos"]["trades"] >= 15
                    and row["oos"]["profit_factor"] >= 1.0
                    and row["oos"]["total_pnl"] > 0
                ):
                    row["grade"] = "WEAK"
                    weak.append(row)
                else:
                    sc = (
                        row["oos"]["expectancy"],
                        row["oos"]["profit_factor"],
                        row["oos"]["total_pnl"],
                    )
                    if best_short_fail is None or sc > (
                        best_short_fail["oos"]["expectancy"],
                        best_short_fail["oos"]["profit_factor"],
                        best_short_fail["oos"]["total_pnl"],
                    ):
                        best_short_fail = row
            elif cfg["side"] == "both" and is_pass(row["is"], row["oos"]):
                row["grade"] = "BOTH_PASS"
                both_pass.append(row)
                print(f"*** BOTH PASS *** {row['name']}", flush=True)

            if cfg["side"] == "short" and row["oos"]["total_pnl"] > 0 and row["is"]["total_pnl"] > 0:
                combined = long_trades + trades
                cins, coos = edge.split_is_oos(combined)
                c_is = resp.group_metrics(cins)
                c_oos = resp.group_metrics(coos)
                if is_pass(c_is, c_oos):
                    dual_pass.append({
                        **row,
                        "dual_overall": resp.group_metrics(combined),
                        "dual_is": c_is,
                        "dual_oos": c_oos,
                        "grade": "DUAL_PASS",
                    })
                    print(f"*** DUAL PASS (long+short) *** short={row['name'][:60]}", flush=True)
                    print(
                        f"  Combined OOS PF={c_oos['profit_factor']:.2f} ${c_oos['total_pnl']:.0f} "
                        f"n={c_oos['trades']}",
                        flush=True,
                    )

            if tested % 100 == 0:
                print(
                    f"  ... {tested} tested short_pass={len(short_pass)} dual={len(dual_pass)}",
                    flush=True,
                )

            if stop_on_first and dual_pass:
                break

        if stop_on_first and (dual_pass or any(r.get("grade") == "PASS" for r in short_pass)):
            break
        if not short_pass and not dual_pass:
            print(f"Phase {pi} done — no full PASS yet\n", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone {elapsed:.0f}s tested={tested}")
    print(f"short_pass={len(short_pass)} both_pass={len(both_pass)} dual_pass={len(dual_pass)} weak={len(weak)}")

    if dual_pass:
        b = max(dual_pass, key=lambda r: (r["dual_oos"]["profit_factor"], r["dual_oos"]["total_pnl"]))
        print("\nBEST DUAL (long locked + short leg):")
        print(f"  Short recipe: {b['name']}")
        print(f"  Short OOS: PF={b['oos']['profit_factor']:.2f} ${b['oos']['total_pnl']:.0f} n={b['oos']['trades']}")
        print(f"  Combined OOS: PF={b['dual_oos']['profit_factor']:.2f} ${b['dual_oos']['total_pnl']:.0f}")
    elif short_pass:
        b = max(short_pass, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print(f"\nBEST SHORT-ONLY: {b['name']}")
        print(f"  OOS PF={b['oos']['profit_factor']:.2f} ${b['oos']['total_pnl']:.0f}")
    elif weak:
        w = max(weak, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print(f"\nNo PASS. Best WEAK short: {w['name']} OOS PF={w['oos']['profit_factor']:.2f}")
    elif best_short_fail:
        print(f"\nLeast-bad short: {best_short_fail['name']} OOS Exp=${best_short_fail['oos']['expectancy']:.2f}")

    payload = {
        "long_winner": long_row,
        "tested": tested,
        "short_pass": short_pass[:15],
        "both_pass": both_pass[:10],
        "dual_pass": dual_pass[:10],
        "weak_top": sorted(weak, key=lambda r: r["oos"]["profit_factor"], reverse=True)[:20],
        "best_short_fail": best_short_fail,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {out}")
    return 0 if (short_pass or dual_pass or both_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
