#!/usr/bin/env python3
"""
PASS hunter — prioritized MNQ recipe search (strategy logic only).

PASS: OOS n>=30, OOS PF>=1.15, OOS$>0, IS PF>=1.05, IS$>0
Stops at first PASS unless --all.
Expands phases until a PASS is found or phases are exhausted.
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
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    atr15_on_1m,
    daily_trend_on_1m,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_index,
    window_label,
    window_requires_quality,
)

ET = resp.ET
WARMUP = 80
PT, COMM = official.PT, official.COMM

WINSETS = {
    "w3_classic": frozenset({0, 2, 4}),
    "w3_morning": frozenset({0, 1, 2}),
    "w2_open": frozenset({0, 2}),
    "w1_935": frozenset({0}),
    "w1_1100": frozenset({2}),
    "w2_am_pm": frozenset({0, 4}),
    "w4_no_noon": frozenset({0, 1, 2, 4}),
    "w6_all": frozenset({0, 1, 2, 3, 4, 5}),
    "w2_935_1015": frozenset({0, 1}),
}

STOPS = {
    "atr15_2_20_60": ("atr15", 2.0, 20.0, 60.0),
    "atr15_2.5_25_70": ("atr15", 2.5, 25.0, 70.0),
    "atr15_1.5_15_45": ("atr15", 1.5, 15.0, 45.0),
    "atr1_1.5_6_25": ("atr1", 1.5, 6.0, 25.0),
    "atr1_2_8_30": ("atr1", 2.0, 8.0, 30.0),
    "fix20": ("fixed", 20.0, 0.0, 0.0),
    "fix25": ("fixed", 25.0, 0.0, 0.0),
    "fix30": ("fixed", 30.0, 0.0, 0.0),
    "fix40": ("fixed", 40.0, 0.0, 0.0),
    "fix50": ("fixed", 50.0, 0.0, 0.0),
}

TPS = {
    "eod500": ("cap", 500.0),
    "1R": ("r", 1.0),
    "1.5R": ("r", 1.5),
    "2R": ("r", 2.0),
    "3R": ("r", 3.0),
}


def is_pass(is_m: Dict, oos_m: Dict) -> bool:
    return (
        oos_m["trades"] >= 30
        and oos_m["profit_factor"] >= 1.15
        and oos_m["total_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["total_pnl"] > 0
    )


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    et = dt.tz_convert(ET)
    return {
        "dt": dt,
        "t15": trend_15m_on_1m(work).to_numpy(dtype=int),
        "td": daily_trend_on_1m(work).to_numpy(dtype=int),
        "t60": trend_60m_on_1m(work).to_numpy(dtype=int),
        "t5": resp.trend_5m_on_1m(work).to_numpy(dtype=int),
        "sep": sep15_on_1m(work).to_numpy(dtype=float),
        "atr15": atr15_on_1m(work).to_numpy(dtype=float),
        "atr1": resp.atr1_on_1m(work).to_numpy(dtype=float),
        "op": work["open"].to_numpy(dtype=float),
        "hi": work["high"].to_numpy(dtype=float),
        "lo": work["low"].to_numpy(dtype=float),
        "cl": work["close"].to_numpy(dtype=float),
        "mins": (et.hour * 60 + et.minute).to_numpy(),
        "dow": et.weekday.to_numpy(),
        "dates": np.array(et.date),
    }


def stack_side(stack: str, i: int, a: Dict) -> int:
    s15 = int(a["t15"][i])
    if s15 == 0:
        return 0
    if stack == "C_15":
        return s15
    if stack == "B_15_5":
        return s15 if s15 == int(a["t5"][i]) else 0
    if stack == "A_d15_60":
        return s15 if (s15 == int(a["td"][i]) and s15 == int(a["t60"][i])) else 0
    if stack == "D_d15_5":
        return s15 if (s15 == int(a["td"][i]) and s15 == int(a["t5"][i])) else 0
    if stack == "E_d15":
        return s15 if s15 == int(a["td"][i]) else 0
    if stack == "F_15_60":
        return s15 if s15 == int(a["t60"][i]) else 0
    raise ValueError(stack)


def stop_pts(cfg: Tuple, atr15: float, atr1: float) -> float:
    kind, mult, lo, hi = cfg
    if kind == "fixed":
        return float(mult)
    raw = atr15 if kind == "atr15" else atr1
    if raw != raw or raw <= 0:
        return float("nan")
    return sl_pts_from_atr(float(raw), mult=mult, sl_min=lo, sl_max=hi)


def simulate(
    a: Dict[str, Any],
    *,
    stack: str,
    winset: frozenset,
    stop_cfg: Tuple,
    tp_cfg: Tuple,
    side_mode: str,
    quiet: bool,
    sep_noon: bool,
    max_lots: int = 2,
) -> List:
    tp_kind, tp_val = tp_cfg
    trades = []
    opens: List[Dict] = []
    fired: Set[int] = set()
    counts: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])
    for i in range(WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
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
                if p.get("win") is not None:
                    fired.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if a["dow"][i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue
        win = window_index(m)
        if win is None or win not in winset or win in fired:
            continue

        side = stack_side(stack, i, a)
        if side == 0:
            continue
        if side_mode == "long" and side != 1:
            continue
        if side_mode == "short" and side != -1:
            continue
        if quiet and not quiet_1m_ok(side, a["op"][i], a["cl"][i]):
            continue

        sep = float(a["sep"][i])
        conf = (sep == sep) and sep >= SEP_MIN
        if sep_noon and window_requires_quality(win) and not conf:
            continue
        if len(opens) >= max_lots:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = stop_pts(stop_cfg, float(a["atr15"][i]), float(a["atr1"][i]))
        if use != use or use <= 0:
            continue
        entry = float(a["cl"][i])
        sl = entry - use if side == 1 else entry + use
        if tp_kind == "r":
            dist = use * tp_val
            tp_px = entry + dist if side == 1 else entry - dist
        else:
            tp_px = entry + tp_val if side == 1 else entry - tp_val
        opens.append({
            "ts": ts, "dir": side, "entry": entry, "sl": sl, "tp": tp_px,
            "sl_pts": use, "win": win, "win_name": window_label(win),
        })
        fired.add(win)
        counts[day] = counts.get(day, 0) + 1
    return trades


def build_phases() -> List[List[Dict[str, Any]]]:
    """Expanding phases of candidate recipes (most likely first)."""
    phase1 = []
    # Classic hold-to-EOD family that docs claimed was profitable
    for stack in ("A_d15_60", "E_d15", "F_15_60", "D_d15_5"):
        for win in ("w3_classic", "w2_open", "w2_am_pm", "w1_935", "w3_morning"):
            for stop in ("atr15_2_20_60", "fix30", "fix40", "fix25", "atr15_2.5_25_70"):
                for side in ("long", "both"):
                    for quiet in (False, True):
                        for sep in (True, False):
                            phase1.append({
                                "stack": stack, "win": win, "stop": stop,
                                "tp": "eod500", "side": side, "quiet": quiet, "sep": sep,
                            })

    phase2 = []
    # R-multiple with wider stops / long-only mornings
    for stack in ("A_d15_60", "B_15_5", "E_d15", "D_d15_5"):
        for win in ("w3_morning", "w2_open", "w3_classic", "w2_935_1015"):
            for stop in ("atr15_2_20_60", "fix30", "fix40", "atr1_2_8_30"):
                for tp in ("2R", "1.5R", "3R", "1R"):
                    for side in ("long", "both"):
                        phase2.append({
                            "stack": stack, "win": win, "stop": stop,
                            "tp": tp, "side": side, "quiet": False, "sep": True,
                        })
                        phase2.append({
                            "stack": stack, "win": win, "stop": stop,
                            "tp": tp, "side": side, "quiet": True, "sep": True,
                        })

    phase3 = []
    # Shorts-only / all windows / tighter stops (long shot)
    for stack in ("A_d15_60", "B_15_5", "C_15"):
        for win in ("w6_all", "w4_no_noon", "w3_classic"):
            for stop in ("atr15_2_20_60", "fix30", "atr1_1.5_6_25", "fix50"):
                for tp in ("eod500", "2R"):
                    for side in ("short", "both", "long"):
                        phase3.append({
                            "stack": stack, "win": win, "stop": stop,
                            "tp": tp, "side": side, "quiet": False, "sep": True,
                        })

    # Deduplicate while preserving order
    def dedupe(rows: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in rows:
            key = (r["stack"], r["win"], r["stop"], r["tp"], r["side"], r["quiet"], r["sep"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    return [dedupe(phase1), dedupe(phase2), dedupe(phase3)]


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    stop_on_first = "--all" not in sys.argv
    out = os.path.join("data", "pass_hunt_results.json")

    print("PASS HUNT (phased)")
    print("Gates: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0")
    print(f"stop_on_first={stop_on_first}")

    t0 = time.time()
    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    a = prepare(df)
    print(f"prepare done in {time.time()-t0:.1f}s\n")

    phases = build_phases()
    tested = 0
    passed: List[Dict] = []
    weak: List[Dict] = []
    best_fail: Optional[Dict] = None

    for pi, phase in enumerate(phases, 1):
        print(f"=== PHASE {pi}: {len(phase)} candidates ===", flush=True)
        for cfg in phase:
            name = (
                f"{cfg['stack']}|{cfg['win']}|{cfg['stop']}|{cfg['tp']}|"
                f"{cfg['side']}|q{int(cfg['quiet'])}|sep{int(cfg['sep'])}"
            )
            trades = simulate(
                a,
                stack=cfg["stack"],
                winset=WINSETS[cfg["win"]],
                stop_cfg=STOPS[cfg["stop"]],
                tp_cfg=TPS[cfg["tp"]],
                side_mode=cfg["side"],
                quiet=cfg["quiet"],
                sep_noon=cfg["sep"],
            )
            overall = resp.group_metrics(trades)
            ins, oos = edge.split_is_oos(trades)
            is_m = resp.group_metrics(ins)
            oos_m = resp.group_metrics(oos)
            tested += 1
            row = {
                "name": name,
                **cfg,
                "overall": overall,
                "is": is_m,
                "oos": oos_m,
            }
            if is_pass(is_m, oos_m):
                row["grade"] = "PASS"
                passed.append(row)
                print(
                    f"*** PASS *** {name}\n"
                    f"  IS  n={is_m['trades']} PF={is_m['profit_factor']:.2f} ${is_m['total_pnl']:.0f}\n"
                    f"  OOS n={oos_m['trades']} PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} "
                    f"Exp=${oos_m['expectancy']:.2f} DD=${oos_m['max_drawdown']:.0f}",
                    flush=True,
                )
                if stop_on_first:
                    break
            elif (
                oos_m["trades"] >= 20
                and oos_m["profit_factor"] >= 1.0
                and oos_m["total_pnl"] > 0
                and is_m["total_pnl"] > 0
            ):
                row["grade"] = "WEAK"
                weak.append(row)
                print(
                    f"WEAK {name} OOS PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} n={oos_m['trades']}",
                    flush=True,
                )
            else:
                row["grade"] = "FAIL"
                score = (oos_m["expectancy"], oos_m["profit_factor"], oos_m["total_pnl"])
                if best_fail is None or score > (
                    best_fail["oos"]["expectancy"],
                    best_fail["oos"]["profit_factor"],
                    best_fail["oos"]["total_pnl"],
                ):
                    best_fail = row

            if tested % 25 == 0:
                print(
                    f"  ... {tested} tested | pass={len(passed)} weak={len(weak)} | {name}",
                    flush=True,
                )
        if passed and stop_on_first:
            break
        if not passed:
            print(f"Phase {pi} done — no PASS yet (weak={len(weak)})\n", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s | tested={tested} PASS={len(passed)} WEAK={len(weak)}")

    if passed:
        best = max(passed, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print(f"\nBEST PASS: {best['name']}")
    elif weak:
        weak.sort(key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True)
        print(f"\nNo PASS. Best WEAK: {weak[0]['name']}")
        print(f"  OOS PF={weak[0]['oos']['profit_factor']:.2f} ${weak[0]['oos']['total_pnl']:.0f}")
    elif best_fail:
        print(f"\nNo PASS/WEAK. Least-bad: {best_fail['name']}")
        print(f"  OOS Exp=${best_fail['oos']['expectancy']:.2f} PF={best_fail['oos']['profit_factor']:.2f}")

    payload = {
        "elapsed_sec": round(elapsed, 1),
        "tested": tested,
        "pass_count": len(passed),
        "weak_count": len(weak),
        "passed": passed,
        "weak_top": sorted(
            weak, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True
        )[:30],
        "best_fail": best_fail,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
