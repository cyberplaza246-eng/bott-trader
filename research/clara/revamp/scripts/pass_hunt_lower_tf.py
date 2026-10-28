#!/usr/bin/env python3
"""
Lower-TF PASS hunt: 15m + 5m + 1m ONLY (no daily, no 60m).

Tests multiple entry loops so you are not waiting on slow filters:
  - window_once   first qualifying bar per slot
  - window_loop   re-enter same slot after flat (gapped re-entry)
  - rth_loop      scan every 1m bar in RTH with cooldown
  - morning_loop  scan 9:35-12:00 with cooldown

PASS: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0
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
    EMA_FAST,
    EMA_SLOW,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    ONE_MIN_MAX_BODY,
    SEP_MIN,
    TP_R,
    _as_utc_frame,
    ema_series,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    window_index,
    window_label,
    window_requires_quality,
)

ET = resp.ET
WARMUP = 80
MORNING_END = 12 * 60

WINSETS = {
    "w3_morning": frozenset({0, 1, 2}),
    "w2_open": frozenset({0, 2}),
    "w2_935_1015": frozenset({0, 1}),
    "w1_935": frozenset({0}),
    "w1_1100": frozenset({2}),
    "w4_no_noon": frozenset({0, 1, 2, 4}),
}

STOPS = {
    "atr1_1.5_6_25": ("atr1", 1.5, 6.0, 25.0),
    "atr1_2_8_30": ("atr1", 2.0, 8.0, 30.0),
    "atr5_1.5_8_30": ("atr5", 1.5, 8.0, 30.0),
    "fix15": ("fixed", 15.0, 0.0, 0.0),
    "fix20": ("fixed", 20.0, 0.0, 0.0),
    "fix25": ("fixed", 25.0, 0.0, 0.0),
    "fix30": ("fixed", 30.0, 0.0, 0.0),
}

TPS = {
    "1R": ("r", 1.0),
    "1.5R": ("r", 1.5),
    "2R": ("r", 2.0),
    "3R": ("r", 3.0),
    "eod500": ("cap", 500.0),
}


def is_pass(is_m: Dict, oos_m: Dict) -> bool:
    return (
        oos_m["trades"] >= 30
        and oos_m["profit_factor"] >= 1.15
        and oos_m["total_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["total_pnl"] > 0
    )


def trend_1m_completed(df_1m: pd.DataFrame) -> np.ndarray:
    tmp = _as_utc_frame(df_1m)
    fast = ema_series(tmp["close"], EMA_FAST)
    slow = ema_series(tmp["close"], EMA_SLOW)
    trend = pd.Series(0, index=tmp.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    return trend.shift(1).fillna(0).astype(int).to_numpy(dtype=int)


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    et = dt.tz_convert(ET)
    return {
        "dt": dt,
        "t15": trend_15m_on_1m(work).to_numpy(dtype=int),
        "t5": resp.trend_5m_on_1m(work).to_numpy(dtype=int),
        "t1": trend_1m_completed(work),
        "sep": sep15_on_1m(work).to_numpy(dtype=float),
        "atr1": resp.atr1_on_1m(work).to_numpy(dtype=float),
        "atr5": resp.atr5_on_1m(work).to_numpy(dtype=float),
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
    if stack == "15_only":
        return s15
    if stack == "15_5":
        return s15 if s15 == int(a["t5"][i]) else 0
    if stack == "15_5_1":
        t1 = int(a["t1"][i])
        return s15 if (s15 == int(a["t5"][i]) and s15 == t1) else 0
    raise ValueError(stack)


def entry_ok(
    mode: str,
    side: int,
    i: int,
    a: Dict,
    *,
    body_max: float = ONE_MIN_MAX_BODY,
) -> bool:
    if mode == "any":
        return True
    if mode == "quiet":
        return quiet_1m_ok(side, a["op"][i], a["cl"][i], max_body=body_max)
    if mode == "quiet2":
        return quiet_1m_ok(side, a["op"][i], a["cl"][i], max_body=2.0)
    if mode == "1m_pullback":
        # small bar closing with trend after prior bar against (micro pullback)
        if not quiet_1m_ok(side, a["op"][i], a["cl"][i], max_body=body_max):
            return False
        if i < 2:
            return False
        prev = float(a["cl"][i - 1]) - float(a["cl"][i - 2])
        if side > 0:
            return prev <= 0
        return prev >= 0
    return False


def stop_pts(cfg: Tuple, atr1: float, atr5: float) -> float:
    kind, mult, lo, hi = cfg
    if kind == "fixed":
        return float(mult)
    raw = atr1 if kind == "atr1" else atr5
    if raw != raw or raw <= 0:
        return float("nan")
    return sl_pts_from_atr(float(raw), mult=mult, sl_min=lo, sl_max=hi)


def in_winset(m: int, winset: Optional[frozenset]) -> bool:
    if winset is None:
        return True
    w = window_index(m)
    return w is not None and w in winset


def simulate(
    a: Dict[str, Any],
    *,
    stack: str,
    loop: str,
    entry: str,
    winset_name: Optional[str],
    stop_key: str,
    tp_key: str,
    side_mode: str,
    sep_noon: bool,
    cooldown_min: int,
    max_day: int,
    max_lots: int,
    morning_only: bool,
) -> List:
    winset = WINSETS.get(winset_name) if winset_name else None
    stop_cfg = STOPS[stop_key]
    tp_cfg = TPS[tp_key]
    tp_kind, tp_val = tp_cfg

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
                if loop == "window_loop" and p.get("win") is not None:
                    fired.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if a["dow"][i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if morning_only and m >= MORNING_END:
            continue
        if counts.get(day, 0) >= max_day:
            continue

        # Loop mode gating
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
            if winset is not None and not in_winset(m, winset):
                continue
            if cooldown_min > 0 and last_entry is not None:
                ld, lm = last_entry
                if ld == day and m - lm < cooldown_min:
                    continue
        else:
            raise ValueError(loop)

        side = stack_side(stack, i, a)
        if side == 0:
            continue
        if side_mode == "long" and side != 1:
            continue
        if side_mode == "short" and side != -1:
            continue
        if not entry_ok(entry, side, i, a):
            continue

        sep = float(a["sep"][i])
        conf = (sep == sep) and sep >= SEP_MIN
        if sep_noon and window_requires_quality(win if win is not None else window_index(m) or -1):
            if win is None:
                w = window_index(m)
                if w is not None and window_requires_quality(w) and not conf:
                    continue
            elif window_requires_quality(win) and not conf:
                continue
        if len(opens) >= max_lots:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
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
            "ts": ts,
            "dir": side,
            "entry": entry_px,
            "sl": sl,
            "tp": tp_px,
            "sl_pts": use,
            "win": widx,
            "win_name": window_label(widx) if widx is not None else "loop",
        })
        if loop in ("window_once", "window_loop") and widx is not None:
            fired.add(int(widx))
        counts[day] = counts.get(day, 0) + 1
        last_entry = (day, m)
    return trades


def build_phases() -> List[List[Dict[str, Any]]]:
    # Phase 0 — fast loop / long-only / morning (most likely to pass lower-TF)
    p0 = []
    for stack in ("15_5_1", "15_5"):
        for loop in ("window_loop", "morning_loop", "window_once"):
            for entry in ("quiet", "quiet2", "1m_pullback"):
                for stop in ("fix15", "fix20", "fix25", "atr1_1.5_6_25"):
                    for tp in ("1.5R", "2R", "3R", "1R"):
                        for win in ("w2_open", "w3_morning", "w1_935"):
                            for cd in (0, 10, 15):
                                p0.append({
                                    "stack": stack, "loop": loop, "entry": entry,
                                    "win": win, "stop": stop, "tp": tp, "side": "long",
                                    "sep": False, "cooldown": cd, "max_day": 4,
                                    "max_lots": 1, "morning": loop != "window_once" or win != "w2_open",
                                })
    # trim morning flag: morning_loop always morning; window_once on w3/w1 only
    for c in p0:
        if c["loop"] == "morning_loop":
            c["morning"] = True
        elif c["loop"] == "window_once" and c["win"] == "w2_open":
            c["morning"] = False

    p1 = []
    for stack in ("15_5", "15_5_1", "15_only"):
        for loop in ("window_once", "window_loop"):
            for win in ("w2_open", "w3_morning", "w2_935_1015", "w1_935"):
                for entry in ("quiet", "quiet2", "1m_pullback", "any"):
                    for stop in ("atr1_1.5_6_25", "fix20", "fix25"):
                        for tp in ("1R", "1.5R", "2R"):
                            for side in ("long", "both"):
                                p1.append({
                                    "stack": stack, "loop": loop, "entry": entry,
                                    "win": win, "stop": stop, "tp": tp, "side": side,
                                    "sep": True, "cooldown": 0, "max_day": 6,
                                    "max_lots": 1, "morning": False,
                                })

    p2 = []
    for stack in ("15_5", "15_5_1"):
        for loop in ("rth_loop", "morning_loop"):
            for win in ("w2_open", "w3_morning", None):
                for entry in ("quiet", "quiet2", "1m_pullback"):
                    for stop in ("atr1_1.5_6_25", "fix20", "fix25", "fix15"):
                        for tp in ("1R", "1.5R", "2R", "eod500"):
                            for side in ("long",):
                                for cd in (5, 10, 15, 20):
                                    for max_day in (3, 6, 10):
                                        p2.append({
                                            "stack": stack, "loop": loop, "entry": entry,
                                            "win": win, "stop": stop, "tp": tp, "side": side,
                                            "sep": False, "cooldown": cd, "max_day": max_day,
                                            "max_lots": 1, "morning": loop == "morning_loop",
                                        })

    p3 = []
    # Long-only morning, tight risk, higher R
    for stack in ("15_5", "15_5_1"):
        for loop in ("window_loop", "morning_loop"):
            for entry in ("quiet", "1m_pullback"):
                for stop in ("fix15", "fix20", "atr1_1.5_6_25"):
                    for tp in ("2R", "3R", "1.5R"):
                        p3.append({
                            "stack": stack, "loop": loop, "entry": entry,
                            "win": "w3_morning", "stop": stop, "tp": tp, "side": "long",
                            "sep": False, "cooldown": 10, "max_day": 4,
                            "max_lots": 1, "morning": True,
                        })

    def dedupe(rows):
        seen = set()
        out = []
        for r in rows:
            k = tuple(sorted(r.items()))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    return [dedupe(p0), dedupe(p1), dedupe(p2), dedupe(p3)]


def cfg_name(c: Dict) -> str:
    w = c["win"] or "rth"
    return (
        f"{c['stack']}|{c['loop']}|{c['entry']}|{w}|{c['stop']}|{c['tp']}|"
        f"{c['side']}|cd{c['cooldown']}|md{c['max_day']}|sep{int(c['sep'])}"
    )


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    stop_on_first = "--all" not in sys.argv
    out = os.path.join("data", "pass_hunt_lower_tf.json")

    print("LOWER-TF PASS HUNT (15m + 5m + 1m only — NO daily/60m)")
    print("Loops: window_once | window_loop | rth_loop | morning_loop")
    print("Gates: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0")
    print(f"stop_on_first={stop_on_first}\n")

    t0 = time.time()
    df = official.load_1m(csv)
    a = prepare(df)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    print(f"prepare {time.time()-t0:.1f}s\n")

    phases = build_phases()
    tested = 0
    passed: List[Dict] = []
    weak: List[Dict] = []
    best_fail: Optional[Dict] = None

    for pi, phase in enumerate(phases, 1):
        print(f"=== PHASE {pi}: {len(phase)} candidates ===", flush=True)
        for cfg in phase:
            name = cfg_name(cfg)
            trades = simulate(
                a,
                stack=cfg["stack"],
                loop=cfg["loop"],
                entry=cfg["entry"],
                winset_name=cfg["win"],
                stop_key=cfg["stop"],
                tp_key=cfg["tp"],
                side_mode=cfg["side"],
                sep_noon=cfg["sep"],
                cooldown_min=cfg["cooldown"],
                max_day=cfg["max_day"],
                max_lots=cfg["max_lots"],
                morning_only=cfg["morning"],
            )
            overall = resp.group_metrics(trades)
            ins, oos = edge.split_is_oos(trades)
            is_m = resp.group_metrics(ins)
            oos_m = resp.group_metrics(oos)
            tested += 1
            row = {"name": name, **cfg, "overall": overall, "is": is_m, "oos": oos_m}

            if is_pass(is_m, oos_m):
                row["grade"] = "PASS"
                passed.append(row)
                print(
                    f"*** PASS *** {name}\n"
                    f"  IS  n={is_m['trades']} PF={is_m['profit_factor']:.2f} ${is_m['total_pnl']:.0f}\n"
                    f"  OOS n={oos_m['trades']} PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} "
                    f"Exp=${oos_m['expectancy']:.2f}",
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
            else:
                row["grade"] = "FAIL"
                sc = (oos_m["expectancy"], oos_m["profit_factor"], oos_m["total_pnl"])
                if best_fail is None or sc > (
                    best_fail["oos"]["expectancy"],
                    best_fail["oos"]["profit_factor"],
                    best_fail["oos"]["total_pnl"],
                ):
                    best_fail = row

            if tested % 100 == 0:
                print(f"  ... {tested} tested pass={len(passed)} weak={len(weak)}", flush=True)

        if passed and stop_on_first:
            break
        if not passed:
            print(f"Phase {pi} done — no PASS (weak={len(weak)})\n", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone {elapsed:.0f}s | tested={tested} PASS={len(passed)} WEAK={len(weak)}")

    if passed:
        b = max(passed, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print(f"\nBEST PASS: {b['name']}")
        print(f"  OOS PF={b['oos']['profit_factor']:.2f} ${b['oos']['total_pnl']:.0f} n={b['oos']['trades']}")
    elif weak:
        weak.sort(key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True)
        w = weak[0]
        print(f"\nNo PASS. Best WEAK: {w['name']}")
        print(f"  OOS PF={w['oos']['profit_factor']:.2f} ${w['oos']['total_pnl']:.0f}")
    elif best_fail:
        print(f"\nLeast-bad FAIL: {best_fail['name']}")
        print(f"  OOS Exp=${best_fail['oos']['expectancy']:.2f} PF={best_fail['oos']['profit_factor']:.2f}")

    payload = {
        "scope": "15m+5m+1m only, no daily/60m",
        "elapsed_sec": round(elapsed, 1),
        "tested": tested,
        "passed": passed,
        "weak_top": sorted(
            weak, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True
        )[:25],
        "best_fail": best_fail,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
