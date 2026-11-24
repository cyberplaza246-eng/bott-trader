#!/usr/bin/env python3
"""Quick MTF hunt for MNQ micro-scalp: 15m/5m bias + 1m (or impulse) entry.

Compares baseline-style impulse entries with higher-TF EMA filters.
Uses data/MNQ_1m.csv. RTH only. Comm $1.24/rt. Slip 0.25 pts/side.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.strategy.day_hybrid import prepare_arrays  # noqa: E402
from src.strategy.mnq_15m_ema_eod import (  # noqa: E402
    ENTRY_MINUTE,
    FLAT_MINUTE,
    quiet_1m_ok,
)

COMM = 1.24
PT = 2.0
SLIP = 0.25  # pts per side


@dataclass
class Recipe:
    name: str
    # bias: none | t5 | t15 | t15_t5 | stack151
    bias: str = "none"
    # entry: impulse | pullback | ema1
    entry: str = "impulse"
    impulse_bars: int = 3  # 1m bars lookback for impulse
    impulse_pts: float = 3.0
    sl_pts: float = 10.0
    tp_pts: float = 10.0  # $20
    max_hold_bars: int = 15  # minutes
    cooldown_bars: int = 1
    require_sep15: bool = False


def metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "exp": 0.0,
            "dd": 0.0, "tp": 0, "sl": 0, "mh": 0, "tprd": 0.0,
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    days = {t["day"] for t in trades}
    return {
        "n": len(trades),
        "wr": round(100.0 * len(wins) / len(trades), 1),
        "pf": round((gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0), 2),
        "net": round(sum(pnls), 2),
        "exp": round(sum(pnls) / len(trades), 2),
        "dd": round(dd, 2),
        "tp": sum(1 for t in trades if t["reason"] == "TP"),
        "sl": sum(1 for t in trades if t["reason"] == "SL"),
        "mh": sum(1 for t in trades if t["reason"] == "MAX_HOLD"),
        "tprd": round(len(trades) / max(len(days), 1), 2),
    }


def bias_side(i: int, a: Dict[str, Any], mode: str) -> int:
    t15 = int(a["t15"][i])
    t5 = int(a["t5"][i])
    t1 = int(a["t1"][i])
    if mode == "none":
        return 0  # caller decides from entry alone
    if mode == "t5":
        return t5
    if mode == "t15":
        return t15
    if mode == "t15_t5":
        return t15 if t15 != 0 and t15 == t5 else 0
    if mode == "stack151":
        return t15 if t15 != 0 and t15 == t5 and t15 == t1 else 0
    return 0


def entry_side(i: int, a: Dict[str, Any], r: Recipe, forced: int = 0) -> int:
    """Return +1 long / -1 short / 0 none. forced=bias when bias filters direction."""
    cl = a["cl"]
    if r.entry == "impulse":
        j = i - r.impulse_bars
        if j < 0:
            return 0
        move = float(cl[i]) - float(cl[j])
        if forced > 0:
            return 1 if move >= r.impulse_pts else 0
        if forced < 0:
            return -1 if move <= -r.impulse_pts else 0
        # no bias: either way
        if move >= r.impulse_pts:
            return 1
        if move <= -r.impulse_pts:
            return -1
        return 0
    if r.entry == "pullback":
        side = forced if forced != 0 else int(a["t1"][i])
        if side == 0:
            return 0
        if not quiet_1m_ok(side, a["op"][i], a["cl"][i]):
            return 0
        if i < 2:
            return 0
        prev = float(cl[i - 1]) - float(cl[i - 2])
        if side > 0 and prev <= 0:
            return 1
        if side < 0 and prev >= 0:
            return -1
        return 0
    if r.entry == "ema1":
        t1 = int(a["t1"][i])
        if forced != 0:
            return forced if forced == t1 else 0
        return t1
    return 0


def simulate(a: Dict[str, Any], r: Recipe) -> List[Dict[str, Any]]:
    n = len(a["cl"])
    trades: List[Dict[str, Any]] = []
    open_side = 0
    entry = sl = tp = 0.0
    entry_i = 0
    cool_until = -1
    for i in range(80, n):
        if int(a["dow"][i]) >= 5:
            continue
        m = int(a["mins"][i])
        # manage open
        if open_side != 0:
            hi = float(a["hi"][i])
            lo = float(a["lo"][i])
            cl = float(a["cl"][i])
            reason = None
            exit_px = cl
            if open_side > 0:
                if lo <= sl:
                    reason, exit_px = "SL", sl
                elif hi >= tp:
                    reason, exit_px = "TP", tp
                elif (i - entry_i) >= r.max_hold_bars or m >= FLAT_MINUTE:
                    reason, exit_px = "MAX_HOLD", cl
            else:
                if hi >= sl:
                    reason, exit_px = "SL", sl
                elif lo <= tp:
                    reason, exit_px = "TP", tp
                elif (i - entry_i) >= r.max_hold_bars or m >= FLAT_MINUTE:
                    reason, exit_px = "MAX_HOLD", cl
            if reason:
                fill_e = entry + SLIP if open_side > 0 else entry - SLIP
                fill_x = exit_px - SLIP if open_side > 0 else exit_px + SLIP
                pts = (fill_x - fill_e) if open_side > 0 else (fill_e - fill_x)
                trades.append({
                    "day": str(a["dates"][i]),
                    "direction": "long" if open_side > 0 else "short",
                    "entry": round(entry, 2),
                    "exit": round(exit_px, 2),
                    "reason": reason,
                    "pnl": round(pts * PT - COMM, 2),
                    "hold": i - entry_i,
                })
                open_side = 0
                cool_until = i + r.cooldown_bars
            continue

        if i < cool_until:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if r.require_sep15 and float(a["sep"][i]) < 2.0:
            continue

        b = bias_side(i, a, r.bias)
        if r.bias != "none" and b == 0:
            continue
        side = entry_side(i, a, r, forced=b if r.bias != "none" else 0)
        if side == 0:
            continue

        entry = float(a["cl"][i])
        if side > 0:
            sl = entry - r.sl_pts
            tp = entry + r.tp_pts
        else:
            sl = entry + r.sl_pts
            tp = entry - r.tp_pts
        open_side = side
        entry_i = i
    return trades


def window(trades: List[Dict[str, Any]], start: str, end: Optional[str] = None):
    out = []
    for t in trades:
        d = t["day"]
        if d < start:
            continue
        if end and d >= end:
            continue
        out.append(t)
    return out


def recipes() -> List[Recipe]:
    out: List[Recipe] = []
    # User stack idea: 15m regime, 5m main direction, 1m entry
    out.append(Recipe("baseline_impulse_1m", bias="none", entry="impulse",
                      impulse_bars=3, impulse_pts=3.0, sl_pts=15.0, tp_pts=10.0))
    out.append(Recipe("impulse_live_defaults", bias="none", entry="impulse",
                      impulse_bars=1, impulse_pts=2.5, sl_pts=15.0, tp_pts=10.0,
                      max_hold_bars=3))
    # 5m alone as direction
    out.append(Recipe("t5_dir_impulse", bias="t5", entry="impulse",
                      impulse_bars=2, impulse_pts=2.5, sl_pts=12.0, tp_pts=10.0))
    out.append(Recipe("t5_dir_impulse_tight", bias="t5", entry="impulse",
                      impulse_bars=1, impulse_pts=3.0, sl_pts=8.0, tp_pts=12.0,
                      max_hold_bars=10))
    out.append(Recipe("t5_dir_ema1", bias="t5", entry="ema1",
                      sl_pts=10.0, tp_pts=12.0, max_hold_bars=15))
    out.append(Recipe("t5_dir_pullback", bias="t5", entry="pullback",
                      sl_pts=8.0, tp_pts=12.0, max_hold_bars=12))
    # 15m general + 5m direction
    out.append(Recipe("t15_t5_impulse", bias="t15_t5", entry="impulse",
                      impulse_bars=2, impulse_pts=2.0, sl_pts=10.0, tp_pts=10.0))
    out.append(Recipe("t15_t5_impulse_rr12", bias="t15_t5", entry="impulse",
                      impulse_bars=2, impulse_pts=2.5, sl_pts=8.0, tp_pts=12.0,
                      max_hold_bars=12))
    out.append(Recipe("t15_t5_ema1", bias="t15_t5", entry="ema1",
                      sl_pts=10.0, tp_pts=10.0, max_hold_bars=20))
    out.append(Recipe("t15_t5_pullback", bias="t15_t5", entry="pullback",
                      sl_pts=10.0, tp_pts=10.0, require_sep15=True))
    out.append(Recipe("t15_bias_impulse", bias="t15", entry="impulse",
                      impulse_bars=2, impulse_pts=2.5, sl_pts=12.0, tp_pts=10.0))
    out.append(Recipe("stack151_pullback", bias="stack151", entry="pullback",
                      sl_pts=10.0, tp_pts=10.0, max_hold_bars=20))
    out.append(Recipe("t5_pullback", bias="t5", entry="pullback",
                      sl_pts=10.0, tp_pts=10.0))
    # risk/reward grid under MTF
    for bias, entry, sl, tp, mh, imp in [
        ("t5", "impulse", 8.0, 12.0, 10, 2.5),
        ("t5", "impulse", 10.0, 15.0, 15, 3.0),
        ("t5", "ema1", 8.0, 12.0, 20, 2.5),
        ("t15_t5", "impulse", 8.0, 12.0, 12, 2.5),
        ("t15_t5", "impulse", 10.0, 15.0, 20, 2.5),
        ("t15_t5", "pullback", 8.0, 12.0, 15, 2.5),
        ("t15_t5", "ema1", 8.0, 12.0, 20, 2.5),
        ("t15", "ema1", 10.0, 15.0, 30, 2.5),
        ("stack151", "pullback", 8.0, 12.0, 15, 2.5),
        ("stack151", "impulse", 8.0, 12.0, 12, 2.5),
        ("t15_t5", "impulse", 12.0, 10.0, 10, 2.5),
        ("t15_t5", "impulse", 10.0, 20.0, 30, 2.5),
        ("t5", "impulse", 6.0, 10.0, 8, 3.0),
        ("t15_t5", "impulse", 6.0, 10.0, 8, 3.0),
    ]:
        out.append(Recipe(
            name=f"{bias}_{entry}_sl{sl:g}_tp{tp:g}_mh{mh}",
            bias=bias, entry=entry, impulse_bars=2, impulse_pts=imp,
            sl_pts=sl, tp_pts=tp, max_hold_bars=mh,
            require_sep15=(bias == "t15_t5" and entry == "pullback"),
        ))
    return out


def gate(m: Dict[str, Any], *, min_n: int = 40) -> bool:
    return m["n"] >= min_n and m["pf"] >= 1.15 and m["net"] > 0 and m["exp"] >= 0.5


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "MNQ_1m.csv")
    print(f"Loading {path} ...", flush=True)
    df = pd.read_csv(path)
    a = prepare_arrays(df)
    print(f"bars={len(a['cl']):,} recipes={len(recipes())}", flush=True)

    rows = []
    for r in recipes():
        trades = simulate(a, r)
        jj = metrics(window(trades, "2026-06-01", "2026-08-01"))
        aug = metrics(window(trades, "2026-08-01", "2026-09-01"))
        full = metrics(window(trades, "2026-06-01"))
        ok = gate(jj) and gate(aug, min_n=20)
        row = {
            "name": r.name,
            "recipe": asdict(r),
            "jun_jul": jj,
            "aug": aug,
            "full": full,
            "pass": ok,
            "score": round(jj["exp"] * 0.6 + aug["exp"] * 0.4, 3) if ok else -999,
        }
        rows.append(row)
        flag = "PASS" if ok else "fail"
        print(
            f"{flag:4} {r.name:40} "
            f"JJ n={jj['n']:4} PF={jj['pf']:5} ${jj['net']:8} exp={jj['exp']:6} | "
            f"Aug n={aug['n']:3} PF={aug['pf']:5} ${aug['net']:8}",
            flush=True,
        )

    rows.sort(key=lambda x: (x["pass"], x["score"], x["full"]["net"]), reverse=True)
    out_path = os.path.join(ROOT, "data", "scalp_mtf_quick_hunt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows[:30], "all": rows}, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    print("\n=== TOP ===", flush=True)
    for row in rows[:8]:
        print(
            f"  {row['name']}: pass={row['pass']} "
            f"JJ PF={row['jun_jul']['pf']} ${row['jun_jul']['net']} | "
            f"Aug PF={row['aug']['pf']} ${row['aug']['net']} | "
            f"Full ${row['full']['net']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
