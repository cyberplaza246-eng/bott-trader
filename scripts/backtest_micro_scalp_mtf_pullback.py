#!/usr/bin/env python3
"""Backtest existing 15/5/1 micro_scalp + a few R:R / ATR stop tweaks on MNQ_1m."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.strategy.day_hybrid import entry_pullback, prepare_arrays, stack_side_15_5_1
from src.strategy.mnq_15m_ema_eod import ENTRY_MINUTE, FLAT_MINUTE, sl_pts_from_atr

COMM = 1.24
PT = 2.0
SLIP = 0.25


def metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "exp": 0.0, "dd": 0.0,
                "tp": 0, "sl": 0, "mh": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
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
    }


def window(trades, start, end=None):
    out = []
    for t in trades:
        if t["day"] < start:
            continue
        if end and t["day"] >= end:
            continue
        out.append(t)
    return out


def run(a, *, tp_pts=10.0, sl_mode="atr", sl_fixed=10.0, atr_mult=1.0,
        sl_min=6.0, sl_max=12.0, max_hold=20, need_t15_t5_only=False,
        cooldown=1) -> List[Dict[str, Any]]:
    n = len(a["cl"])
    trades = []
    side = 0
    entry = sl = tp = 0.0
    ei = 0
    cool = -1
    for i in range(80, n):
        if int(a["dow"][i]) >= 5:
            continue
        m = int(a["mins"][i])
        if side != 0:
            hi, lo, cl = float(a["hi"][i]), float(a["lo"][i]), float(a["cl"][i])
            reason = None
            xp = cl
            if side > 0:
                if lo <= sl:
                    reason, xp = "SL", sl
                elif hi >= tp:
                    reason, xp = "TP", tp
                elif (i - ei) >= max_hold or m >= FLAT_MINUTE:
                    reason, xp = "MAX_HOLD", cl
            else:
                if hi >= sl:
                    reason, xp = "SL", sl
                elif lo <= tp:
                    reason, xp = "TP", tp
                elif (i - ei) >= max_hold or m >= FLAT_MINUTE:
                    reason, xp = "MAX_HOLD", cl
            if reason:
                fe = entry + SLIP if side > 0 else entry - SLIP
                fx = xp - SLIP if side > 0 else xp + SLIP
                pts = (fx - fe) if side > 0 else (fe - fx)
                trades.append({
                    "day": str(a["dates"][i]), "reason": reason,
                    "pnl": round(pts * PT - COMM, 2),
                })
                side = 0
                cool = i + cooldown
            continue
        if i < cool or m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if need_t15_t5_only:
            s15, s5 = int(a["t15"][i]), int(a["t5"][i])
            if s15 == 0 or s15 != s5:
                continue
            stack = s15
        else:
            stack = stack_side_15_5_1(i, a)
        if stack == 0:
            continue
        if not entry_pullback(stack, i, a):
            continue
        if sl_mode == "atr":
            atr1 = float(a["atr1"][i])
            sl_pts = sl_pts_from_atr(atr1, mult=atr_mult, sl_min=sl_min, sl_max=sl_max)
            if sl_pts != sl_pts or sl_pts <= 0:
                continue
        else:
            sl_pts = sl_fixed
        entry = float(a["cl"][i])
        if stack > 0:
            sl, tp = entry - sl_pts, entry + tp_pts
        else:
            sl, tp = entry + sl_pts, entry - tp_pts
        side, ei = stack, i
    return trades


def main() -> int:
    path = os.path.join(ROOT, "data", "MNQ_1m.csv")
    print(f"Loading {path}", flush=True)
    a = prepare_arrays(pd.read_csv(path))
    configs = [
        ("micro_scalp_default_atr", dict(tp_pts=10, sl_mode="atr", atr_mult=1.0, sl_min=6, sl_max=12, max_hold=20)),
        ("atr1.5_tp10", dict(tp_pts=10, sl_mode="atr", atr_mult=1.5, sl_min=6, sl_max=15, max_hold=20)),
        ("atr1_tp15", dict(tp_pts=15, sl_mode="atr", atr_mult=1.0, sl_min=6, sl_max=12, max_hold=30)),
        ("fixed8_tp12", dict(tp_pts=12, sl_mode="fixed", sl_fixed=8, max_hold=15)),
        ("fixed10_tp20", dict(tp_pts=20, sl_mode="fixed", sl_fixed=10, max_hold=30)),
        ("t15t5_pull_atr", dict(tp_pts=10, sl_mode="atr", atr_mult=1.0, need_t15_t5_only=True, max_hold=20)),
        ("t15t5_pull_sl8_tp12", dict(tp_pts=12, sl_mode="fixed", sl_fixed=8, need_t15_t5_only=True, max_hold=15)),
        ("stack_sl8_tp16_mh25", dict(tp_pts=16, sl_mode="fixed", sl_fixed=8, max_hold=25)),
    ]
    rows = []
    for name, kw in configs:
        trades = run(a, **kw)
        jj = metrics(window(trades, "2026-06-01", "2026-08-01"))
        aug = metrics(window(trades, "2026-08-01", "2026-09-01"))
        full = metrics(window(trades, "2026-06-01"))
        ok = jj["n"] >= 40 and jj["pf"] >= 1.15 and jj["net"] > 0 and aug["n"] >= 15 and aug["pf"] >= 1.05 and aug["net"] > 0
        rows.append({"name": name, "jun_jul": jj, "aug": aug, "full": full, "pass": ok})
        print(
            f"{'PASS' if ok else 'fail':4} {name:28} "
            f"JJ n={jj['n']:4} PF={jj['pf']:5} ${jj['net']:9} | "
            f"Aug n={aug['n']:3} PF={aug['pf']:5} ${aug['net']:9}",
            flush=True,
        )
    out = os.path.join(ROOT, "data", "scalp_mtf_pullback_hunt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
