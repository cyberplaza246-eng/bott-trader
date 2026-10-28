#!/usr/bin/env python3
"""Quick continuous RTH hunt vs slotted PASS (same stack/stop/tp)."""
from __future__ import annotations

import os
import sys
from datetime import date
from typing import Dict, List, Optional, Set

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.pass_hunt_mnq as hunt
from src.strategy.mnq_15m_ema_eod import (
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    quiet_1m_ok,
    window_index,
    window_requires_quality,
)

ET = resp.ET
WARMUP = 80


def simulate_continuous(
    a: Dict,
    *,
    quiet: bool = True,
    cooldown_min: int = 15,
    max_day: int = 12,
    sep_noon: bool = True,
) -> List:
    stop_cfg = hunt.STOPS["fix25"]
    tp_cfg = hunt.TPS["eod500"]
    tp_kind, tp_val = tp_cfg
    trades = []
    opens: List[Dict] = []
    counts: Dict = {}
    last_entry_m: Dict = {}
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
            else:
                still.append(p)
        opens = still

        if a["dow"][i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= max_day:
            continue
        if day in last_entry_m and m < last_entry_m[day] + cooldown_min:
            continue
        if len(opens) >= 2:
            continue

        side = hunt.stack_side("A_d15_60", i, a)
        if side != 1:  # long only
            continue
        if quiet and not quiet_1m_ok(side, a["op"][i], a["cl"][i]):
            continue

        sep = float(a["sep"][i])
        conf = (sep == sep) and sep >= SEP_MIN
        win = window_index(m)
        if sep_noon and (
            (win is not None and window_requires_quality(win)) or m >= 12 * 60
        ) and not conf:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = hunt.stop_pts(stop_cfg, float(a["atr15"][i]), float(a["atr1"][i]))
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
            "win": win, "i": i,
        })
        counts[day] = counts.get(day, 0) + 1
        last_entry_m[day] = m
    return trades


def win(trades, start: date, end: Optional[date] = None):
    out = []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        d = ts.tz_convert(ET).date()
        if d < start:
            continue
        if end and d >= end:
            continue
        out.append(t)
    return out


def show(label, trades):
    m = resp.group_metrics(trades)
    print(
        f"{label:22} n={m['trades']:4} PF={m['profit_factor']:5.2f} "
        f"Net=${m['total_pnl']:8.1f} Exp=${m['expectancy']:6.2f}"
    )
    return m


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = hunt.prepare(df)

    print("=== SLOTTED PASS (w2_open, quiet=off) ===")
    slotted = hunt.simulate(
        a, stack="A_d15_60", winset=hunt.WINSETS["w2_open"],
        stop_cfg=hunt.STOPS["fix25"], tp_cfg=hunt.TPS["eod500"],
        side_mode="long", quiet=False, sep_noon=True,
    )
    show("IS", win(slotted, date(2020, 1, 1), date(2026, 6, 1)))
    show("VAL", win(slotted, date(2026, 6, 1), date(2026, 8, 1)))
    show("AUG", win(slotted, date(2026, 8, 1), date(2026, 9, 1)))

    print("\n=== CONTINUOUS RTH + quiet 1m (cd=15, max12/day) ===")
    cont = simulate_continuous(a, quiet=True, cooldown_min=15, max_day=12)
    is_m = show("IS", win(cont, date(2020, 1, 1), date(2026, 6, 1)))
    val_m = show("VAL", win(cont, date(2026, 6, 1), date(2026, 8, 1)))
    aug_m = show("AUG", win(cont, date(2026, 8, 1), date(2026, 9, 1)))
    ok = (
        is_m["profit_factor"] >= 1.15 and is_m["total_pnl"] > 0
        and val_m["profit_factor"] >= 1.15 and val_m["total_pnl"] > 0
        and aug_m["profit_factor"] >= 1.10 and aug_m["total_pnl"] > 0
    )
    print(f"\nCLEAN_FORWARD continuous: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
