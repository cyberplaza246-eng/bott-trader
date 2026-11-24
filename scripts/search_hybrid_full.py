#!/usr/bin/env python3
"""
Broad hybrid search — long/short recipes, one position, unlimited daily.

Goal: PASS OOS gates + maximize trades without w6-quiet long bleed.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import product
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth

ET = resp.ET
LONG_WINS = {
    "w2": hunt.WINSETS["w2_open"],
    "w3": hunt.WINSETS["w3_morning"],
    "w4": base.WINSETS["w4_no_noon"],
    "w6": hunt.WINSETS["w6_all"],
}
SHORT_WINS = hunt.WINSETS["w6_all"]


def long_entry(mode: str, i: int, a: Dict) -> bool:
    if mode == "quiet":
        return combo.long_entry_ok(i, a)
    if mode == "pullback":
        return base.entry_ok("1m_pullback", 1, i, a)
    if mode == "quiet_pullback":
        return combo.long_entry_ok(i, a) or base.entry_ok("1m_pullback", 1, i, a)
    raise ValueError(mode)


def long_quality(filter_: str, i: int, a: Dict) -> bool:
    if filter_ == "none":
        return True
    if filter_ == "hh_hl":
        return srth.structure_ok("hh_hl", 1, i, a)
    if filter_ == "sep_pm":
        m = int(a["mins"][i])
        if m >= base.MORNING_END:
            sep = float(a["sep"][i])
            return sep == sep and sep >= base.SEP_MIN
        return True
    if filter_ == "hh_sep_pm":
        if not srth.structure_ok("hh_hl", 1, i, a):
            return False
        m = int(a["mins"][i])
        if m >= base.MORNING_END:
            sep = float(a["sep"][i])
            return sep == sep and sep >= base.SEP_MIN
        return True
    raise ValueError(filter_)


def simulate(
    a: Dict[str, Any],
    long_win_key: str,
    long_entry_mode: str,
    long_quality_filter: str,
    short_tp: str = "1R",
) -> List:
    long_wins = LONG_WINS[long_win_key]
    stop_cfg = base.STOPS["atr1_1.5_6_25"]
    tp_kind, tp_val = base.TPS[short_tp]
    s_tp_kind, s_tp_val = base.TPS["1R"]

    trades: List = []
    pos = None
    dt, n = a["dt"], len(a["cl"])

    for i in range(hunt.WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]

        if pos is not None:
            reason = exit_px = None
            if m >= base.FLAT_MINUTE:
                reason, exit_px = "EOD", float(a["cl"][i])
            elif pos["dir"] == 1:
                if a["lo"][i] <= pos["sl"]:
                    reason, exit_px = "SL", pos["sl"]
                elif a["hi"][i] >= pos["tp"]:
                    reason, exit_px = "TP", pos["tp"]
            else:
                if a["hi"][i] >= pos["sl"]:
                    reason, exit_px = "SL", pos["sl"]
                elif a["lo"][i] <= pos["tp"]:
                    reason, exit_px = "TP", pos["tp"]
            if reason:
                trades.append(gapped._close_position(pos, ts, float(exit_px), reason))
                pos = None

        if a["dow"][i] >= 5 or m < base.ENTRY_MINUTE or m >= base.FLAT_MINUTE or pos is not None:
            continue

        win = base.window_index(m)
        if win is None:
            continue

        side = 0
        if win in long_wins and base.stack_side("15_5_1", i, a) == 1:
            if long_entry(long_entry_mode, i, a) and long_quality(long_quality_filter, i, a):
                side = 1
        elif win in SHORT_WINS and combo.short_entry_ok(i, a):
            side = -1

        if side == 0:
            continue

        use = base.stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
        if use != use or use <= 0:
            continue
        entry_px = float(a["cl"][i])
        if side == 1:
            sl, tp_px = entry_px - use, entry_px + use
        else:
            sl = entry_px + use
            dist = use * (s_tp_val if short_tp == "1R" else tp_val)
            tp_px = entry_px - dist

        pos = {"ts": ts, "dir": side, "entry": entry_px, "sl": sl, "tp": tp_px, "sl_pts": use}

    return trades


def et_date(t) -> Any:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ET).date()


def tpd(trades: List) -> Tuple[int, int, float]:
    days = Counter(et_date(t) for t in trades)
    if not days:
        return 0, 0, 0.0
    c = list(days.values())
    return max(c), sum(1 for x in c if x >= 5), len(trades) / len(days)


def score(trades: List) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    im, om = resp.group_metrics(ins), resp.group_metrics(oos)
    lo = resp.group_metrics([t for t in oos if t.direction == "long"])
    so = resp.group_metrics([t for t in oos if t.direction == "short"])
    mx, d5, avg = tpd(trades)
    return {
        "all_n": len(trades),
        "oos_n": om["trades"],
        "oos_pf": round(om["profit_factor"], 3),
        "oos_net": round(om["total_pnl"], 1),
        "is_pf": round(im["profit_factor"], 3),
        "is_net": round(im["total_pnl"], 1),
        "long_oos": lo["trades"],
        "long_net": round(lo["total_pnl"], 1),
        "short_oos": so["trades"],
        "short_net": round(so["total_pnl"], 1),
        "max_day": mx,
        "avg_active_day": round(avg, 2),
        "days_5plus": d5,
        "pass": hunt.is_pass(im, om),
    }


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = combo.prepare(df)

    long_wins = ("w2", "w3", "w4", "w6")
    entries = ("quiet", "pullback", "quiet_pullback")
    qualities = ("none", "hh_hl", "sep_pm", "hh_sep_pm")
    short_tps = ("1R", "2R")

    rows: List[Dict] = []
    total = len(long_wins) * len(entries) * len(qualities) * len(short_tps)
    n = 0
    for lw, ent, qual, stp in product(long_wins, entries, qualities, short_tps):
        n += 1
        if n % 20 == 0:
            print(f"  ... {n}/{total}", flush=True)
        tr = simulate(a, lw, ent, qual, stp)
        name = f"L={lw}|{ent}|{qual}|S_tp={stp}"
        rows.append({"name": name, "long_win": lw, "long_entry": ent, "long_qual": qual, "short_tp": stp, **score(tr)})

    passing = [r for r in rows if r["pass"]]
    passing.sort(key=lambda r: (-r["oos_n"], -r["oos_pf"], -r["oos_net"]))

    best_pass_trades = passing[0] if passing else None
    best_pass_pf = max(passing, key=lambda r: (r["oos_pf"], r["oos_net"])) if passing else None

    # high activity + positive oos
    hot = [r for r in rows if r["oos_net"] > 200 and r["oos_pf"] >= 1.1]
    hot.sort(key=lambda r: (-r["oos_n"], -r["oos_pf"]))

    print("\n" + "=" * 80)
    print(f"SEARCH DONE: {total} configs | PASS={len(passing)}")
    print("=" * 80)

    if passing:
        print("\nTOP PASS (by OOS trade count):")
        for r in passing[:12]:
            print(
                f"  {r['name']:<45} OOS n={r['oos_n']:3d} PF={r['oos_pf']:.2f} ${r['oos_net']:6.0f} "
                f"L={r['long_oos']:2d}/${r['long_net']:5.0f} S={r['short_oos']:2d}/${r['short_net']:5.0f} "
                f"max/d={r['max_day']} avg/d={r['avg_active_day']}"
            )

    if hot:
        print("\nHIGH ACTIVITY + OOS PF>=1.1 + net>$200:")
        for r in hot[:8]:
            flag = "PASS" if r["pass"] else "FAIL"
            print(f"  [{flag}] {r['name']:<45} n={r['oos_n']:3d} PF={r['oos_pf']:.2f} ${r['oos_net']:6.0f}")

    pick = best_pass_trades or (hot[0] if hot else max(rows, key=lambda r: r["oos_net"]))
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIG:")
    print(f"  {pick['name']}")
    print(f"  OOS: n={pick['oos_n']} PF={pick['oos_pf']} ${pick['oos_net']} PASS={pick['pass']}")
    print(f"  Activity: max {pick['max_day']}/day, avg {pick['avg_active_day']}/active day")
    print("=" * 80)

    out = os.path.join("data", "hybrid_search_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "passing": passing[:20],
            "hot": hot[:15],
            "recommendation": pick,
            "all_count": len(rows),
        }, f, indent=2)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
