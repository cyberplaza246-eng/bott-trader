#!/usr/bin/env python3
"""
Search hybrid configs: more long activity without w6_all quiet-1m bleed.

Fixed: short w6_all LH_LL pullback 1R unlimited, one position.
Vary long windows + optional quality filters.
"""
from __future__ import annotations

import os
import sys
from itertools import product
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth

WIN_OPTIONS = {
    "w2_open": hunt.WINSETS["w2_open"],
    "w3_morning": hunt.WINSETS["w3_morning"],
    "w4_no_noon": base.WINSETS["w4_no_noon"],
    "w6_all": hunt.WINSETS["w6_all"],
}


def simulate_variant(
    a: Dict[str, Any],
    long_win_key: str,
    long_filter: str,
) -> List:
    """One-position combined; long filter variants."""
    long_wins = WIN_OPTIONS[long_win_key]
    short_wins = hunt.WINSETS["w6_all"]
    stop_cfg = base.STOPS["atr1_1.5_6_25"]
    tp_kind, tp_val = base.TPS["1R"]

    trades = []
    pos = None
    counts_long: Dict = {}
    counts_short: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])

    def long_ok(i: int) -> bool:
        if not combo.long_entry_ok(i, a):
            return False
        if long_filter == "quiet":
            return True
        if long_filter == "hh_hl":
            return srth.structure_ok("hh_hl", 1, i, a)
        if long_filter == "sep_noon":
            m = int(a["mins"][i])
            if m >= base.MORNING_END:
                sep = float(a["sep"][i])
                return sep == sep and sep >= base.SEP_MIN
            return True
        if long_filter == "hh_sep_noon":
            if not srth.structure_ok("hh_hl", 1, i, a):
                return False
            m = int(a["mins"][i])
            if m >= base.MORNING_END:
                sep = float(a["sep"][i])
                return sep == sep and sep >= base.SEP_MIN
            return True
        raise ValueError(long_filter)

    for i in range(hunt.WARMUP, n):
        ts = __import__("pandas").Timestamp(dt[i])
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
                import scripts.backtest_ema15_gapped_vs_full as gapped
                trades.append(gapped._close_position(pos, ts, float(exit_px), reason))
                pos = None

        if a["dow"][i] >= 5:
            continue
        if m < base.ENTRY_MINUTE or m >= base.FLAT_MINUTE:
            continue
        if pos is not None:
            continue

        win = base.window_index(m)
        if win is None:
            continue

        side = 0
        if win in long_wins and base.stack_side("15_5_1", i, a) == 1 and long_ok(i):
            side = 1
        elif win in short_wins and combo.short_entry_ok(i, a):
            side = -1

        if side == 0:
            continue

        use = base.stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
        if use != use or use <= 0:
            continue
        entry_px = float(a["cl"][i])
        sl = entry_px - use if side == 1 else entry_px + use
        tp_px = entry_px + use if side == 1 else entry_px - use

        pos = {
            "ts": ts, "dir": side, "entry": entry_px, "sl": sl, "tp": tp_px, "sl_pts": use,
        }
        if side == 1:
            counts_long[day] = counts_long.get(day, 0) + 1
        else:
            counts_short[day] = counts_short.get(day, 0) + 1

    return trades


def score(trades: List) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    im, om = resp.group_metrics(ins), resp.group_metrics(oos)
    longs = [t for t in oos if t.direction == "long"]
    shorts = [t for t in oos if t.direction == "short"]
    lm, sm = resp.group_metrics(longs), resp.group_metrics(shorts)
    return {
        "all_n": len(trades),
        "oos_n": om["trades"],
        "oos_pf": om["profit_factor"],
        "oos_net": om["total_pnl"],
        "oos_wr": om["win_rate"],
        "is_pf": im["profit_factor"],
        "is_net": im["total_pnl"],
        "long_oos_n": lm["trades"],
        "long_oos_pf": lm["profit_factor"],
        "long_oos_net": lm["total_pnl"],
        "short_oos_n": sm["trades"],
        "short_oos_pf": sm["profit_factor"],
        "short_oos_net": sm["total_pnl"],
        "pass": hunt.is_pass(im, om),
    }


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = combo.prepare(df)

    filters = ("quiet", "hh_hl", "sep_noon", "hh_sep_noon")
    rows = []
    for win_key, filt in product(WIN_OPTIONS.keys(), filters):
        tr = simulate_variant(a, win_key, filt)
        s = score(tr)
        rows.append({"long_win": win_key, "long_filter": filt, **s})

    # baseline references
    base_w2 = score(combo.simulate_combined_long_w2(a))
    base_w6 = score(combo.simulate_combined_active(a))

    passing = [r for r in rows if r["pass"]]
    passing.sort(key=lambda r: (-r["oos_pf"], -r["oos_net"], -r["oos_n"]))

    interesting = [
        r for r in rows
        if not r["pass"] and r["oos_net"] > 0 and r["oos_pf"] >= 1.05 and r["oos_n"] >= 100
    ]
    interesting.sort(key=lambda r: (-r["oos_n"], -r["oos_pf"]))

    print("HYBRID SEARCH — short fixed (w6 LH_LL pullback unlimited)")
    print(f"Baseline w2 quiet: OOS n={base_w2['oos_n']} PF={base_w2['oos_pf']:.2f} ${base_w2['oos_net']:.0f}")
    print(f"Baseline w6 quiet: OOS n={base_w6['oos_n']} PF={base_w6['oos_pf']:.2f} ${base_w6['oos_net']:.0f}")
    print()

    if passing:
        print("PASS configs:")
        for r in passing[:8]:
            print(
                f"  long={r['long_win']:12s} filter={r['long_filter']:12s} "
                f"OOS n={r['oos_n']:3d} PF={r['oos_pf']:.2f} ${r['oos_net']:6.0f} "
                f"L={r['long_oos_n']:3d}/${r['long_oos_net']:5.0f} S={r['short_oos_n']:3d}/${r['short_oos_net']:5.0f}"
            )

    print("\nTop activity + positive OOS (not full PASS):")
    for r in interesting[:6]:
        print(
            f"  long={r['long_win']:12s} filter={r['long_filter']:12s} "
            f"OOS n={r['oos_n']:3d} PF={r['oos_pf']:.2f} ${r['oos_net']:6.0f}"
        )

    best_pass = passing[0] if passing else None
    best_activity = max(
        [r for r in rows if r["oos_net"] > 0],
        key=lambda r: (r["oos_n"], r["oos_pf"]),
        default=None,
    )

    import json
    out = os.path.join("data", "hybrid_recommendation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_w2_quiet": base_w2,
            "baseline_w6_quiet": base_w6,
            "passing": passing[:10],
            "interesting": interesting[:10],
            "recommendation": best_pass or best_activity,
        }, f, indent=2, default=str)
    print(f"\nSaved {out}")

    if best_pass:
        print(
            f"RECOMMEND (PASS): long={best_pass['long_win']} filter={best_pass['long_filter']} "
            f"— OOS PF {best_pass['oos_pf']:.2f}, ${best_pass['oos_net']:.0f}, n={best_pass['oos_n']}"
        )
    elif best_activity:
        print(
            f"RECOMMEND (best positive): long={best_activity['long_win']} "
            f"filter={best_activity['long_filter']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
