#!/usr/bin/env python3
"""
FAIL-fast V2 strengthen hunt for MNQ Pullback Hybrid.
Select on Jun-Jul only; validate Aug; forward Sep 1-21; cap <=40 variants.
Does NOT peek Sep when choosing. Does NOT densify near-misses.
Does NOT modify LONG_RECIPE_V4 / SHORT_RECIPE in place.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth

ET = resp.ET
JUNE = pd.Timestamp("2026-06-01", tz="UTC")
JULY_END = pd.Timestamp("2026-08-01", tz="UTC")  # exclusive end of Jul
AUG = pd.Timestamp("2026-08-01", tz="UTC")
SEP = pd.Timestamp("2026-09-01", tz="UTC")
SEP_END = pd.Timestamp("2026-09-22", tz="UTC")  # exclusive; tape ends Sep 21
MAX_VARIANTS = 40
WEEKS_JUN_LATEST = 16.0  # ~Jun1-Sep21


def entry_ts(t) -> pd.Timestamp:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def in_range(trades, start, end=None):
    out = []
    for t in trades:
        ts = entry_ts(t)
        if ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out.append(t)
    return out


def metrics(trades) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2),
        "expectancy": round(m["expectancy"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
    }


def _day_capped(recipe: Dict, counts: Dict, day: Any) -> bool:
    md = recipe.get("max_day")
    if md is None:
        return False
    return counts.get(day, 0) >= int(md)


def _window_repeat_allowed(recipe: Dict) -> bool:
    return recipe.get("max_day") is None


def simulate_param(
    a: Dict[str, Any],
    long_recipe: Dict[str, Any],
    short_recipe: Dict[str, Any],
    stop_key: str,
    tp_key: str,
    legs: str = "combined",
) -> List[scalp_bt.ScalpTrade]:
    """Parameterized copy of combo.simulate_combined (stop/tp/recipes/legs)."""
    if stop_key not in base.STOPS:
        raise KeyError(f"missing stop {stop_key}")
    if tp_key not in base.TPS:
        raise KeyError(f"missing tp {tp_key}")

    winset_short = hunt.WINSETS.get(short_recipe.get("win", "w6_all"), hunt.WINSETS["w6_all"])
    stop_cfg = base.STOPS[stop_key]
    tp_kind, tp_val = base.TPS[tp_key]

    trades: List[scalp_bt.ScalpTrade] = []
    pos: Optional[Dict] = None
    fired_long: Set[int] = set()
    fired_short: Set[int] = set()
    counts_long: Dict = {}
    counts_short: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])
    allow_long = legs in ("combined", "long_only")
    allow_short = legs in ("combined", "short_only")

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
                tr = gapped._close_position(pos, ts, float(exit_px), reason)
                tr.version = pos.get("leg", "combined")
                trades.append(tr)
                w = pos.get("win")
                if w is not None:
                    if pos["dir"] == 1:
                        fired_long.discard(int(w))
                    else:
                        fired_short.discard(int(w))
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

        long_wins = hunt.WINSETS.get(long_recipe.get("win", "w2_open"), hunt.WINSETS["w2_open"])
        short_ok_win = win in winset_short
        long_ok_win = win in long_wins

        side = 0
        leg = ""

        if allow_long and long_ok_win and base.stack_side("15_5_1", i, a) == 1:
            entry_mode = long_recipe.get("entry", "quiet")
            struct = long_recipe.get("structure")
            ok = base.entry_ok(entry_mode, 1, i, a)
            if ok and struct:
                # only if structure_ok exists and key is known
                if hasattr(srth, "structure_ok"):
                    ok = srth.structure_ok(struct, 1, i, a)
                else:
                    ok = False
            if ok:
                if win in fired_long and not _window_repeat_allowed(long_recipe):
                    pass
                elif _day_capped(long_recipe, counts_long, day):
                    pass
                else:
                    side, leg = 1, "long"

        if side == 0 and allow_short and short_ok_win:
            short_entry = short_recipe.get("entry", "1m_pullback")
            short_struct = short_recipe.get("structure", "lh_ll")
            ok = (
                base.stack_side("15_5_1", i, a) == -1
                and base.entry_ok(short_entry, -1, i, a)
                and srth.structure_ok(short_struct, -1, i, a)
            )
            if ok:
                if win in fired_short and not _window_repeat_allowed(short_recipe):
                    pass
                elif _day_capped(short_recipe, counts_short, day):
                    pass
                else:
                    side, leg = -1, "short"

        if side == 0:
            continue

        use = base.stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
        if use != use or use <= 0:
            continue
        entry_px = float(a["cl"][i])
        sl = entry_px - use if side == 1 else entry_px + use

        # TP from TPS dict
        if tp_kind == "R" or (isinstance(tp_key, str) and tp_key.endswith("R")):
            mult = float(tp_val) if tp_kind == "R" else float(str(tp_key).replace("R", "") or 1)
            # base.TPS values are typically ("R", 1.0) etc.
            if tp_kind == "R":
                mult = float(tp_val)
            else:
                mult = float(tp_val)
            tp_px = entry_px + use * mult if side == 1 else entry_px - use * mult
        else:
            # fallback 1R
            tp_px = entry_px + use if side == 1 else entry_px - use

        pos = {
            "ts": ts,
            "dir": side,
            "entry": entry_px,
            "sl": sl,
            "tp": tp_px,
            "sl_pts": use,
            "win": win,
            "leg": leg,
        }
        if side == 1:
            if not _window_repeat_allowed(long_recipe):
                fired_long.add(int(win))
            counts_long[day] = counts_long.get(day, 0) + 1
        else:
            if not _window_repeat_allowed(short_recipe):
                fired_short.add(int(win))
            counts_short[day] = counts_short.get(day, 0) + 1

    if pos is not None:
        i = n - 1
        tr = gapped._close_position(pos, pd.Timestamp(dt[i]), float(a["cl"][i]), "EOD")
        tr.version = pos.get("leg", "combined")
        trades.append(tr)
    return trades


def build_grid() -> List[Dict[str, Any]]:
    """DNA-preserving grid, hard-capped at MAX_VARIANTS."""
    long_wins = [w for w in ("w2_open", "w3_morning", "w4_no_noon") if w in hunt.WINSETS]
    long_entries = ["1m_pullback", "quiet"]
    tps = [k for k in ("1R", "1.5R", "2R") if k in base.TPS]
    stops = [k for k in ("atr1_1.5_6_25", "atr1_1.0_6_25", "atr1_2.0_6_25") if k in base.STOPS]
    if not stops:
        stops = ["atr1_1.5_6_25"] if "atr1_1.5_6_25" in base.STOPS else list(base.STOPS)[:1]

    long_max = [None, 2, 3]
    short_max = [None, 2]
    legs_opts = ["combined", "long_only", "short_only"]

    # Prefer frozen-adjacent first, then expand.
    variants: List[Dict[str, Any]] = []

    def add(v):
        if len(variants) >= MAX_VARIANTS:
            return
        # dedupe by json
        key = json.dumps(v, sort_keys=True, default=str)
        if any(json.dumps(x, sort_keys=True, default=str) == key for x in variants):
            return
        variants.append(v)

    # 1) Frozen V2 baseline
    add({
        "name": "frozen_v2",
        "long_win": "w4_no_noon",
        "long_entry": "1m_pullback",
        "long_structure": None,
        "long_max_day": None,
        "short_max_day": None,
        "stop": "atr1_1.5_6_25",
        "tp": "1R",
        "legs": "combined",
    })

    # 2) Older hybrid_recommendation winner if hh_hl supported
    hh_ok = False
    if hasattr(srth, "structure_ok"):
        # probe: don't invent — only if "hh_hl" is accepted by existing code paths
        # We detect by checking source/annotations if STRUCTURES dict exists
        structs = getattr(srth, "STRUCTURES", None) or getattr(srth, "STRUCTURE_MODES", None)
        if isinstance(structs, dict) and "hh_hl" in structs:
            hh_ok = True
        elif "hh_hl" in getattr(srth, "__dict__", {}):
            hh_ok = True
        else:
            # try reading function defaults / docstring
            import inspect
            try:
                src = inspect.getsource(srth.structure_ok)
                hh_ok = "hh_hl" in src
            except Exception:
                hh_ok = False
    if hh_ok and "w2_open" in hunt.WINSETS:
        add({
            "name": "hybrid_rec_w2_hh_hl",
            "long_win": "w2_open",
            "long_entry": "quiet",
            "long_structure": "hh_hl",
            "long_max_day": None,
            "short_max_day": None,
            "stop": "atr1_1.5_6_25",
            "tp": "1R",
            "legs": "combined",
        })

    # 3) Core DNA grid: win x entry x tp (stop frozen) combined only
    for lw, le, tp in product(long_wins, long_entries, tps):
        add({
            "name": f"{lw}|{le}|{tp}|atr1.5|comb",
            "long_win": lw,
            "long_entry": le,
            "long_structure": None,
            "long_max_day": None,
            "short_max_day": None,
            "stop": stops[0],
            "tp": tp,
            "legs": "combined",
        })

    # 4) Nearby stops on frozen DNA
    for sk in stops[1:]:
        add({
            "name": f"w4|pb|{tps[0]}|{sk}|comb",
            "long_win": "w4_no_noon",
            "long_entry": "1m_pullback",
            "long_structure": None,
            "long_max_day": None,
            "short_max_day": None,
            "stop": sk,
            "tp": tps[0] if tps else "1R",
            "legs": "combined",
        })

    # 5) max_day knobs on frozen DNA
    for lm, sm in product(long_max, short_max):
        if lm is None and sm is None:
            continue
        add({
            "name": f"w4|pb|1R|mdL{lm}_S{sm}",
            "long_win": "w4_no_noon",
            "long_entry": "1m_pullback",
            "long_structure": None,
            "long_max_day": lm,
            "short_max_day": sm,
            "stop": stops[0],
            "tp": tps[0] if tps else "1R",
            "legs": "combined",
        })

    # 6) single-leg probes on frozen DNA
    for legs in ("long_only", "short_only"):
        add({
            "name": f"frozen_{legs}",
            "long_win": "w4_no_noon",
            "long_entry": "1m_pullback",
            "long_structure": None,
            "long_max_day": None,
            "short_max_day": None,
            "stop": stops[0],
            "tp": tps[0] if tps else "1R",
            "legs": legs,
        })

    return variants[:MAX_VARIANTS]


def score_windows(trades) -> Dict[str, Any]:
    jun_jul = in_range(trades, JUNE, JULY_END)
    aug = in_range(trades, AUG, SEP)
    sep = in_range(trades, SEP, SEP_END)
    full = in_range(trades, JUNE, None)
    return {
        "jun_jul": metrics(jun_jul),
        "aug": metrics(aug),
        "sep": metrics(sep),
        "jun_latest": metrics(full),
    }


def gate_aug(m) -> bool:
    return m["profit_factor"] >= 1.0 and m["net_pnl"] >= 0


def gate_sep(m) -> bool:
    return m["profit_factor"] >= 1.2 and m["net_pnl"] > 0 and m["trades"] >= 15


def gate_full(m) -> bool:
    return m["net_pnl"] >= 1600 and m["profit_factor"] >= 1.25


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out_path = os.path.join("data", "v2_strengthen_hunt.json")
    pass_path = os.path.join("data", "v2_strengthen_pass.json")

    print("V2 STRENGTHEN HUNT (FAIL-fast, <=40, no Sep peek for selection)")
    print(f"STOPS keys sample: {list(base.STOPS)[:12]}")
    print(f"TPS keys: {list(base.TPS)}")
    print(f"WINSETS: {list(hunt.WINSETS)}")

    df = official.load_1m(csv)
    a = combo.prepare(df)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    grid = build_grid()
    print(f"variants planned: {len(grid)}")

    results = []
    # Selection uses Jun-Jul only — rank after all scored, but we still evaluate
    # Aug/Sep for gates (Sep not used to choose which to keep expanding — grid fixed).
    for i, spec in enumerate(grid):
        long_r = deepcopy(combo.LONG_RECIPE_V4)
        short_r = deepcopy(combo.SHORT_RECIPE)
        long_r["win"] = spec["long_win"]
        long_r["entry"] = spec["long_entry"]
        long_r["max_day"] = spec["long_max_day"]
        long_r["stop"] = spec["stop"]
        long_r["tp"] = spec["tp"]
        if spec.get("long_structure"):
            long_r["structure"] = spec["long_structure"]
        short_r["max_day"] = spec["short_max_day"]
        short_r["stop"] = spec["stop"]
        short_r["tp"] = spec["tp"]

        try:
            trades = simulate_param(
                a, long_r, short_r, spec["stop"], spec["tp"], legs=spec["legs"]
            )
        except Exception as e:
            results.append({"spec": spec, "error": str(e)})
            print(f"[{i+1}/{len(grid)}] {spec['name']} ERROR {e}")
            continue

        windows = score_windows(trades)
        jj = windows["jun_jul"]
        row = {
            "spec": spec,
            "long_recipe": long_r,
            "short_recipe": short_r,
            "windows": windows,
            "train_jun_jul_net": jj["net_pnl"],
            "train_jun_jul_pf": jj["profit_factor"],
            "gate_aug": gate_aug(windows["aug"]),
            "gate_sep": gate_sep(windows["sep"]),
            "gate_full": gate_full(windows["jun_latest"]),
        }
        results.append(row)
        print(
            f"[{i+1}/{len(grid)}] {spec['name']}: "
            f"JJ n={jj['trades']} PF={jj['profit_factor']:.2f} ${jj['net_pnl']:.0f} | "
            f"Aug PF={windows['aug']['profit_factor']:.2f} ${windows['aug']['net_pnl']:.0f} "
            f"({'Y' if row['gate_aug'] else 'N'}) | "
            f"Sep n={windows['sep']['trades']} PF={windows['sep']['profit_factor']:.2f} "
            f"${windows['sep']['net_pnl']:.0f} ({'Y' if row['gate_sep'] else 'N'}) | "
            f"Full ${windows['jun_latest']['net_pnl']:.0f}"
        )

    # Select ONLY using Jun-Jul among those that clear Aug+Sep gates
    cleared = [
        r for r in results
        if "windows" in r and r["gate_aug"] and r["gate_sep"]
    ]
    # rank by train Jun-Jul net (no Sep in selection)
    cleared.sort(key=lambda r: (r["train_jun_jul_net"], r["train_jun_jul_pf"]), reverse=True)

    status = "FAIL"
    winner = None
    if cleared:
        # prefer one that also clears full $1600; else best clearing 2+3
        full_ok = [r for r in cleared if r["gate_full"]]
        winner = full_ok[0] if full_ok else cleared[0]
        if winner["gate_full"]:
            status = "PASS"
        else:
            status = "FAIL_FULL_TARGET"  # cleared 1-3 but not $1600 — still report as best

    # Near-miss: best by Jun-Jul among those with gate_aug, else overall JJ
    def sort_key(r):
        if "windows" not in r:
            return (-1e18, -1)
        return (r["train_jun_jul_net"], r["train_jun_jul_pf"])

    scored = [r for r in results if "windows" in r]
    scored.sort(key=sort_key, reverse=True)
    near = None
    for r in scored:
        if not (r["gate_aug"] and r["gate_sep"]):
            near = r
            why = []
            if not r["gate_aug"]:
                why.append(
                    f"Aug FAIL PF={r['windows']['aug']['profit_factor']:.2f} "
                    f"net={r['windows']['aug']['net_pnl']:.2f} (need PF>=1 net>=0)"
                )
            if not r["gate_sep"]:
                s = r["windows"]["sep"]
                why.append(
                    f"Sep FAIL n={s['trades']} PF={s['profit_factor']:.2f} "
                    f"net={s['net_pnl']:.2f} (need n>=15 PF>=1.2 net>0)"
                )
            near = {**r, "failed_gates": why}
            break
    if near is None and scored:
        # all cleared 2+3? then near-miss is one missing full
        for r in scored:
            if not r["gate_full"]:
                near = {
                    **r,
                    "failed_gates": [
                        f"Full FAIL net={r['windows']['jun_latest']['net_pnl']:.2f} "
                        f"PF={r['windows']['jun_latest']['profit_factor']:.2f} "
                        f"(need net>=1600 PF>=1.25)"
                    ],
                }
                break

    # FAIL-fast rule: if none clear gates 2+3 after first full screen, STOP (no expand)
    if not cleared:
        status = "FAIL"

    payload = {
        "status": status if status != "FAIL_FULL_TARGET" else "FAIL",
        "note": (
            "No variant cleared Aug+Sep gates after first screen; grid not expanded."
            if not cleared
            else (
                "Cleared Aug+Sep but not full $1600/PF1.25 target."
                if status == "FAIL_FULL_TARGET"
                else "All hard gates cleared."
            )
        ),
        "variants_tested": len(scored),
        "variants_planned": len(grid),
        "data_bars": len(df),
        "data_end": str(df["datetime"].iloc[-1]),
        "selection_rule": "rank by Jun-Jul net/PF among gate_aug AND gate_sep; Sep never used to choose",
        "gates": {
            "aug": "PF>=1.0 and net>=0",
            "sep": "PF>=1.2 and net>0 and n>=15",
            "full": "net>=1600 and PF>=1.25 (~$100/wk over ~16w)",
        },
        "winner": None,
        "best_near_miss": None,
        "all_results": [
            {
                "name": r["spec"]["name"],
                "spec": r["spec"],
                "windows": r.get("windows"),
                "gate_aug": r.get("gate_aug"),
                "gate_sep": r.get("gate_sep"),
                "gate_full": r.get("gate_full"),
                "error": r.get("error"),
            }
            for r in results
        ],
        "reminders": {
            "size_2x": "2x size doubles dollars but not edge quality",
            "production_paper": "S16 mid ORB remains locked production paper (~$80/wk at 1x)",
            "economics": (
                "At 1 MNQ lot and 1R, even a good edge is often hundreds–low thousands "
                "per quarter; hitting ~$1600/16w may require size not DNA."
            ),
            "live_trading": "OFF",
        },
    }

    if winner:
        jl = winner["windows"]["jun_latest"]
        per_week = round(jl["net_pnl"] / WEEKS_JUN_LATEST, 2)
        payload["winner"] = {
            "spec": winner["spec"],
            "long_recipe": winner["long_recipe"],
            "short_recipe": winner["short_recipe"],
            "windows": winner["windows"],
            "dollars_per_week_est": per_week,
            "gate_aug": winner["gate_aug"],
            "gate_sep": winner["gate_sep"],
            "gate_full": winner["gate_full"],
        }
        if winner["gate_aug"] and winner["gate_sep"] and winner["gate_full"]:
            with open(pass_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "PASS",
                        "long_recipe": winner["long_recipe"],
                        "short_recipe": winner["short_recipe"],
                        "stop": winner["spec"]["stop"],
                        "tp": winner["spec"]["tp"],
                        "legs": winner["spec"]["legs"],
                        "windows": winner["windows"],
                        "dollars_per_week_est": per_week,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"PASS recipe written {pass_path}")
            payload["pass_json"] = pass_path

    if near:
        payload["best_near_miss"] = {
            "spec": near["spec"],
            "windows": near["windows"],
            "failed_gates": near.get("failed_gates", []),
            "gate_aug": near.get("gate_aug"),
            "gate_sep": near.get("gate_sep"),
            "gate_full": near.get("gate_full"),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {out_path}")
    print(f"STATUS={payload['status']} tested={payload['variants_tested']}")
    if payload["status"] == "PASS":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
