#!/usr/bin/env python3
"""
Fresh-period short validation + unified combined bidirectional backtest.

Locked research spec (no parameter changes):
  LONG: 15/5/1 bullish, quiet 1m, 1R, ATR×1.5 clip [6-25], LONG_WINNER entry rules
  SHORT: 15/5/1 bearish, LH_LL structure, 1m pullback, 1R, same stop, no VWAP, w6_all

Combined rules:
  - One position at a time
  - All six RTH windows active for BOTH directions (no directional time ban)
  - Clock controls scan windows only; stack + setup determine direction

Fresh holdout: entries >= FRESH_START (Aug 2026 — not used to select short recipe).
Development OOS: entries >= 2026-06-01 (standard split).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_level_strategy_search as lvl
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth

ET = resp.ET
OOS_START = edge.OOS_START
FRESH_START = pd.Timestamp("2026-08-01", tz="UTC")
MORNING_END = base.MORNING_END
MIDDAY_END = 14 * 60

SHORT_RECIPE = {
    "stack": "15_5_1",
    "loop": "window_loop",
    "win": "w6_all",
    "stop": "atr1_1.5_6_25",
    "side": "short",
    "max_day": None,
    "max_lots": 1,
    "sep_min": base.SEP_MIN,
    "entry": "1m_pullback",
    "tp": "1R",
    "vwap": "none",
    "structure": "lh_ll",
    "sep_mode": "none",
}

LONG_RECIPE_W2 = dict(hunt.LONG_WINNER)
LONG_RECIPE_W2["max_day"] = None
LONG_RECIPE_W6 = dict(hunt.LONG_WINNER)
LONG_RECIPE_W6["win"] = "w6_all"
LONG_RECIPE_W6["max_day"] = None
LONG_RECIPE_V4 = {
    "stack": "15_5_1",
    "loop": "window_loop",
    "entry": "1m_pullback",
    "win": "w4_no_noon",
    "stop": "atr1_1.5_6_25",
    "tp": "1R",
    "side": "long",
    "sep": False,
    "cooldown": 0,
    "max_day": None,
    "max_lots": 1,
    "morning": False,
    "vwap": "none",
    "sep_short_min": None,
}
LONG_RECIPE = LONG_RECIPE_V4


def simulate_combined_long_w2(a: Dict[str, Any]) -> List[scalp_bt.ScalpTrade]:
    """Combined: long w2_open only, short w6_all, one position."""
    global LONG_RECIPE
    saved = LONG_RECIPE
    LONG_RECIPE = LONG_RECIPE_W2
    trades = simulate_combined(a)
    LONG_RECIPE = saved
    return trades


def simulate_combined_active(a: Dict[str, Any]) -> List[scalp_bt.ScalpTrade]:
    """Current spec: long w6_all + short w6_all, unlimited daily, one position."""
    global LONG_RECIPE
    saved = LONG_RECIPE
    LONG_RECIPE = LONG_RECIPE_W6
    trades = simulate_combined(a)
    LONG_RECIPE = saved
    return trades


def simulate_combined_recommended(a: Dict[str, Any]) -> List[scalp_bt.ScalpTrade]:
    """Search winner: long w4 pullback + short w6 LH_LL pullback."""
    global LONG_RECIPE
    saved = LONG_RECIPE
    LONG_RECIPE = LONG_RECIPE_V4
    trades = simulate_combined(a)
    LONG_RECIPE = saved
    return trades


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    return srth.prepare(df)


def long_entry_ok(i: int, a: Dict) -> bool:
    mode = LONG_RECIPE.get("entry", "quiet")
    return base.entry_ok(mode, 1, i, a)


def short_entry_ok(i: int, a: Dict) -> bool:
    return (
        base.stack_side("15_5_1", i, a) == -1
        and base.entry_ok("1m_pullback", -1, i, a)
        and srth.structure_ok("lh_ll", -1, i, a)
    )


def _day_capped(recipe: Dict, counts: Dict, day: Any) -> bool:
    md = recipe.get("max_day")
    if md is None:
        return False
    return counts.get(day, 0) >= int(md)


def _window_repeat_allowed(recipe: Dict) -> bool:
    """When max_day is None, re-enter same window whenever setup fires again."""
    return recipe.get("max_day") is None


def simulate_combined(a: Dict[str, Any]) -> List[scalp_bt.ScalpTrade]:
    """One account, one position, w6_all windows, both directions."""
    winset = hunt.WINSETS["w6_all"]
    stop_cfg = base.STOPS["atr1_1.5_6_25"]
    tp_kind, tp_val = base.TPS["1R"]

    trades: List[scalp_bt.ScalpTrade] = []
    pos: Optional[Dict] = None
    fired_long: Set[int] = set()
    fired_short: Set[int] = set()
    counts_long: Dict = {}
    counts_short: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])

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

        long_wins = hunt.WINSETS.get(LONG_RECIPE.get("win", "w2_open"), hunt.WINSETS["w2_open"])
        short_ok_win = win in winset
        long_ok_win = win in long_wins

        side = 0
        leg = ""
        if long_ok_win and base.stack_side("15_5_1", i, a) == 1 and long_entry_ok(i, a):
            if win in fired_long and not _window_repeat_allowed(LONG_RECIPE):
                pass
            elif _day_capped(LONG_RECIPE, counts_long, day):
                pass
            else:
                side, leg = 1, "long"
        elif short_ok_win and short_entry_ok(i, a):
            if win in fired_short and not _window_repeat_allowed(SHORT_RECIPE):
                pass
            elif _day_capped(SHORT_RECIPE, counts_short, day):
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
            if not _window_repeat_allowed(LONG_RECIPE):
                fired_long.add(int(win))
            counts_long[day] = counts_long.get(day, 0) + 1
        else:
            if not _window_repeat_allowed(SHORT_RECIPE):
                fired_short.add(int(win))
            counts_short[day] = counts_short.get(day, 0) + 1

    if pos is not None:
        i = n - 1
        tr = gapped._close_position(pos, pd.Timestamp(dt[i]), float(a["cl"][i]), "EOD")
        tr.version = pos.get("leg", "combined")
        trades.append(tr)

    return trades


def filter_trades(trades: List[scalp_bt.ScalpTrade], start: pd.Timestamp) -> List[scalp_bt.ScalpTrade]:
    out = []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts >= start:
            out.append(t)
    return out


def session_bucket(trade: scalp_bt.ScalpTrade) -> str:
    ts = pd.Timestamp(trade.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert(ET)
    m = et.hour * 60 + et.minute
    if m < MORNING_END:
        return "morning"
    if m < MIDDAY_END:
        return "midday"
    return "afternoon"


def monthly_pnl(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, float]:
    buckets: Dict[str, float] = defaultdict(float)
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        buckets[key] += t.pnl
    return dict(sorted(buckets.items()))


def metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    if not trades:
        z = resp.group_metrics([])
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "max_consec_losses": 0,
        }
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2),
        "expectancy": round(m["expectancy"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
        "max_consec_losses": m.get("max_consec_losses", 0),
    }


def max_consec_losses(trades: List[scalp_bt.ScalpTrade]) -> int:
    best = cur = 0
    for t in trades:
        if t.pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def enrich_metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    m = metrics(trades)
    m["max_consec_losses"] = max_consec_losses(trades)
    m["total_pnl"] = m["net_pnl"]
    return m


def session_splits(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Dict]:
    return {
        label: enrich_metrics([t for t in trades if session_bucket(t) == label])
        for label in ("morning", "midday", "afternoon")
    }


def full_report(label: str, trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    fresh = filter_trades(trades, FRESH_START)
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    oos_longs = [t for t in oos if t.direction == "long"]
    oos_shorts = [t for t in oos if t.direction == "short"]

    rep = {
        "label": label,
        "overall": enrich_metrics(trades),
        "is": enrich_metrics(ins),
        "oos": enrich_metrics(oos),
        "fresh": enrich_metrics(fresh),
        "long": enrich_metrics(longs),
        "short": enrich_metrics(shorts),
        "oos_long": enrich_metrics(oos_longs),
        "oos_short": enrich_metrics(oos_shorts),
        "session_oos": session_splits(oos),
        "monthly_all": monthly_pnl(trades),
        "monthly_oos": monthly_pnl(oos),
        "pass_oos": hunt.is_pass(enrich_metrics(ins), enrich_metrics(oos)),
        "pass_fresh": (
            fresh and enrich_metrics(fresh)["profit_factor"] >= 1.0
            and enrich_metrics(fresh)["net_pnl"] > 0
        ),
    }
    return rep


def print_metrics_block(title: str, m: Dict[str, Any]) -> None:
    print(
        f"  {title}: n={m['trades']:3d} WR={m['win_rate']:5.1f}% "
        f"PF={m['profit_factor']:5.2f} Net=${m['net_pnl']:8.1f} "
        f"DD=${m['max_drawdown']:7.1f} maxLL={m['max_consec_losses']}"
    )


def print_report(r: Dict[str, Any]) -> None:
    flag = "PASS" if r["pass_oos"] else "FAIL"
    print(f"\n{'=' * 90}")
    print(f"{r['label']}  [OOS {flag}]")
    print(f"{'=' * 90}")
    print_metrics_block("ALL", r["overall"])
    print_metrics_block("IS", r["is"])
    print_metrics_block("OOS", r["oos"])
    print_metrics_block("FRESH (>= Aug 2026)", r["fresh"])
    print_metrics_block("Long ALL", r["long"])
    print_metrics_block("Short ALL", r["short"])
    print_metrics_block("OOS Long", r["oos_long"])
    print_metrics_block("OOS Short", r["oos_short"])
    print("  OOS session split:")
    for period in ("morning", "midday", "afternoon"):
        s = r["session_oos"][period]
        sign = "+" if s["net_pnl"] > 0 else "−"
        print(
            f"    {period:10s} n={s['trades']:3d} PF={s['profit_factor']:.2f} "
            f"${s['net_pnl']:7.0f} ({sign})"
        )
    print("  Monthly OOS P&L:")
    for month, pnl in r["monthly_oos"].items():
        sign = "+" if pnl > 0 else "−"
        print(f"    {month}: ${pnl:7.0f} ({sign})")


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "combined_bidirectional_validation.json")

    print("FRESH-PERIOD SHORT VALIDATION + COMBINED BIDIRECTIONAL")
    print(f"Costs ${official.PT}/pt + ${official.COMM} RT")
    print(f"OOS split >= {OOS_START.date()} | Fresh holdout >= {FRESH_START.date()}")
    print()

    df = official.load_1m(csv)
    a = prepare(df)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    # --- 1) Short standalone (locked recipe) ---
    short_trades = srth.simulate_short(a, SHORT_RECIPE)
    short_rep = full_report("SHORT standalone (locked recipe)", short_trades)
    print_report(short_rep)

    # --- 2) Long standalone on w6_all (bidirectional scan windows) ---
    long_trades = hunt.simulate_ext(hunt.prepare(df), LONG_RECIPE)
    long_rep = full_report("LONG standalone (quiet 1m, w6_all windows)", long_trades)
    print_report(long_rep)

    # --- 3) Combined unified one-position (long w6_all scan) ---
    combo_trades = simulate_combined(a)
    combo_rep = full_report("COMBINED (one pos, long w6_all + short w6_all)", combo_trades)
    print_report(combo_rep)

    # --- 3b) Combined: long w2_open + short w6_all (locked long windows) ---
    combo_w2_trades = simulate_combined_long_w2(a)
    combo_w2_rep = full_report(
        "COMBINED (one pos, long w2_open + short w6_all — locked recipes)",
        combo_w2_trades,
    )
    print_report(combo_w2_rep)

    # --- 4) Stacked trades diagnostic ---
    long_w2 = hunt.simulate_ext(hunt.prepare(df), LONG_RECIPE_W2)
    stacked = long_w2 + short_trades
    stacked_oos = edge.split_is_oos(stacked)[1]
    stacked_m = enrich_metrics(stacked_oos)
    print(f"\n{'=' * 90}")
    print("DIAGNOSTIC: stacked long+short sims (separate accounts — NOT one position)")
    print(f"{'=' * 90}")
    print_metrics_block("OOS stacked", stacked_m)

    masking = (
        combo_w2_rep["oos_long"]["net_pnl"] > 0
        and combo_w2_rep["oos_short"]["net_pnl"] > 0
    )
    best_combo = combo_w2_rep if combo_w2_rep["oos"]["profit_factor"] >= combo_rep["oos"]["profit_factor"] else combo_rep
    verdict_parts = []
    if short_rep["pass_oos"]:
        verdict_parts.append("Short recipe PASSes standard OOS gates.")
    elif short_rep["fresh"]["net_pnl"] > 0:
        verdict_parts.append("Short fresh holdout positive but full OOS gates not met.")
    else:
        verdict_parts.append("Short does not hold on fresh/OOS — needs more data before implementation.")

    if combo_w2_rep["pass_oos"] or best_combo["pass_oos"]:
        verdict_parts.append("Combined system PASSes OOS — candidate for implementation path.")
    elif best_combo["oos"]["net_pnl"] > 0:
        verdict_parts.append(
            f"Combined OOS positive (PF {best_combo['oos']['profit_factor']:.2f}, "
            f"${best_combo['oos']['net_pnl']:.0f}) but below full PASS gates."
        )
    else:
        verdict_parts.append("Combined system does not PASS — do not implement yet.")

    verdict = " ".join(verdict_parts)
    print(f"\nVerdict: {verdict}")

    payload = {
        "spec": {
            "long_w2": LONG_RECIPE_W2,
            "long_w6": LONG_RECIPE_W6,
            "short": SHORT_RECIPE,
            "combined_w6_long": "one_position long+short w6_all",
            "combined_w2_long": "one_position long w2_open short w6_all",
        },
        "oos_start": str(OOS_START),
        "fresh_start": str(FRESH_START),
        "short_standalone": short_rep,
        "long_standalone_w6": long_rep,
        "combined_w6_long": combo_rep,
        "combined_w2_long": combo_w2_rep,
        "stacked_oos_diagnostic": stacked_m,
        "verdict": verdict,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0 if (combo_w2_rep["pass_oos"] or combo_rep["pass_oos"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
