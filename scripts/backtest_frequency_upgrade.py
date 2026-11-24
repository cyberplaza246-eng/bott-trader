#!/usr/bin/env python3
"""
Frequency upgrade search — Jun 2026+ on Databento MNQ 1m.

Paths:
  1) Continuous RTH scan (rth_loop) pullback-only
  2) Faster turnover (smaller R targets, optional max-hold minutes)
  3) Second leg: 5m scalp momentum alongside stack engine
  4) Stacked books: selective long pullback + aggressive short
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_short_rth_all_windows as srth
import scripts.pass_hunt_shorts as hunt

ET = resp.ET
JUNE = pd.Timestamp("2026-06-01", tz="UTC")
PT, COMM = official.PT, official.COMM

TPS = dict(base.TPS)
TPS["0.5R"] = ("r", 0.5)
TPS["0.75R"] = ("r", 0.75)
TPS["fix8"] = ("fixed", 8.0)
TPS["fix12"] = ("fixed", 12.0)


def entry_ts(t) -> pd.Timestamp:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def jun_filter(trades: List) -> List:
    return [t for t in trades if entry_ts(t) >= JUNE]


def weekday_stats(trades: List, end_d: date) -> Dict[str, Any]:
    trades = jun_filter(trades)
    days = Counter(entry_ts(t).tz_convert(ET).date() for t in trades)
    cal = sum(
        1
        for i in range((end_d - date(2026, 6, 1)).days + 1)
        if (date(2026, 6, 1) + timedelta(i)).weekday() < 5
    )
    counts = list(days.values()) if days else []
    ge6 = sum(1 for c in counts if c >= 6)
    return {
        "trades": len(trades),
        "weekdays": cal,
        "avg_per_weekday": round(len(trades) / cal, 2) if cal else 0.0,
        "avg_active_day": round(len(trades) / len(days), 2) if days else 0.0,
        "max_day": max(counts) if counts else 0,
        "days_ge6": ge6,
        "pct_weekdays_ge6": round(100 * ge6 / cal, 1) if cal else 0.0,
    }


def metrics(trades: List) -> Dict[str, Any]:
    trades = jun_filter(trades)
    if not trades:
        return {"trades": 0, "pf": 0.0, "net": 0.0, "wr": 0.0, "exp": 0.0, "dd": 0.0}
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "pf": round(m["profit_factor"], 2),
        "net": round(m["total_pnl"], 1),
        "wr": round(m["win_rate"], 1),
        "exp": round(m["expectancy"], 2),
        "dd": round(m["max_drawdown"], 1),
    }


def simulate_leg(
    a: Dict[str, Any],
    *,
    side_mode: str,
    loop: str,
    entry: str,
    winset_name: Optional[str],
    stop_key: str = "atr1_1.5_6_25",
    tp_key: str = "1R",
    cooldown_min: int = 0,
    max_day: int = 99,
    max_lots: int = 1,
    structure: Optional[str] = None,
    max_hold_min: Optional[int] = None,
) -> List:
    """Single-book leg: rth_loop or window_loop, pullback entries, optional structure filter."""
    winset = base.WINSETS.get(winset_name) if winset_name else None
    if winset_name == "w6_all":
        winset = hunt.WINSETS["w6_all"]
    stop_cfg = base.STOPS[stop_key]
    tp_kind, tp_val = TPS[tp_key]

    trades: List = []
    opens: List[Dict] = []
    fired: Set[int] = set()
    counts: Dict = {}
    last_entry: Optional[Tuple] = None
    dt, n = a["dt"], len(a["cl"])

    for i in range(base.WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]
        still = []
        for p in opens:
            reason = exit_px = None
            entry_m = int(p["entry_min"])
            if max_hold_min and m - entry_m >= max_hold_min:
                reason, exit_px = "MAX_HOLD", float(a["cl"][i])
            elif m >= base.FLAT_MINUTE:
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

        if a["dow"][i] >= 5 or m < base.ENTRY_MINUTE or m >= base.FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= max_day:
            continue

        win: Optional[int] = None
        if loop in ("window_once", "window_loop"):
            win = base.window_index(m)
            if win is None or win not in (winset or set()):
                continue
            if loop == "window_once" and win in fired:
                continue
            if loop == "window_loop" and win in fired:
                continue
        elif loop in ("rth_loop", "morning_loop"):
            if winset is not None and not base.in_winset(m, winset):
                continue
            if cooldown_min > 0 and last_entry is not None:
                ld, lm = last_entry
                if ld == day and m - lm < cooldown_min:
                    continue
        else:
            raise ValueError(loop)

        side = base.stack_side("15_5_1", i, a)
        if side == 0:
            continue
        if side_mode == "long" and side != 1:
            continue
        if side_mode == "short" and side != -1:
            continue
        if not base.entry_ok(entry, side, i, a):
            continue
        if structure and not srth.structure_ok(structure, side, i, a):
            continue
        if len(opens) >= max_lots:
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

        widx = win if win is not None else base.window_index(m)
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry_px,
            "sl": sl,
            "tp": tp_px,
            "sl_pts": use,
            "win": widx,
            "entry_min": m,
        })
        if loop in ("window_once", "window_loop") and widx is not None:
            fired.add(int(widx))
        counts[day] = counts.get(day, 0) + 1
        last_entry = (day, m)

    return trades


def run_scalp_leg(df: pd.DataFrame, sl: float = 8.0, tp: float = 12.0, max_hold: int = 90) -> List[scalp_bt.ScalpTrade]:
    df_n = scalp_bt.add_1m_indicators(scalp_bt.normalize_dt(df))
    df_5m = scalp_bt.add_5m_indicators(scalp_bt.resample_5m(df_n))
    variant = scalp_bt.VariantConfig(
        name="C1m", use_30s_trigger=False, no_chase=True, adaptive_trend=False
    )
    return scalp_bt.simulate_variant_trades(variant, df_n, df_5m, df_n, "MNQ", sl, tp, max_hold)


def eval_combo(name: str, trades: List, end_d: date) -> Dict[str, Any]:
    return {
        "name": name,
        "activity": weekday_stats(trades, end_d),
        "pnl": metrics(trades),
        "long": metrics([t for t in trades if getattr(t, "direction", None) == "long"]),
        "short": metrics([t for t in trades if getattr(t, "direction", None) == "short"]),
    }


def build_candidates() -> List[Dict[str, Any]]:
    """Hand-picked meaningful frequency-upgrade recipes."""
    rows: List[Dict[str, Any]] = []

    def stacked(name: str, long: Dict, short: Dict):
        rows.append({"name": name, "mode": "stacked", "long": long, "short": short})

    # --- Path 1: Continuous RTH pullback ---
    stacked(
        "RTH pullback full | L+S pullback cd0 tp1R ml2",
        {"loop": "rth_loop", "entry": "1m_pullback", "winset": None, "max_lots": 1, "max_day": 99},
        {"loop": "rth_loop", "entry": "1m_pullback", "winset": None, "max_lots": 2, "structure": "lh_ll", "max_day": 99},
    )
    stacked(
        "RTH pullback full | tp0.75R max_hold10",
        {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None,
            "max_lots": 1, "max_day": 99, "tp_key": "0.75R", "max_hold_min": 10,
        },
        {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None,
            "max_lots": 2, "structure": "lh_ll", "max_day": 99, "tp_key": "0.75R", "max_hold_min": 10,
        },
    )
    stacked(
        "RTH pullback full | tp0.5R max_hold8",
        {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None,
            "max_lots": 1, "max_day": 99, "tp_key": "0.5R", "max_hold_min": 8,
        },
        {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None,
            "max_lots": 2, "structure": "lh_ll", "max_day": 99, "tp_key": "0.5R", "max_hold_min": 8,
        },
    )
    stacked(
        "RTH pullback | cd3 tp1R",
        {"loop": "rth_loop", "entry": "1m_pullback", "cooldown_min": 3, "max_lots": 1, "max_day": 99},
        {"loop": "rth_loop", "entry": "1m_pullback", "cooldown_min": 3, "max_lots": 2, "structure": "lh_ll", "max_day": 99},
    )

    # --- Path 4: Stacked selective long + aggressive short ---
    stacked(
        "HYBRID v4 long w4 pullback + short RTH quiet x2",
        {"loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99},
        {"loop": "rth_loop", "entry": "quiet", "winset": None, "max_lots": 2, "max_day": 12, "structure": None},
    )
    stacked(
        "HYBRID v4 long + short RTH quiet x2 tp0.75R hold10",
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99,
        },
        {
            "loop": "rth_loop", "entry": "quiet", "winset": None, "max_lots": 2, "max_day": 12,
            "structure": None, "tp_key": "0.75R", "max_hold_min": 10,
        },
    )
    stacked(
        "HYBRID v4 long + short RTH pullback lh_ll x2",
        {"loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99},
        {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None, "max_lots": 2,
            "structure": "lh_ll", "max_day": 99,
        },
    )
    stacked(
        "HYBRID v4 long + short w6 window pullback x2",
        {"loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99},
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w6_all", "max_lots": 2,
            "structure": "lh_ll", "max_day": 99,
        },
    )

    # --- Path 2: Faster turnover on window loop ---
    stacked(
        "w6 window pullback x2 tp0.75R hold10",
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w6_all", "max_lots": 2,
            "max_day": 99, "tp_key": "0.75R", "max_hold_min": 10,
        },
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w6_all", "max_lots": 2,
            "structure": "lh_ll", "max_day": 99, "tp_key": "0.75R", "max_hold_min": 10,
        },
    )
    stacked(
        "w6 window pullback x2 tp0.5R hold8",
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w6_all", "max_lots": 2,
            "max_day": 99, "tp_key": "0.5R", "max_hold_min": 8,
        },
        {
            "loop": "window_loop", "entry": "1m_pullback", "winset": "w6_all", "max_lots": 2,
            "structure": "lh_ll", "max_day": 99, "tp_key": "0.5R", "max_hold_min": 8,
        },
    )

    # --- Path 3: + scalp second leg (on top of hybrid base) ---
    rows.append({
        "name": "HYBRID v4 long + short RTH quiet x2 + scalp C",
        "mode": "stacked_plus_scalp",
        "long": {"loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99},
        "short": {"loop": "rth_loop", "entry": "quiet", "winset": None, "max_lots": 2, "max_day": 12, "structure": None},
        "scalp": {"sl": 8, "tp": 12, "max_hold": 90},
    })
    rows.append({
        "name": "v4 long only + scalp C",
        "mode": "long_plus_scalp",
        "long": {"loop": "window_loop", "entry": "1m_pullback", "winset": "w4_no_noon", "max_lots": 1, "max_day": 99},
        "scalp": {"sl": 8, "tp": 12, "max_hold": 90},
    })
    rows.append({
        "name": "RTH pullback stacked + scalp C",
        "mode": "stacked_plus_scalp",
        "long": {"loop": "rth_loop", "entry": "1m_pullback", "winset": None, "max_lots": 1, "max_day": 99},
        "short": {
            "loop": "rth_loop", "entry": "1m_pullback", "winset": None, "max_lots": 2,
            "structure": "lh_ll", "max_day": 99, "tp_key": "0.75R", "max_hold_min": 10,
        },
        "scalp": {"sl": 8, "tp": 10, "max_hold": 60},
    })

    return rows


def run_candidate(a: Dict, df: pd.DataFrame, cfg: Dict) -> List:
    trades: List = []
    mode = cfg["mode"]

    def leg(side: str, spec: Dict) -> List:
        return simulate_leg(
            a,
            side_mode=side,
            loop=spec.get("loop", "window_loop"),
            entry=spec.get("entry", "1m_pullback"),
            winset_name=spec.get("winset"),
            tp_key=spec.get("tp_key", "1R"),
            cooldown_min=int(spec.get("cooldown_min", 0)),
            max_day=int(spec.get("max_day", 99)),
            max_lots=int(spec.get("max_lots", 1)),
            structure=spec.get("structure"),
            max_hold_min=spec.get("max_hold_min"),
        )

    if mode in ("stacked", "stacked_plus_scalp", "long_plus_scalp"):
        if "long" in cfg:
            trades.extend(leg("long", cfg["long"]))
        if "short" in cfg:
            trades.extend(leg("short", cfg["short"]))

    if mode in ("stacked_plus_scalp", "long_plus_scalp") and cfg.get("scalp"):
        sc = cfg["scalp"]
        scalp_trades = run_scalp_leg(
            df,
            sl=float(sc.get("sl", 8)),
            tp=float(sc.get("tp", 12)),
            max_hold=int(sc.get("max_hold", 90)),
        )
        for t in scalp_trades:
            t.version = "scalp"
        trades.extend(scalp_trades)

    return trades


def main() -> int:
    csv = os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "frequency_upgrade_search.json")
    df = official.load_1m(csv)
    a = srth.prepare(df)
    end_d = df["datetime"].iloc[-1].tz_convert(ET).date()

    candidates = build_candidates()
    results: List[Dict] = []

    print("FREQUENCY UPGRADE SEARCH (Jun 2026+)")
    print(f"Data: {len(df):,} bars -> {df['datetime'].iloc[-1]}")
    print(f"Candidates: {len(candidates)}\n")

    for i, cfg in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] {cfg['name']}...", flush=True)
        trades = run_candidate(a, df, cfg)
        row = eval_combo(cfg["name"], trades, end_d)
        row["mode"] = cfg["mode"]
        results.append(row)

    by_freq = sorted(results, key=lambda r: (-r["activity"]["avg_per_weekday"], -r["pnl"]["pf"]))
    profitable = [r for r in results if r["pnl"]["pf"] >= 1.0 and r["pnl"]["net"] > 0]
    hit6 = [r for r in results if r["activity"]["avg_per_weekday"] >= 6.0]
    hit6_profit = [r for r in hit6 if r["pnl"]["pf"] >= 1.0 and r["pnl"]["net"] > 0]

    print("\n" + "=" * 90)
    print("TOP BY FREQUENCY (avg trades/weekday)")
    print("=" * 90)
    for r in by_freq[:12]:
        act, pnl = r["activity"], r["pnl"]
        print(
            f"  avg={act['avg_per_weekday']:4.2f}/wd ge6={act['days_ge6']:2d}/{act['weekdays']} "
            f"n={pnl['trades']:4d} PF={pnl['pf']:4.2f} ${pnl['net']:7.0f} | {r['name'][:55]}"
        )

    print("\n" + "=" * 90)
    print("PROFITABLE (PF>=1.0, net>0) sorted by frequency")
    print("=" * 90)
    for r in sorted(profitable, key=lambda x: (-x["activity"]["avg_per_weekday"], -x["pnl"]["pf"]))[:10]:
        act, pnl = r["activity"], r["pnl"]
        print(
            f"  avg={act['avg_per_weekday']:4.2f}/wd ge6={act['days_ge6']:2d} "
            f"PF={pnl['pf']:4.2f} ${pnl['net']:7.0f} L=${r['long']['net']:5.0f} S=${r['short']['net']:5.0f} | {r['name']}"
        )

    print("\n" + "=" * 90)
    print(">=6 TRADES/WEEKDAY AND PROFITABLE")
    print("=" * 90)
    if not hit6_profit:
        print("  NONE — best profitable near target:")
        for r in sorted(profitable, key=lambda x: -x["activity"]["avg_per_weekday"])[:3]:
            act, pnl = r["activity"], r["pnl"]
            print(f"    avg={act['avg_per_weekday']:.2f}/wd PF={pnl['pf']:.2f} ${pnl['net']:.0f} | {r['name']}")
    else:
        for r in hit6_profit:
            act, pnl = r["activity"], r["pnl"]
            print(f"  avg={act['avg_per_weekday']:.2f}/wd PF={pnl['pf']:.2f} ${pnl['net']:.0f} | {r['name']}")

    best_freq = by_freq[0] if by_freq else None
    best_profit_freq = (
        sorted(profitable, key=lambda x: (-x["activity"]["avg_per_weekday"], -x["pnl"]["pf"]))[0]
        if profitable
        else None
    )

    payload = {
        "period": f"{JUNE.date()} through {end_d}",
        "candidates": len(candidates),
        "results": results,
        "best_frequency": best_freq,
        "best_profitable_frequency": best_profit_freq,
        "hit6_profitable": hit6_profit,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
