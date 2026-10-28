#!/usr/bin/env python3
"""FAIL-fast hunt: same official EMA15 entries, runner-preserving exits only."""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.test_ema15_tp_modes as tpmod
from src.strategy.mnq_15m_ema_eod import FLAT_MINUTE

ET = tpmod.ET
PT, COMM = official.PT, official.COMM
JUNE = pd.Timestamp("2026-06-01", tz="UTC")
JULY_END = pd.Timestamp("2026-08-01", tz="UTC")
AUG = pd.Timestamp("2026-08-01", tz="UTC")
SEP = pd.Timestamp("2026-09-01", tz="UTC")
SEP_END = pd.Timestamp("2026-09-22", tz="UTC")
MAX_VARIANTS = 20


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
            "median_pnl": 0.0,
            "eod": 0,
            "sl": 0,
            "tp": 0,
            "other": 0,
            "top5_share_of_wins": 0.0,
            "top5_pnl": 0.0,
            "rest_pnl": 0.0,
        }
    pnls = [t.pnl for t in trades]
    wins = sorted([p for p in pnls if p > 0], reverse=True)
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    net = sum(pnls)
    top5 = wins[:5]
    top5_pnl = sum(top5)
    share = (top5_pnl / gross_win) if gross_win > 0 else 0.0
    # drawdown
    eq = peak = 0.0
    dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    reasons = {"EOD": 0, "SL": 0, "TP": 0, "other": 0}
    for t in trades:
        r = getattr(t, "exit_reason", "") or ""
        if r in reasons:
            reasons[r] += 1
        else:
            reasons["other"] += 1
    wr = 100.0 * len(wins) / len(trades)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    return {
        "trades": len(trades),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "net_pnl": round(net, 2),
        "expectancy": round(net / len(trades), 2),
        "max_drawdown": round(dd, 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "eod": reasons["EOD"],
        "sl": reasons["SL"],
        "tp": reasons["TP"],
        "other": reasons["other"],
        "top5_share_of_wins": round(share, 3),
        "top5_pnl": round(top5_pnl, 2),
        "rest_pnl": round(net - top5_pnl, 2),
    }


def make_trade(p, ts, exit_px, reason, version, lot_frac=1.0):
    pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
    return scalp_bt.ScalpTrade(
        entry_time=p["ts"],
        direction="long" if p["dir"] == 1 else "short",
        entry_price=p["entry"],
        sl=p["sl"],
        tp=p.get("tp", p["entry"]),
        exit_time=ts,
        exit_price=exit_px,
        pnl=(pts * PT - COMM) * lot_frac,
        exit_reason=reason,
        version=version,
        hold_seconds=(ts - p["ts"]).total_seconds(),
    )


def simulate_exit(df: pd.DataFrame, entries: list, spec: Dict[str, Any]) -> List:
    """Replay same fills; exit policy from spec."""
    mode = spec["mode"]
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()

    pending = []
    for e in entries:
        pend = deepcopy(e)
        pend["tp"] = e["entry"] + 500 if e["dir"] == 1 else e["entry"] - 500
        pend["peak"] = e["entry"]
        pend["armed"] = False
        pend["be"] = False
        pend["partial_done"] = False
        pend["lot_left"] = 1.0
        pend["entry_i"] = e["i"]
        pend["sl0"] = e["sl"]
        pend["r"] = float(e["sl_pts"])
        pending.append(pend)
    by_i: Dict[int, list] = {}
    for p in pending:
        by_i.setdefault(p["i"], []).append(p)

    trades: List = []
    opens: List = []

    for i in range(80, len(df)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        still = []
        for p in opens:
            d = p["dir"]
            # update peak
            if d == 1:
                p["peak"] = max(p["peak"], float(hi[i]))
                fav = p["peak"] - p["entry"]
            else:
                p["peak"] = min(p["peak"], float(lo[i]))
                fav = p["entry"] - p["peak"]
            r = max(p["r"], 1e-9)

            # arm / BE / trail updates before exit checks
            if mode == "hold":
                pass
            elif mode == "be_1r":
                if (not p["be"]) and fav >= r:
                    p["sl"] = p["entry"]  # break-even (ignore tiny buffer)
                    p["be"] = True
            elif mode.startswith("trail_"):
                # trail_1r_1.0atr / trail_1r_1.5atr / trail_0.5r_2.0atr
                parts = mode.split("_")
                # trail_<arm>_<mult>atr  e.g. trail_1r_1.0atr
                arm_s = parts[1]  # 1r or 0.5r
                mult = float(parts[2].replace("atr", ""))
                arm = 1.0 * r if arm_s.startswith("1") else 0.5 * r
                if fav >= arm:
                    p["armed"] = True
                if p["armed"]:
                    trail = mult * r
                    if d == 1:
                        p["sl"] = max(p["sl"], p["peak"] - trail)
                    else:
                        p["sl"] = min(p["sl"], p["peak"] + trail)
            elif mode.startswith("time_"):
                # time_60 / time_120 — exit if not +0.5R by N minutes from entry bar
                nmin = int(mode.split("_")[1])
                need = 0.5 * r
                held_min = (i - p["entry_i"])
                if held_min >= nmin and fav < need and m < FLAT_MINUTE:
                    trades.append(make_trade(p, ts, float(cl[i]), "TP", mode, p["lot_left"]))
                    continue
            elif mode == "partial_1r_eod":
                if (not p["partial_done"]) and fav >= r:
                    # take half at 1R price
                    tp_px = p["entry"] + r if d == 1 else p["entry"] - r
                    # if price already past, fill at 1R
                    trades.append(make_trade(p, ts, tp_px, "TP", mode, 0.5))
                    p["partial_done"] = True
                    p["lot_left"] = 0.5
                    # move stop to BE on remainder
                    p["sl"] = p["entry"]

            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif d == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    # official hold uses a 500pt runaway cap; keep it for fair compare
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]

            if reason:
                trades.append(make_trade(p, ts, float(exit_px), reason, mode, p["lot_left"]))
            else:
                still.append(p)
        opens = still
        for p in by_i.get(i, []):
            opens.append(p)
    return trades


def build_specs() -> List[Dict[str, Any]]:
    specs = [
        {"name": "hold", "mode": "hold"},
        {"name": "be_after_1r", "mode": "be_1r"},
        {"name": "trail_1r_1.0atr", "mode": "trail_1r_1.0atr"},
        {"name": "trail_1r_1.5atr", "mode": "trail_1r_1.5atr"},
        {"name": "trail_0.5r_2.0atr", "mode": "trail_0.5r_2.0atr"},
        {"name": "partial_1r_rest_eod", "mode": "partial_1r_eod"},
        {"name": "time_60_need_0.5r", "mode": "time_60"},
        {"name": "time_120_need_0.5r", "mode": "time_120"},
    ]
    return specs[:MAX_VARIANTS]


def gate_aug(m):
    return m["profit_factor"] >= 1.0 and m["net_pnl"] >= 0


def gate_sep(m):
    return m["profit_factor"] >= 1.0 and m["net_pnl"] > 0


def gate_full(m):
    return m["net_pnl"] >= 1000


def fragile(m):
    return m["top5_share_of_wins"] > 0.70


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "runners_exit_hunt.json")
    print("RUNNERS EXIT HUNT (same official entries; FAIL-fast <=20)")
    df = official.load_1m(csv)
    print(f"bars={len(df):,} {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    print("collecting official entries...")
    entries = tpmod.collect_official_entries(df)
    print(f"fills={len(entries)}")
    specs = build_specs()
    results = []
    for i, spec in enumerate(specs):
        trades = simulate_exit(df, entries, spec)
        jj = metrics(in_range(trades, JUNE, JULY_END))
        aug = metrics(in_range(trades, AUG, SEP))
        sep = metrics(in_range(trades, SEP, SEP_END))
        full = metrics(in_range(trades, JUNE, None))
        pre = metrics(in_range(trades, pd.Timestamp("2024-09-22", tz="UTC"), JUNE))
        row = {
            "spec": spec,
            "jun_jul": jj,
            "aug": aug,
            "sep": sep,
            "jun_latest": full,
            "pre_jun": pre,
            "gate_aug": gate_aug(aug),
            "gate_sep": gate_sep(sep),
            "gate_full": gate_full(full),
            "fragile": fragile(full),
            "train_net": jj["net_pnl"],
            "train_pf": jj["profit_factor"],
        }
        results.append(row)
        print(
            f"[{i+1}/{len(specs)}] {spec['name']}: "
            f"JJ ${jj['net_pnl']:.0f} PF{jj['profit_factor']:.2f} | "
            f"Aug ${aug['net_pnl']:.0f} PF{aug['profit_factor']:.2f} "
            f"({'Y' if row['gate_aug'] else 'N'}) | "
            f"Sep ${sep['net_pnl']:.0f} PF{sep['profit_factor']:.2f} "
            f"({'Y' if row['gate_sep'] else 'N'}) | "
            f"Full ${full['net_pnl']:.0f} top5={full['top5_share_of_wins']:.0%} "
            f"med={full['median_pnl']:.0f}"
        )

    cleared = [r for r in results if r["gate_aug"] and r["gate_sep"]]
    cleared.sort(key=lambda r: (r["train_net"], r["train_pf"]), reverse=True)
    status = "FAIL"
    winner = None
    if cleared:
        full_ok = [r for r in cleared if r["gate_full"] and not r["fragile"]]
        if full_ok:
            winner = full_ok[0]
            status = "PASS"
        else:
            # cleared months but fragile or under $1000
            winner = cleared[0]
            status = "FRAGILE" if winner["fragile"] or not winner["gate_full"] else "PASS"

    # near miss: best train among not clearing
    scored = sorted(results, key=lambda r: (r["train_net"], r["train_pf"]), reverse=True)
    near = None
    for r in scored:
        if not (r["gate_aug"] and r["gate_sep"]):
            why = []
            if not r["gate_aug"]:
                why.append(f"Aug FAIL {r['aug']}")
            if not r["gate_sep"]:
                why.append(f"Sep FAIL {r['sep']}")
            near = {"spec": r["spec"], "failed": why, "jun_jul": r["jun_jul"], "jun_latest": r["jun_latest"]}
            break

    payload = {
        "status": status,
        "variants_tested": len(results),
        "note": (
            "No runner-exit cleared Aug+Sep after first screen."
            if status == "FAIL"
            else (
                "Cleared Aug+Sep but fragile concentration or under $1000 — do not lock."
                if status == "FRAGILE"
                else "Cleared gates without extreme top-5 concentration."
            )
        ),
        "winner": winner,
        "near_miss": near,
        "hold_baseline": next((r for r in results if r["spec"]["name"] == "hold"), None),
        "all": results,
        "gates": {
            "aug": "PF>=1 and net>=0",
            "sep": "PF>=1 and net>0",
            "full": "net>=1000",
            "fragile": "top5_share_of_wins>0.70",
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {out}")
    print(f"STATUS={status}")
    if winner:
        print("WINNER", winner["spec"], "full", winner["jun_latest"])
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
