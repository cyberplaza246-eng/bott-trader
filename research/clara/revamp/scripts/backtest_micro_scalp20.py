#!/usr/bin/env python3
"""Backtest micro_scalp_20 on 1-second MNQ closes (real 10s min-hold)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.strategy.micro_scalp_20 import (
    COOLDOWN_SEC,
    MAX_HOLD_SEC,
    MIN_HOLD_SEC,
    MNQ_POINT_VALUE,
    OpenPos,
    SL_PTS,
    ScalpState,
    TP_USD,
    check_entry,
    check_exit,
    on_closed,
    unrealized_usd,
)

# Match other MNQ paper backtests (approx round-turn)
COMM = float(os.getenv("MICRO_SCALP20_COMM", "1.24"))
PT = MNQ_POINT_VALUE


def metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_sec": 0.0,
            "median_hold_sec": 0.0,
            "tp": 0,
            "sl": 0,
            "max_hold": 0,
            "holds_under_10s": 0,
        }
    pnls = [t["pnl"] for t in trades]
    holds = [t["hold_sec"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    reasons = Counter(t["reason"] for t in trades)
    return {
        "trades": len(trades),
        "win_rate": round(100.0 * len(wins) / len(trades), 1),
        "profit_factor": round((gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0), 2),
        "net_pnl": round(sum(pnls), 2),
        "expectancy": round(sum(pnls) / len(trades), 2),
        "max_drawdown": round(dd, 2),
        "avg_hold_sec": round(float(np.mean(holds)), 1),
        "median_hold_sec": round(float(np.median(holds)), 1),
        "tp": int(reasons.get("TP", 0)),
        "sl": int(reasons.get("SL", 0)),
        "max_hold": int(reasons.get("MAX_HOLD", 0)),
        "holds_under_10s": int(sum(1 for h in holds if h < MIN_HOLD_SEC - 1e-6)),
    }


def run_backtest(csv_path: str, *, rth_only: bool = False) -> Dict[str, Any]:
    print(f"Loading {csv_path} ...", flush=True)
    # stream in chunks to keep memory reasonable
    usecols = ["datetime", "close"]
    state = ScalpState()
    trades: List[Dict[str, Any]] = []
    n_bars = 0
    # epoch seconds relative: use pandas timestamps as float unix
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=500_000):
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
        if rth_only:
            et = chunk["datetime"].dt.tz_convert("US/Eastern")
            mins = et.dt.hour * 60 + et.dt.minute
            # keep Globex too if not rth_only; for rth filter weekdays 9:35-15:50
            mask = (et.dt.weekday < 5) & (mins >= 9 * 60 + 35) & (mins < 15 * 60 + 50)
            chunk = chunk.loc[mask]
        for ts, close in zip(chunk["datetime"].to_numpy(), chunk["close"].to_numpy()):
            n_bars += 1
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            unix = float(ts.timestamp())
            px = float(close)
            if px <= 0 or px != px:
                continue
            state.buf.push(px, ts=unix)

            if state.open is not None:
                hit = check_exit(state.open, px, now=unix)
                if hit:
                    reason, exit_px = hit
                    pos = state.open
                    hold = unix - pos.entry_ts
                    pts = (exit_px - pos.entry) if pos.direction == "long" else (pos.entry - exit_px)
                    # For SL use stop price; for TP/MAX use bar close
                    if reason == "SL":
                        pts = (pos.sl - pos.entry) if pos.direction == "long" else (pos.entry - pos.sl)
                        exit_px = pos.sl
                    pnl = pts * PT - COMM
                    trades.append(
                        {
                            "entry_time": pd.Timestamp(pos.entry_ts, unit="s", tz="UTC").isoformat(),
                            "exit_time": ts.isoformat(),
                            "direction": pos.direction,
                            "entry": pos.entry,
                            "exit": float(exit_px),
                            "hold_sec": round(hold, 2),
                            "reason": reason,
                            "pnl": round(pnl, 2),
                        }
                    )
                    on_closed(state, reason, now=unix)
                continue

            # cooldown handled inside check_entry via state.cooldown_until
            sig = check_entry(state, now=unix)
            if not sig:
                continue
            # fill at this close
            fill = px
            sl = sig["sl"] if sig["direction"] == "long" else sig["sl"]
            # recompute brackets from fill
            if sig["direction"] == "long":
                sl = fill - SL_PTS
                tp_far = fill + 500.0
            else:
                sl = fill + SL_PTS
                tp_far = fill - 500.0
            state.open = OpenPos(
                direction=sig["direction"],
                entry=fill,
                sl=sl,
                tp_far=tp_far,
                entry_ts=unix,
            )
            # consume cooldown only on close; entry clears nothing

        print(f"  processed ~{n_bars:,} bars, trades so far {len(trades)}", flush=True)

    # force flat at end
    if state.open is not None:
        pos = state.open
        # use last pushed
        last = state.buf.last() or pos.entry
        hold = (state.buf.ticks[-1][0] if state.buf.ticks else pos.entry_ts) - pos.entry_ts
        pts = (last - pos.entry) if pos.direction == "long" else (pos.entry - last)
        trades.append(
            {
                "entry_time": pd.Timestamp(pos.entry_ts, unit="s", tz="UTC").isoformat(),
                "exit_time": "EOD_DATA",
                "direction": pos.direction,
                "entry": pos.entry,
                "exit": last,
                "hold_sec": round(hold, 2),
                "reason": "DATA_END",
                "pnl": round(pts * PT - COMM, 2),
            }
        )
        on_closed(state, "DATA_END")

    def window(start: str, end: Optional[str] = None):
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC") if end else None
        out = []
        for t in trades:
            et = pd.Timestamp(t["entry_time"])
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            if et < s:
                continue
            if e is not None and et >= e:
                continue
            out.append(t)
        return out

    rep = {
        "spec": "micro_scalp_20",
        "data": csv_path,
        "bars": n_bars,
        "params": {
            "min_hold_sec": MIN_HOLD_SEC,
            "tp_usd": TP_USD,
            "sl_pts": SL_PTS,
            "comm": COMM,
            "rth_only": rth_only,
            "max_hold_sec": MAX_HOLD_SEC,
            "cooldown_sec": COOLDOWN_SEC,
        },
        "all": metrics(trades),
        "jun_jul": metrics(window("2026-06-01", "2026-08-01")),
        "aug": metrics(window("2026-08-01", "2026-09-01")),
        "jun_through_data_end": metrics(window("2026-06-01")),
        "note": "1s close proxy for quote stream; Sep 2026 not in this 1s file (ends ~2026-08-16).",
    }
    # TP hold sanity
    tp_trades = [t for t in trades if t["reason"] == "TP"]
    rep["tp_hold_min"] = round(min((t["hold_sec"] for t in tp_trades), default=0.0), 2)
    rep["tp_hold_ok"] = all(t["hold_sec"] + 1e-6 >= MIN_HOLD_SEC for t in tp_trades)
    return rep


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1s_databento.csv")
    rth = str(os.getenv("MICRO_SCALP20_BT_RTH", "true")).lower() in ("1", "true", "yes")
    rep = run_backtest(csv, rth_only=rth)
    out = os.path.join("data", "micro_scalp20_backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("\n=== MICRO SCALP $20 BACKTEST (1s) ===")
    print(f"bars={rep['bars']:,} rth_only={rep['params']['rth_only']}")
    for label in ("jun_jul", "aug", "jun_through_data_end", "all"):
        m = rep[label]
        print(
            f"{label}: n={m['trades']} WR={m['win_rate']}% PF={m['profit_factor']} "
            f"Net=${m['net_pnl']} Exp=${m['expectancy']} DD=${m['max_drawdown']} "
            f"hold_med={m['median_hold_sec']}s TP={m['tp']} SL={m['sl']} "
            f"under10s={m['holds_under_10s']}"
        )
    print(f"tp_hold_min={rep['tp_hold_min']} tp_hold_ok={rep['tp_hold_ok']}")
    print("Saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
