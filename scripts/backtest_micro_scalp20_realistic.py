#!/usr/bin/env python3
"""Realistic costs + small improve hunt using the proven per-bar Timestamp loop."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.strategy.micro_scalp_20 as ms
from src.strategy.micro_scalp_20 import OpenPos, ScalpState

COMM = 1.24
PT = 2.0
MAX_VARIANTS = 18  # keep runtime sane on full 1s RTH


@dataclass
class Params:
    name: str
    min_hold: float = 10.0
    tp_usd: float = 20.0
    sl_pts: float = 10.0
    lookback: float = 8.0
    impulse: float = 3.0
    cooldown: float = 15.0
    max_hold: float = 180.0
    slip_rt_pts: float = 0.5
    next_bar_fill: bool = True
    ema_fast: int = 5
    ema_slow: int = 12


def metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0,
            "expectancy": 0.0, "max_drawdown": 0.0, "median_hold_sec": 0.0,
            "tp": 0, "sl": 0, "max_hold": 0, "tprd": 0.0, "active_days": 0,
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
    days = {t["entry_time"][:10] for t in trades}
    return {
        "trades": len(trades),
        "win_rate": round(100.0 * len(wins) / len(trades), 1),
        "profit_factor": round((gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0), 2),
        "net_pnl": round(sum(pnls), 2),
        "expectancy": round(sum(pnls) / len(trades), 2),
        "max_drawdown": round(dd, 2),
        "median_hold_sec": round(float(np.median(holds)), 1),
        "tp": sum(1 for t in trades if t["reason"] == "TP"),
        "sl": sum(1 for t in trades if t["reason"] == "SL"),
        "max_hold": sum(1 for t in trades if t["reason"] == "MAX_HOLD"),
        "tprd": round(len(trades) / max(len(days), 1), 2),
        "active_days": len(days),
    }


def apply_params(p: Params) -> None:
    ms.MIN_HOLD_SEC = p.min_hold
    ms.TP_USD = p.tp_usd
    ms.TP_PTS = p.tp_usd / PT
    ms.SL_PTS = p.sl_pts
    ms.LOOKBACK_SEC = p.lookback
    ms.IMPULSE_PTS = p.impulse
    ms.COOLDOWN_SEC = p.cooldown
    ms.MAX_HOLD_SEC = p.max_hold
    ms.EMA_FAST = p.ema_fast
    ms.EMA_SLOW = p.ema_slow
    ms.MNQ_POINT_VALUE = PT


def entry_signal(state: ScalpState, unix: float) -> Optional[str]:
    if state.open is not None or unix < state.cooldown_until:
        return None
    if state.buf.last() is None:
        return None
    impulse = state.buf.impulse_pts(ms.LOOKBACK_SEC)
    fast, slow = state.buf.ema_pair(ms.EMA_FAST, ms.EMA_SLOW)
    if fast is None or slow is None:
        return None
    if impulse >= ms.IMPULSE_PTS and fast > slow:
        return "long"
    if impulse <= -ms.IMPULSE_PTS and fast < slow:
        return "short"
    return None


def exit_signal(pos: OpenPos, px: float, unix: float):
    hold = unix - pos.entry_ts
    if pos.direction == "long" and px <= pos.sl:
        return "SL", pos.sl
    if pos.direction == "short" and px >= pos.sl:
        return "SL", pos.sl
    if hold >= ms.MAX_HOLD_SEC:
        return "MAX_HOLD", px
    if hold >= ms.MIN_HOLD_SEC:
        u = (px - pos.entry) * PT if pos.direction == "long" else (pos.entry - px) * PT
        if u >= ms.TP_USD:
            return "TP", px
    return None


def simulate_csv(csv_path: str, p: Params) -> List[Dict[str, Any]]:
    apply_params(p)
    state = ScalpState()
    trades: List[Dict[str, Any]] = []
    pending: Optional[str] = None
    half = p.slip_rt_pts / 2.0
    n = 0

    for chunk in pd.read_csv(csv_path, usecols=["datetime", "close"], chunksize=400_000):
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
        et = chunk["datetime"].dt.tz_convert("US/Eastern")
        mins = et.dt.hour * 60 + et.dt.minute
        mask = (et.dt.weekday < 5) & (mins >= 9 * 60 + 35) & (mins < 15 * 60 + 50)
        chunk = chunk.loc[mask]
        for ts, close in zip(chunk["datetime"].to_numpy(), chunk["close"].to_numpy()):
            n += 1
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            unix = float(ts.timestamp())
            px = float(close)
            if not (px > 0):
                continue
            state.buf.push(px, ts=unix)

            if pending is not None and state.open is None:
                d = pending
                fill = px + half if d == "long" else px - half
                sl = fill - p.sl_pts if d == "long" else fill + p.sl_pts
                state.open = OpenPos(d, fill, sl, fill, unix)
                pending = None

            if state.open is not None:
                hit = exit_signal(state.open, px, unix)
                if hit:
                    reason, exit_px = hit
                    pos = state.open
                    hold = unix - pos.entry_ts
                    if reason == "SL":
                        exit_px = pos.sl
                    else:
                        exit_px = exit_px - half if pos.direction == "long" else exit_px + half
                    pts = (exit_px - pos.entry) if pos.direction == "long" else (pos.entry - exit_px)
                    trades.append({
                        "entry_time": pd.Timestamp(pos.entry_ts, unit="s", tz="UTC").isoformat(),
                        "direction": pos.direction,
                        "hold_sec": round(hold, 2),
                        "reason": reason,
                        "pnl": round(pts * PT - COMM, 2),
                    })
                    state.open = None
                    state.cooldown_until = unix + ms.COOLDOWN_SEC
                continue

            if pending is not None:
                continue
            d = entry_signal(state, unix)
            if not d:
                continue
            if p.next_bar_fill:
                pending = d
            else:
                fill = px + half if d == "long" else px - half
                sl = fill - p.sl_pts if d == "long" else fill + p.sl_pts
                state.open = OpenPos(d, fill, sl, fill, unix)

    print(f"  {p.name}: bars~{n:,} trades={len(trades)}", flush=True)
    return trades


def window(trades, start, end=None):
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


def build_grid() -> List[Params]:
    out: List[Params] = []

    def add(p: Params):
        if len(out) >= MAX_VARIANTS:
            return
        out.append(p)

    add(Params("soft_no_slip_samebar", slip_rt_pts=0.0, next_bar_fill=False))
    add(Params("realistic_0.5_nextbar", slip_rt_pts=0.5, next_bar_fill=True))
    add(Params("harsh_1.0_nextbar", slip_rt_pts=1.0, next_bar_fill=True))
    # tighter / better candidates under realistic costs
    for impulse, lookback, sl, cooldown, tp in [
        (3.0, 8.0, 10.0, 15.0, 20.0),
        (4.0, 8.0, 10.0, 30.0, 20.0),
        (5.0, 8.0, 10.0, 30.0, 20.0),
        (5.0, 12.0, 10.0, 30.0, 20.0),
        (6.0, 12.0, 10.0, 60.0, 20.0),
        (4.0, 8.0, 8.0, 30.0, 20.0),
        (4.0, 8.0, 12.0, 30.0, 20.0),
        (5.0, 8.0, 10.0, 30.0, 24.0),
        (5.0, 12.0, 12.0, 60.0, 24.0),
        (6.0, 12.0, 12.0, 60.0, 24.0),
        (4.0, 12.0, 10.0, 45.0, 20.0),
        (5.0, 10.0, 10.0, 45.0, 20.0),
        (3.0, 12.0, 10.0, 30.0, 20.0),
        (6.0, 8.0, 10.0, 30.0, 20.0),
        (5.0, 8.0, 8.0, 60.0, 20.0),
    ]:
        add(Params(
            name=f"imp{impulse}_lb{lookback}_sl{sl}_cd{cooldown}_tp{tp}",
            impulse=impulse, lookback=lookback, sl_pts=sl,
            cooldown=cooldown, tp_usd=tp,
            slip_rt_pts=0.5, next_bar_fill=True,
        ))
    return out[:MAX_VARIANTS]


def gate_train(m) -> bool:
    return m["trades"] >= 100 and m["profit_factor"] >= 1.1 and m["net_pnl"] > 0 and m["expectancy"] >= 0.25


def gate_aug(m) -> bool:
    return m["trades"] >= 30 and m["profit_factor"] >= 1.05 and m["net_pnl"] > 0 and m["expectancy"] > 0


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1s_databento.csv")
    grid = build_grid()
    print(f"variants={len(grid)} file={csv}", flush=True)
    rows = []
    for i, p in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] {p.name}", flush=True)
        trades = simulate_csv(csv, p)
        jj = metrics(window(trades, "2026-06-01", "2026-08-01"))
        aug = metrics(window(trades, "2026-08-01", "2026-09-01"))
        full = metrics(window(trades, "2026-06-01"))
        row = {
            "name": p.name,
            "params": p.__dict__,
            "jun_jul": jj,
            "aug": aug,
            "full": full,
            "gate_train": gate_train(jj),
            "gate_aug": gate_aug(aug),
            "train_exp": jj["expectancy"],
            "train_net": jj["net_pnl"],
        }
        rows.append(row)
        print(
            f"  JJ n={jj['trades']} PF={jj['profit_factor']} ${jj['net_pnl']} exp={jj['expectancy']} | "
            f"Aug n={aug['trades']} PF={aug['profit_factor']} ${aug['net_pnl']} "
            f"({'Y' if row['gate_aug'] else 'N'}) | Full ${full['net_pnl']}",
            flush=True,
        )

    soft = next(r for r in rows if r["name"] == "soft_no_slip_samebar")
    realistic = next(r for r in rows if r["name"] == "realistic_0.5_nextbar")
    harsh = next(r for r in rows if r["name"] == "harsh_1.0_nextbar")
    cleared = [r for r in rows if r["params"]["slip_rt_pts"] > 0 and r["gate_train"] and r["gate_aug"]]
    cleared.sort(key=lambda r: (r["train_exp"], r["train_net"]), reverse=True)
    status = "PASS" if cleared else "FAIL"
    winner = cleared[0] if cleared else None
    realistic_rows = [r for r in rows if r["params"]["slip_rt_pts"] > 0]
    realistic_rows.sort(key=lambda r: (r["gate_aug"], r["gate_train"], r["train_exp"], r["train_net"]), reverse=True)
    near = None if winner else realistic_rows[0]

    payload = {
        "status": status,
        "soft_baseline": soft,
        "realistic_baseline": realistic,
        "harsh_baseline": harsh,
        "winner": winner,
        "near_miss": near,
        "variants_tested": len(rows),
        "all": rows,
        "note": "Commission $1.24; realistic=0.5pt RT slip + next 1s fill; harsh=1.0pt RT.",
    }
    out = os.path.join("data", "micro_scalp20_realistic_hunt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n=== REALISTIC RETEST ===")
    print("SOFT:", soft["full"])
    print("REALISTIC 0.5:", realistic["full"], "Aug", realistic["aug"])
    print("HARSH 1.0:", harsh["full"])
    print("STATUS", status)
    if winner:
        print("WINNER", winner["name"], winner["params"])
        print("JJ", winner["jun_jul"], "Aug", winner["aug"], "Full", winner["full"])
        rec = {
            "status": "recommended_paper",
            "from_hunt": winner["name"],
            "MICRO_SCALP20_MIN_HOLD_SEC": winner["params"]["min_hold"],
            "MICRO_SCALP20_TP_USD": winner["params"]["tp_usd"],
            "MICRO_SCALP20_SL_PTS": winner["params"]["sl_pts"],
            "MICRO_SCALP20_LOOKBACK_SEC": winner["params"]["lookback"],
            "MICRO_SCALP20_IMPULSE_PTS": winner["params"]["impulse"],
            "MICRO_SCALP20_COOLDOWN_SEC": winner["params"]["cooldown"],
            "assumed_slip_rt_pts": winner["params"]["slip_rt_pts"],
            "metrics": {"jun_jul": winner["jun_jul"], "aug": winner["aug"], "full": winner["full"]},
        }
        with open(os.path.join("data", "micro_scalp20_recommended.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        with open(os.path.join("data", "micro_scalp20_paper.env"), "w", encoding="utf-8") as f:
            for k, v in rec.items():
                if str(k).startswith("MICRO_"):
                    f.write(f"{k}={v}\n")
        print("Wrote data/micro_scalp20_recommended.json and paper.env")
    else:
        print("NEAR", near["name"], near["full"], "Aug", near["aug"])
    print("Saved", out)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
