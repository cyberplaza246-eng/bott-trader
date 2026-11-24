#!/usr/bin/env python3
"""1s impulse entry + 5m/15m EMA bias filter (from 1m bars).

Keeps current micro_scalp_20 entry (impulse + EMA5/12) and only trades
with the higher-TF direction.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.strategy.micro_scalp_20 as ms
from src.strategy.day_hybrid import prepare_arrays
from src.strategy.micro_scalp_20 import OpenPos, ScalpState, on_closed

COMM = 1.24
PT = 2.0
SLIP = 0.25


def metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "exp": 0.0, "dd": 0.0}
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
    }


def bias_map(df_1m: pd.DataFrame) -> Dict[pd.Timestamp, Tuple[int, int, int]]:
    a = prepare_arrays(df_1m)
    out: Dict[pd.Timestamp, Tuple[int, int, int]] = {}
    for i in range(len(a["dt"])):
        ts = pd.Timestamp(a["dt"][i]).tz_convert("UTC").floor("min")
        out[ts] = (int(a["t15"][i]), int(a["t5"][i]), int(a["t1"][i]))
    return out


def allow(mode: str, d: int, t15: int, t5: int, t1: int) -> bool:
    if mode == "none":
        return True
    if mode == "t5":
        return t5 == d
    if mode == "t15":
        return t15 == d
    if mode == "t15_t5":
        return t15 == d and t5 == d
    if mode == "stack151":
        return t15 == d and t5 == d and t1 == d
    if mode == "t15_ctx_t5":
        # 15m sets regime; 5m is the trade direction (must agree with 15 or be same as d)
        return t15 != 0 and t5 == d and (t15 == d or abs(t15) == 1)
    return False


def run(
    csv_1s: str,
    bias: Dict[pd.Timestamp, Tuple[int, int, int]],
    *,
    mode: str,
    impulse: float,
    lookback: float,
    sl_pts: float,
    tp_usd: float,
    max_hold: float,
    cooldown: float,
) -> List[Dict[str, Any]]:
    ms.IMPULSE_PTS = impulse
    ms.LOOKBACK_SEC = lookback
    ms.SL_PTS = sl_pts
    ms.TP_USD = tp_usd
    ms.TP_PTS = tp_usd / PT
    ms.MAX_HOLD_SEC = max_hold
    ms.COOLDOWN_SEC = cooldown
    ms.MIN_HOLD_SEC = 10.0
    ms.EMA_FAST, ms.EMA_SLOW = 5, 12

    state = ScalpState()
    trades: List[Dict[str, Any]] = []
    for chunk in pd.read_csv(csv_1s, usecols=["datetime", "close"], chunksize=400_000):
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
        et = chunk["datetime"].dt.tz_convert("US/Eastern")
        mins = et.dt.hour * 60 + et.dt.minute
        mask = (et.dt.weekday < 5) & (mins >= 9 * 60 + 35) & (mins < 15 * 60 + 50)
        chunk = chunk.loc[mask]
        for ts, close in zip(chunk["datetime"].to_numpy(), chunk["close"].to_numpy()):
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            unix = float(ts.timestamp())
            px = float(close)
            if px <= 0 or px != px:
                continue
            state.buf.push(px, ts=unix)
            if state.open is not None:
                hit = ms.check_exit(state.open, px, now=unix)
                if hit:
                    reason, exit_px = hit
                    pos = state.open
                    if reason == "SL":
                        exit_px = pos.sl
                    fe = pos.entry + SLIP if pos.direction == "long" else pos.entry - SLIP
                    fx = exit_px - SLIP if pos.direction == "long" else exit_px + SLIP
                    pts = (fx - fe) if pos.direction == "long" else (fe - fx)
                    trades.append({
                        "day": ts.tz_convert("US/Eastern").strftime("%Y-%m-%d"),
                        "reason": reason,
                        "pnl": round(pts * PT - COMM, 2),
                    })
                    on_closed(state, reason, now=unix)
                continue

            sig = ms.check_entry(state, now=unix)
            if not sig:
                continue
            minute = ts.floor("min")
            t15, t5, t1 = bias.get(minute, (0, 0, 0))
            d = 1 if sig["direction"] == "long" else -1
            if not allow(mode, d, t15, t5, t1):
                continue
            fill = px
            if d > 0:
                sl = fill - sl_pts
                tp = fill + tp_usd / PT
            else:
                sl = fill + sl_pts
                tp = fill - tp_usd / PT
            state.open = OpenPos(
                direction=sig["direction"], entry=fill, sl=sl, tp_far=tp, entry_ts=unix,
            )
    return trades


def win(trades, start, end=None):
    return [t for t in trades if t["day"] >= start and (end is None or t["day"] < end)]


def main() -> int:
    path_1s = os.path.join(ROOT, "data", "MNQ_1s_databento.csv")
    path_1m = os.path.join(ROOT, "data", "MNQ_1m.csv")
    print("Building 15m/5m/1m bias map from 1m ...", flush=True)
    bias = bias_map(pd.read_csv(path_1m))
    print(f"bias minutes={len(bias):,}", flush=True)

    # (name, mode, impulse, lookback, sl, tp_usd, max_hold, cooldown)
    variants = [
        ("raw_live_defaults", "none", 2.5, 25.0, 15.0, 20.0, 180.0, 15.0),
        ("raw_tight_impulse", "none", 3.0, 8.0, 10.0, 20.0, 180.0, 15.0),
        ("t5_dir_live", "t5", 2.5, 25.0, 15.0, 20.0, 180.0, 15.0),
        ("t5_dir_imp3", "t5", 3.0, 8.0, 10.0, 20.0, 180.0, 15.0),
        ("t15_ctx_t5_live", "t15_ctx_t5", 2.5, 25.0, 15.0, 20.0, 180.0, 15.0),
        ("t15_t5_live", "t15_t5", 2.5, 25.0, 15.0, 20.0, 180.0, 15.0),
        ("t15_t5_imp3", "t15_t5", 3.0, 8.0, 10.0, 20.0, 180.0, 20.0),
        ("t15_t5_sl8_tp12", "t15_t5", 3.0, 8.0, 8.0, 24.0, 120.0, 30.0),
        ("t5_sl8_tp12", "t5", 3.0, 10.0, 8.0, 24.0, 120.0, 30.0),
        ("stack151_imp3", "stack151", 3.0, 8.0, 10.0, 20.0, 180.0, 30.0),
        ("t15_only_live", "t15", 2.5, 25.0, 15.0, 20.0, 180.0, 15.0),
    ]

    rows = []
    print(f"variants={len(variants)}", flush=True)
    for name, mode, imp, lb, sl, tp, mh, cd in variants:
        print(f"\n=== {name} mode={mode} ===", flush=True)
        trades = run(
            path_1s, bias, mode=mode,
            impulse=imp, lookback=lb, sl_pts=sl, tp_usd=tp,
            max_hold=mh, cooldown=cd,
        )
        jj = metrics(win(trades, "2026-06-01", "2026-08-01"))
        aug = metrics(win(trades, "2026-08-01", "2026-09-01"))
        full = metrics(win(trades, "2026-06-01"))
        ok = (
            jj["n"] >= 40 and jj["pf"] >= 1.15 and jj["net"] > 0 and jj["exp"] >= 0.5
            and aug["n"] >= 20 and aug["pf"] >= 1.0 and aug["net"] > 0
        )
        flag = "PASS" if ok else "fail"
        print(
            f"  {flag} JJ n={jj['n']} PF={jj['pf']} ${jj['net']} exp={jj['exp']} | "
            f"Aug n={aug['n']} PF={aug['pf']} ${aug['net']}",
            flush=True,
        )
        rows.append({
            "name": name, "mode": mode, "pass": ok,
            "jun_jul": jj, "aug": aug, "full": full,
            "params": {
                "impulse": imp, "lookback": lb, "sl": sl,
                "tp_usd": tp, "max_hold": mh, "cooldown": cd,
            },
        })

    rows.sort(key=lambda r: (r["pass"], r["jun_jul"]["exp"] + r["aug"]["exp"]), reverse=True)
    out = os.path.join(ROOT, "data", "scalp_1s_mtf_filter_hunt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nSaved {out}", flush=True)
    print("\n=== TOP ===", flush=True)
    for r in rows[:6]:
        print(
            f"  {r['name']}: pass={r['pass']} "
            f"JJ PF={r['jun_jul']['pf']} ${r['jun_jul']['net']} | "
            f"Aug PF={r['aug']['pf']} ${r['aug']['net']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
