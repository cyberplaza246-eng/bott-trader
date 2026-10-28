"""
COMPRESSION_EXPANSION — new architecture (not a repair of prior kills).

Hypothesis:
  A compressed market is transitioning into an expansion regime, and the
  initial directional expansion has continuation potential.

Frozen pipeline:
  1) Compression: recent range (or ATR) at/below a low percentile for COMPRESS_BARS
  2) Build the compression box (high/low over that window)
  3) Expansion: close breaks outside the box by EXPAND_BUFFER pts
  4) Confirmation: next CONFIRM bars hold outside on that side
  5) Entry: first with-trend resume bar within RESUME_WITHIN
  6) Exit: SL 8 / TP 12 / max 10 bars (signal test only)
     Same-bar: stop before target. Fill next open ±1 tick. COMM 1.24.

Stages (Aug+ NEVER chooses):
  IS  = before 2026-06-01
  VAL = 2026-06-01 .. 2026-07-31
  OOS = 2026-08-01 onward

Hard gate unchanged. Control baseline not modified.
If FAIL: kill — do not invent another pattern; step back to data-state analysis.
"""
from __future__ import annotations

import json
import os
sys_path_insert = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, sys_path_insert)

from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategy.mnq_15m_ema_eod import ENTRY_MINUTE, FLAT_MINUTE

PT = 2.0
COMM = 1.24
SLIP = 0.25
IS_END = date(2026, 6, 1)
VAL_END = date(2026, 8, 1)
HORIZON_MAE = 15

RAW = {
    "name": "raw_compress_expand",
    "range_lookback": 20,       # bars for rolling range / ATR proxy
    "pct_lookback": 200,        # bars for percentile of compression
    "compress_pct": 25.0,       # range must be <= this percentile
    "compress_bars": 8,         # consecutive bars in compression
    "expand_buffer": 1.0,       # pts beyond box to count expansion
    "confirm": 1,
    "resume_within": 5,
    "body_max": 6.0,
    "sl_pts": 8.0,
    "tp_pts": 12.0,
    "max_bars": 10,
}

NEIGHBORS = [
    {**RAW, "name": "pct20", "compress_pct": 20.0},
    {**RAW, "name": "pct30", "compress_pct": 30.0},
    {**RAW, "name": "comp6", "compress_bars": 6},
    {**RAW, "name": "comp10", "compress_bars": 10},
    {**RAW, "name": "rb15", "range_lookback": 15},
    {**RAW, "name": "rb30", "range_lookback": 30},
    {**RAW, "name": "confirm0", "confirm": 0},
    {**RAW, "name": "confirm2", "confirm": 2},
    {**RAW, "name": "buf2", "expand_buffer": 2.0},
]

YEAR_CUTS = [
    (date(2024, 9, 22), date(2025, 1, 1), "2024q4"),
    (date(2025, 1, 1), date(2025, 7, 1), "2025h1"),
    (date(2025, 7, 1), date(2026, 1, 1), "2025h2"),
    (date(2026, 1, 1), date(2026, 6, 1), "2026pre"),
    (date(2026, 6, 1), date(2026, 8, 1), "2026val"),
    (date(2026, 8, 1), date(2026, 9, 22), "2026oos"),
]


def load_arrays(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    et = dt.tz_convert("America/New_York")
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    # rolling range (completed): max(hi)-min(lo) over lookback ending at i-1
    # precompute for default lookbacks used in RAW/neighbors
    return {
        "op": df["open"].to_numpy(dtype=float),
        "hi": hi,
        "lo": lo,
        "cl": cl,
        "mins": (et.hour * 60 + et.minute).to_numpy(),
        "dow": et.weekday.to_numpy(),
        "dates": np.array(et.date),
        "n": len(df),
    }


def rolling_range(hi: np.ndarray, lo: np.ndarray, lb: int) -> np.ndarray:
    """range[i] = max(hi[i-lb:i]) - min(lo[i-lb:i]); nan until enough history."""
    n = len(hi)
    out = np.full(n, np.nan)
    # simple loop — 664k * 30 is fine
    for i in range(lb, n):
        out[i] = float(np.max(hi[i - lb : i]) - np.min(lo[i - lb : i]))
    return out


def percentile_rank(series: np.ndarray, window: int) -> np.ndarray:
    """Percentile of series[i] among series[i-window:i] (exclusive of current)."""
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(window + 1, n):
        v = series[i]
        if v != v:
            continue
        hist = series[i - window : i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < max(20, window // 5):
            continue
        out[i] = 100.0 * float(np.mean(hist <= v))
    return out


def find_signals(a: Dict[str, Any], cfg: Dict[str, Any], rng: np.ndarray, pct: np.ndarray) -> List[Dict[str, Any]]:
    """
    IDLE → COMPRESSING (compress_bars of low-pct range) → BOX set
         → EXPANDED (close outside box) → CONFIRM → RESUME entry
    """
    compress_bars = int(cfg["compress_bars"])
    compress_pct = float(cfg["compress_pct"])
    expand_buffer = float(cfg["expand_buffer"])
    confirm_need = int(cfg["confirm"])
    resume_within = int(cfg["resume_within"])
    body_max = float(cfg["body_max"])
    lb = int(cfg["range_lookback"])
    hi, lo, cl, op = a["hi"], a["lo"], a["cl"], a["op"]
    n = a["n"]
    out: List[Dict[str, Any]] = []

    state = "IDLE"
    side = 0
    box_hi = box_lo = 0.0
    compress_run = 0
    compress_start = -1
    confirm_count = 0
    resume_count = 0
    last_day = None
    cool_until = 0

    for i in range(lb + 2, n - 1):
        day = a["dates"][i]
        if day != last_day:
            state, side, compress_run = "IDLE", 0, 0
            confirm_count = resume_count = 0
            last_day = day

        if i < cool_until:
            continue
        if int(a["dow"][i]) >= 5:
            state = "IDLE"
            compress_run = 0
            continue
        m = int(a["mins"][i])
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            if m >= FLAT_MINUTE:
                state = "IDLE"
                compress_run = 0
            continue

        p = pct[i]
        in_compress = p == p and p <= compress_pct

        if state == "IDLE":
            if in_compress:
                compress_run = 1
                compress_start = i
                state = "COMPRESSING"
            continue

        if state == "COMPRESSING":
            if not in_compress:
                state = "IDLE"
                compress_run = 0
                continue
            compress_run += 1
            if compress_run >= compress_bars:
                # freeze box from the compression window (completed bars)
                w0 = compress_start
                w1 = i + 1  # include current compress bar
                box_hi = float(np.max(hi[w0:w1]))
                box_lo = float(np.min(lo[w0:w1]))
                if box_hi - box_lo < 1.0:
                    state = "IDLE"
                    compress_run = 0
                    continue
                state = "BOX"
            continue

        if state == "BOX":
            # wait for expansion beyond box
            if cl[i] > box_hi + expand_buffer:
                state = "CONFIRM" if confirm_need > 0 else "RESUME"
                side = 1
                confirm_count = 0
                resume_count = 0
            elif cl[i] < box_lo - expand_buffer:
                state = "CONFIRM" if confirm_need > 0 else "RESUME"
                side = -1
                confirm_count = 0
                resume_count = 0
            else:
                # if compression breaks down without expansion for too long, reset
                # allow up to 30 bars waiting
                if i - compress_start > compress_bars + 30:
                    state = "IDLE"
                    compress_run = 0
            continue

        if state == "CONFIRM":
            if side > 0 and lo[i] < box_hi:
                # failed hold — back to box watch if still inside
                if cl[i] <= box_hi + expand_buffer:
                    state = "BOX"
                    side = 0
                continue
            if side < 0 and hi[i] > box_lo:
                if cl[i] >= box_lo - expand_buffer:
                    state = "BOX"
                    side = 0
                continue
            confirm_count += 1
            if confirm_count >= confirm_need:
                state = "RESUME"
                resume_count = 0
            continue

        if state == "RESUME":
            if side > 0 and lo[i] < box_hi:
                state = "IDLE"
                continue
            if side < 0 and hi[i] > box_lo:
                state = "IDLE"
                continue
            resume_count += 1
            if resume_count > resume_within:
                state = "IDLE"
                continue
            body = abs(cl[i] - op[i])
            if body > body_max:
                continue
            long_ok = side > 0 and cl[i] > op[i] and cl[i] > hi[i - 1]
            short_ok = side < 0 and cl[i] < op[i] and cl[i] < lo[i - 1]
            if long_ok or short_ok:
                j = i + 1
                if j < n and a["dates"][j] == day and int(a["mins"][j]) < FLAT_MINUTE:
                    out.append({
                        "side": int(side),
                        "sig": i,
                        "fill_i": j,
                        "date": a["dates"][j],
                        "raw_open": float(op[j]),
                        "box_hi": float(box_hi),
                        "box_lo": float(box_lo),
                        "box_width": float(box_hi - box_lo),
                    })
                    state = "IDLE"
                    compress_run = 0
                    cool_until = j + 5
            continue

    return out


def simulate_trade(a, sig, cfg, *, slip, comm) -> Dict[str, Any]:
    j = sig["fill_i"]
    side = sig["side"]
    sl_pts = float(cfg["sl_pts"])
    tp_pts = float(cfg["tp_pts"])
    max_bars = int(cfg["max_bars"])
    fill = sig["raw_open"] + (slip if side > 0 else -slip)
    stop = fill - sl_pts if side > 0 else fill + sl_pts
    tgt = fill + tp_pts if side > 0 else fill - tp_pts
    n = a["n"]
    last = min(j + max_bars, n)
    mfe = mae = 0.0
    reason = "time"
    k_exit = j
    pnl = 0.0
    for k in range(j, last):
        if a["dates"][k] != a["dates"][j] or int(a["mins"][k]) >= FLAT_MINUTE:
            k_exit = max(j, k - 1)
            clv = float(a["cl"][k_exit])
            px = clv - slip if side > 0 else clv + slip
            pts = (px - fill) if side > 0 else (fill - px)
            pnl = pts * PT - comm
            reason = "eod"
            break
        hv, lv = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            mfe = max(mfe, hv - fill)
            mae = max(mae, fill - lv)
            hit_sl, hit_tp = lv <= stop, hv >= tgt
        else:
            mfe = max(mfe, fill - lv)
            mae = max(mae, hv - fill)
            hit_sl, hit_tp = hv >= stop, lv <= tgt
        if hit_sl:
            px = stop - slip if side > 0 else stop + slip
            pts = (px - fill) if side > 0 else (fill - px)
            pnl = pts * PT - comm
            reason = "stop_ambig" if hit_tp else "stop"
            k_exit = k
            break
        if hit_tp:
            px = tgt - slip if side > 0 else tgt + slip
            pts = (px - fill) if side > 0 else (fill - px)
            pnl = pts * PT - comm
            reason = "target"
            k_exit = k
            break
    else:
        k_exit = last - 1
        clv = float(a["cl"][k_exit])
        px = clv - slip if side > 0 else clv + slip
        pts = (px - fill) if side > 0 else (fill - px)
        pnl = pts * PT - comm
        reason = "time"

    path_last = min(j + HORIZON_MAE, n)
    for k in range(j, path_last):
        if a["dates"][k] != a["dates"][j]:
            break
        hv, lv = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            mfe = max(mfe, hv - fill)
            mae = max(mae, fill - lv)
        else:
            mfe = max(mfe, fill - lv)
            mae = max(mae, hv - fill)

    return {
        "pnl": float(pnl), "reason": reason, "hold": int(k_exit - j + 1),
        "date": sig["date"], "side": side, "mfe": float(mfe), "mae": float(mae),
    }


def run_cfg(a, cfg, rng_cache, pct_cache, *, slip, comm):
    lb = int(cfg["range_lookback"])
    plb = int(cfg.get("pct_lookback", RAW["pct_lookback"]))
    key = (lb, plb)
    if key not in rng_cache:
        rng = rolling_range(a["hi"], a["lo"], lb)
        pct = percentile_rank(rng, plb)
        rng_cache[key] = rng
        pct_cache[key] = pct
    sigs = find_signals(a, cfg, rng_cache[key], pct_cache[key])
    trades = []
    next_free = 0
    for s in sigs:
        if s["fill_i"] < next_free:
            continue
        t = simulate_trade(a, s, cfg, slip=slip, comm=comm)
        trades.append(t)
        next_free = s["fill_i"] + t["hold"] + 1
    return trades, sigs


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0, "exp": 0.0, "pf": 0.0, "wr": 0.0, "net": 0.0, "dd": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "med_hold": 0, "mfe": None, "mae": None, "edge": None,
        }
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    wins = float(w.sum()) if len(w) else 0.0
    losses = float(l.sum()) if len(l) else 0.0
    pf = wins / abs(losses) if losses < 0 else (99.0 if wins > 0 else 0.0)
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    mfe = np.array([t["mfe"] for t in trades])
    mae = np.array([t["mae"] for t in trades])
    return {
        "n": int(len(pnl)),
        "exp": round(float(pnl.mean()), 2),
        "pf": round(pf, 3),
        "wr": round(100.0 * float((pnl > 0).mean()), 1),
        "net": round(float(pnl.sum()), 2),
        "dd": round(dd, 2),
        "avg_win": round(float(w.mean()), 2) if len(w) else 0.0,
        "avg_loss": round(float(l.mean()), 2) if len(l) else 0.0,
        "med_hold": int(np.median([t["hold"] for t in trades])),
        "mfe": round(float(np.median(mfe)), 2),
        "mae": round(float(np.median(mae)), 2),
        "edge": round(float(np.median(mfe) - np.median(mae)), 2),
    }


def split(trades, start: Optional[date], end: Optional[date]):
    return [
        t for t in trades
        if (start is None or t["date"] >= start) and (end is None or t["date"] < end)
    ]


def year_nets(trades) -> Dict[str, float]:
    out = {}
    for start, end, label in YEAR_CUTS:
        out[label] = round(sum(t["pnl"] for t in trades if start <= t["date"] < end), 2)
    return out


def stage_report(trades) -> Dict[str, Any]:
    return {
        "is": summarize(split(trades, None, IS_END)),
        "val": summarize(split(trades, IS_END, VAL_END)),
        "oos": summarize(split(trades, VAL_END, None)),
        "years": year_nets(trades),
    }


def passes_hard(rep: Dict[str, Any], cost2: Dict[str, Any]) -> bool:
    for key in ("is", "val", "oos"):
        s = rep[key]
        if s["n"] < 25 or s["exp"] <= 0 or s["pf"] < 1.20:
            return False
    if cost2["oos"]["exp"] <= 0 or cost2["oos"]["pf"] < 1.0:
        return False
    if sum(1 for v in rep["years"].values() if v > 0) < 3:
        return False
    if min(rep["years"].values()) < -5000:
        return False
    return True


def print_stage(name: str, rep: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===", flush=True)
    for stage in ("is", "val", "oos"):
        s = rep[stage]
        print(
            f"{stage.upper():4} n={s['n']:4} exp=${s['exp']:+7.2f} PF={s['pf']:.3f} "
            f"WR={s['wr']:5.1f}% avgW=${s['avg_win']:+.2f} avgL=${s['avg_loss']:+.2f} "
            f"DD=${s['dd']:.2f} hold={s['med_hold']} MFE={s['mfe']} MAE={s['mae']} edge={s['edge']}",
            flush=True,
        )
    print(f"years {rep['years']}", flush=True)


def main():
    control_path = "data/micro_scalp_control_baseline.json"
    control = {}
    if os.path.exists(control_path):
        with open(control_path, encoding="utf-8") as f:
            control = json.load(f)
        print(f"CONTROL (untouched): OOS exp={control.get('oos_exp')} PF={control.get('oos_pf')}", flush=True)

    a = load_arrays("data/MNQ_1m.csv")
    print(f"bars={a['n']} HYPOTHESIS=COMPRESSION_EXPANSION RAW={RAW['name']}", flush=True)
    print("precomputing rolling ranges…", flush=True)
    rng_cache: Dict = {}
    pct_cache: Dict = {}

    trades_1x, sigs = run_cfg(a, RAW, rng_cache, pct_cache, slip=SLIP, comm=COMM)
    print(f"raw_signals={len(sigs)} trades={len(trades_1x)}", flush=True)
    rep_1x = stage_report(trades_1x)
    print_stage("RAW 1x costs", rep_1x)

    cost_reps = {"1x": rep_1x}
    for mult, label in ((2, "2x"), (3, "3x")):
        tr, _ = run_cfg(a, RAW, rng_cache, pct_cache, slip=SLIP * mult, comm=COMM * mult)
        cost_reps[label] = stage_report(tr)
        print_stage(f"RAW {label} costs", cost_reps[label])

    hard = passes_hard(rep_1x, cost_reps["2x"])
    print(f"\nRAW_HARD_PASS {hard}", flush=True)

    print("\n=== ROBUSTNESS NEIGHBORS (1x) ===", flush=True)
    neigh_out = []
    for cfg in NEIGHBORS:
        tr, _ = run_cfg(a, cfg, rng_cache, pct_cache, slip=SLIP, comm=COMM)
        rep = stage_report(tr)
        ok = (
            rep["is"]["exp"] > 0 and rep["val"]["exp"] > 0 and rep["oos"]["exp"] > 0
            and rep["is"]["pf"] >= 1.1 and rep["oos"]["pf"] >= 1.1
        )
        neigh_out.append({"name": cfg["name"], "rep": rep, "stable_flag": ok})
        print(
            f"{cfg['name']:12} IS exp={rep['is']['exp']:+.2f} pf={rep['is']['pf']:.2f} n={rep['is']['n']} | "
            f"VAL exp={rep['val']['exp']:+.2f} pf={rep['val']['pf']:.2f} n={rep['val']['n']} | "
            f"OOS exp={rep['oos']['exp']:+.2f} pf={rep['oos']['pf']:.2f} n={rep['oos']['n']} "
            f"{'~stable' if ok else ''}",
            flush=True,
        )

    stable_count = sum(1 for n in neigh_out if n["stable_flag"])
    beats_control = (
        rep_1x["oos"]["exp"] > float(control.get("oos_exp", -3.02))
        and rep_1x["oos"]["pf"] > float(control.get("oos_pf", 0.71))
    )

    if hard and stable_count >= 2:
        verdict = "PASS"
    elif hard:
        verdict = "PASS_FRAGILE"
    elif (
        rep_1x["is"]["exp"] > 0 and rep_1x["val"]["exp"] > 0 and rep_1x["oos"]["exp"] > 0
        and rep_1x["oos"]["pf"] >= 1.1
    ):
        verdict = "PROMISING_BELOW_BAR"
    else:
        verdict = "FAIL"

    report = {
        "hypothesis": "COMPRESSION_EXPANSION",
        "raw_cfg": RAW,
        "control_path": control_path,
        "control": {k: control.get(k) for k in ("oos_exp", "oos_pf", "oos_wr", "mfe_median", "mae_median")},
        "beats_control_oos": beats_control,
        "raw_signals": len(sigs),
        "raw_1x": rep_1x,
        "costs": {k: v for k, v in cost_reps.items()},
        "raw_hard_pass": hard,
        "neighbors": [
            {"name": n["name"], "stable_flag": n["stable_flag"],
             "is": n["rep"]["is"], "val": n["rep"]["val"], "oos": n["rep"]["oos"]}
            for n in neigh_out
        ],
        "stable_neighbor_count": stable_count,
        "verdict": verdict,
        "killed_prior": [
            "micro_scalp_entry",
            "exit_repair",
            "conditional_filters",
            "pullback_continuation",
            "liquidity_sweep_reversal",
        ],
        "if_fail_next": (
            "Do not invent another pattern. Step back: measure which MNQ market "
            "states precede positive expectancy with OOS protected."
        ),
        "note": "Aug+ never chooses RAW. Pass bar not lowered. Control not modified.",
    }
    out = "data/compression_expansion_test.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nVERDICT {verdict}", flush=True)
    print(f"BEATS_CONTROL_OOS {beats_control}", flush=True)
    if verdict == "FAIL":
        print("NEXT: stop inventing patterns -> data-state expectancy map (OOS protected)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
