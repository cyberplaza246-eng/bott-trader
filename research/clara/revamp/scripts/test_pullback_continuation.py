"""
Pullback Continuation — new architecture (not a micro-scalp repair).

Hypothesis:
  A directional breakout occurred, price accepted the new level, and a
  controlled pullback failed to invalidate it — then price resumes.

Long (short = inverse):
  1) Close breaks prior swing high (lookback N).
  2) Confirmation: next HOLD bars stay at/above that break level.
  3) Pullback: within PB_MIN..PB_MAX bars, a quiet dip that stays above break.
  4) Resume: with-trend bar that closes back through the pullback extreme.
  5) Fixed SL / fixed TP. Stop before target on same 1m bar.

Stages (Aug+ NEVER used to choose):
  Discovery IS  = before 2026-06-01
  Validation    = 2026-06-01 .. 2026-07-31
  Untouched OOS = 2026-08-01 onward

Control to beat: data/micro_scalp_control_baseline.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.mnq_15m_ema_eod import ENTRY_MINUTE, FLAT_MINUTE

PT = 2.0
COMM = 1.24
SLIP = 0.25
IS_END = date(2026, 6, 1)
VAL_END = date(2026, 8, 1)
HORIZON_MAE = 15

# --- RAW architecture (pre-registered, one shot) ---
RAW = {
    "name": "raw_pb_cont",
    "swing": 10,
    "hold": 2,
    "pb_min": 2,
    "pb_max": 5,
    "sl_pts": 8.0,
    "tp_pts": 12.0,
    "max_bars": 10,
    "body_max": 4.0,
}

# Robustness neighbors — only interpreted AFTER raw results; not used to invent raw.
NEIGHBORS = [
    {**RAW, "name": "pb_2_5", "pb_min": 2, "pb_max": 5},
    {**RAW, "name": "pb_3_6", "pb_min": 3, "pb_max": 6},
    {**RAW, "name": "pb_4_7", "pb_min": 4, "pb_max": 7},
    {**RAW, "name": "swing8", "swing": 8},
    {**RAW, "name": "swing12", "swing": 12},
    {**RAW, "name": "sl6_tp9", "sl_pts": 6.0, "tp_pts": 9.0},
    {**RAW, "name": "sl8_tp16", "sl_pts": 8.0, "tp_pts": 16.0},
    {**RAW, "name": "hold1", "hold": 1},
    {**RAW, "name": "hold3", "hold": 3},
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
    return {
        "dt": dt,
        "op": df["open"].to_numpy(dtype=float),
        "hi": df["high"].to_numpy(dtype=float),
        "lo": df["low"].to_numpy(dtype=float),
        "cl": df["close"].to_numpy(dtype=float),
        "mins": (et.hour * 60 + et.minute).to_numpy(),
        "dow": et.weekday.to_numpy(),
        "dates": np.array(et.date),
        "n": len(df),
    }


def find_signals(a: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    State machine per RTH day:

      IDLE → BREAKOUT (close breaks swing) → CONFIRM (hold bars accept level)
           → PULLBACK (controlled dip respecting break) → RESUME entry
    """
    swing = int(cfg["swing"])
    hold_need = int(cfg["hold"])
    pb_min = int(cfg["pb_min"])
    pb_max = int(cfg["pb_max"])
    body_max = float(cfg["body_max"])
    hi, lo, cl, op = a["hi"], a["lo"], a["cl"], a["op"]
    n = a["n"]
    out: List[Dict[str, Any]] = []

    state = "IDLE"
    side = 0
    brk = 0.0
    break_i = -1
    hold_count = 0
    pb_count = 0
    dipped = False
    last_day = None

    for i in range(swing + 1, n - 1):
        day = a["dates"][i]
        if day != last_day:
            state, side, hold_count, pb_count, dipped = "IDLE", 0, 0, 0, False
            last_day = day

        if int(a["dow"][i]) >= 5:
            state = "IDLE"
            continue
        m = int(a["mins"][i])
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            if m >= FLAT_MINUTE:
                state = "IDLE"
            continue

        prior_hi = float(np.max(hi[i - swing : i]))
        prior_lo = float(np.min(lo[i - swing : i]))

        if state == "IDLE":
            if cl[i] > prior_hi:
                state = "CONFIRM"
                side = 1
                brk = prior_hi
                break_i = i
                hold_count = 0
                pb_count = 0
                dipped = False
            elif cl[i] < prior_lo:
                state = "CONFIRM"
                side = -1
                brk = prior_lo
                break_i = i
                hold_count = 0
                pb_count = 0
                dipped = False
            continue

        if state == "CONFIRM":
            # acceptance: stay on the breakout side of the level
            if side > 0 and lo[i] < brk:
                state = "IDLE"
                continue
            if side < 0 and hi[i] > brk:
                state = "IDLE"
                continue
            hold_count += 1
            if hold_count >= hold_need:
                state = "PULLBACK"
                pb_count = 0
                dipped = False
            continue

        if state == "PULLBACK":
            if side > 0 and lo[i] < brk:
                state = "IDLE"
                continue
            if side < 0 and hi[i] > brk:
                state = "IDLE"
                continue
            pb_count += 1
            if side > 0 and cl[i] < op[i]:
                dipped = True
            if side < 0 and cl[i] > op[i]:
                dipped = True
            if pb_count > pb_max:
                state = "IDLE"
                continue
            if pb_count >= pb_min and dipped:
                state = "RESUME"
            continue

        if state == "RESUME":
            if side > 0 and lo[i] < brk:
                state = "IDLE"
                continue
            if side < 0 and hi[i] > brk:
                state = "IDLE"
                continue
            pb_count += 1
            if pb_count > pb_max:
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
                        "break_level": float(brk),
                        "break_i": break_i,
                    })
                    state = "IDLE"
            continue

    return out


def simulate_trade(
    a: Dict[str, Any],
    sig: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    slip: float,
    comm: float,
) -> Dict[str, Any]:
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
            cl = float(a["cl"][k_exit])
            px = cl - slip if side > 0 else cl + slip
            pts = (px - fill) if side > 0 else (fill - px)
            pnl = pts * PT - comm
            reason = "eod"
            break
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            mfe = max(mfe, hi - fill)
            mae = max(mae, fill - lo)
            hit_sl = lo <= stop
            hit_tp = hi >= tgt
        else:
            mfe = max(mfe, fill - lo)
            mae = max(mae, hi - fill)
            hit_sl = hi >= stop
            hit_tp = lo <= tgt
        if hit_sl:  # stop first if both
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
        cl = float(a["cl"][k_exit])
        px = cl - slip if side > 0 else cl + slip
        pts = (px - fill) if side > 0 else (fill - px)
        pnl = pts * PT - comm
        reason = "time"

    # extend MAE/MFE path a bit for reporting even after exit
    path_last = min(j + HORIZON_MAE, n)
    for k in range(j, path_last):
        if a["dates"][k] != a["dates"][j]:
            break
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            mfe = max(mfe, hi - fill)
            mae = max(mae, fill - lo)
        else:
            mfe = max(mfe, fill - lo)
            mae = max(mae, hi - fill)

    return {
        "pnl": float(pnl),
        "reason": reason,
        "hold": int(k_exit - j + 1),
        "date": sig["date"],
        "side": side,
        "mfe": float(mfe),
        "mae": float(mae),
    }


def run_cfg(a, cfg, *, slip, comm) -> List[Dict[str, Any]]:
    sigs = find_signals(a, cfg)
    trades = []
    next_free = 0
    for s in sigs:
        if s["fill_i"] < next_free:
            continue
        t = simulate_trade(a, s, cfg, slip=slip, comm=comm)
        trades.append(t)
        next_free = s["fill_i"] + t["hold"] + 1
    return trades


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0, "exp": 0.0, "pf": 0.0, "wr": 0.0, "net": 0.0, "dd": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "med_hold": 0, "mfe": None, "mae": None, "edge": None,
        }
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wins = float(w.sum()) if len(w) else 0.0
    losses = float(l.sum()) if len(l) else 0.0
    pf = wins / abs(losses) if losses < 0 else (99.0 if wins > 0 else 0.0)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max()) if len(eq) else 0.0
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
    out = []
    for t in trades:
        d = t["date"]
        if start is not None and d < start:
            continue
        if end is not None and d >= end:
            continue
        out.append(t)
    return out


def year_nets(trades) -> Dict[str, float]:
    out = {}
    for start, end, label in YEAR_CUTS:
        sub = [t for t in trades if start <= t["date"] < end]
        out[label] = round(sum(t["pnl"] for t in sub), 2)
    return out


def stage_report(trades, slip_mult: float = 1.0) -> Dict[str, Any]:
    is_t = split(trades, None, IS_END)
    val_t = split(trades, IS_END, VAL_END)
    oos_t = split(trades, VAL_END, None)
    return {
        "is": summarize(is_t),
        "val": summarize(val_t),
        "oos": summarize(oos_t),
        "years": year_nets(trades),
        "slip_mult": slip_mult,
    }


def passes_hard(rep: Dict[str, Any], cost2: Dict[str, Any]) -> bool:
    """Pass criteria from the research brief. OOS must be Aug+ (untouched)."""
    for key in ("is", "val", "oos"):
        s = rep[key]
        if s["n"] < 25:
            return False
        if s["exp"] <= 0 or s["pf"] < 1.20:
            return False
    # 2x costs still profitable on OOS
    if cost2["oos"]["exp"] <= 0 or cost2["oos"]["pf"] < 1.0:
        return False
    yn = rep["years"]
    pos = [v for k, v in yn.items() if v > 0]
    if len(pos) < 3:
        return False
    # not catastrophic worst year among those with trades
    worst = min(yn.values()) if yn else 0
    if worst < -5000:  # soft catastrophe guard relative to sample
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
    print(f"bars={a['n']} RAW={RAW['name']}", flush=True)

    # Stage 1–3 at 1x
    sigs = find_signals(a, RAW)
    print(f"raw_signals={len(sigs)}", flush=True)
    trades_1x = run_cfg(a, RAW, slip=SLIP, comm=COMM)
    rep_1x = stage_report(trades_1x, 1.0)
    print_stage("RAW 1x costs", rep_1x)

    # Stage 4 costs
    cost_reps = {"1x": rep_1x}
    for mult, label in ((2, "2x"), (3, "3x")):
        tr = run_cfg(a, RAW, slip=SLIP * mult, comm=COMM * mult)
        cost_reps[label] = stage_report(tr, float(mult))
        print_stage(f"RAW {label} costs", cost_reps[label])

    hard = passes_hard(rep_1x, cost_reps["2x"])
    print(f"\nRAW_HARD_PASS {hard}", flush=True)

    # Stage 5 robustness — report always; only "stable" if raw passed or near-pass
    print("\n=== ROBUSTNESS NEIGHBORS (1x) ===", flush=True)
    neigh_out = []
    for cfg in NEIGHBORS:
        tr = run_cfg(a, cfg, slip=SLIP, comm=COMM)
        rep = stage_report(tr)
        ok = (
            rep["is"]["exp"] > 0 and rep["val"]["exp"] > 0 and rep["oos"]["exp"] > 0
            and rep["is"]["pf"] >= 1.1 and rep["oos"]["pf"] >= 1.1
        )
        neigh_out.append({"name": cfg["name"], "cfg": cfg, "rep": rep, "stable_flag": ok})
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

    verdict = "FAIL"
    if hard and stable_count >= 2:
        verdict = "PASS"
    elif hard:
        verdict = "PASS_FRAGILE"
    elif (
        rep_1x["is"]["exp"] > 0 and rep_1x["val"]["exp"] > 0 and rep_1x["oos"]["exp"] > 0
        and rep_1x["oos"]["pf"] >= 1.1
    ):
        verdict = "PROMISING_BELOW_BAR"

    report = {
        "architecture": "pullback_continuation",
        "hypothesis": (
            "Breakout → acceptance hold → controlled pullback that respects break → resume entry"
        ),
        "raw_cfg": RAW,
        "control_path": control_path,
        "control": {k: control.get(k) for k in ("oos_exp", "oos_pf", "oos_wr", "mfe_median", "mae_median")},
        "beats_control_oos": beats_control,
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
        "note": (
            "Aug+ OOS never used to define RAW. Same-bar: stop before target. "
            "Do not lower pass bar. Control baseline file was not modified."
        ),
    }
    out = "data/pullback_continuation_test.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nVERDICT {verdict}", flush=True)
    print(f"BEATS_CONTROL_OOS {beats_control}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
