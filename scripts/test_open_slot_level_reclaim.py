#!/usr/bin/env python3
"""
OPEN_SLOT_LEVEL_RECLAIM — new hypothesis (not a micro-scalp repair).

Suggestion under test:
  Trade only the opening slot (09:35–10:15 ET). Enter when price tags a
  major session level, fails beyond it, then reclaims (failed auction).

Long (short = inverse):
  1) Clock in OPEN_START..OPEN_END
  2) Level in {PDH, PDL, ONH, ONL} (ORH/ORL after OR forms; RTH open)
  3) Bar trades beyond level by >= SWEEP_MIN pts (or tags within TOUCH)
  4) Same or next bars: close back through the level (reclaim)
  5) Optional: 5m EMA bias agrees with reclaim side
  6) Fill next bar open. Fixed SL/TP. Stop before target same bar.

Stages (Aug+ NEVER chooses):
  IS  = before 2026-06-01
  VAL = 2026-06-01 .. 2026-07-31
  OOS = 2026-08-01 onward

Hard gate: IS + VAL both green (n, PF, net, exp); OOS report-only but must
not be deeply red for PASS. Slip + commission from bar 1. Kill if med MFE<=MAE.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.day_hybrid import prepare_arrays
from src.strategy.mnq_15m_ema_eod import ENTRY_MINUTE, FLAT_MINUTE, _as_utc_frame

PT = 2.0
COMM = 1.24
SLIP = 0.25
IS_END = date(2026, 6, 1)
VAL_END = date(2026, 8, 1)
HORIZON_MAE = 20
ET = pytz.timezone("US/Eastern")

OPEN_START = 9 * 60 + 35
OPEN_END = 10 * 60 + 15  # exclusive upper for new entries
OR_END = 9 * 60 + 45

RAW = {
    "name": "raw_open_reclaim",
    "sweep_min": 1.0,
    "touch_pts": 4.0,
    "confirm": 0,
    "sl_pts": 8.0,
    "tp_pts": 12.0,
    "max_bars": 15,
    "require_t5": False,
    "levels": ("pdh", "pdl", "onh", "onl", "rth_open"),
    "use_or": True,
    "max_day": 2,
}

# Pre-registered neighbors only — no free search
NEIGHBORS = [
    {**RAW, "name": "t5_filter", "require_t5": True},
    {**RAW, "name": "sweep2", "sweep_min": 2.0},
    {**RAW, "name": "sl10_tp15", "sl_pts": 10.0, "tp_pts": 15.0, "max_bars": 20},
    {**RAW, "name": "open_to_1030", "open_end": 10 * 60 + 30},
    {**RAW, "name": "pd_only", "levels": ("pdh", "pdl"), "use_or": False},
    {**RAW, "name": "or_only", "levels": (), "use_or": True},
    {**RAW, "name": "max1", "max_day": 1},
    {**RAW, "name": "confirm1", "confirm": 1},
]


def _prior_rth_day(et_day: date) -> Optional[date]:
    d = et_day - timedelta(days=1)
    for _ in range(8):
        if d.weekday() < 5:
            return d
        d -= timedelta(days=1)
    return None


def _globex_start(et_day: date) -> datetime:
    start_day = et_day - timedelta(days=1)
    while start_day.weekday() == 5:
        start_day -= timedelta(days=1)
    return ET.localize(datetime(start_day.year, start_day.month, start_day.day, 18, 0))


def day_static_levels(tmp: pd.DataFrame) -> Dict[date, Dict[str, Optional[float]]]:
    et = tmp.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    wd = et.weekday
    rth = (wd < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    day_arr = pd.Series(et.date, index=tmp.index)
    out: Dict[date, Dict[str, Optional[float]]] = {}
    for et_day in sorted(set(day_arr[rth].tolist())):
        prior = _prior_rth_day(et_day)
        pdh = pdl = None
        if prior is not None:
            pm = rth & (day_arr == prior)
            pb = tmp.loc[pm]
            if not pb.empty:
                pdh = float(pb["high"].max())
                pdl = float(pb["low"].min())
        g0 = _globex_start(et_day).astimezone(pytz.UTC)
        rth0 = ET.localize(datetime(et_day.year, et_day.month, et_day.day, 9, 30)).astimezone(pytz.UTC)
        on = tmp.loc[(tmp.index >= g0) & (tmp.index < rth0)]
        onh = float(on["high"].max()) if not on.empty else None
        onl = float(on["low"].min()) if not on.empty else None
        today = tmp.loc[rth & (day_arr == et_day)]
        rth_open = float(today.iloc[0]["open"]) if not today.empty else None
        out[et_day] = {"pdh": pdh, "pdl": pdl, "onh": onh, "onl": onl, "rth_open": rth_open}
    return out


def load(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    tmp = _as_utc_frame(df)
    a = prepare_arrays(df)
    levels = day_static_levels(tmp)
    return {
        "op": a["op"],
        "hi": a["hi"],
        "lo": a["lo"],
        "cl": a["cl"],
        "mins": a["mins"],
        "dow": a["dow"],
        "dates": a["dates"],
        "t5": a["t5"],
        "n": len(a["cl"]),
        "levels": levels,
        "tmp": tmp,
    }


def active_levels(
    a: Dict[str, Any],
    i: int,
    cfg: Dict[str, Any],
    or_cache: Dict[date, Tuple[Optional[float], Optional[float]]],
) -> List[Tuple[str, float]]:
    d = a["dates"][i]
    static = a["levels"].get(d, {})
    out: List[Tuple[str, float]] = []
    for k in cfg.get("levels", ()):
        v = static.get(k)
        if v is not None:
            out.append((k, float(v)))
    if cfg.get("use_or"):
        orh, orl = or_cache.get(d, (None, None))
        m = int(a["mins"][i])
        if m >= OR_END:
            if orh is not None:
                out.append(("orh", float(orh)))
            if orl is not None:
                out.append(("orl", float(orl)))
    return out


def update_or(a: Dict[str, Any], i: int, or_cache: Dict[date, Tuple[Optional[float], Optional[float]]]) -> None:
    d = a["dates"][i]
    m = int(a["mins"][i])
    if int(a["dow"][i]) >= 5:
        return
    if m < 9 * 60 + 30 or m >= OR_END:
        return
    hi, lo = float(a["hi"][i]), float(a["lo"][i])
    prev = or_cache.get(d, (None, None))
    orh = hi if prev[0] is None else max(prev[0], hi)
    orl = lo if prev[1] is None else min(prev[1], lo)
    or_cache[d] = (orh, orl)


def find_signals(a: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = a["n"]
    open_end = int(cfg.get("open_end", OPEN_END))
    sweep_min = float(cfg["sweep_min"])
    touch = float(cfg["touch_pts"])
    confirm = int(cfg["confirm"])
    require_t5 = bool(cfg["require_t5"])
    max_day = int(cfg["max_day"])
    or_cache: Dict[date, Tuple[Optional[float], Optional[float]]] = {}
    sigs: List[Dict[str, Any]] = []
    day_count: Dict[date, int] = {}
    # state: waiting reclaim after pierce
    pending: Optional[Dict[str, Any]] = None

    for i in range(80, n - 2):
        update_or(a, i, or_cache)
        if int(a["dow"][i]) >= 5:
            pending = None
            continue
        m = int(a["mins"][i])
        d = a["dates"][i]
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            pending = None
            continue

        # expire pending across days / after slot
        if pending is not None and (pending["date"] != d or m >= open_end + 30):
            pending = None

        hi, lo, cl = float(a["hi"][i]), float(a["lo"][i]), float(a["cl"][i])

        if pending is not None:
            side = pending["side"]
            lvl = pending["level_px"]
            # confirm bars hold reclaim side
            held = True
            if confirm > 0:
                # after reclaim bar, need confirm more bars
                if i < pending["reclaim_i"] + confirm:
                    if side > 0 and lo < lvl:
                        held = False
                    if side < 0 and hi > lvl:
                        held = False
                    if not held:
                        pending = None
                    continue
            # resume: with-trend close past reclaim bar extreme
            if side > 0 and cl >= pending["reclaim_hi"] and cl > lvl:
                ok = True
            elif side < 0 and cl <= pending["reclaim_lo"] and cl < lvl:
                ok = True
            else:
                ok = False
            if not ok:
                if i > pending["reclaim_i"] + 8:
                    pending = None
                continue
            if require_t5 and int(a["t5"][i]) != side:
                pending = None
                continue
            if day_count.get(d, 0) >= max_day:
                pending = None
                continue
            fill_i = i + 1
            if fill_i >= n or a["dates"][fill_i] != d:
                pending = None
                continue
            sigs.append({
                "i": i,
                "fill_i": fill_i,
                "side": side,
                "date": d,
                "level": pending["level_name"],
                "level_px": lvl,
            })
            day_count[d] = day_count.get(d, 0) + 1
            pending = None
            continue

        if m < OPEN_START or m >= open_end:
            continue
        if day_count.get(d, 0) >= max_day:
            continue

        for name, lvl in active_levels(a, i, cfg, or_cache):
            # long: pierce below support-like or tag resistance fail
            # Failed auction long: take out low of level then reclaim above
            # Treat any level as a magnet: pierce beyond then reclaim through
            pierced_long = lo <= lvl - sweep_min and cl >= lvl
            pierced_short = hi >= lvl + sweep_min and cl <= lvl
            tagged_long = abs(lo - lvl) <= touch and cl > lvl and float(a["op"][i]) <= lvl + touch
            tagged_short = abs(hi - lvl) <= touch and cl < lvl and float(a["op"][i]) >= lvl - touch

            side = 0
            if pierced_long or (not pierced_short and tagged_long and lo <= lvl):
                side = 1
            elif pierced_short or (not pierced_long and tagged_short and hi >= lvl):
                side = -1
            if side == 0:
                continue
            if require_t5 and int(a["t5"][i]) != side:
                continue
            if confirm == 0:
                # enter on reclaim close → next open
                fill_i = i + 1
                if fill_i >= n or a["dates"][fill_i] != d:
                    continue
                if day_count.get(d, 0) >= max_day:
                    break
                sigs.append({
                    "i": i,
                    "fill_i": fill_i,
                    "side": side,
                    "date": d,
                    "level": name,
                    "level_px": lvl,
                })
                day_count[d] = day_count.get(d, 0) + 1
                break
            pending = {
                "side": side,
                "level_name": name,
                "level_px": lvl,
                "date": d,
                "reclaim_i": i,
                "reclaim_hi": hi,
                "reclaim_lo": lo,
            }
            break
    return sigs


def simulate_trade(a, sig, cfg, *, slip, comm) -> Dict[str, Any]:
    j = sig["fill_i"]
    side = sig["side"]
    fill = float(a["op"][j]) + (slip if side > 0 else -slip)
    sl = cfg["sl_pts"]
    tp = cfg["tp_pts"]
    stop = fill - sl if side > 0 else fill + sl
    tgt = fill + tp if side > 0 else fill - tp
    n = a["n"]
    last = min(j + int(cfg["max_bars"]), n)
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
        hi, lo = float(a["hi"][k]), float(a["lo"][k])
        if side > 0:
            mfe = max(mfe, hi - fill)
            mae = max(mae, fill - lo)
            hit_sl, hit_tp = lo <= stop, hi >= tgt
        else:
            mfe = max(mfe, fill - lo)
            mae = max(mae, hi - fill)
            hit_sl, hit_tp = hi >= stop, lo <= tgt
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
        "level": sig["level"],
        "mfe": float(mfe),
        "mae": float(mae),
    }


def run_cfg(a, cfg) -> Tuple[List[Dict], List[Dict]]:
    sigs = find_signals(a, cfg)
    trades = []
    next_free = 0
    for s in sigs:
        if s["fill_i"] < next_free:
            continue
        t = simulate_trade(a, s, cfg, slip=SLIP, comm=COMM)
        trades.append(t)
        next_free = s["fill_i"] + t["hold"] + 1
    return trades, sigs


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "exp": 0.0, "dd": 0.0,
            "mfe": 0.0, "mae": 0.0, "edge": 0.0,
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    mfe = float(np.median([t["mfe"] for t in trades]))
    mae = float(np.median([t["mae"] for t in trades]))
    return {
        "n": len(trades),
        "wr": round(100.0 * len(wins) / len(trades), 1),
        "pf": round((gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0), 2),
        "net": round(sum(pnls), 2),
        "exp": round(sum(pnls) / len(trades), 2),
        "dd": round(dd, 2),
        "mfe": round(mfe, 2),
        "mae": round(mae, 2),
        "edge": round(mfe - mae, 2),
    }


def window(trades, start: date, end: Optional[date] = None):
    out = []
    for t in trades:
        d = t["date"]
        if isinstance(d, str):
            d = date.fromisoformat(d)
        if d < start:
            continue
        if end and d >= end:
            continue
        out.append(t)
    return out


def gate(m: Dict[str, Any], *, min_n: int) -> bool:
    return (
        m["n"] >= min_n
        and m["pf"] >= 1.15
        and m["net"] > 0
        and m["exp"] >= 0.5
        and m["edge"] > 0
    )


def eval_cfg(a, cfg) -> Dict[str, Any]:
    trades, sigs = run_cfg(a, cfg)
    # serialize dates
    for t in trades:
        if hasattr(t["date"], "isoformat"):
            t["date"] = t["date"].isoformat()
    is_ = summarize(window(trades, date(2020, 1, 1), IS_END))
    val = summarize(window(trades, IS_END, VAL_END))
    oos = summarize(window(trades, VAL_END))
    ok = gate(is_, min_n=40) and gate(val, min_n=15) and oos["net"] > -200 and oos["pf"] >= 0.95
    return {
        "name": cfg["name"],
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "signals": len(sigs),
        "is": is_,
        "val": val,
        "oos": oos,
        "pass": ok,
        "full": summarize(trades),
    }


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print(f"Loading {path} ...", flush=True)
    a = load(path)
    print(f"bars={a['n']:,} days_with_levels={len(a['levels']):,}", flush=True)

    cfgs = [RAW] + NEIGHBORS
    rows = []
    for cfg in cfgs:
        row = eval_cfg(a, cfg)
        rows.append(row)
        flag = "PASS" if row["pass"] else "fail"
        print(
            f"{flag:4} {row['name']:20} "
            f"IS n={row['is']['n']:4} PF={row['is']['pf']:5} ${row['is']['net']:8} "
            f"edge={row['is']['edge']:5} | "
            f"VAL n={row['val']['n']:3} PF={row['val']['pf']:5} ${row['val']['net']:7} | "
            f"OOS n={row['oos']['n']:3} PF={row['oos']['pf']:5} ${row['oos']['net']:7}",
            flush=True,
        )

    rows.sort(key=lambda r: (r["pass"], r["val"]["exp"], r["is"]["exp"]), reverse=True)
    out = os.path.join("data", "open_slot_level_reclaim_test.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"hypothesis": "open_slot_level_reclaim", "rows": rows}, f, indent=2)
    print(f"\nSaved {out}", flush=True)
    print("\n=== TOP ===", flush=True)
    for r in rows[:5]:
        print(
            f"  {r['name']}: pass={r['pass']} "
            f"IS edge={r['is']['edge']} PF={r['is']['pf']} | "
            f"VAL PF={r['val']['pf']} ${r['val']['net']} | "
            f"OOS PF={r['oos']['pf']} ${r['oos']['net']}",
            flush=True,
        )
    passed = [r for r in rows if r["pass"]]
    print(f"\nPASS count: {len(passed)} / {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
