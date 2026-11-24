#!/usr/bin/env python3
"""
Sweep recipes for ~6 quality RTH entries/day on locked ema15_eod.
Real Databento 1m only. IS < 2026-06-01, OOS >= 2026-06-01.
Does not change the locked strategy — research only.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    FLAT_MINUTE,
    SEP_MIN,
    atr15_on_1m,
    daily_trend_on_1m,
    high_confidence,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
)

ET = pytz.timezone("US/Eastern")
PT, COMM = 2.0, 1.24
ENTRY_MINUTE = 9 * 60 + 35

WIN3 = (
    (9 * 60 + 35, 10 * 60),
    (11 * 60, 11 * 60 + 30),
    (13 * 60 + 30, 14 * 60),
)
WIN4A = (
    (9 * 60 + 35, 10 * 60),
    (11 * 60, 11 * 60 + 30),
    (13 * 60, 13 * 60 + 30),
    (14 * 60 + 30, 15 * 60),
)
WIN4B = (
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 45, 11 * 60 + 15),
    (12 * 60 + 30, 13 * 60),
    (14 * 60, 14 * 60 + 30),
)
WIN5 = (
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 30, 11 * 60),
    (11 * 60 + 30, 12 * 60),
    (13 * 60 + 30, 14 * 60),
    (14 * 60 + 30, 15 * 60),
)
WIN6 = (
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 15, 10 * 60 + 40),
    (11 * 60, 11 * 60 + 30),
    (12 * 60, 12 * 60 + 30),
    (13 * 60 + 30, 14 * 60),
    (14 * 60 + 30, 15 * 60),
)
WIN6B = (
    (9 * 60 + 35, 9 * 60 + 55),
    (10 * 60 + 20, 10 * 60 + 40),
    (11 * 60 + 5, 11 * 60 + 25),
    (12 * 60 + 5, 12 * 60 + 25),
    (13 * 60 + 20, 13 * 60 + 40),
    (14 * 60 + 20, 14 * 60 + 40),
)


def win_idx(mins: int, windows) -> Optional[int]:
    for i, (a, b) in enumerate(windows):
        if a <= mins < b:
            return i
    return None


def load_arrays(csv: str):
    raw, _, _, _ = scalp_bt.load_data(csv, None, None, allow_synthetic_30s=True)
    df = scalp_bt.add_1m_indicators(raw)
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    rth = (et.weekday < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    dates = et.date
    rth_days = sorted({d for d, ok in zip(dates, rth) if ok})
    oos = pd.Timestamp("2026-06-01").date()
    return {
        "dt": dt,
        "t15": trend_15m_on_1m(work).to_numpy(dtype=int),
        "td": daily_trend_on_1m(work).to_numpy(dtype=int),
        "t60": trend_60m_on_1m(work).to_numpy(dtype=int),
        "sep": sep15_on_1m(work).to_numpy(dtype=float),
        "atr": atr15_on_1m(work).to_numpy(dtype=float),
        "hi": work["high"].to_numpy(dtype=float),
        "lo": work["low"].to_numpy(dtype=float),
        "cl": work["close"].to_numpy(dtype=float),
        "mins": mins,
        "dow": et.weekday.to_numpy(),
        "dates": dates,
        "n_rth_days": len(rth_days),
        "n_rth_is": sum(1 for d in rth_days if d < oos),
        "n_rth_oos": sum(1 for d in rth_days if d >= oos),
    }


def simulate(
    a: Dict,
    *,
    windows,
    max_day: int,
    max_open: int = 2,
    conf_2lot: bool = True,
    conf_from: int = 99,
    extra_win_conf: int = 99,
    extra_win_set=(),
    cooldown_min: int = 0,
    max_hold_min: Optional[int] = None,
    reenter_aligned: bool = False,
    sl_reenter_only: bool = False,
) -> List:
    dt = a["dt"]
    t15, td, t60, sep, atr = a["t15"], a["td"], a["t60"], a["sep"], a["atr"]
    hi, lo, cl = a["hi"], a["lo"], a["cl"]
    mins, dow, dates = a["mins"], a["dow"], a["dates"]
    trades = []
    opens = []
    fired = {}
    counts = {}
    last_exit_min = {}
    last_exit_reason = {}

    for i in range(80, len(dt)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            held = (ts - p["ts"]).total_seconds() / 60.0
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif max_hold_min is not None and held >= max_hold_min:
                reason, exit_px = "TIME", float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
                trades.append(
                    scalp_bt.ScalpTrade(
                        entry_time=p["ts"],
                        direction="long" if p["dir"] == 1 else "short",
                        entry_price=p["entry"],
                        sl=p["sl"],
                        tp=p["tp"],
                        exit_time=ts,
                        exit_price=exit_px,
                        pnl=pts * PT - COMM,
                        exit_reason=reason,
                        version="search",
                        hold_seconds=(ts - p["ts"]).total_seconds(),
                    )
                )
                last_exit_min[day] = m
                last_exit_reason[day] = reason
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        side = int(t15[i])
        if side == 0 or side != int(td[i]):
            continue
        if counts.get(day, 0) >= max_day:
            continue
        n_today = counts.get(day, 0)
        conf = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        if n_today >= conf_from and not conf:
            continue
        win = win_idx(m, windows)
        if cooldown_min and day in last_exit_min and (m - last_exit_min[day]) < cooldown_min:
            continue
        if reenter_aligned:
            first_ok = win is not None and win not in fired.get(day, set())
            re_ok = n_today > 0 and (
                not sl_reenter_only or last_exit_reason.get(day) == "SL"
            )
            if not (first_ok or re_ok):
                continue
        else:
            if win is None:
                continue
            if win in fired.get(day, set()):
                continue
            if (win >= extra_win_conf or win in extra_win_set) and not conf:
                continue
        if len(opens) >= max_open:
            continue
        if len(opens) >= 1:
            if conf_2lot and (not conf or opens[0]["dir"] != side):
                continue
            if not conf_2lot:
                continue
        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp = entry + 500 if side == 1 else entry - 500
        opens.append({"ts": ts, "dir": side, "entry": entry, "sl": sl, "tp": tp})
        if win is not None:
            fired.setdefault(day, set()).add(win)
        counts[day] = n_today + 1
    return trades


def verdict(ins, oos) -> str:
    i, o = edge.summarize(ins), edge.summarize(oos)
    ok = (
        o["trades"] >= 30
        and o["profit_factor"] >= 1.15
        and o["total_pnl"] > 0
        and i["profit_factor"] >= 1.05
        and i["total_pnl"] > 0
    )
    return "PASS" if ok else "FAIL"


def per_day(trades, n_days: int) -> float:
    return (len(trades) / n_days) if n_days else 0.0


def dump(name: str, trades, a, *, show=True) -> Tuple[str, Dict]:
    ins, oos = edge.split_is_oos(trades)
    i, o, all_s = edge.summarize(ins), edge.summarize(oos), edge.summarize(trades)
    v = verdict(ins, oos)
    tpd = per_day(trades, a["n_rth_days"])
    tpd_oos = per_day(oos, a["n_rth_oos"])
    row = {
        "name": name,
        "verdict": v,
        "is_n": i["trades"],
        "is_pf": i["profit_factor"],
        "is_pnl": i["total_pnl"],
        "oos_n": o["trades"],
        "oos_pf": o["profit_factor"],
        "oos_pnl": o["total_pnl"],
        "all_pnl": all_s["total_pnl"],
        "dd": all_s["max_drawdown"],
        "tpd": tpd,
        "tpd_oos": tpd_oos,
        "wr": all_s["win_rate"],
    }
    if show:
        print(
            f"{v:4} {name:<52} IS {i['trades']:3d} {i['profit_factor']:5.2f} ${i['total_pnl']:7.0f}  "
            f"OOS {o['trades']:3d} {o['profit_factor']:5.2f} ${o['total_pnl']:7.0f}  "
            f"t/d {tpd:4.2f}/{tpd_oos:4.2f}  DD ${all_s['max_drawdown']:6.0f}",
            flush=True,
        )
    return v, row


def hist_day(trades, a):
    from collections import Counter
    days = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        d = ts.tz_convert(ET).date()
        days[d] = days.get(d, 0) + 1
    c = Counter(days.values())
    zeros = a["n_rth_days"] - len(days)
    print(f"  day-fill hist (incl {zeros} zero days): 0={zeros}", end="")
    for k in range(1, 7):
        print(f"  {k}={c.get(k, 0)}", end="")
    ge6 = sum(v for k, v in c.items() if k >= 6)
    mx = max(days.values()) if days else 0
    print(f"  6+={ge6}  max={mx}")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    refine = "--refine" in sys.argv
    csv = args[0] if args else os.path.join("data", "MNQ_1m.csv")
    print("Loading", csv, flush=True)
    a = load_arrays(csv)
    print(
        f"RTH days ALL={a['n_rth_days']} IS={a['n_rth_is']} OOS={a['n_rth_oos']}",
        flush=True,
    )
    print(f"{'':4} {'Rule':<52} {'IS':>16} {'OOS':>22}  t/d all/oos", flush=True)
    print("-" * 140, flush=True)
    rows: List[Dict] = []

    extra135 = {1, 3, 5}  # 10:15, 12:00, 14:30 in WIN6
    if refine:
        recipes = [
            ("BASE 3win 2lot", dict(windows=WIN3, max_day=3)),
            ("6win ungated", dict(windows=WIN6, max_day=6)),
            ("6win extra slots 10:15/12/14:30 conf", dict(windows=WIN6, max_day=6, extra_win_set=extra135)),
            ("6win last3 windows conf", dict(windows=WIN6, max_day=6, extra_win_conf=3)),
            ("6win trades4+ conf", dict(windows=WIN6, max_day=6, conf_from=3)),
            ("4win B spaced", dict(windows=WIN4B, max_day=4)),
            ("4win A 935/11/13/1430", dict(windows=WIN4A, max_day=4)),
            ("orig3 + 14:30 conf", dict(windows=WIN3 + ((14 * 60 + 30, 15 * 60),), max_day=4, extra_win_set={3})),
            ("orig3 + 12:00 + 14:30 conf", dict(
                windows=WIN3[:2] + ((12 * 60, 12 * 60 + 30),) + WIN3[2:] + ((14 * 60 + 30, 15 * 60),),
                max_day=5,
                extra_win_set={2, 4},
            )),
            ("6win extra conf + trades4+ conf", dict(windows=WIN6, max_day=6, extra_win_set=extra135, conf_from=3)),
        ]
    else:
        recipes = [
        ("BASE 3win 2lot", dict(windows=WIN3, max_day=3)),
        ("6win same filters", dict(windows=WIN6, max_day=6)),
        ("6winB same filters", dict(windows=WIN6B, max_day=6)),
        ("6win extras need conf", dict(windows=WIN6, max_day=6, extra_win_conf=3)),
        ("6win trades4+ need conf", dict(windows=WIN6, max_day=6, conf_from=3)),
        ("6win all need conf", dict(windows=WIN6, max_day=6, conf_from=0)),
        ("5win same filters", dict(windows=WIN5, max_day=5)),
        ("5win extras need conf", dict(windows=WIN5, max_day=5, extra_win_conf=3)),
        ("4win A 935/11/13/1430", dict(windows=WIN4A, max_day=4)),
        ("4win B spaced", dict(windows=WIN4B, max_day=4)),
        ("4win extras need conf", dict(windows=WIN4A, max_day=4, extra_win_conf=3)),
        ("6win hold45", dict(windows=WIN6, max_day=6, max_hold_min=45)),
        ("6win hold60", dict(windows=WIN6, max_day=6, max_hold_min=60)),
        ("6win hold90", dict(windows=WIN6, max_day=6, max_hold_min=90)),
        ("6win hold120", dict(windows=WIN6, max_day=6, max_hold_min=120)),
        ("6win hold180", dict(windows=WIN6, max_day=6, max_hold_min=180)),
        ("6win hold90 extras conf", dict(windows=WIN6, max_day=6, max_hold_min=90, extra_win_conf=3)),
        ("6win hold90 all conf", dict(windows=WIN6, max_day=6, max_hold_min=90, conf_from=0)),
        ("3win hold90 cap6 reenter", dict(windows=WIN3, max_day=6, max_hold_min=90, reenter_aligned=True, cooldown_min=30)),
        ("reenter after SL cd20 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, sl_reenter_only=True, cooldown_min=20)),
        ("reenter after SL cd30 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, sl_reenter_only=True, cooldown_min=30)),
        ("reenter after SL cd45 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, sl_reenter_only=True, cooldown_min=45)),
        ("reenter SL cd30 cap6 conf", dict(windows=WIN3, max_day=6, reenter_aligned=True, sl_reenter_only=True, cooldown_min=30, conf_from=0)),
        ("reenter SL cd30 cap6 t4conf", dict(windows=WIN3, max_day=6, reenter_aligned=True, sl_reenter_only=True, cooldown_min=30, conf_from=3)),
        ("aligned cd20 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=20)),
        ("aligned cd30 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=30)),
        ("aligned cd45 cap6", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=45)),
        ("aligned cd30 cap6 allconf", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=30, conf_from=0)),
        ("aligned cd45 cap6 allconf", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=45, conf_from=0)),
        ("aligned cd30 cap6 t4conf", dict(windows=WIN3, max_day=6, reenter_aligned=True, cooldown_min=30, conf_from=3)),
        ("6win 1open only", dict(windows=WIN6, max_day=6, max_open=1)),
        ("6win hold90 1open", dict(windows=WIN6, max_day=6, max_hold_min=90, max_open=1)),
        ("6win hold60 1open extras conf", dict(windows=WIN6, max_day=6, max_hold_min=60, max_open=1, extra_win_conf=3)),
        ("4win hold90", dict(windows=WIN4A, max_day=4, max_hold_min=90)),
        ("4win hold90 extras conf", dict(windows=WIN4A, max_day=4, max_hold_min=90, extra_win_conf=3)),
        ("5win hold90 extras conf", dict(windows=WIN5, max_day=5, max_hold_min=90, extra_win_conf=3)),
        ("3win hold60 reenter cd30 cap6", dict(windows=WIN3, max_day=6, max_hold_min=60, reenter_aligned=True, cooldown_min=30)),
        ("6win hold90 t4conf", dict(windows=WIN6, max_day=6, max_hold_min=90, conf_from=3)),
        ("WIN6 + cd30 no reenter", dict(windows=WIN6, max_day=6, cooldown_min=30)),
        ("WIN4A + cd30", dict(windows=WIN4A, max_day=4, cooldown_min=30)),
    ]

    for name, kw in recipes:
        trades = simulate(a, **kw)
        _, row = dump(name, trades, a)
        rows.append(row)
        if refine:
            hist_day(trades, a)

    print("\n=== PASS sorted by OOS $ ===", flush=True)
    passed = [r for r in rows if r["verdict"] == "PASS"]
    passed.sort(key=lambda r: (-r["oos_pnl"], -r["oos_pf"]))
    if not passed:
        print("NONE PASSED", flush=True)
    for r in passed:
        print(
            f"  {r['name']:<52} OOS PF {r['oos_pf']:.2f} ${r['oos_pnl']:.0f}  "
            f"IS PF {r['is_pf']:.2f} ${r['is_pnl']:.0f}  t/d {r['tpd']:.2f}",
            flush=True,
        )
    print("\nClosest to 6/day among PASS (t/d desc):", flush=True)
    by_tpd = sorted(passed, key=lambda r: -r["tpd"])
    for r in by_tpd[:8]:
        print(f"  {r['name']:<52} t/d {r['tpd']:.2f} OOS {r['oos_pf']:.2f} ${r['oos_pnl']:.0f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
