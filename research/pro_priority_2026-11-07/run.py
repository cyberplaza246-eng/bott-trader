"""Research priorities from the 2026-11-07 brief. Rules frozen before the run.

No bid/ask or aggressor-side data exists here, so absorption and delta divergence
are not tested. These are the price-only versions of the other priorities.

  ORB15     09:30-09:45 high/low, first 5-min close outside, stop at the other
            side, target 1.5R or 2R, flat 15:55. One trade.
  ORB15_F   same, plus all three filters: 60-min trend (08:30 bar vs 05:30 bar,
            both complete at or before 09:30), skip a 15-min range above the
            prior 20-day 75th percentile, no new signal at or after 11:00.
  IB        09:30-10:30 range. Acceptance = first 5-min close outside after
            10:30. Stop back at the broken edge. Target one range beyond it.
  IB_RE     after that close, wait for a retest of the edge that closes back
            outside. Same stop and target. No new signal at or after 14:00.
  FAIL_PD   wick through prior RTH high/low, close back inside, 1.5R, one/side.
  FAIL_ON   same at the overnight high/low (18:00 to 09:30).
  VWAP_HOLD last six 5-min closes on one side of VWAP, that side agrees with
            the 09:30-10:00 move, first pullback that touches VWAP and closes
            back with the trend, 1.5R, one trade, no signal at or after 14:00.
  ON_CONT   overnight 18:00->09:25 direction, enter 09:30, flat 11:30.
  ON_REV    the same, faded.

Train 2025 picks the cell inside a family. It then has to be positive in 2026 H1,
positive Jul 1-Sep 21, positive in at least 4 of 6 walk-forward folds, and still
positive in both 2026 windows at 3 ticks. Sep 22-Nov 6 is not used to pick.
Costs $4 round trip + slippage, 1 MNQ, stop before target.

The risk overlay (skip a trade risking more than $150, max 2 trades/day, stop
new trades after -$300 that day) is applied after selection. It is not used to pick.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import backtest_revamp_live_stack as bt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ET = "America/New_York"
PV, COMM, TICK = 2.0, 4.0, 0.25
TRAIN = ("2025-01-01", "2026-01-01")
VAL = ("2026-01-01", "2026-07-01")
TEST = ("2026-07-01", "2026-09-22")
HOLD = ("2026-09-22", "2026-11-07")
FOLDS = [
    ("2025-01-02", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("2025-01-02", "2025-07-01", "2025-07-01", "2025-10-01"),
    ("2025-04-01", "2025-10-01", "2025-10-01", "2026-01-01"),
    ("2025-07-01", "2026-01-01", "2026-01-01", "2026-04-01"),
    ("2025-10-01", "2026-04-01", "2026-04-01", "2026-07-01"),
    ("2026-01-01", "2026-07-01", "2026-07-01", "2026-09-22"),
]


def load() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"), usecols=["datetime", "open", "high", "low", "close", "volume"])
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index().loc["2024-11-01":]
    m = m[~m.index.duplicated(keep="last")].astype(float)
    m, dropped = bt.clean_1m(m)
    x = m.tz_convert(ET)
    g = x.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna(subset=["open"])
    y = pd.read_csv(os.path.join(ROOT, "research", "live_audit_2026-11-07", "NQ_5m_yahoo.csv"))
    y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
    y = y.set_index("datetime").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    y = y[~y.index.duplicated(keep="last")]
    y, _ = bt.clean_1m(y)
    y = y[y.index > m.index.max()].tz_convert(ET)
    g = pd.concat([g, y]).sort_index()
    g = g[~g.index.duplicated(keep="first")]
    g["mins"] = g.index.hour * 60 + g.index.minute
    g["day"] = g.index.strftime("%Y-%m-%d")
    print(f"5m {g.index[0]} .. {g.index[-1]} n={len(g):,} dropped={dropped}")
    return g


def _idx(day: str, minute: int) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}", tz=ET)


def prep(g: pd.DataFrame) -> dict:
    o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins, day = g.mins.to_numpy(int), g.day.to_numpy()
    # 60-minute closes. Bar labeled T covers [T, T+60). Complete when T+60 <= 09:30.
    h1 = g.resample("60min", label="left", closed="left").agg(close=("close", "last")).dropna()
    trend = {}
    or15 = {}
    ib = {}
    rth = {}
    on = {}
    day = np.array([str(x) for x in day])
    days = list(dict.fromkeys(day))
    loc = {(day[i], int(mins[i])): i for i in range(len(g))}
    for d in days:
        t930 = _idx(d, 570)
        done = h1[h1.index + pd.Timedelta(minutes=60) <= t930]
        if len(done) >= 4:
            trend[d] = float(np.sign(done.close.iloc[-1] - done.close.iloc[-4]))
        # 15-min OR and 60-min IB from 5-min bars
        i0 = loc.get((d, 570))
        if i0 is None:
            continue
        def span(mins_end):
            hh = ll = None
            k = i0
            while k < len(g) and day[k] == d and mins[k] < mins_end:
                hh = h[k] if hh is None else max(hh, h[k])
                ll = l[k] if ll is None else min(ll, l[k])
                k += 1
            return hh, ll, k
        rh, rl, _ = span(585)
        if rh is not None and rh > rl:
            or15[d] = (rh, rl)
        ih, il, k_ib = span(630)
        if ih is not None and ih > il:
            ib[d] = (ih, il, k_ib)
        hh = ll = cl = None
        k = i0
        while k < len(g) and day[k] == d and mins[k] <= 955:
            hh = h[k] if hh is None else max(hh, h[k])
            ll = l[k] if ll is None else min(ll, l[k])
            if mins[k] == 955:
                cl = c[k]
            k += 1
        if hh is not None:
            rth[d] = (hh, ll, cl if cl is not None else c[min(k - 1, len(g) - 1)])
    # prior-20 q75 of the 15-min range, using only earlier days
    keys = [d for d in days if d in or15]
    widths = pd.Series({d: or15[d][0] - or15[d][1] for d in keys})
    q75 = widths.shift(1).rolling(20, min_periods=20).quantile(0.75)
    # overnight leading into each RTH day: 18:00 previous calendar through 09:25
    for i, d in enumerate(keys):
        start = _idx(d, 570) - pd.Timedelta(hours=15, minutes=30)  # 18:00 prior calendar day
        # Sunday open is 18:00; a Friday->Monday gap is fine
        sl = g[(g.index >= start) & (g.index < _idx(d, 570))]
        if len(sl) < 10:
            continue
        ent = loc.get((d, 965))  # not used
        on[d] = (float(sl.high.max()), float(sl.low.min()), float(sl.open.iloc[0]), float(sl.close.iloc[-1]))
    return {"o": o, "h": h, "l": l, "c": c, "mins": mins, "day": day, "loc": loc,
            "trend": trend, "or15": or15, "q75": q75.to_dict(), "ib": ib, "rth": rth, "on": on,
            "index": g.index, "vwap_src": g}


def vwap_by_day(g: pd.DataFrame) -> np.ndarray:
    tp = ((g.high + g.low + g.close) / 3.0).to_numpy()
    vol = g.volume.to_numpy()
    vol = np.where(vol > 0, vol, 1.0)
    mins, day = g.mins.to_numpy(int), g.day.to_numpy()
    out = np.full(len(g), np.nan)
    cv = cp = 0.0
    prev = None
    for i in range(len(g)):
        if not (570 <= mins[i] <= 955):
            prev = None
            continue
        if day[i] != prev:
            cv = cp = 0.0
            prev = day[i]
        cv += tp[i] * vol[i]
        cp += vol[i]
        out[i] = cv / cp
    return out


def _trade(P, ent, side, stop, target, exit_min, slip, tag) -> dict | None:
    o, h, l = P["o"], P["h"], P["l"]
    mins, day = P["mins"], P["day"]
    if ent >= len(o) or ent < 1:
        return None
    sp = slip * TICK
    entry = o[ent] + side * sp
    if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
        return None
    if target is not None and ((side > 0 and target <= entry) or (side < 0 and target >= entry)):
        return None
    reason, exit_px = "time", None
    j = ent
    while j < len(o):
        if mins[j] >= exit_min or (j > ent and day[j] != day[ent]):
            reason, exit_px = "time", o[j] - side * sp
            break
        hit_s = l[j] <= stop if side > 0 else h[j] >= stop
        hit_t = target is not None and (h[j] >= target if side > 0 else l[j] <= target)
        if hit_s:
            raw = o[j] if (side > 0 and o[j] < stop) or (side < 0 and o[j] > stop) else stop
            reason, exit_px = "stop", raw - side * sp
            break
        if hit_t:
            raw = o[j] if (side > 0 and o[j] > target) or (side < 0 and o[j] < target) else target
            reason, exit_px = "target", raw - side * sp
            break
        j += 1
    if exit_px is None:
        return None
    risk = abs(entry - stop) * PV
    return {"day": str(day[ent]), "pnl": side * (exit_px - entry) * PV - COMM,
            "reason": reason, "dir": side, "risk": risk, "tag": tag}


def run_orb(P, slip, target_r, filtered) -> pd.DataFrame:
    out = []
    mins, day, c = P["mins"], P["day"], P["c"]
    seen = set()
    for i in range(len(c) - 1):
        d = str(day[i])
        if d in seen or mins[i] < 585:
            continue
        if d not in P["or15"]:
            continue
        rh, rl = P["or15"][d]
        if filtered:
            cap = P["q75"].get(d)
            if cap is None or not np.isfinite(cap) or (rh - rl) > cap:
                continue
            tr = P["trend"].get(d, 0.0)
            if tr == 0:
                continue
            if mins[i] >= 660:
                continue
        if c[i] > rh:
            side = 1
        elif c[i] < rl:
            side = -1
        else:
            continue
        if filtered and side != P["trend"].get(d, 0.0):
            continue
        if day[i + 1] != d or mins[i + 1] >= 955:
            continue
        entry_stop = rl if side > 0 else rh
        # target from the actual fill is applied inside after we know entry; approximate with close then redo
        rec = _trade(P, i + 1, side, entry_stop, None, 955, slip, "orb")
        if rec is None:
            seen.add(d)
            continue
        # rebuild target off the fill. _trade with target None exits only on time/we need target.
        # Recompute properly:
        sp = slip * TICK
        entry = P["o"][i + 1] + side * sp
        risk_pts = abs(entry - entry_stop)
        tgt = entry + side * target_r * risk_pts
        rec = _trade(P, i + 1, side, entry_stop, tgt, 955, slip, "orb")
        if rec:
            out.append(rec)
        seen.add(d)
    return pd.DataFrame(out)


def run_ib(P, slip, retest: bool) -> pd.DataFrame:
    out = []
    h, l, c = P["h"], P["l"], P["c"]
    mins, day = P["mins"], P["day"]
    for d, (ih, il, k0) in P["ib"].items():
        rng = ih - il
        if rng <= 0:
            continue
        broke = None
        k = k0
        while k < len(c) and day[k] == d and mins[k] < (840 if retest else 900):
            if c[k] > ih:
                broke = (k, 1, ih)
                break
            if c[k] < il:
                broke = (k, -1, il)
                break
            k += 1
        if broke is None:
            continue
        sig, side, edge = broke
        if retest:
            sig = None
            j = broke[0] + 1
            while j < len(c) and day[j] == d and mins[j] < 840:
                if side > 0 and l[j] <= edge and c[j] > edge:
                    sig = j
                    break
                if side < 0 and h[j] >= edge and c[j] < edge:
                    sig = j
                    break
                j += 1
            if sig is None:
                continue
        ent = sig + 1
        if ent >= len(c) or day[ent] != d or mins[ent] >= 955:
            continue
        stop = edge - side * TICK
        target = (ih + rng) if side > 0 else (il - rng)
        rec = _trade(P, ent, side, stop, target, 955, slip, "ib")
        if rec:
            out.append(rec)
    return pd.DataFrame(out)


def run_fail(P, slip, which: str) -> pd.DataFrame:
    out = []
    h, l, c = P["h"], P["l"], P["c"]
    mins, day = P["mins"], P["day"]
    rth_days = [d for d in dict.fromkeys(P["rth"]) ]
    prev = {rth_days[i]: rth_days[i - 1] for i in range(1, len(rth_days))}
    used = set()
    for i in range(len(c) - 1):
        d = str(day[i])
        if not (575 <= mins[i] <= 900):
            continue
        levels = []
        if which in ("pd", "both"):
            p = prev.get(d)
            if p in P["rth"]:
                levels.append(("pd", P["rth"][p][0], P["rth"][p][1]))
        if which in ("on", "both") and d in P["on"]:
            levels.append(("on", P["on"][d][0], P["on"][d][1]))
        for name, hi, lo in levels:
            if (d, name, -1) not in used and h[i] > hi and c[i] < hi:
                rec = _finish_fail(P, i, -1, h[i] + TICK, slip)
                used.add((d, name, -1))
                if rec:
                    out.append(rec)
            elif (d, name, 1) not in used and l[i] < lo and c[i] > lo:
                rec = _finish_fail(P, i, 1, l[i] - TICK, slip)
                used.add((d, name, 1))
                if rec:
                    out.append(rec)
    return pd.DataFrame(out)


def _finish_fail(P, i, side, stop, slip) -> dict | None:
    ent = i + 1
    if P["day"][ent] != P["day"][i] or P["mins"][ent] >= 955:
        return None
    entry = P["o"][ent] + side * slip * TICK
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return _trade(P, ent, side, stop, entry + side * 1.5 * risk, 955, slip, "fail")


def run_vwap(P, slip) -> pd.DataFrame:
    g = P["vwap_src"]
    vw = vwap_by_day(g)
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]
    mins, day = P["mins"], P["day"]
    open930, close1000 = {}, {}
    for i in range(len(c)):
        if mins[i] == 570:
            open930[day[i]] = o[i]
        if mins[i] == 595:
            close1000[day[i]] = c[i]
    out = []
    seen = set()
    for i in range(6, len(c) - 1):
        d = day[i]
        if d in seen or not (630 <= mins[i] < 840):
            continue
        if d not in open930 or d not in close1000 or not np.isfinite(vw[i]):
            continue
        mom = close1000[d] - open930[d]
        window = c[i - 5:i + 1]
        wvw = vw[i - 5:i + 1]
        if not np.isfinite(wvw).all():
            continue
        if mom > 0 and np.all(window > wvw) and l[i] <= vw[i] and c[i] > vw[i]:
            side = 1
            stop = min(l[i], vw[i]) - TICK
        elif mom < 0 and np.all(window < wvw) and h[i] >= vw[i] and c[i] < vw[i]:
            side = -1
            stop = max(h[i], vw[i]) + TICK
        else:
            continue
        ent = i + 1
        if day[ent] != d:
            continue
        entry = o[ent] + side * slip * TICK
        risk = abs(entry - stop)
        if risk <= 0:
            seen.add(d)
            continue
        rec = _trade(P, ent, side, stop, entry + side * 1.5 * risk, 955, slip, "vwap")
        if rec:
            out.append(rec)
        seen.add(d)
    return pd.DataFrame(out)


def run_on(P, slip, fade: bool) -> pd.DataFrame:
    out = []
    for d, (hh, ll, op, cl) in P["on"].items():
        if cl == op:
            continue
        side = -1 if fade else 1
        side *= 1 if cl > op else -1
        ent = P["loc"].get((d, 570))
        if ent is None:
            continue
        dist = float(np.clip(0.5 * abs(cl - op), 15.0, 75.0))
        entry = P["o"][ent] + side * slip * TICK
        stop = entry - side * dist
        rec = _trade(P, ent, side, stop, None, 690, slip, "on")
        if rec:
            out.append(rec)
    return pd.DataFrame(out)


def overlay(tr: pd.DataFrame) -> pd.DataFrame:
    """Skip risk > $150, max 2 trades a day, no new trade after the day is -$300."""
    if tr.empty:
        return tr
    kept = []
    day_n: dict[str, int] = {}
    day_pnl: dict[str, float] = {}
    for rec in tr.itertuples(index=False):
        d = rec.day
        if rec.risk > 150 or day_n.get(d, 0) >= 2 or day_pnl.get(d, 0.0) <= -300:
            continue
        kept.append(rec._asdict() if hasattr(rec, "_asdict") else rec)
        day_n[d] = day_n.get(d, 0) + 1
        day_pnl[d] = day_pnl.get(d, 0.0) + rec.pnl
    if not kept:
        return tr.iloc[0:0]
    return pd.DataFrame(kept)


def pack(tr: pd.DataFrame, a: str, b: str) -> dict:
    if tr is None or len(tr) == 0:
        return {"net": 0.0, "n": 0, "pf": 0.0, "dd": 0.0, "win": 0.0}
    t = tr[(tr.day >= a) & (tr.day < b)]
    if t.empty:
        return {"net": 0.0, "n": 0, "pf": 0.0, "dd": 0.0, "win": 0.0}
    pnl = t.pnl.to_numpy()
    gl = float(-pnl[pnl < 0].sum())
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"net": round(float(pnl.sum()), 1), "n": int(len(t)),
            "pf": round(float(pnl[pnl > 0].sum() / gl), 3) if gl else 0.0,
            "dd": round(dd, 1), "win": round(float((pnl > 0).mean()), 3)}


def main() -> None:
    g = load()
    P = prep(g)
    print(f"days OR {len(P['or15'])} IB {len(P['ib'])} ON {len(P['on'])} trend {len(P['trend'])}")
    builders = {
        "ORB15_1.5": lambda s: run_orb(P, s, 1.5, False),
        "ORB15_2": lambda s: run_orb(P, s, 2.0, False),
        "ORB15_2_F": lambda s: run_orb(P, s, 2.0, True),
        "ORB15_1.5_F": lambda s: run_orb(P, s, 1.5, True),
        "IB": lambda s: run_ib(P, s, False),
        "IB_RE": lambda s: run_ib(P, s, True),
        "FAIL_PD": lambda s: run_fail(P, s, "pd"),
        "FAIL_ON": lambda s: run_fail(P, s, "on"),
        "FAIL_BOTH": lambda s: run_fail(P, s, "both"),
        "VWAP_HOLD": lambda s: run_vwap(P, s),
        "ON_CONT": lambda s: run_on(P, s, False),
        "ON_REV": lambda s: run_on(P, s, True),
    }
    cells = {}
    for k, fn in builders.items():
        cells[k] = fn(1)
        print(f"  {k} {len(cells[k])} trades", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for name, tr in cells.items():
        rec = {"cell": name}
        for lab, sp in ("train", TRAIN), ("val", VAL), ("test", TEST), ("hold", HOLD):
            s = pack(tr, *sp)
            for key, val in s.items():
                rec[f"{lab}_{key}"] = val
        ov = overlay(tr)
        so = pack(ov, *TRAIN)
        sv = pack(ov, *VAL)
        st = pack(ov, *TEST)
        rec["ov_train"], rec["ov_val"], rec["ov_test"] = so["net"], sv["net"], st["net"]
        rec["ov_test_n"] = st["n"]
        rows.append(rec)
        if len(tr):
            tr.to_csv(os.path.join(HERE, "results", f"trades_{name}.csv"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, "results", "cells.csv"), index=False)
    show = ["cell", "train_net", "train_n", "train_pf", "val_net", "test_net", "hold_net", "train_dd", "ov_train", "ov_val", "ov_test"]
    print("\n" + tab[show].to_string(index=False))
    families = {
        "ORB": ["ORB15_1.5", "ORB15_2", "ORB15_2_F", "ORB15_1.5_F"],
        "IB": ["IB", "IB_RE"],
        "FAIL": ["FAIL_PD", "FAIL_ON", "FAIL_BOTH"],
        "VWAP_HOLD": ["VWAP_HOLD"],
        "OVERNIGHT": ["ON_CONT", "ON_REV"],
    }
    print("\nCHOSEN ON 2025")
    for fam, members in families.items():
        nets = []
        for ta, tb, va, vb in FOLDS:
            pick = max(members, key=lambda m: pack(cells[m], ta, tb)["net"])
            nets.append(pack(cells[pick], va, vb)["net"])
        chosen = max(members, key=lambda m: float(tab.loc[tab.cell == m, "train_net"].iloc[0]))
        row = tab[tab.cell == chosen].iloc[0]
        s3 = builders[chosen](3)
        v3, t3 = pack(s3, *VAL)["net"], pack(s3, *TEST)["net"]
        pos = sum(x > 0 for x in nets)
        ok = row.val_net > 0 and row.test_net > 0 and sum(nets) > 0 and pos >= 4 and v3 > 0 and t3 > 0
        print(f"{fam:12} {chosen:12} train {row.train_net:8} val {row.val_net:8} test {row.test_net:8} "
              f"wf {sum(nets):8.0f} {pos}/6 hold {row.hold_net:8} 3tick {v3}/{t3} "
              f"overlay val/test {row.ov_val}/{row.ov_test} candidate {ok}")


if __name__ == "__main__":
    main()
