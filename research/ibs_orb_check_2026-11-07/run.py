"""Frozen check of two rules from the 2026-11-07 writeup. No threshold is chosen after seeing results.

IBS (the claim: daily IBS extreme, then the overnight drift):
  IBS = (RTH close - RTH low) / (RTH high - RTH low), known at the 15:55 ET close.
  IBS20 / IBS30: long at 16:05 when IBS <= 0.20 / 0.30, flat at 09:25, 300-pt disaster stop.
  IBS20_DAY: same entry, flat at the next 15:55 instead of 09:25.
  IBS80_SHORT: short at 16:05 when IBS >= 0.80, flat at 09:25, 300-pt stop.
  ON_ALL: long every 16:05, same stop and exit, so IBS has to beat "just be long".

15-minute NY ORB, as written (one cell, no grid):
  Range = high/low of 09:30-09:45 ET. First 5-minute close outside that range
  from 09:45 to 11:25 enters at the next 5-minute open. Stop = the other side
  of the range. Flat at 11:30. One trade a day.

Costs $4 round trip + slippage, 1 MNQ, stop before time exit. Train 2025,
validate 2026 H1, test Jul 1-Sep 21, holdout reported and not used to pick.
A long-only cell must also beat ON_ALL on validate+test.
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
TRAIN, VAL, TEST, HOLD = ("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-07-01"), ("2026-07-01", "2026-09-22"), ("2026-09-22", "2026-11-07")
FOLDS = [
    ("2025-01-02", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("2025-01-02", "2025-07-01", "2025-07-01", "2025-10-01"),
    ("2025-04-01", "2025-10-01", "2025-10-01", "2026-01-01"),
    ("2025-07-01", "2026-01-01", "2026-01-01", "2026-04-01"),
    ("2025-10-01", "2026-04-01", "2026-04-01", "2026-07-01"),
    ("2026-01-01", "2026-07-01", "2026-07-01", "2026-09-22"),
]


def load_bars() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"))
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index()
    m = m.loc["2024-11-01":, ["open", "high", "low", "close", "volume"]].astype(float)
    m = m[~m.index.duplicated(keep="last")]
    m, dropped = bt.clean_1m(m)
    audit = os.path.join(ROOT, "research", "live_audit_2026-11-07")
    parts = [m]
    for name in ("NQ_5m_yahoo.csv", "NQ_1m_yahoo.csv"):
        y = pd.read_csv(os.path.join(audit, name))
        y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
        y = y.set_index("datetime").sort_index()
        y = y.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
        y = y[~y.index.duplicated(keep="last")]
        y, _ = bt.clean_1m(y)
        parts.append(y[y.index > m.index.max()])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    print(f"bars {out.index.min()} .. {out.index.max()} n={len(out):,} dropped={dropped}")
    return out


def sessions(bars: pd.DataFrame) -> pd.DataFrame:
    et = bars.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    loc = {}
    for i, k in enumerate(zip(et.strftime("%Y-%m-%d"), mins)):
        loc.setdefault(k, i)
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    rows = []
    for d in sorted(set(et.strftime("%Y-%m-%d"))):
        i930, i955 = loc.get((d, 570)), loc.get((d, 955))
        if i930 is None or i955 is None or i955 <= i930:
            continue
        sl = slice(i930, i955 + 1)
        hi, lo, cl = float(h[sl].max()), float(l[sl].min()), float(c[i955])
        ibs = (cl - lo) / (hi - lo) if hi > lo else np.nan
        rows.append({"day": d, "i930": i930, "i955": i955, "i1605": loc.get((d, 965)),
                     "i925": loc.get((d, 565)), "ibs": ibs})
    return pd.DataFrame(rows)


def _px(raw: float, side: int, slip: int) -> float:
    return raw + side * slip * TICK


def hold_trade(bars, ent_i: int, exit_i: int, d: int, slip: int, stop_pts: float, day: str) -> dict | None:
    if exit_i <= ent_i or (bars.index[exit_i] - bars.index[ent_i]) > pd.Timedelta(days=5):
        return None
    o, h, l = (bars[k].to_numpy() for k in ("open", "high", "low"))
    entry = _px(o[ent_i], d, slip)
    stop = entry - d * stop_pts
    reason, exit_px, exit_at = "time", _px(o[exit_i], -d, slip), exit_i
    for j in range(ent_i, exit_i + 1):
        hit = l[j] <= stop if d > 0 else h[j] >= stop
        if not hit:
            continue
        raw = o[j] if (d > 0 and o[j] < stop) or (d < 0 and o[j] > stop) else stop
        reason, exit_px, exit_at = "stop", _px(raw, -d, slip), j
        break
    return {"day": day, "pnl": d * (exit_px - entry) * PV - COMM, "reason": reason,
            "entry_i": ent_i, "exit_i": exit_at, "entry": entry, "dir": d, "stop": stop}


def run_ibs(bars, sess, thr: float, side: int, slip: int, exit_at: str) -> pd.DataFrame:
    days = sess.day.tolist()
    nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    by = sess.set_index("day")
    out = []
    for rec in sess.itertuples(index=False):
        if not np.isfinite(rec.ibs) or pd.isna(rec.i1605):
            continue
        if side > 0 and rec.ibs > thr:
            continue
        if side < 0 and rec.ibs < thr:
            continue
        nd = nxt.get(rec.day)
        if nd is None:
            continue
        exit_i = by.at[nd, "i925" if exit_at == "open" else "i955"]
        if pd.isna(exit_i):
            continue
        tr = hold_trade(bars, int(rec.i1605), int(exit_i), side, slip, 300.0, rec.day)
        if tr:
            out.append(tr)
    return pd.DataFrame(out)


def run_on_all(bars, sess, slip: int) -> pd.DataFrame:
    days = sess.day.tolist()
    nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    by = sess.set_index("day")
    out = []
    for rec in sess.itertuples(index=False):
        nd = nxt.get(rec.day)
        if nd is None or pd.isna(rec.i1605) or pd.isna(by.at[nd, "i925"]):
            continue
        tr = hold_trade(bars, int(rec.i1605), int(by.at[nd, "i925"]), 1, slip, 300.0, rec.day)
        if tr:
            out.append(tr)
    return pd.DataFrame(out)


def run_orb(bars, sess, slip: int) -> pd.DataFrame:
    et = bars.index.tz_convert(ET)
    m5 = bars.tz_convert(ET).resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    m5 = m5[m5.index.tz_convert(ET).indexer_at_time] if False else m5
    # map each 5-min bar to the first 1m bar at that timestamp, for stop path
    idx = bars.index
    out = []
    for rec in sess.itertuples(index=False):
        day = pd.Timestamp(rec.day, tz=ET)
        win = m5[(m5.index >= day + pd.Timedelta(hours=9, minutes=30)) &
                 (m5.index < day + pd.Timedelta(hours=11, minutes=30))]
        rng = win[(win.index >= day + pd.Timedelta(hours=9, minutes=30)) &
                  (win.index < day + pd.Timedelta(hours=9, minutes=45))]
        if len(rng) < 3:
            continue
        rh, rl = float(rng.high.max()), float(rng.low.min())
        if rh <= rl:
            continue
        later = win[win.index >= day + pd.Timedelta(hours=9, minutes=45)]
        sig = None
        for ts, row in later.iterrows():
            if row.close > rh:
                sig = (ts, 1)
                break
            if row.close < rl:
                sig = (ts, -1)
                break
        if sig is None:
            continue
        ent_ts = sig[0] + pd.Timedelta(minutes=5)
        if ent_ts.tz_convert(ET) >= day + pd.Timedelta(hours=11, minutes=30):
            continue
        flat_ts = day + pd.Timedelta(hours=11, minutes=30)
        ent_i = idx.searchsorted(ent_ts.tz_convert("UTC"))
        exit_i = idx.searchsorted(flat_ts.tz_convert("UTC"))
        if ent_i >= len(idx) or exit_i >= len(idx) or exit_i <= ent_i:
            continue
        d = sig[1]
        o, h, l = (bars[k].to_numpy() for k in ("open", "high", "low"))
        entry = _px(float(o[ent_i]), d, slip)
        stop = rl if d > 0 else rh
        # a stop already through the fill is not a trade
        if (d > 0 and stop >= entry) or (d < 0 and stop <= entry):
            continue
        reason, exit_px, exit_at = "time", _px(float(o[exit_i]), -d, slip), exit_i
        for j in range(ent_i, exit_i + 1):
            hit = l[j] <= stop if d > 0 else h[j] >= stop
            if not hit:
                continue
            raw = o[j] if (d > 0 and o[j] < stop) or (d < 0 and o[j] > stop) else stop
            reason, exit_px, exit_at = "stop", _px(float(raw), -d, slip), j
            break
        out.append({"day": rec.day, "pnl": d * (exit_px - entry) * PV - COMM, "reason": reason,
                    "entry_i": ent_i, "exit_i": exit_at, "entry": entry, "dir": d, "stop": stop})
    return pd.DataFrame(out)


def mtm_dd(bars, trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    h, l, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
    realized, peak, worst = 0.0, 0.0, 0.0
    for rec in trades.itertuples(index=False):
        d = int(rec.dir)
        for j in range(int(rec.entry_i), int(rec.exit_i) + 1):
            adv = l[j] if d > 0 else h[j]
            worst = min(worst, realized + (d * (adv - rec.entry) * PV - COMM) - peak)
            peak = max(peak, realized + (d * (c[j] - rec.entry) * PV - COMM))
        realized += rec.pnl
        peak = max(peak, realized)
        worst = min(worst, realized - peak)
    return float(-worst)


def pack(trades, bars, a, b) -> dict:
    t = trades[(trades.day >= a) & (trades.day < b)] if len(trades) else trades
    if t.empty:
        return {"net": 0.0, "n": 0, "pf": 0.0, "dd": 0.0, "win": 0.0}
    pnl = t.pnl.to_numpy()
    gl = -pnl[pnl < 0].sum()
    return {"net": round(float(pnl.sum()), 1), "n": int(len(t)),
            "pf": round(float(pnl[pnl > 0].sum() / gl), 3) if gl else 0.0,
            "dd": round(mtm_dd(bars, t), 1), "win": round(float((pnl > 0).mean()), 3)}


def main() -> None:
    bars = load_bars()
    sess = sessions(bars)
    print(f"sessions {sess.day.iloc[0]} .. {sess.day.iloc[-1]} n={len(sess)} ibs median {sess.ibs.median():.2f}")
    cells = {
        "IBS20_ON": run_ibs(bars, sess, 0.20, 1, 1, "open"),
        "IBS30_ON": run_ibs(bars, sess, 0.30, 1, 1, "open"),
        "IBS20_DAY": run_ibs(bars, sess, 0.20, 1, 1, "close"),
        "IBS80_SHORT": run_ibs(bars, sess, 0.80, -1, 1, "open"),
        "ON_ALL": run_on_all(bars, sess, 1),
        "ORB15_1130": run_orb(bars, sess, 1),
    }
    slip3 = {k: fn for k, fn in {
        "IBS20_ON": run_ibs(bars, sess, 0.20, 1, 3, "open"),
        "IBS30_ON": run_ibs(bars, sess, 0.30, 1, 3, "open"),
        "ORB15_1130": run_orb(bars, sess, 3),
    }.items()}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for name, tr in cells.items():
        rec = {"cell": name}
        for lab, sp in ("train", TRAIN), ("val", VAL), ("test", TEST), ("hold", HOLD):
            s = pack(tr, bars, *sp)
            rec[f"{lab}_net"], rec[f"{lab}_n"], rec[f"{lab}_pf"], rec[f"{lab}_dd"], rec[f"{lab}_win"] = (
                s["net"], s["n"], s["pf"], s["dd"], s["win"])
        rows.append(rec)
        if len(tr):
            tr.to_csv(os.path.join(HERE, "results", f"trades_{name}.csv"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, "results", "cells.csv"), index=False)
    print(tab.to_string(index=False))
    # walk-forward only over the two overnight IBS thresholds; pick on train fold
    members = ["IBS20_ON", "IBS30_ON"]
    folds = []
    for i, (ta, tb, va, vb) in enumerate(FOLDS, 1):
        pick = max(members, key=lambda m: pack(cells[m], bars, ta, tb)["net"])
        te = pack(cells[pick], bars, va, vb)
        folds.append((i, pick, te["net"]))
        print(f"fold {i} {pick} test {te['net']}")
    base = tab[tab.cell == "ON_ALL"].iloc[0]
    ibs_pick = max(members, key=lambda m: tab[tab.cell == m].iloc[0].train_net)
    p = tab[tab.cell == ibs_pick].iloc[0]
    vt_ibs = pack(cells[ibs_pick], bars, VAL[0], TEST[1])["net"]
    vt_on = pack(cells["ON_ALL"], bars, VAL[0], TEST[1])["net"]
    pos = sum(x[2] > 0 for x in folds)
    print(f"picked {ibs_pick} wf {sum(x[2] for x in folds):.0f} {pos}/6  val+test {vt_ibs} vs always-long {vt_on}")
    print("3tick", {k: (pack(v, bars, *VAL)["net"], pack(v, bars, *TEST)["net"]) for k, v in slip3.items()})
    print(f"ON_ALL train {base.train_net} val {base.val_net} test {base.test_net} hold {base.hold_net}")


if __name__ == "__main__":
    main()
