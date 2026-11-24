"""Pre-registered search with the $2,000 drawdown cap removed. Rules frozen before the run.

Question: is there a timing edge on MNQ beyond "the market went up", at 1 contract,
with costs, that holds up out of sample?

Ideas (not tuned on the result):
  ON     Long from 16:05 ET to 09:25 ET next session. Stops 80 / 150 / 300 points.
  ON_F   Same, stop 150, only after an up RTH day or only after a down RTH day.
  ON18   Long from 18:00 ET to 09:25, stop 150.
  MOM    At 09:30, long if prior RTH close is above the close 10 or 20 sessions ago,
         otherwise short. Flat at 15:55. Stop = 0.75 x prior RTH range, clipped to 20-120 pts.
  MOML   Same lookbacks, long only (skip the short).
  RTH    Control: long every RTH session 09:30 to 15:55, same stop rule as MOM with
         the prior-day range. This is bull-market beta, not a candidate.

A candidate must, on the cell chosen by TRAIN net only:
  validate > 0, test > 0, walk-forward total > 0, at least 4 of 6 folds > 0,
  and validate and test still > 0 at 3 ticks of slippage.
Drawdown is reported, not used as a pass/fail gate.
Holdout (2026-09-22 to 2026-11-06) is reported for every cell and is not used to pick.

An overnight cell is only called an edge if it also beats the RTH long control on
validate+test combined. Otherwise it is the same bull market, held at night.

Costs: $4 round trip, slippage per side, MNQ $2/point. Stop is filled before a
time exit when both happen on the same bar. Entry is the bar open +/- slippage.
No live or broker file is read or written.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import backtest_revamp_live_stack as bt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ET = "America/New_York"
PV = 2.0
COMM = 4.0
TICK = 0.25

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


def load_bars() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"))
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index()
    m = m[~m.index.duplicated(keep="last")][["open", "high", "low", "close", "volume"]].astype(float)
    m = m[m.index >= "2024-11-01"]
    m, dropped = bt.clean_1m(m)
    audit = os.path.join(ROOT, "research", "live_audit_2026-11-07")
    parts = [m]
    for name in ("NQ_5m_yahoo.csv", "NQ_1m_yahoo.csv"):
        y = pd.read_csv(os.path.join(audit, name))
        y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
        y = y.set_index("datetime").sort_index()
        y = y[~y.index.duplicated(keep="last")][["open", "high", "low", "close", "volume"]].astype(float)
        y, _ = bt.clean_1m(y)
        parts.append(y[y.index > m.index.max()])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    print(f"bars {out.index.min()} .. {out.index.max()}  n={len(out):,}  main_dropped={dropped}")
    return out


def clock_index(bars: pd.DataFrame) -> dict[tuple, int]:
    et = bars.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    key = list(zip(et.strftime("%Y-%m-%d"), mins))
    loc = {}
    for i, k in enumerate(key):
        loc.setdefault(k, i)  # first bar at that minute
    return loc


def sessions(bars: pd.DataFrame, loc: dict) -> pd.DataFrame:
    et = bars.index.tz_convert(ET)
    days = sorted(set(et.strftime("%Y-%m-%d")))
    rows = []
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    for d in days:
        i930 = loc.get((d, 570))
        i955 = loc.get((d, 955))
        if i930 is None or i955 is None or i955 <= i930:
            continue
        # RTH range from 09:30 through the 15:55 bar
        sl = slice(i930, i955 + 1)
        rows.append({
            "day": d,
            "i930": i930,
            "i955": i955,
            "i1605": loc.get((d, 965)),
            "i1800": loc.get((d, 18 * 60)),
            "i925": loc.get((d, 565)),
            "rth_open": o[i930],
            "rth_close": c[i955],
            "rth_high": float(h[sl].max()),
            "rth_low": float(l[sl].min()),
        })
    s = pd.DataFrame(rows)
    s["prev_close"] = s["rth_close"].shift(1)
    s["prev_range"] = (s["rth_high"] - s["rth_low"]).shift(1)
    s["rth_up"] = s["rth_close"] > s["rth_open"]
    return s


def _fill(px: float, side: int, slip_ticks: int) -> float:
    """side +1 means we are buying, -1 selling."""
    return px + side * slip_ticks * TICK


def run_overnight(bars, sess, entry_col: str, stop_pts: float, slip: int, when: str) -> pd.DataFrame:
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    days = sess["day"].tolist()
    nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    by_day = sess.set_index("day")
    trades = []
    for rec in sess.itertuples(index=False):
        raw_ent = getattr(rec, entry_col)
        if raw_ent is None or pd.isna(raw_ent):
            continue
        nd = nxt.get(rec.day)
        if nd is None:
            continue
        exit_i = by_day.at[nd, "i925"]
        if exit_i is None or pd.isna(exit_i):
            continue
        ent_i = int(raw_ent)
        exit_i = int(exit_i)
        if exit_i <= ent_i:
            continue
        if (bars.index[exit_i] - bars.index[ent_i]) > pd.Timedelta(days=5):
            continue
        up = bool(rec.rth_up) if rec.rth_close == rec.rth_close else False
        if when == "up" and not up:
            continue
        if when == "down" and up:
            continue
        if when != "all" and not (rec.rth_close == rec.rth_close):
            continue
        entry = _fill(o[ent_i], +1, slip)
        stop = entry - stop_pts
        reason, exit_px, exit_at = "time", _fill(o[exit_i], -1, slip), exit_i
        for j in range(ent_i, exit_i + 1):
            if l[j] <= stop:
                # gap through: fill at the open, worse of open and stop, then slippage
                raw = o[j] if o[j] < stop else stop
                reason, exit_px, exit_at = "stop", _fill(raw, -1, slip), j
                break
        pnl = (exit_px - entry) * PV - COMM
        trades.append({
            "day": rec.day, "exit_day": bars.index[exit_at].tz_convert(ET).strftime("%Y-%m-%d"),
            "entry": entry, "exit": exit_px, "pnl": pnl, "reason": reason, "dir": 1,
            "entry_i": ent_i, "exit_i": exit_at, "stop": stop,
        })
    return pd.DataFrame(trades)


def run_mom(bars, sess, lookback: int, slip: int, long_only: bool, rth_control: bool) -> pd.DataFrame:
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    closes = sess["rth_close"].to_numpy()
    trades = []
    for i, rec in enumerate(sess.itertuples(index=False)):
        if i < lookback + 1:
            continue
        if not np.isfinite(rec.prev_range) or rec.prev_range <= 0:
            continue
        if rth_control:
            d = 1
        else:
            moved = closes[i - 1] - closes[i - 1 - lookback]
            if moved == 0 or not np.isfinite(moved):
                continue
            d = 1 if moved > 0 else -1
            if long_only and d < 0:
                continue
        ent_i, exit_i = int(rec.i930), int(rec.i955)
        entry = _fill(o[ent_i], d, slip)
        dist = float(np.clip(0.75 * rec.prev_range, 20.0, 120.0))
        stop = entry - d * dist
        reason, exit_px, exit_at = "time", _fill(o[exit_i], -d, slip), exit_i
        for j in range(ent_i, exit_i + 1):
            hit = l[j] <= stop if d > 0 else h[j] >= stop
            if hit:
                if d > 0:
                    raw = o[j] if o[j] < stop else stop
                    exit_px = _fill(raw, -1, slip)
                else:
                    raw = o[j] if o[j] > stop else stop
                    exit_px = _fill(raw, +1, slip)
                reason, exit_at = "stop", j
                break
        pnl = d * (exit_px - entry) * PV - COMM
        trades.append({
            "day": rec.day, "exit_day": rec.day, "entry": entry, "exit": exit_px,
            "pnl": pnl, "reason": reason, "dir": d, "entry_i": ent_i, "exit_i": exit_at, "stop": stop,
        })
    return pd.DataFrame(trades)


def mtm_dd(bars, trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    realized = 0.0
    peak = 0.0
    worst = 0.0
    for rec in trades.itertuples(index=False):
        a, b = int(rec.entry_i), int(rec.exit_i)
        d = int(rec.dir)
        for j in range(a, b + 1):
            adv = l[j] if d > 0 else h[j]
            mark = realized + (d * (adv - rec.entry) * PV - COMM)
            worst = min(worst, mark - peak)
            mark_c = realized + (d * (c[j] - rec.entry) * PV - COMM)
            peak = max(peak, mark_c)
        realized += rec.pnl
        peak = max(peak, realized)
        worst = min(worst, realized - peak)
    return float(-worst)


def in_split(trades: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    if trades.empty:
        return trades
    # score by the session that was known when the trade was opened
    return trades[(trades.day >= a) & (trades.day < b)]


def summarize(trades: pd.DataFrame, bars, a: str, b: str) -> dict:
    t = in_split(trades, a, b)
    if t.empty:
        return {"net": 0.0, "trades": 0, "pf": 0.0, "dd": 0.0, "win": 0.0, "t": 0.0}
    pnl = t.pnl.to_numpy()
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    days = pd.bdate_range(a, pd.Timestamp(b) - pd.Timedelta(days=1))
    by = t.groupby("day").pnl.sum()
    daily = pd.Series(0.0, index=days.strftime("%Y-%m-%d"))
    daily.loc[daily.index.intersection(by.index)] = by
    sd = float(daily.std(ddof=1))
    tstat = float(daily.mean() / sd * np.sqrt(len(daily))) if sd > 0 else 0.0
    return {
        "net": round(float(pnl.sum()), 1),
        "trades": int(len(t)),
        "pf": round(float(gp / gl), 3) if gl > 0 else 0.0,
        "dd": round(mtm_dd(bars, t), 1),
        "win": round(float((pnl > 0).mean()), 3),
        "t": round(tstat, 2),
    }


def main() -> None:
    bars = load_bars()
    loc = clock_index(bars)
    sess = sessions(bars, loc)
    print(f"rth sessions {sess.day.iloc[0]} .. {sess.day.iloc[-1]}  n={len(sess)}")

    cells = {
        "ON_s80": run_overnight(bars, sess, "i1605", 80, 1, "all"),
        "ON_s150": run_overnight(bars, sess, "i1605", 150, 1, "all"),
        "ON_s300": run_overnight(bars, sess, "i1605", 300, 1, "all"),
        "ON_s150_up": run_overnight(bars, sess, "i1605", 150, 1, "up"),
        "ON_s150_down": run_overnight(bars, sess, "i1605", 150, 1, "down"),
        "ON18_s150": run_overnight(bars, sess, "i1800", 150, 1, "all"),
        "MOM10": run_mom(bars, sess, 10, 1, False, False),
        "MOM20": run_mom(bars, sess, 20, 1, False, False),
        "MOML10": run_mom(bars, sess, 10, 1, True, False),
        "MOML20": run_mom(bars, sess, 20, 1, True, False),
        "RTH_LONG": run_mom(bars, sess, 20, 1, True, True),
    }
    slip3 = {
        "ON_s80": run_overnight(bars, sess, "i1605", 80, 3, "all"),
        "ON_s150": run_overnight(bars, sess, "i1605", 150, 3, "all"),
        "ON_s300": run_overnight(bars, sess, "i1605", 300, 3, "all"),
        "ON_s150_up": run_overnight(bars, sess, "i1605", 150, 3, "up"),
        "ON_s150_down": run_overnight(bars, sess, "i1605", 150, 3, "down"),
        "ON18_s150": run_overnight(bars, sess, "i1800", 150, 3, "all"),
        "MOM10": run_mom(bars, sess, 10, 3, False, False),
        "MOM20": run_mom(bars, sess, 20, 3, False, False),
        "MOML10": run_mom(bars, sess, 10, 3, True, False),
        "MOML20": run_mom(bars, sess, 20, 3, True, False),
        "RTH_LONG": run_mom(bars, sess, 20, 3, True, True),
    }
    families = {
        "ON_STOP": ["ON_s80", "ON_s150", "ON_s300"],
        "ON_FILTER": ["ON_s150", "ON_s150_up", "ON_s150_down"],
        "ON18": ["ON18_s150"],
        "MOM": ["MOM10", "MOM20"],
        "MOML": ["MOML10", "MOML20"],
    }

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for name, tr in cells.items():
        rec = {"cell": name}
        for label, (a, b) in ("train", TRAIN), ("validate", VAL), ("test", TEST), ("holdout", HOLD):
            s = summarize(tr, bars, a, b)
            for k, v in s.items():
                rec[f"{label}_{k}"] = v
        s3v = summarize(slip3[name], bars, *VAL)
        s3t = summarize(slip3[name], bars, *TEST)
        rec["validate_net_3t"] = s3v["net"]
        rec["test_net_3t"] = s3t["net"]
        rows.append(rec)
        tr.assign(cell=name).to_csv(os.path.join(HERE, "results", f"trades_{name}.csv"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, "results", "cells.csv"), index=False)

    wf_rows = []
    summary = []
    rth_vt = summarize(cells["RTH_LONG"], bars, VAL[0], TEST[1])
    for fam, members in families.items():
        fold_nets = []
        for i, (ta, tb, va, vb) in enumerate(FOLDS, 1):
            scored = [(m, summarize(cells[m], bars, ta, tb)["net"]) for m in members]
            pick = max(scored, key=lambda kv: kv[1])[0]
            te = summarize(cells[pick], bars, va, vb)
            fold_nets.append(te["net"])
            wf_rows.append({"family": fam, "fold": i, "picked": pick, "test_net": te["net"], "test_trades": te["trades"], "test_dd": te["dd"]})
        # cell chosen on the full train year
        pick = max(members, key=lambda m: summarize(cells[m], bars, *TRAIN)["net"])
        base = tab[tab.cell == pick].iloc[0]
        vt = summarize(cells[pick], bars, VAL[0], TEST[1])
        pos = sum(x > 0 for x in fold_nets)
        beats_rth = bool(vt["net"] > rth_vt["net"])
        ok = (
            base.validate_net > 0 and base.test_net > 0 and sum(fold_nets) > 0 and pos >= 4
            and base.validate_net_3t > 0 and base.test_net_3t > 0
        )
        # long-only overnight / long-only momentum must beat holding the cash session
        if pick.startswith("ON") or pick.startswith("MOML"):
            ok = ok and beats_rth
        summary.append({
            "family": fam, "picked": pick, "train": base.train_net, "validate": base.validate_net,
            "test": base.test_net, "wf": round(sum(fold_nets), 1), "folds_pos": f"{pos}/6",
            "holdout": base.holdout_net, "dd_validate_test": vt["dd"],
            "val_test_net": vt["net"], "rth_control_val_test": rth_vt["net"],
            "beats_rth_control": beats_rth, "candidate": bool(ok),
            "pf_train": base.train_pf, "t_train": base.train_t, "t_validate": base.validate_t,
        })
    pd.DataFrame(wf_rows).to_csv(os.path.join(HERE, "results", "walkforward.csv"), index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(HERE, "results", "summary.csv"), index=False)
    print("\nCELLS")
    print(tab[["cell", "train_net", "validate_net", "test_net", "holdout_net", "train_dd", "validate_dd", "test_dd", "holdout_dd", "train_trades", "train_pf"]].to_string(index=False))
    print("\nFAMILIES")
    print(sm.to_string(index=False))
    print("\nRTH control", rth_vt)
    (HERE and open(os.path.join(HERE, "results", "summary.json"), "w")).write(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
