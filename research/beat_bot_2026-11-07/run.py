"""New entry rules, frozen before looking at results. None of these were in the earlier tests.

Bar to beat: the 3R bot, about +$29,300 from Nov 2024 through Sep 21 2026, 1 MNQ.
A rule counts only if it also made money in 2010-2019 and in 2020-2022.
Dollars are MNQ: $2 per point, $4 commission, 1 tick per side.

NR7     Narrowest range of 7 days. Next day, if the open gaps out of that range,
        enter the gap and exit at the close. Stop is the other side of the range.
TOM     Long the open-to-close of the month's last day and the first 3 days.
PULL    Close above its 20-day average, then two down closes. Buy the next open,
        exit at that day's close.
INSIDE  Today sits inside the prior day's range. Next open, if it gaps out, enter
        and exit at the close.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PV, COMM, TICK = 2.0, 4.0, 0.25
BOT = 29313.0
WINDOWS = (
    ("2010s", "2010-01-01", "2020-01-01"),
    ("2020-22", "2020-01-01", "2023-01-01"),
    ("2023-24", "2023-01-01", "2025-01-01"),
    ("bot_window", "2024-11-01", "2026-09-22"),
)


def load() -> pd.DataFrame:
    raw = yf.download("NQ=F", start="2009-06-01", interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna().astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    rng = df.High - df.Low
    bad = rng > 6 * rng.rolling(20, min_periods=10).median()
    df = df.loc[~bad.fillna(False)].copy()
    print(f"{df.index.min().date()} .. {df.index.max().date()}  n={len(df)}  dropped {int(bad.fillna(False).sum())}")
    return df


def add(rows, day, entry, exit_px, stop=None):
    if stop is not None and ((exit_px - entry) * (1 if entry else 0) or True):
        # stop is a price. If the day's range crosses it we cannot know the order, so
        # the caller only passes a stop when the exit decision is already made.
        pass
    pnl = (exit_px - entry) * PV - COMM
    rows.append({"day": day, "pnl": pnl, "ret": (exit_px - entry) / entry})


def run_nr7(df):
    h, l, o, c = (df[k].to_numpy() for k in ("High", "Low", "Open", "Close"))
    rng = h - l
    rows = []
    for i in range(7, len(df) - 1):
        if rng[i] > rng[i - 6:i + 1].min() + 1e-9:
            continue
        op, hi, lo, cl = o[i + 1], h[i + 1], l[i + 1], c[i + 1]
        if op > h[i]:
            side, entry, stop = 1, op + TICK, l[i]
        elif op < l[i]:
            side, entry, stop = -1, op - TICK, h[i]
        else:
            continue
        # both sides of the stop and the close are known only as a daily bar:
        # if the stop was touched, take the stop.
        stopped = lo <= stop if side > 0 else hi >= stop
        exit_px = (stop - side * TICK) if stopped else (cl - side * TICK)
        rows.append({"day": df.index[i + 1].strftime("%Y-%m-%d"),
                     "pnl": side * (exit_px - entry) * PV - COMM,
                     "ret": side * (exit_px - entry) / entry})
    return pd.DataFrame(rows)


def run_inside(df):
    h, l, o, c = (df[k].to_numpy() for k in ("High", "Low", "Open", "Close"))
    rows = []
    for i in range(1, len(df) - 1):
        if not (h[i] < h[i - 1] and l[i] > l[i - 1]):
            continue
        op, cl = o[i + 1], c[i + 1]
        if op > h[i]:
            side, entry = 1, op + TICK
        elif op < l[i]:
            side, entry = -1, op - TICK
        else:
            continue
        exit_px = cl - side * TICK
        rows.append({"day": df.index[i + 1].strftime("%Y-%m-%d"),
                     "pnl": side * (exit_px - entry) * PV - COMM,
                     "ret": side * (exit_px - entry) / entry})
    return pd.DataFrame(rows)


def run_tom(df):
    o, c = df.Open.to_numpy(), df.Close.to_numpy()
    idx = df.index
    last = pd.Series(idx).groupby(pd.Series(idx).dt.to_period("M")).transform("max")
    last = set(pd.DatetimeIndex(last))
    rows = []
    for i in range(len(df)):
        d = idx[i]
        if d.day > 3 and d not in last:
            continue
        entry, exit_px = o[i] + TICK, c[i] - TICK
        rows.append({"day": d.strftime("%Y-%m-%d"),
                     "pnl": (exit_px - entry) * PV - COMM,
                     "ret": (exit_px - entry) / entry})
    return pd.DataFrame(rows)


def run_pull(df):
    c, o = df.Close.to_numpy(), df.Open.to_numpy()
    sma = pd.Series(c).rolling(20).mean().to_numpy()
    rows = []
    for i in range(21, len(df) - 1):
        if not (np.isfinite(sma[i]) and c[i] > sma[i] and c[i] < c[i - 1] < c[i - 2]):
            continue
        entry, exit_px = o[i + 1] + TICK, c[i + 1] - TICK
        rows.append({"day": df.index[i + 1].strftime("%Y-%m-%d"),
                     "pnl": (exit_px - entry) * PV - COMM,
                     "ret": (exit_px - entry) / entry})
    return pd.DataFrame(rows)


def run_hold(df):
    o, c = df.Open.to_numpy(), df.Close.to_numpy()
    rows = []
    for i in range(len(df)):
        entry, exit_px = o[i] + TICK, c[i] - TICK
        rows.append({"day": df.index[i].strftime("%Y-%m-%d"),
                     "pnl": (exit_px - entry) * PV - COMM,
                     "ret": (exit_px - entry) / entry})
    return pd.DataFrame(rows)


def pack(tr, a, b):
    t = tr[(tr.day >= a) & (tr.day < b)] if len(tr) else tr
    if t.empty:
        return 0.0, 0, 0.0, 0.0
    pnl = t.pnl.to_numpy()
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return round(float(pnl.sum()), 0), int(len(t)), round(float(t.ret.sum() * 100), 1), round(dd, 0)


def main():
    df = load()
    cells = {
        "NR7": run_nr7(df),
        "INSIDE": run_inside(df),
        "TOM": run_tom(df),
        "PULL": run_pull(df),
        "HOLD_EVERY_DAY": run_hold(df),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    print(f"bot bar on bot_window: +{BOT:.0f}\n")
    print(f"{'rule':16} {'2010s $ / %':>18} {'2020-22 $ / %':>18} {'2023-24 $ / %':>18} {'Nov24-Sep26 $':>16} {'DD':>8}")
    for name, tr in cells.items():
        bits = []
        scores = {}
        for lab, a, b in WINDOWS:
            net, n, pct, dd = pack(tr, a, b)
            scores[lab] = (net, pct, dd, n)
            bits.append((net, pct, dd))
        a, b, c, d = bits
        ok = scores["2010s"][0] > 0 and scores["2020-22"][0] > 0 and scores["bot_window"][0] > BOT
        print(f"{name:16} {a[0]:+8.0f} {a[1]:+5}%   {b[0]:+8.0f} {b[1]:+5}%   {c[0]:+8.0f} {c[1]:+5}%   {d[0]:+8.0f}  dd {d[2]:.0f}  n={scores['bot_window'][3]}  beats {ok}")
        tr.to_csv(os.path.join(HERE, "results", f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
