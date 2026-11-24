"""More trades from the Bollinger + RSI dip. Frozen before the run.

Daily bars, 2010-2019 / 2020-2022 / 2023-2024. 2025-2026 is not used to pick.
A looser daily rule is kept only if all three blocks are positive and it trades
more often than the saved version (20-day, 2 std, RSI under 20).

  D20_2_R20   the current rule
  D20_2_R30   RSI under 30
  D20_2_R40   RSI under 40
  D20_15_R30  band at 1.5 std, RSI under 30
  D10_2_R30   10-day band, RSI under 30

Mini versions use the same idea on shorter bars: close under the 20-bar lower
band and RSI(2) under the threshold, buy the next bar, sell when RSI(2) closes
above 65 or after 5 bars. 60-minute can hold across the session. 15-minute is
regular hours only and is flat by 16:00. Intraday splits are 2025, 2026 H1,
Jul 1-Sep 21. The weeks after Sep 21 are not used to pick.
Costs are $4 and 1 tick per side, 1 MNQ.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fresh_rules_2026-11-07"))
import run as base  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PV, COMM, TICK = base.PV, base.COMM, base.TICK
ET = "America/New_York"


def trade_daily(df, mask, rsi) -> pd.DataFrame:
    o, c = df.Open.to_numpy(), df.Close.to_numpy()
    idx = df.index
    rows, i = [], 200
    while i < len(df) - 1:
        if not mask[i]:
            i += 1
            continue
        entry = o[i + 1] + TICK
        last = min(i + 6, len(df) - 1)
        for j in range(i + 1, last + 1):
            if rsi[j] > 65 or j == last:
                rows.append({"day": idx[i + 1].strftime("%Y-%m-%d"),
                             "pnl": (c[j] - TICK - entry) * PV - COMM})
                i = j
                break
        i += 1
    return pd.DataFrame(rows)


def mask_of(close, length, k, rsi_thr):
    mid = pd.Series(close).rolling(length).mean().to_numpy()
    sd = pd.Series(close).rolling(length).std().to_numpy()
    rsi = base.rsi2(close)
    ok = np.isfinite(sd) & (close < mid - k * sd) & (rsi < rsi_thr)
    return ok, rsi


def report(name, tr, windows, need):
    bits, oks, ns = [], [], []
    for lab, a, b in windows:
        t = tr[(tr.day >= a) & (tr.day < b)] if len(tr) else tr
        pnl = t.pnl.to_numpy() if len(t) else np.array([])
        gl = -pnl[pnl < 0].sum() if len(pnl) else 0
        pf = round(float(pnl[pnl > 0].sum() / gl), 2) if gl else 0
        net = round(float(pnl.sum())) if len(pnl) else 0
        bits.append(f"{lab} {net:+.0f} n{len(t)} pf{pf}")
        oks.append(net > 0)
        ns.append(len(t))
    mark = "KEEP" if all(oks[:need]) and ns[0] > 53 else "no"
    print(f"{name:12} {mark:4} " + " | ".join(bits))
    return mark


def load_bars(minutes: int) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"), usecols=["datetime", "open", "high", "low", "close"])
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index().loc["2024-11-01":]
    m = m[~m.index.duplicated(keep="last")].astype(float)
    import backtest_revamp_live_stack as bt
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    m, _ = bt.clean_1m(m)
    x = m.tz_convert(ET)
    g = x.resample(f"{minutes}min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    y = pd.read_csv(os.path.join(ROOT, "research", "live_audit_2026-11-07", "NQ_5m_yahoo.csv"))
    y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
    y = y.set_index("datetime").sort_index()[["open", "high", "low", "close"]].astype(float)
    y, _ = bt.clean_1m(y)
    y = y[y.index > m.index.max()].tz_convert(ET)
    if minutes != 5:
        y = y.resample(f"{minutes}min", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    g = pd.concat([g, y]).sort_index()
    g = g[~g.index.duplicated(keep="first")]
    g["mins"] = g.index.hour * 60 + g.index.minute
    g["day"] = g.index.strftime("%Y-%m-%d")
    return g


def trade_intra(g, mask, rsi, rth_only: bool) -> pd.DataFrame:
    o, c = g.open.to_numpy(), g.close.to_numpy()
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    rows, i = [], 30
    n = len(g)
    while i < n - 1:
        if not mask[i] or (rth_only and not (570 <= mins[i] < 955)):
            i += 1
            continue
        ent = i + 1
        if rth_only and (day[ent] != day[i] or mins[ent] >= 955):
            i += 1
            continue
        entry = o[ent] + TICK
        last = min(ent + 4, n - 1)
        for j in range(ent, last + 1):
            session_end = rth_only and (day[j] != day[ent] or mins[j] >= 955)
            if rsi[j] > 65 or j == last or session_end:
                px = o[j] if session_end and day[j] != day[ent] else c[j]
                if session_end and day[j] == day[ent] and mins[j] >= 955:
                    px = o[j]
                rows.append({"day": str(day[ent]), "pnl": (px - TICK - entry) * PV - COMM})
                i = j
                break
        i += 1
    return pd.DataFrame(rows)


def main() -> None:
    df = base.load_daily()
    c = df.Close.to_numpy()
    daily_windows = (("2010s", "2010-01-01", "2020-01-01"), ("2020-22", "2020-01-01", "2023-01-01"),
                     ("2023-24", "2023-01-01", "2025-01-01"), ("2025-26", "2025-01-01", "2027-01-01"))
    print("DAILY  (current rule had 53 trades in the 2010s)")
    specs = [("D20_2_R20", 20, 2.0, 20), ("D20_2_R30", 20, 2.0, 30), ("D20_2_R40", 20, 2.0, 40),
             ("D20_15_R30", 20, 1.5, 30), ("D10_2_R30", 10, 2.0, 30)]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, length, k, thr in specs:
        ok, rsi = mask_of(c, length, k, thr)
        tr = trade_daily(df, ok, rsi)
        report(name, tr, daily_windows, 3)
        if len(tr):
            tr.to_csv(os.path.join(HERE, "results", f"more_{name}.csv"), index=False)

    intra_windows = (("2025", "2025-01-01", "2026-01-01"), ("2026H1", "2026-01-01", "2026-07-01"),
                     ("JulSep", "2026-07-01", "2026-09-22"), ("hold", "2026-09-22", "2027-01-01"))
    for minutes, rth, label in ((60, False, "60-MIN"), (15, True, "15-MIN RTH")):
        g = load_bars(minutes)
        print(f"\n{label} {g.index.min()} .. {g.index.max()} n={len(g)}")
        for thr in (20, 30):
            ok, rsi = mask_of(g.close.to_numpy(), 20, 2.0, thr)
            tr = trade_intra(g, ok, rsi, rth)
            report(f"M{minutes}_R{thr}", tr, intra_windows, 3)
            if len(tr):
                tr.to_csv(os.path.join(HERE, "results", f"more_M{minutes}_R{thr}.csv"), index=False)


if __name__ == "__main__":
    main()
