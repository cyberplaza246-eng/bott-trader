"""Bollinger dip, rules frozen before the run.

Buy the next open after a close under the lower band. Worth a second look only if
2010-2019, 2020-2022, and 2023-2024 are all positive. 2025-2026 is not used to pick.

  BB20        20-day, 2 standard deviations. Sell at the 20-day average, or after 3 days.
  BB20_UP     Same, and the close is above the 200-day average.
  BB20_DEEP   20-day, 2.5 standard deviations. Same exit as BB20.
  BB20_RSI    20-day, 2 std, and RSI(2) < 20. Sell when RSI(2) > 65, or after 5 days.
  BB20_NEXT   20-day, 2 std. Sell the next close. No other exit.
  BB10        10-day, 2 std. Sell at the 10-day average, or after 3 days.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fresh_rules_2026-11-07"))
import run as base  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PV, COMM, TICK = base.PV, base.COMM, base.TICK
WINDOWS = (("2010s", "2010-01-01", "2020-01-01"), ("2020", "2020-01-01", "2023-01-01"),
           ("2023", "2023-01-01", "2025-01-01"), ("2025", "2025-01-01", "2027-01-01"))


def run(df, mask, mid, rsi, mode: str) -> pd.DataFrame:
    o, c = df.Open.to_numpy(), df.Close.to_numpy()
    idx = df.index
    rows, i = [], 200
    while i < len(df) - 1:
        if not mask[i]:
            i += 1
            continue
        entry = o[i + 1] + TICK
        last = min(i + (6 if mode == "rsi" else 4), len(df) - 1)
        for j in range(i + 1, last + 1):
            if mode == "next":
                done = True
            elif mode == "rsi":
                done = rsi[j] > 65 or j == last
            else:
                done = c[j] >= mid[j] or j == last
            if done:
                rows.append({"day": idx[i + 1].strftime("%Y-%m-%d"),
                             "pnl": (c[j] - TICK - entry) * PV - COMM,
                             "ret": (c[j] - TICK - entry) / entry})
                i = j
                break
        i += 1
    return pd.DataFrame(rows)


def main() -> None:
    df = base.load_daily()
    c, sd, mid, sma, rsi = (df[k].to_numpy() for k in ("Close", "sd", "mid", "sma", "rsi"))
    mid10 = df.Close.rolling(10).mean().to_numpy()
    sd10 = df.Close.rolling(10).std().to_numpy()
    up = np.isfinite(sma) & (c > sma)
    under = np.isfinite(sd) & (c < mid - 2 * sd)
    specs = {
        "BB20": (under, mid, rsi, "mid"),
        "BB20_UP": (under & up, mid, rsi, "mid"),
        "BB20_DEEP": (np.isfinite(sd) & (c < mid - 2.5 * sd), mid, rsi, "mid"),
        "BB20_RSI": (under & (rsi < 20), mid, rsi, "rsi"),
        "BB20_NEXT": (under, mid, rsi, "next"),
        "BB10": (np.isfinite(sd10) & (c < mid10 - 2 * sd10), mid10, rsi, "mid"),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, (mask, m, r, mode) in specs.items():
        tr = run(df, mask, m, r, mode)
        bits = []
        ok = []
        for lab, a, b in WINDOWS:
            t = tr[(tr.day >= a) & (tr.day < b)] if len(tr) else tr
            pnl = t.pnl.to_numpy() if len(t) else np.array([])
            gl = -pnl[pnl < 0].sum() if len(pnl) else 0
            pf = round(float(pnl[pnl > 0].sum() / gl), 2) if gl else 0
            net = round(float(pnl.sum()), 0) if len(pnl) else 0
            bits.append(f"{lab} {net:+.0f} n{len(t)} pf{pf}")
            ok.append(net > 0)
        years = tr.groupby(tr.day.str[:4]).pnl.sum() if len(tr) else pd.Series(dtype=float)
        red = int((years < 0).sum())
        top3 = (round(float(tr.pnl.nlargest(3).sum() / tr.pnl.sum()), 2) if len(tr) and tr.pnl.sum() else 0)
        y22 = round(float(years.get("2022", 0)), 0)
        mark = "KEEP" if all(ok[:3]) else "no"
        print(f"{name:10} {mark:4} red_years {red} top3 {top3} 2022 {y22:+.0f} | " + " | ".join(bits))
        if len(tr):
            tr.to_csv(os.path.join(HERE, "results", f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
