"""Build on the three dip rules. Chosen before the run. Same exit for every row so
only the entry changes: buy the next open, sell when RSI(2) closes above 65, or after 5 days.

  BASE     RSI(2) < 10 and close above the 200-day average. This is the one to beat.
  DEEPER   BASE, and today closed down.
  BOTH     BASE, and close under the lower Bollinger band.
  DOWN_UP  Three lower closes, and close above the 200-day average.
  DOWN_RSI Three lower closes, and RSI(2) < 15.
  EITHER   RSI(2) < 10 or close under the band, and close above the 200-day average.
  TWO      At least two of {RSI<10, three down closes, under the band}, and above the 200-day.

Worth keeping only if 2010-2019, 2020-2022, and 2023-2024 are positive, and the
2010-2019 result beats BASE. 2025-2026 is not used to pick.
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


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c = df.Close.to_numpy()
    r, sma, mid, sd = df.rsi.to_numpy(), df.sma.to_numpy(), df.mid.to_numpy(), df.sd.to_numpy()
    up = np.isfinite(sma) & (c > sma)
    rsi = (r < 10) & up
    down3 = np.zeros(len(c), dtype=bool)
    down3[3:] = (c[3:] < c[2:-1]) & (c[2:-1] < c[1:-2]) & (c[1:-2] < c[:-3])
    band = np.isfinite(sd) & (c < mid - 2 * sd)
    down_close = np.zeros(len(c), dtype=bool)
    down_close[1:] = c[1:] < c[:-1]
    two = (rsi.astype(int) + down3.astype(int) + (band & up).astype(int)) >= 2
    return {
        "BASE": rsi,
        "DEEPER": rsi & down_close,
        "BOTH": rsi & band,
        "DOWN_UP": down3 & up,
        "DOWN_RSI": down3 & (r < 15),
        "EITHER": (rsi | (band & up)),
        "TWO": two & up,
    }


def trade(df, mask) -> pd.DataFrame:
    o, c, r = df.Open.to_numpy(), df.Close.to_numpy(), df.rsi.to_numpy()
    idx = df.index
    rows, i = [], 200
    while i < len(df) - 1:
        if mask[i]:
            entry = o[i + 1] + TICK
            exit_i = min(i + 6, len(df) - 1)
            for j in range(i + 1, exit_i + 1):
                if r[j] > 65 or j == exit_i:
                    rows.append({
                        "day": idx[i + 1].strftime("%Y-%m-%d"),
                        "pnl": (c[j] - TICK - entry) * PV - COMM,
                        "ret": (c[j] - TICK - entry) / entry,
                    })
                    i = j
                    break
        i += 1
    return pd.DataFrame(rows)


def summarize(tr: pd.DataFrame, a: str, b: str) -> dict:
    t = tr[(tr.day >= a) & (tr.day < b)] if len(tr) else tr
    if t.empty:
        return {"net": 0, "n": 0, "pf": 0, "pct": 0, "red_years": 0}
    pnl = t.pnl.to_numpy()
    gl = -pnl[pnl < 0].sum()
    years = t.groupby(t.day.str[:4]).pnl.sum()
    return {
        "net": round(float(pnl.sum())),
        "n": int(len(t)),
        "pf": round(float(pnl[pnl > 0].sum() / gl), 2) if gl else 0,
        "pct": round(float(t.ret.sum() * 100), 1),
        "red_years": int((years < 0).sum()),
    }


def main() -> None:
    df = base.load_daily()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    windows = (("2010s", "2010-01-01", "2020-01-01"), ("2020", "2020-01-01", "2023-01-01"),
               ("2023", "2023-01-01", "2025-01-01"), ("2025", "2025-01-01", "2027-01-01"))
    rows = []
    built = {}
    for name, mask in signals(df).items():
        tr = trade(df, mask)
        built[name] = tr
        rec = {"rule": name}
        for lab, a, b in windows:
            s = summarize(tr, a, b)
            for k, v in s.items():
                rec[f"{lab}_{k}"] = v
        tr.to_csv(os.path.join(HERE, "results", f"{name}.csv"), index=False)
        rows.append(rec)
        print(f"{name:8} 2010s {rec['2010s_net']:+7} n{rec['2010s_n']:3} pf{rec['2010s_pf']:4} {rec['2010s_pct']:+5}% red{rec['2010s_red_years']} | "
              f"2020-22 {rec['2020_net']:+7} pf{rec['2020_pf']:4} | 2023-24 {rec['2023_net']:+7} pf{rec['2023_pf']:4} | "
              f"2025-26 {rec['2025_net']:+7}")
    tab = pd.DataFrame(rows)
    base_pct = float(tab.loc[tab.rule == "BASE", "2010s_pct"].iloc[0])
    tab["beats_base_2010s"] = tab["2010s_pct"] > base_pct
    tab["holds"] = (tab["2010s_net"] > 0) & (tab["2020_net"] > 0) & (tab["2023_net"] > 0) & tab["beats_base_2010s"]
    tab.to_csv(os.path.join(HERE, "results", "summary.csv"), index=False)
    print("\nbeats the plain RSI rule on 2010-2019 and stays positive later:")
    hit = tab[tab.holds]
    print("none" if hit.empty else hit.rule.tolist())


if __name__ == "__main__":
    main()
