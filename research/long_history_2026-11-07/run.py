"""Long-history check. Rules frozen before the run.

Daily NQ=F, 2010-2026, scored in MNQ dollars ($2/point, $4 commission, 1 tick/side).
Train 2010-2019, validate 2020-2022, test 2023-2024. 2025-2026 is reported, not used to pick.

IBS: (close-low)/(high-low) <= 0.15, 0.20, or 0.30. Buy that close, sell the next open.
IBS_DAY: threshold 0.20, sell the next close instead.
ON_ALL: buy every close, sell the next open. A long-only IBS rule has to beat this.
MOM: long the next session if the close is above the close 200 sessions ago, otherwise flat.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PV, COMM, TICK = 2.0, 4.0, 0.25
SPLITS = {
    "train": ("2010-01-01", "2020-01-01"),
    "val": ("2020-01-01", "2023-01-01"),
    "test": ("2023-01-01", "2025-01-01"),
    "hold": ("2025-01-01", "2027-01-01"),
}


def load() -> pd.DataFrame:
    raw = yf.download("NQ=F", start="2010-01-01", interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna().astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    rng = df.High - df.Low
    med = rng.rolling(20, min_periods=10).median()
    bad = rng > (6 * med)
    print(f"days {df.index.min().date()} .. {df.index.max().date()} n={len(df)} dropped {int(bad.sum())}")
    return df.loc[~bad].copy()


def run(df: pd.DataFrame, mode: str, thr: float = 0.2) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    ibs = np.where(h > l, (c - l) / (h - l), np.nan)
    rows = []
    for i in range(200, len(df) - 1):
        if mode == "mom":
            if not np.isfinite(c[i - 200]) or c[i] <= c[i - 200]:
                continue
            entry, exit_px = c[i] + TICK, c[i + 1] - TICK
        elif mode == "all_day":
            entry, exit_px = c[i] + TICK, c[i + 1] - TICK
        elif mode == "all":
            entry, exit_px = c[i] + TICK, o[i + 1] - TICK
        else:
            if not np.isfinite(ibs[i]) or ibs[i] > thr:
                continue
            entry = c[i] + TICK
            exit_px = (c[i + 1] if mode == "day" else o[i + 1]) - TICK
        rows.append({
            "day": df.index[i].strftime("%Y-%m-%d"),
            "pnl": (exit_px - entry) * PV - COMM,
            "ret": (exit_px - entry) / entry,
        })
    return pd.DataFrame(rows)


def pack(tr: pd.DataFrame, a: str, b: str) -> tuple[float, int, float]:
    t = tr[(tr.day >= a) & (tr.day < b)] if len(tr) else tr
    if t.empty:
        return 0.0, 0, 0.0
    pnl = t.pnl.to_numpy()
    gl = -pnl[pnl < 0].sum()
    pf = float(pnl[pnl > 0].sum() / gl) if gl else 0.0
    pct = round(float(t.ret.sum() * 100), 1)
    return round(float(pnl.sum()), 0), int(len(t)), round(pf, 2), pct


def main() -> None:
    df = load()
    cells = {
        "IBS15": run(df, "on", 0.15),
        "IBS20": run(df, "on", 0.20),
        "IBS30": run(df, "on", 0.30),
        "IBS20_DAY": run(df, "day", 0.20),
        "ON_ALL": run(df, "all"),
        "LONG_EVERY_DAY": run(df, "all_day"),
        "MOM200": run(df, "mom"),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    print(f"{'cell':12} {'train':>14} {'val':>14} {'test':>14} {'hold':>14}")
    for name, tr in cells.items():
        bits = []
        for lab, (a, b) in SPLITS.items():
            net, n, pf, pct = pack(tr, a, b)
            bits.append(f"{net:+.0f}/{n} pf{pf} {pct:+}%")
        print(f"{name:12} " + "  ".join(f"{x:>16}" for x in bits))
        tr.to_csv(os.path.join(HERE, "results", f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
