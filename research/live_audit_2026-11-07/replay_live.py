"""Replay the backtest model of the live revamp stack over the live period and compare
it with what the live bot actually did.

Signals: research ensemble + add-ons on Yahoo NQ=F 5m (warm-up from 2026-08-28).
Fills: Yahoo NQ=F 1m from 2026-10-12. Settings = scripts/backtest_revamp_live_stack.py,
which mirrors start_mnq_live_revamp.bat (breakout 2R, trend 1.5R, MR far band, $ lock,
max 3 open, session limits, overnight all sleeves).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import backtest_revamp_live_stack as bt  # noqa: E402

ET = "America/New_York"
START = pd.Timestamp("2026-10-12 22:00", tz="UTC")


def load(name):
    df = pd.read_csv(os.path.join(HERE, name))
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime").sort_index().astype(float)


m1, bad1 = bt.clean_1m(load("NQ_1m_yahoo.csv"))
m5, bad5 = bt.clean_1m(load("NQ_5m_yahoo.csv"))
print(f"spike cleaner dropped {bad1} bad 1m bars and {bad5} bad 5m bars")

live5 = pd.read_csv(os.path.join(ROOT, "data", "live_bars_MNQ_5m.csv"))
live5["datetime"] = pd.to_datetime(live5["ts_et"], utc=True)
live5 = live5.set_index("datetime").sort_index()
j = m5[["close"]].join(live5[["close"]], rsuffix="_live", how="inner")
diff = (j.close - j.close_live).abs()
print(f"Yahoo NQ vs bot MNQ 5m closes: {len(j)} shared bars, median |diff| {diff.median():.2f} pts, p95 {diff.quantile(.95):.2f}")

ens = bt.ensemble_signals(m5)
b = bt.featured(m5)
adds = bt.addon_rows(b)
sig = pd.concat([ens] + list(adds.values()))
sig = sig[sig.index >= START - pd.Timedelta(minutes=5)]

trades = bt.simulate(m1[m1.index >= START], m5.index[m5.index >= START], sig, bt.Rules(overnight_sleeves="all"))
tr = pd.DataFrame([t.__dict__ for t in trades])
tr["et"] = pd.to_datetime(tr.exit_time).dt.tz_convert(ET)
tr["date"] = tr.et.dt.date
tr.to_csv(os.path.join(HERE, "replay_trades.csv"), index=False)


def summary(name, p):
    p = np.asarray(p, dtype=float)
    if not len(p):
        print(f"{name}: no trades")
        return
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max())
    g, l = p[p > 0].sum(), -p[p <= 0].sum()
    print(f"{name}: n={len(p)} net=${p.sum():,.2f} win={np.mean(p > 0):.0%} PF={g / l if l else float('inf'):.2f} "
          f"avgW=${p[p > 0].mean() if (p > 0).any() else 0:.2f} avgL=${p[p <= 0].mean() if (p <= 0).any() else 0:.2f} DD=${dd:,.2f}")


live = pd.read_csv(os.path.join(HERE, "live_trades.csv"))
live["exit_time"] = pd.to_datetime(live.exit_time, utc=True, format="mixed")
live = live[live.exit_time >= START]
live["date"] = live.exit_time.dt.tz_convert(ET).dt.date

print(f"\nwindow {START} -> {m1.index[-1]}")
summary("BACKTEST MODEL", tr.pnl)
summary("LIVE BOT      ", live.pnl)

print("\nbacktest by tag:")
print(tr.groupby("tag").pnl.agg(["count", "sum", lambda s: (s > 0).mean()]).rename(columns={"<lambda_0>": "win"}).to_string())
print("\nbacktest exit reasons:", tr.reason.value_counts().to_dict())

daily = pd.DataFrame({"backtest": tr.groupby("date").pnl.sum(), "bt_n": tr.groupby("date").size(),
                      "live": live.groupby("date").pnl.sum(), "live_n": live.groupby("date").size()}).fillna(0)
print("\nDAILY (exit date, ET)")
print(daily.round(2).to_string())
print("\ncorrelation of daily P&L:", round(daily.backtest.corr(daily.live), 2))
