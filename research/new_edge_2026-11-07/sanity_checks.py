"""Engine sanity checks on results/trades_all.csv and a synthetic bar series (no tuning)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

t = pd.read_csv(os.path.join(HERE, "results", "trades_all.csv"))
t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True).dt.tz_convert(E.ET)
t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True).dt.tz_convert(E.ET)
b = t[t.slip_ticks == 1]
print("exit reasons by strategy\n", pd.crosstab(b.strategy, b.reason))
em = b.entry_time.dt.hour * 60 + b.entry_time.dt.minute
xm = b.exit_time.dt.hour * 60 + b.exit_time.dt.minute
print("\nentry ET minute range by strategy\n", em.groupby(b.strategy).agg(["min", "max"]))
print("\nexit ET minute max by strategy\n", xm.groupby(b.strategy).max())
print("\nrisk pts (max must be <= 60)\n", b.groupby("strategy")["risk_pts"].describe()[["mean", "50%", "max"]])
per_day = b.groupby(["cell", "segment", b.entry_time.dt.date]).size()
print("\nmax trades/day per cell\n", per_day.groupby(level=0).max())
dl = b.groupby(["cell", b.entry_time.dt.date])["pnl"].sum()
print("\nworst day per cell (should be > about -400 - one trade's loss)\n", dl.groupby(level=0).min())

# synthetic: long at next open, stop/target both touched on the same bar -> stop first
idx = pd.date_range("2025-03-03 14:30", periods=6, freq="5min", tz="UTC")
x = pd.DataFrame({"open": [100, 100, 101, 101, 101, 101.0], "high": [100, 101, 112, 101, 101, 101.0],
                  "low": [100, 99, 90, 101, 101, 101.0], "close": [100, 100.5, 101, 101, 101, 101.0],
                  "volume": 1.0}, index=idx)
x = E.rth(x)
n = len(x)
s = E.Signals(np.r_[1, np.zeros(n - 1)], np.r_[95, np.full(n - 1, np.nan)], np.full(n, np.nan),
              np.r_[110, np.full(n - 1, np.nan)], None, 955, 930, 2)
r = E.simulate(x, s, 1)
tr = r.trades.iloc[0]
assert tr.entry == 100.25 and tr.reason == "stop" and tr.exit == 94.75, tr
assert abs(tr.pnl - ((94.75 - 100.25) * 2 - 4)) < 1e-9
print("\nsynthetic fill test OK:", tr[["entry", "exit", "reason", "pnl"]].to_dict())
