"""Is the holdout loss unusual for the strategy's own backtest?

Takes the 3R baseline trades (Nov 2024 - Sep 2026, research run) and measures every
rolling window of the same length as the holdout (34 trading days)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SR = os.path.join(ROOT, "research", "og_profit_search_2026-10-27", "strategy_research_2026-10-28")

tr = pd.read_csv(os.path.join(SR, "trades_base.csv"))
tr["exit_time"] = pd.to_datetime(tr.exit_time, utc=True)
day = tr.groupby(tr.exit_time.dt.tz_convert("America/New_York").dt.date).pnl.sum()
day.index = pd.to_datetime(day.index)
day = day.reindex(pd.bdate_range(day.index.min(), day.index.max()), fill_value=0.0)

W = 34
nets, dds = [], []
v = day.values
for i in range(len(v) - W + 1):
    seg = np.cumsum(v[i:i + W])
    nets.append(seg[-1])
    dds.append(float((np.maximum.accumulate(np.maximum(seg, 0)) - seg).max()))
nets, dds = np.array(nets), np.array(dds)

print(f"backtest 3R baseline: {len(tr)} trades, net ${tr.pnl.sum():,.0f}; {len(nets)} rolling {W}-day windows")
print(f"window net: median ${np.median(nets):,.0f}  p10 ${np.percentile(nets, 10):,.0f}  p5 ${np.percentile(nets, 5):,.0f}  min ${nets.min():,.0f}")
print(f"share of windows with net <= -$1,533 (holdout A): {np.mean(nets <= -1533):.1%}")
print(f"window max DD: median ${np.median(dds):,.0f}  p95 ${np.percentile(dds, 95):,.0f}  max ${dds.max():,.0f}")
print(f"share of windows with DD >= $5,136 (holdout A): {np.mean(dds >= 5136):.1%}")
print(f"share of windows with DD >= $2,000: {np.mean(dds >= 2000):.1%}")

eq = np.cumsum(day.values)
print(f"\nfull-history max DD ${(np.maximum.accumulate(np.maximum(eq, 0)) - eq).max():,.0f}; "
      f"days with a new $2k+ drawdown from peak: {(np.maximum.accumulate(np.maximum(eq, 0)) - eq >= 2000).sum()} of {len(eq)}")
