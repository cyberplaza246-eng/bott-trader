import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ET = "America/New_York"

tr = pd.read_csv(os.path.join(HERE, "holdout_A__revamp_3R__24h.csv"))
for c in ("entry_time", "exit_time"):
    tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_convert(ET)
tr["date"] = tr.exit_time.dt.date
tr["risk_pts"] = (tr.entry - tr.exit).abs()

print("by tag / session:")
print(tr.groupby(["tag", "session"]).pnl.agg(["count", "sum", "mean"]).round(1).to_string())
print("\nexit reasons:", tr.reason.value_counts().to_dict())
print(f"stop-loss trades: avg ${tr[tr.reason == 'STOP_LOSS'].pnl.mean():.1f}; worst trade ${tr.pnl.min():.1f}")

daily = tr.groupby("date").pnl.sum().sort_values()
print("\nworst 4 days:")
for d in daily.index[:4]:
    g = tr[tr.date == d]
    print(f"\n{d}  net ${daily[d]:,.1f}  trades {len(g)}")
    print(g[["entry_time", "exit_time", "tag", "dir", "entry", "exit", "pnl", "reason"]].to_string(index=False))
