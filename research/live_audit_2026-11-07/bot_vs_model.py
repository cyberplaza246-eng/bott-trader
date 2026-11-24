import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ET = "America/New_York"

rows = json.load(open(os.path.join(ROOT, "data", "session_pnl.json"), encoding="utf-8"))["rows"]
live = pd.DataFrame(rows)
live["exit_et"] = pd.to_datetime(live.exit_time, utc=True, format="mixed").dt.tz_convert(ET)
live["adopted"] = live.order_id.str.startswith("bot_adopt")
print("LIVE ledger trades:")
print(live[["exit_et", "kind", "order_id", "adopted", "pnl"]].to_string(index=False))

model = pd.read_csv(os.path.join(HERE, "replay_trades.csv"))
for c in ("entry_time", "exit_time"):
    model[c] = pd.to_datetime(model[c], utc=True).dt.tz_convert(ET)
lo, hi = live.exit_et.min() - pd.Timedelta(hours=6), live.exit_et.max() + pd.Timedelta(hours=1)
m = model[(model.exit_time >= lo) & (model.exit_time <= hi)]
print("\nMODEL trades in the same window:")
print(m[["entry_time", "exit_time", "tag", "dir", "entry", "exit", "pnl", "reason"]].round(2).to_string(index=False))
print(f"\nlive net ${live.pnl.sum():,.2f} (bot-opened ${live[~live.adopted].pnl.sum():,.2f}); model net ${m.pnl.sum():,.2f}")
