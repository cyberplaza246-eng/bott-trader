"""Collect every dated live MNQ trade the bot recorded and summarize it.

Sources: per-session logs/live_rithmic_MNQ_*.json (written when a session ends)
and data/session_pnl.json (rolling 7-day ledger of the session still running).
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))

rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "logs", "live_rithmic_MNQ_*.json"))):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for t in d.get("trades", []):
        if t.get("symbol") != "MNQ" or t.get("exit_time") is None:
            continue
        rows.append(
            {
                "exit_time": t["exit_time"],
                "entry_time": t.get("entry_time"),
                "direction": t.get("direction"),
                "exit_type": t.get("exit_type"),
                "pnl": float(t["pnl"]),
                "paper": bool(d.get("paper_mode")),
                "strategy": d.get("strategy"),
                "source": os.path.basename(path),
            }
        )

ledger = json.load(open(os.path.join(ROOT, "data", "session_pnl.json"), encoding="utf-8"))["rows"]
for r in ledger:
    rows.append(
        {
            "exit_time": r["exit_time"],
            "entry_time": None,
            "direction": None,
            "exit_type": None,
            "pnl": float(r["pnl"]),
            "paper": False,
            "strategy": "session_ledger",
            "source": "session_pnl.json:" + r.get("kind", ""),
        }
    )

df = pd.DataFrame(rows)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, format="mixed")
df = df.sort_values("exit_time")
df["key"] = df["exit_time"].dt.floor("s").astype(str) + "|" + df["pnl"].round(2).astype(str)
df = df.drop_duplicates("key")
df["et"] = df["exit_time"].dt.tz_convert("America/New_York")
df["date"] = df["et"].dt.date

live = df[~df["paper"]].copy()
live.to_csv(os.path.join(OUT, "live_trades.csv"), index=False)

print("sources:", df["source"].str.split(":").str[0].str.slice(0, 16).value_counts().to_dict())
print("strategies:", live["strategy"].value_counts().to_dict())
print(f"live trades {len(live)}  from {live['et'].min()}  to {live['et'].max()}")

eq = live["pnl"].cumsum()
dd = (eq.cummax().clip(lower=0) - eq).max()
wins = live[live.pnl > 0].pnl
losses = live[live.pnl <= 0].pnl
pf = wins.sum() / -losses.sum() if len(losses) else float("inf")
print(f"net ${live.pnl.sum():,.2f}  win {len(wins)}/{len(live)} = {len(wins) / len(live):.0%}  PF {pf:.2f}")
print(f"avg win ${wins.mean():.2f}  avg loss ${losses.mean():.2f}  max closed-trade DD ${dd:,.2f}")

daily = live.groupby("date").pnl.agg(["count", "sum"])
print("\nDAILY")
print(daily.to_string())
print(f"\nred days {(daily['sum'] < 0).sum()}/{len(daily)}  worst ${daily['sum'].min():,.2f}  best ${daily['sum'].max():,.2f}")

live["week"] = live["et"].dt.to_period("W").astype(str)
print("\nWEEKLY")
print(live.groupby("week").pnl.agg(["count", "sum"]).to_string())
