"""Seed data/gapfade_state.json with the history of |NY-open gaps| (expanding quantile needs it).
gap = 09:30 ET bar open - previous session 15:59 ET bar close, from data/MNQ_1m.csv (UTC bars).
Usage: python scripts/seed_gapfade_state.py [--csv data/MNQ_1m.csv] [--out data/gapfade_state.json] [--force]
Never connects to anything. Refuses to overwrite an existing state file unless --force."""
import argparse, json, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/MNQ_1m.csv")
ap.add_argument("--out", default="data/gapfade_state.json")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
if os.path.exists(a.out) and not a.force:
    sys.exit(f"{a.out} exists - use --force to overwrite")
df = pd.read_csv(a.csv, usecols=["datetime", "open", "close"])
df["t"] = pd.to_datetime(df["datetime"], utc=True)
df = df.set_index("t").sort_index()
et = df.index.tz_convert("America/New_York")
df["d"] = et.strftime("%Y-%m-%d"); df["m"] = et.hour * 60 + et.minute
open_, close_ = {}, {}
for d, g in df.groupby("d"):
    o = g[(g.m >= 570) & (g.m <= 572)]; c = g[(g.m >= 957) & (g.m <= 959)]
    if o.empty or c.empty: continue
    if (c.index[-1] - o.index[0]).total_seconds() / 60 < 300: continue          # need a real RTH session
    open_[d] = float(o.iloc[0].open); close_[d] = float(c[c.m == c.m.max()].iloc[-1].close)
days = sorted(open_)
gaps = [(days[k], open_[days[k]] - close_[days[k - 1]]) for k in range(1, len(days))]
past = [round(abs(g), 2) for _, g in gaps]
st = {"version": 1, "past_abs": past, "closes": {d: {"px": close_[d], "contract": ""} for d in days[-6:]}, "events": {}, "open": {},
      "seeded_from": a.csv, "seeded_last_day": days[-1]}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(st, open(a.out, "w"), indent=1)
q = float(np.quantile(past, 0.75))
print(f"seeded {len(past)} gaps ({gaps[0][0]}..{gaps[-1][0]}), q75 threshold now {q:.2f} pts, last close {days[-1]} {close_[days[-1]]:.2f} -> {a.out}")
print("trades the expanding rule would have taken:", sum(1 for k in range(20, len(past)) if past[k] >= np.quantile(past[:k], .75)))
