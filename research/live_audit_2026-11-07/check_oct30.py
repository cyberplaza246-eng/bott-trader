import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ET = "America/New_York"

y1 = pd.read_csv(os.path.join(HERE, "NQ_1m_yahoo.csv"), parse_dates=["datetime"]).set_index("datetime")
y1.index = y1.index.tz_convert(ET)
w = y1.loc["2026-10-30 12:45":"2026-10-30 15:10"]
print("Yahoo 1m, Oct 30 12:45-15:10 ET: low", w.low.min(), "high", w.high.max())
jumps = w.close.diff().abs().sort_values(ascending=False).head(5)
print("largest 1m close-to-close moves:\n", jumps.to_string())
print(w.loc["2026-10-30 14:55":"2026-10-30 15:05"].to_string())

live5 = pd.read_csv(os.path.join(ROOT, "data", "live_bars_MNQ_5m.csv"))
live5.index = pd.to_datetime(live5.ts_et, utc=True).tz_convert(ET)
lw = live5.loc["2026-10-30 12:45":"2026-10-30 15:15"]
print("\nbot MNQ 5m, same window: low", lw.low.min(), "high", lw.high.max())
print(lw[["open", "high", "low", "close", "volume"]].iloc[::3].to_string())
