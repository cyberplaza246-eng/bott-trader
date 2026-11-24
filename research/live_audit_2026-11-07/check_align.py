import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

y5 = pd.read_csv(os.path.join(HERE, "NQ_5m_yahoo.csv"), parse_dates=["datetime"]).set_index("datetime")
live5 = pd.read_csv(os.path.join(ROOT, "data", "live_bars_MNQ_5m.csv"))
live5.index = pd.to_datetime(live5.ts_et, utc=True)
live5 = live5[~live5.index.duplicated(keep="last")]
print("bot 5m duplicate timestamps dropped; rows", len(live5))
m1 = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"), parse_dates=["datetime"]).set_index("datetime")
loc5 = m1[m1.index >= "2026-08-28"].resample("5min").agg({"close": "last"}).dropna()

for name, ref in (("bot live 5m", live5.close), ("local MNQ_1m->5m", loc5.close)):
    print(name)
    for k in (-2, -1, 0, 1, 2):
        s = y5.close.copy()
        s.index = s.index + pd.Timedelta(minutes=5 * k)
        j = pd.concat([s, ref], axis=1, join="inner").dropna()
        d = (j.iloc[:, 0] - j.iloc[:, 1])
        print(f"  shift {k:+d} bars: n={len(j)} median diff {d.median():+.2f}  median |diff| {d.abs().median():.2f}")
