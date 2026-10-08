"""Build 5-minute MNQ bars + causal regime-aware features from the cleaned 1m back-adjusted series.  SIM ONLY.
Normalisation stats are fitted on pre-holdout rows only.  Output: feat5m.parquet (+ norm.json)."""
import sys, json
sys.path.insert(0, "/workspace/neat-plan/.pylib")
import numpy as np, pandas as pd

HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
src = pd.read_parquet("/workspace/neat_plan_out/clean.parquet")
src["b"] = src.datetime.dt.floor("5min")
g = src.groupby("b", sort=True)
df = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last(),
                       volume=g.volume.sum(), no_trade=g.no_trade.max(), n1=g.close.size())).reset_index().rename(columns={"b": "datetime"})
et = df.datetime.dt.tz_convert("America/New_York")
df["day"] = (et + pd.Timedelta(hours=6)).dt.date          # Globex day: 18:00 ET -> next calendar date
df["dayid"] = pd.factorize(df["day"])[0].astype(np.int64)
mins = et.dt.hour * 60 + et.dt.minute
c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume.astype(float)

def wilder(x, n): return x.ewm(alpha=1 / n, adjust=False).mean()
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
atr = wilder(tr, 14); df["atr"] = atr
F = pd.DataFrame(index=df.index)
for k in (1, 3, 6, 12, 48, 288):
    F[f"r{k}"] = (c - c.shift(k)) / atr
d = c.diff(); up = wilder(d.clip(lower=0), 14); dn = wilder((-d).clip(lower=0), 14)
F["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50) / 50
ema = {n: c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200, 1200)}
for n in (20, 50, 200, 1200):
    F[f"ema{n}_d"] = (c - ema[n]) / atr
F["ema20_slope"] = (ema[20] - ema[20].shift(6)) / atr
F["ema200_slope"] = (ema[200] - ema[200].shift(24)) / atr
pdm = (h.diff()).where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0)
mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
pdi = 100 * wilder(pdm, 14) / atr; mdi = 100 * wilder(mdm, 14) / atr
dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
F["adx"] = wilder(dx, 14) / 50 - 0.5
F["di_diff"] = (pdi - mdi) / 50
F["atr_ratio"] = np.log(atr / atr.rolling(2760, min_periods=500).median())
r1 = np.log(c).diff()
F["vol_ratio"] = np.log(r1.rolling(12).std() / r1.rolling(72).std())
sma = c.rolling(20).mean(); sd = c.rolling(20).std()
F["bb"] = (c - sma) / (2 * sd)
for n in (20, 96):
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    F[f"don{n}"] = (c - ll) / (hh - ll).replace(0, np.nan) * 2 - 1
F["vlog"] = np.log((v + 1) / (v.rolling(48).mean() + 1))
tod = mins // 5
F["vtod"] = np.log((v + 1) / (v.groupby(tod).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean()) + 1))
F["tod_sin"] = np.sin(2 * np.pi * mins / 1440); F["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
F["rth"] = ((mins >= 570) & (mins < 960)).astype(float)
# day structure
dg = df.groupby("dayid")
day_open = dg.open.transform("first")
day_hi = dg.high.cummax(); day_lo = dg.low.cummin()
dh = dg.high.max(); dl = dg.low.min(); dc = dg.close.last()
prev_hi = df.dayid.map(dh.shift(1)); prev_lo = df.dayid.map(dl.shift(1)); prev_c = df.dayid.map(dc.shift(1))
F["d_open"] = (c - day_open) / atr
F["d_hi"] = (c - day_hi) / atr; F["d_lo"] = (c - day_lo) / atr
F["p_hi"] = (c - prev_hi) / atr; F["p_lo"] = (c - prev_lo) / atr; F["p_c"] = (c - prev_c) / atr
F = F.replace([np.inf, -np.inf], np.nan)
cols = list(F.columns)
raw = ["tod_sin", "tod_cos", "rth"]
pre = df.datetime < HOLDOUT_START
norm = {}
for col in cols:
    if col in raw: continue
    x = F.loc[pre, col]
    lo, hi = x.quantile(0.005), x.quantile(0.995)
    xc = x.clip(lo, hi)
    norm[col] = dict(mu=float(xc.mean()), sd=float(xc.std() + 1e-9), lo=float(lo), hi=float(hi))
    F[col] = ((F[col].clip(lo, hi) - norm[col]["mu"]) / norm[col]["sd"]).clip(-4, 4)
F = F.fillna(0.0)
valid = np.zeros(len(df), bool); valid[1500:] = True        # warm-up
df["no_trade"] = df["no_trade"] | ~valid
out = pd.concat([df, F.add_prefix("f_")], axis=1)
out.to_parquet("/workspace/neat_v2/feat5m.parquet")
json.dump(dict(cols=cols, norm=norm), open("/workspace/neat_v2/norm.json", "w"), indent=1)
print(len(out), out.datetime.min(), out.datetime.max(), "pre-holdout rows", int(pre.sum()), "features", len(cols))
print(F.loc[pre].describe().T[["mean", "std", "min", "max"]].round(2))
