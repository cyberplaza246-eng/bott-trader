"""Research only: do TradingView Kernel Regression (Laplace, h=14) and 1h MACD(12,26,9)
filters improve the revamp ADX ensemble? Self-contained engine; does not touch live code.
Run from repo root: venv\\Scripts\\python.exe research\\tv_filters\\tv_filter_backtest.py
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src.research_ensemble.indicators import adx, atr, bollinger_bands, ema
from src.research_ensemble.trend_following import TrendFollowingStrategy
from src.research_ensemble.mean_reversion import MeanReversionStrategy
from src.research_ensemble.breakout import BreakoutStrategy

PT, TICK = 2.0, 0.25
COST_RT = 2 * 1.5 + 2 * TICK * PT          # $1.5/side commission + 1 tick slip/side = $4
TREND_R, BO_R, TRAIL = 1.5, 2.0, 4.0

def load():
    one = pd.read_csv("data/MNQ_1m.csv", parse_dates=["datetime"]).set_index("datetime").sort_index()
    one = one[~one.index.duplicated()]
    f5 = one.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    p = "data/neat_kid_bars_MNQ_5m.csv" if os.path.exists("data/neat_kid_bars_MNQ_5m.csv") else "data/kid5m.csv"
    k = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime")
    k = k[k.index > f5.index[-1]]
    return pd.concat([f5, k]).sort_index()

def kernel(src, h=14.0, win=100):
    """Non-repainting Nadaraya-Watson, Laplace weights exp(-i/h), past bars only.
    (Approximation of ChartPrime 'Multi Kernel Regression'; exact source not public-verified.)"""
    w = np.exp(-np.arange(win) / h); w /= w.sum()
    v = src.values; out = np.full(len(v), np.nan)
    c = np.convolve(v, w, mode="full")[:len(v)]
    out[win - 1:] = c[win - 1:]
    return pd.Series(out, index=src.index)

def htf(df, rule, offset=None):
    r = df.resample(rule, label="right", closed="left", offset=offset).agg({"close": "last"}).dropna()
    return r["close"]

def to5(s, idx):  # value available at HTF bar close (label=right) -> asof onto 5m bar close times
    bar_close = idx + pd.Timedelta(minutes=5)
    return pd.Series(s.reindex(bar_close, method="ffill").values, index=idx)

def signals(df):
    c = df["close"]; a = adx(df["high"], df["low"], c, 14)
    tr, mr, bo = (S().generate_signals(df) for S in (TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy))
    up, mid, lo = bollinger_bands(c, 20, 2.0)
    n = len(df); e = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan); sv = np.array([""] * n, dtype=object)
    trending, ranging = (a >= 25).values, (a < 20).values
    for sig, mask, name in ((tr, trending, "trend"), (mr, ranging, "mr")):
        m = mask & (sig.entries.values != 0)
        e[m] = sig.entries.values[m]; sl[m] = sig.stop_price.values[m]; sv[m] = name
    m = (bo.entries.values != 0) & ~ranging
    e[m] = bo.entries.values[m]; sl[m] = bo.stop_price.values[m]; sv[m] = "breakout"
    cv = c.values; risk = np.abs(cv - sl)
    for name, r in (("trend", TREND_R), ("breakout", BO_R)):
        m = sv == name; tp[m] = cv[m] + e[m] * r * risk[m]
    m = sv == "mr"; tp[m] = np.where(e[m] > 0, up.values[m], lo.values[m])
    bad = ((e == 1) & (tp <= cv)) | ((e == -1) & (tp >= cv)); tp[bad] = np.nan
    return e, sl, tp, sv

def run(df, e, sl, tp, sv, atr_):
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    trades = []; pos = 0; i = 0; n = len(df)
    while i < n - 1:
        if pos == 0:
            if e[i] != 0 and np.isfinite(sl[i]):
                pos = e[i]; ent = c[i]; stop = sl[i]; tgt = tp[i]; ei = i; sleeve = sv[i]
                best = ent
            i += 1; continue
        # manage on bar i (stop first if both)
        ex = None
        if pos > 0:
            if l[i] <= stop: ex = min(stop, o[i])
            elif np.isfinite(tgt) and h[i] >= tgt: ex = max(tgt, o[i])
        else:
            if h[i] >= stop: ex = max(stop, o[i])
            elif np.isfinite(tgt) and l[i] <= tgt: ex = min(tgt, o[i])
        if ex is None:
            best = max(best, h[i]) if pos > 0 else min(best, l[i])
            ts = best - pos * TRAIL * atr_[i]
            if np.isfinite(ts): stop = max(stop, ts) if pos > 0 else min(stop, ts)
            i += 1; continue
        trades.append((df.index[ei], sleeve, pos, (ex - ent) * pos * PT - COST_RT))
        pos = 0; i += 1  # can re-enter from next bar
    return pd.DataFrame(trades, columns=["time", "sleeve", "dir", "pnl"])

def stats(t):
    if len(t) == 0: return dict(trades=0, win=0, net=0, pf=0, mdd=0)
    p = t.pnl; eq = p.cumsum(); dd = (eq - eq.cummax().clip(lower=0)).min()
    g, L = p[p > 0].sum(), -p[p < 0].sum()
    return dict(trades=len(t), win=round(100 * (p > 0).mean(), 1), net=round(p.sum()), pf=round(g / L, 2) if L else 99, mdd=round(dd))

def main():
    df = load(); print("bars", len(df), df.index[0], "->", df.index[-1])
    c = df["close"]; atr_ = atr(df["high"], df["low"], c, 14).values
    e, sl, tp, sv = signals(df)
    k5 = kernel(c); k5s = np.sign(k5.diff()).values
    h1 = htf(df, "1h"); k1 = kernel(h1); k1s = np.sign(to5(k1.diff(), df.index)).values
    d1 = htf(df, "24h", offset="22h"); kd = kernel(d1); kds = np.sign(to5(kd.diff(), df.index)).values  # daily ~18:00 ET roll (22:00 UTC)
    m = ema(h1, 12) - ema(h1, 26); sg = ema(m, 9); macd = np.sign(to5(m - sg, df.index)).values
    split = df.index[int(len(df) * 0.7)]; print("IS/OOS split", split)
    filt = {"baseline": None, "A_kern5m": k5s, "B1_kern1h": k1s, "B2_kernD": kds, "C_macd1h": macd}
    filt["D1_k5m+macd"] = np.where(k5s == macd, k5s, 0)
    filt["D2_k1h+macd"] = np.where(k1s == macd, k1s, 0)
    filt["D3_kD+macd"] = np.where(kds == macd, kds, 0)
    rows = []
    variants = []
    for name, f in filt.items():
        variants.append((name, f, None))
        if f is not None: variants.append(("E_" + name + "_on_trend+bo", f, {"trend", "breakout"}))
        if f is not None: variants.append(("E_" + name + "_on_mr", f, {"mr"}))
    variants.append(("X_no_breakout", np.zeros(len(e)), {"breakout"}))
    variants.append(("X_mr_only_sleeve", np.zeros(len(e)), {"breakout", "trend"}))
    for name, f, sleeves in variants:
        ee = e.copy()
        if f is not None:
            apply = np.ones(len(ee), bool) if sleeves is None else np.isin(sv, list(sleeves))
            ee[apply & (ee != f)] = 0
        t = run(df, ee, sl, tp, sv, atr_)
        for lab, tt in (("IS", t[t.time < split]), ("OOS", t[t.time >= split])):
            rows.append(dict(variant=name, set=lab, **stats(tt)))
    r = pd.DataFrame(rows); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    w = r.pivot(index="variant", columns="set").swaplevel(axis=1).sort_index(axis=1)
    print(w.to_string()); r.to_csv("research/tv_filters/results.csv", index=False)
    bt = run(df, e, sl, tp, sv, atr_); bt["set"] = np.where(bt.time < split, "IS", "OOS")
    print(bt.groupby(["set", "sleeve"]).pnl.agg(["count", "sum"]))
main()
