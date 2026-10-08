"""RSI-50 momentum variants vs REVAMP baseline (MNQ, 1 lot, $3 RT, $500 daily, $100->$20/$150 trail lock).
SIMULATION ONLY."""
import itertools, json, sys
import numpy as np, pandas as pd
import run_nq as R, engine as E
from data import load_1m, resample_5m
from signals import build_signals, live_atr_sma14, rtick, TICK
sys.path.insert(0, "/workspace/bott-trader-revamp")
from src.research_ensemble.indicators import rsi as rsi_f, ema, adx as adx_f
from src.research_ensemble.session import rth_mask

PV, COST = 2.0, 3.0
CFG = dict(max_conc=3, extra_conf=0.72, cd_win=5, cd_loss=15, daily_loss=500, sizing=False)
E.LOCK_MODE, E.PV = "usd_new", PV
E.USD_ACT, E.USD_KEEP, E.USD_TRAIL, E.USD_PCT = 100.0, 20.0, 150.0, 0.5


def conf(r):
    if r["sleeve"] in ("mean_reversion", "rsi_mom") or not np.isfinite(r["adx"]):
        return 0.66
    return max(0.5, min(0.95, 0.62 + (r["adx"] - 20.0) * 0.01))
R.conf = conf


def rsi_signals(df5, filt, stop, xhi, x50):
    c, h, l = df5["close"], df5["high"], df5["low"]
    r = rsi_f(c, 14)
    atr = live_atr_sma14(df5)
    ax = adx_f(h, l, c, 14)
    e50, e200 = ema(c, 50), ema(c, 200)
    vok = df5["volume"] >= df5["volume"].rolling(20).mean()
    up = (r > 50) & (r.shift() <= 50)
    dn = (r < 50) & (r.shift() >= 50)
    if filt == "none":
        fl = fs = pd.Series(True, index=df5.index)
    elif filt == "ema_vol":       # with the EMA50/200 trend + volume
        fl = (c > e200) & (e50 > e200) & vok
        fs = (c < e200) & (e50 < e200) & vok
    elif filt == "adx20_vol":     # ADX>=20 + volume
        fl = fs = (ax >= 20) & vok
    elif filt == "ema_adx15_vol": # EMA trend, skip ADX<15, volume
        fl = (c > e200) & (e50 > e200) & (ax >= 15) & vok
        fs = (c < e200) & (e50 < e200) & (ax >= 15) & vok
    else:
        raise ValueError(filt)
    d = pd.Series(0, index=df5.index)
    d[up & fl] = 1
    d[dn & fs] = -1
    d = d.where(rth_mask(df5.index), 0)
    if stop == "atr1.5":
        sl = c - d * 1.5 * atr
    else:  # swing: last 6 bars extreme +/- 1 tick, distance clamped to 0.5..3 ATR
        lo6, hi6 = l.rolling(6).min() - TICK, h.rolling(6).max() + TICK
        raw = np.where(d > 0, c - lo6, hi6 - c)
        dist = np.clip(raw, 0.5 * atr, 3.0 * atr)
        sl = c - d * dist
    out = pd.DataFrame({"dir": d, "ref": c, "sl": rtick(sl), "tp": np.nan, "sleeve": "rsi_mom",
                        "adx": ax, "atr_live": atr, "trailing": False, "xhi": float(xhi), "x50": bool(x50)},
                       index=df5.index)
    out = out[(out["dir"] != 0) & np.isfinite(out["sl"])].copy()
    out["signal_time"] = out.index + pd.Timedelta(minutes=5)
    return out


def main():
    d1 = load_1m(); df5 = resample_5m(d1)
    base = build_signals(df5)
    base["xhi"], base["x50"] = np.nan, False
    win = lambda s: s[(s["signal_time"] >= R.START) & (s["signal_time"] < R.END)]
    base = win(base)
    B = E.Bars(d1, df5, live_atr_sma14(df5), rsi_f(df5["close"], 14))
    adx5 = adx_f(df5["high"], df5["low"], df5["close"], 14)

    variants = {"A_baseline": base}
    for filt, stop, xhi, x50 in itertools.product(["none", "ema_vol", "adx20_vol", "ema_adx15_vol"],
                                                   ["atr1.5", "swing"], [65, 70], [False, True]):
        tag = ("B" if filt == "none" else "C") + f"_{filt}_{stop}_x{xhi}" + ("_rev50" if x50 else "")
        rs = win(rsi_signals(df5, filt, stop, xhi, x50))
        variants[tag] = rs
        nomr = base[base["sleeve"] != "mean_reversion"]
        add = rs[~rs.index.isin(base.index)]
        rng = adx5.reindex(add.index) < 20
        variants["D_range_" + tag] = pd.concat([nomr, add[rng]]).sort_index()   # RSI only in MR's slot (ADX<20)
        variants["D_any_" + tag] = pd.concat([nomr, add]).sort_index()          # RSI wherever trend/breakout silent

    rows = []
    cache = {}
    for name, sg in variants.items():
        att = E.attempts(B, sg, "base", False)
        g = R.gate(att, PV, COST, CFG)
        m = {"variant": name}
        for per, lo, hi in (("full", R.START, R.END), ("dev", R.START, R.HOLD), ("hold", R.HOLD, R.END)):
            mm = R.metrics(g, lo, hi)
            for k in ("net", "trades", "win_rate", "pf", "max_dd", "green_months", "worst_month", "avg_month"):
                m[f"{per}_{k}"] = mm[k]
        t = g[g["taken"] & (g["status"] == "filled")]
        m["rsi_trades"] = int((t["sleeve"] == "rsi_mom").sum())
        m["exit_mix"] = t[t["sleeve"] == "rsi_mom"]["exit_reason"].value_counts().to_dict()
        rows.append(m)
        cache[name] = t
    S = pd.DataFrame(rows)
    S.to_csv("/workspace/nq_bt/out/rsi_variants.csv", index=False)
    pd.to_pickle(cache, "/workspace/nq_bt/out/rsi_trades.pkl")
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
    cols = ["variant", "full_net", "full_trades", "full_win_rate", "full_pf", "full_max_dd", "full_green_months",
            "dev_net", "dev_pf", "hold_net", "hold_pf", "hold_max_dd", "hold_green_months"]
    print(S[cols].sort_values("dev_pf", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
