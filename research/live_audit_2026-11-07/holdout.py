"""True holdout: every candidate from the Oct 27-28 research, scored on bars that did not
exist when it was chosen (2026-09-22 .. 2026-11-06).

Bars: Yahoo NQ=F (matches local MNQ to 0.25 pt at the same bar-start label).
Signals use 5m bars from 2026-08-28 for warm-up. Fills use 1m bars from 2026-10-12 and
5m bars before that (coarser: one bar per 5 minutes, stop assumed first).
Costs: $4 round trip + 1 tick slippage per side, 1 MNQ.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import backtest_revamp_live_stack as bt  # noqa: E402

ET = "America/New_York"
START = pd.Timestamp("2026-09-22 00:00", tz="UTC")
M1_FROM = pd.Timestamp("2026-10-12 22:00", tz="UTC")
bt.COMM_RT = 4.0


def load(name):
    df = pd.read_csv(os.path.join(HERE, name))
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime").sort_index().astype(float)


m1y, bad1 = bt.clean_1m(load("NQ_1m_yahoo.csv"))
m5, bad5 = bt.clean_1m(load("NQ_5m_yahoo.csv"))
print(f"spike cleaner dropped {bad1} bad 1m bars and {bad5} bad 5m bars")
fills = pd.concat([m5[(m5.index >= START) & (m5.index < M1_FROM)], m1y[m1y.index >= M1_FROM]]).sort_index()
bars5 = m5.index[m5.index >= START]


def ensemble(bo_r):
    bt.BREAKOUT_TP_R = bo_r
    ens = bt.ensemble_signals(m5)
    adds = bt.addon_rows(bt.featured(m5))
    return pd.concat([ens] + list(adds.values()))


def run(sig, rules):
    tr = bt.simulate(fills, bars5, sig[sig.index >= START - pd.Timedelta(minutes=5)], rules)
    return pd.DataFrame([t.__dict__ for t in tr])


def gap_fade(q=0.75, hold_min=90, stop_pts=300.0):
    """Fade the 09:30 open vs the prior 15:59 close when |gap| >= expanding q of past |gaps|."""
    hist = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"))
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True)
    hist = hist.set_index("datetime").sort_index()
    hist, _ = bt.clean_1m(hist[["open", "high", "low", "close", "volume"]].astype(float))
    bars = pd.concat([hist[hist.index < START], fills]).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    ix = bars.index
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))

    def at(ts, tol=3):
        i = ix.searchsorted(ts)
        return i if i < len(ix) and (ix[i] - ts) <= pd.Timedelta(minutes=tol) else None

    def T(day, hm):
        return pd.Timestamp(f"{day} {hm}", tz=ET).tz_convert("UTC")

    rows, past, prev = [], [], None
    for d in pd.bdate_range("2024-10-01", "2026-11-06"):
        ds = d.strftime("%Y-%m-%d")
        i = at(T(ds, "09:30"))
        j = ix.searchsorted(T(ds, "15:59"), side="right") - 1
        if prev is not None and i is not None:
            g = o[i] - prev
            ok = len(past) >= 20 and abs(g) >= np.quantile(past, q) and g != 0
            past.append(abs(g))
            k = at(T(ds, "09:30") + pd.Timedelta(minutes=hold_min), tol=5)
            if ok and k is not None and ix[i] >= START:
                s = -int(np.sign(g))
                ent = o[i] + s * bt.TICK
                adverse = (ent - l[i:k].min()) if s > 0 else (h[i:k].max() - ent)
                if adverse >= stop_pts:
                    pnl = -(stop_pts + bt.TICK) * bt.PV - bt.COMM_RT
                    reason = "STOP_LOSS"
                else:
                    pnl = s * ((o[k] - s * bt.TICK) - ent) * bt.PV - bt.COMM_RT
                    reason = "TIME"
                rows.append(dict(tag="gap_fade", dir=s, entry_time=ix[i], exit_time=ix[k], pnl=pnl, reason=reason, gap=g))
        last_ok = j >= 0 and abs(ix[j] - T(ds, "15:59")) <= pd.Timedelta(minutes=5)
        prev = c[j] if last_ok else None
    return pd.DataFrame(rows)


def summary(name, df):
    if df is None or not len(df):
        return dict(variant=name, n=0)
    df = df.sort_values("exit_time")
    p = df.pnl.values
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max())
    day = df.groupby(pd.to_datetime(df.exit_time).dt.tz_convert(ET).dt.date).pnl.sum()
    g, lo = p[p > 0].sum(), -p[p <= 0].sum()
    return dict(variant=name, n=len(p), net=round(p.sum()), win=f"{np.mean(p > 0):.0%}",
                pf=round(g / lo, 2) if lo else None, avg=round(p.mean(), 1), max_dd=round(dd),
                worst_day=round(day.min()), red_days=f"{(day < 0).sum()}/{len(day)}",
                breach_2k="YES" if dd >= 2000 else "no")


sig2, sig3 = ensemble(2.0), ensemble(3.0)
variants = {
    "LIVE now: revamp 2R, 24h": run(sig2, bt.Rules(overnight_sleeves="all")),
    "A: revamp 3R, 24h": run(sig3, bt.Rules(overnight_sleeves="all")),
    "D2: day-only 3R, no 6PM": run(sig3, bt.Rules(rth_main_only=True, disabled_tags=("addon_e1",))),
    "day-only 2R, no 6PM": run(sig2, bt.Rules(rth_main_only=True, disabled_tags=("addon_e1",))),
}
gf = gap_fade()
variants["gap fade q75 H90 (alone)"] = gf
variants["A + gap fade"] = pd.concat([variants["A: revamp 3R, 24h"], gf])
variants["D2 + gap fade"] = pd.concat([variants["D2: day-only 3R, no 6PM"], gf])

res = pd.DataFrame([summary(k, v) for k, v in variants.items()])
print(f"HOLDOUT {START.date()} -> {fills.index[-1].date()} (never seen by the research)\n")
worst = variants["A: revamp 3R, 24h"].pnl.min()
print(f"worst single trade in A: ${worst:,.1f}\n")
print(res.to_string(index=False))

print("\nweekly net by variant")
wk = {}
for k, v in variants.items():
    if len(v):
        w = pd.to_datetime(v.exit_time).dt.tz_convert(ET).dt.tz_localize(None).dt.to_period("W").astype(str)
        wk[k[:22]] = v.groupby(w).pnl.sum().round()
print(pd.DataFrame(wk).fillna(0).to_string())

print("\ngap fade trades:")
print(gf[["entry_time", "gap", "dir", "pnl", "reason"]].to_string(index=False) if len(gf) else "none")
for k, v in variants.items():
    v.to_csv(os.path.join(HERE, "holdout_" + "".join(ch if ch.isalnum() else "_" for ch in k)[:40] + ".csv"), index=False)
