"""Shared data/fitness for NEAT v2 ('kid v2').  SIMULATION ONLY.  The holdout (>= 2026-04-01) is never loaded unless holdout=True."""
import sys, json
sys.path.insert(0, "/workspace/neat-plan/.pylib"); sys.path.insert(0, "/workspace/neat_v2/pylib"); sys.path.insert(0, "/workspace/neat_v2")
import numpy as np, pandas as pd
import sim

HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
NORM = json.load(open("/workspace/neat_v2/norm.json"))
COLS = ["f_" + c for c in NORM["cols"]]
CAP = 500.0


class Data:
    def __init__(self, holdout=False):
        df = pd.read_parquet("/workspace/neat_v2/feat5m.parquet")
        if not holdout:
            df = df[df.datetime < HOLDOUT_START].reset_index(drop=True)
        self.df = df
        self.o, self.h, self.l, self.c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
        self.atr = df.atr.to_numpy(np.float64)
        self.nt = df.no_trade.to_numpy(np.bool_).copy()
        self.dayid = df.dayid.to_numpy(np.int64)
        self.F = np.ascontiguousarray(df[COLS].to_numpy(np.float64))
        self.n_days = int(self.dayid.max()) + 1
        # per-day descriptive regime labels (ex post; used only to SCORE, never as inputs)
        g = df.groupby("dayid")
        dd = pd.DataFrame(dict(o=g.open.first(), c=g.close.last(), h=g.high.max(), l=g.low.min(), n=g.size(),
                               t0=g.datetime.first(), day=g.day.first()))
        rng = (dd.h - dd.l); ret = dd.c - dd.o
        med = rng.shift(1).rolling(20, min_periods=5).median().bfill()
        dd["trend"] = np.where(ret >= 0.45 * rng, "up", np.where(ret <= -0.45 * rng, "down", "chop"))
        dd["vol"] = np.where(rng > 1.25 * med, "high", np.where(rng < 0.8 * med, "low", "normal"))
        dd["full"] = dd.n >= 100           # skip holiday/partial days in day-statistics
        wk = ((pd.to_datetime(dd.day) - pd.to_datetime(dd.day.iloc[0])).dt.days // 7).to_numpy()
        dd["val"] = (wk % 5 == 2)          # every 5th week is internal validation (all regimes covered)
        self.days = dd
        self.holdout_day = (dd.t0 >= HOLDOUT_START).to_numpy() if holdout else np.zeros(len(dd), bool)

    def run(self, net, p, start=1600, end=None, log=False, cost_mult=1.0, long_only=False):
        end = len(self.c) if end is None else end
        cap = 20000 if log else 1
        ti = np.zeros(cap, np.int64); to = np.zeros(cap, np.int64); tp = np.zeros(cap); tdir = np.zeros(cap, np.int64)
        pos_out = np.zeros(len(self.c), np.int8)
        ptr, src, w, bias, resp, oi, n_in, n_tot = net
        dp, ntr, mdd, nlog = sim.run(self.o, self.h, self.l, self.c, self.atr, self.nt, self.dayid, self.F, start, end,
                                     ptr, src, w, bias, resp, oi, n_in, n_tot, p["sl"], p["tp"], p["trail"], p["min_sl"],
                                     sim.SLIP, sim.COMM, sim.PV, CAP, self.n_days, long_only, cost_mult,
                                     log, ti, to, tp, tdir, pos_out)
        trades = None
        if log:
            trades = pd.DataFrame(dict(i_in=ti[:nlog], i_out=to[:nlog], pnl=tp[:nlog], dir=tdir[:nlog]))
            trades["t_in"] = self.df.datetime.to_numpy()[trades.i_in]; trades["t_out"] = self.df.datetime.to_numpy()[trades.i_out]
        return dp, ntr, mdd, trades


def score(dp, ntr, D, mask):
    """Robust regime-aware fitness ($/day units) over the days in mask."""
    days = D.days
    m = mask & days.full.to_numpy()
    x = dp[m]
    if len(x) == 0:
        return -99.0, {}
    eq = np.cumsum(x); dd = float(np.max(np.maximum.accumulate(np.maximum(eq, 0)) - eq))
    buckets = {}
    for lab, col in (("up", "trend"), ("down", "trend"), ("chop", "trend"), ("high", "vol"), ("low", "vol")):
        bm = m & (days[col].to_numpy() == lab)
        if bm.sum() >= 5:
            buckets[lab] = float(dp[bm].mean())
    mean = float(x.mean())
    worst = min(buckets.values()) if buckets else mean
    tpd = ntr / max(1, len(dp[days.full.to_numpy()]))
    pen = 15.0 * max(0.0, (0.4 - tpd) / 0.4)
    f = 0.5 * mean + 0.5 * worst - dd / 150.0 - pen
    # fraction of 4-week blocks profitable (consistency)
    blocks = np.array_split(x, max(1, len(x) // 20))
    cons = float(np.mean([b.sum() > 0 for b in blocks]))
    f += 5.0 * (cons - 0.5)
    return f, dict(mean=mean, worst=worst, dd=dd, tpd=tpd, cons=cons, buckets=buckets, net=float(x.sum()))
