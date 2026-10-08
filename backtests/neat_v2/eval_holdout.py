#!/usr/bin/env python
"""Evaluate NEAT v2 hall-of-fame brains on the LOCKED holdout Apr-Sep 2026.  SIMULATION ONLY.
Pre-declared pick: the single HOF genome with the highest selection score min(train, val) across all islands
(selection never looks at the holdout).  Other island champions are reported for information only."""
import sys, glob, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/neat_v2")
import numpy as np, pandas as pd
import common as C, sim, neat

def cfg_for(n_in):
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "/workspace/neat_v2/neat.cfg")
    cfg.genome_config.num_inputs = n_in; cfg.genome_config.input_keys = [-i - 1 for i in range(n_in)]
    return cfg

def report(D, trades, label):
    T = trades.copy()
    et_in = pd.to_datetime(T.t_in, utc=True).dt.tz_convert("America/New_York")
    T["month"] = pd.to_datetime(T.t_in, utc=True).dt.strftime("%Y-%m")
    mins = et_in.dt.hour * 60 + et_in.dt.minute
    T["session"] = np.where((mins >= 570) & (mins < 960), "RTH", "overnight")
    days = D.days
    T["trend"] = days.trend.to_numpy()[D.dayid[T.i_in]]; T["vol"] = days.vol.to_numpy()[D.dayid[T.i_in]]
    eq = T.pnl.cumsum(); dd = float((eq.cummax().clip(lower=0) - eq).max()) if len(T) else 0.0
    gw = T.pnl[T.pnl > 0].sum(); gl = -T.pnl[T.pnl < 0].sum()
    agg = lambda k: {a: dict(net=round(float(b.pnl.sum()), 2), trades=int(len(b)), win=round(float((b.pnl > 0).mean()), 3),
                             pf=round(float(b.pnl[b.pnl > 0].sum() / max(1e-9, -b.pnl[b.pnl < 0].sum())), 2)) for a, b in T.groupby(k)}
    r = dict(label=label, net=round(float(T.pnl.sum()), 2), trades=int(len(T)), win_rate=round(float((T.pnl > 0).mean()), 3) if len(T) else None,
             pf=round(float(gw / gl), 3) if gl else None, max_dd=round(dd, 2), avg_trade=round(float(T.pnl.mean()), 2) if len(T) else None,
             long_short=agg("dir"), monthly=agg("month"), trend=agg("trend"), vol=agg("vol"), session=agg("session"))
    return r

def main():
    hof = []
    for f in sorted(glob.glob("/workspace/neat_v2/runs/*/hof.pkl")):
        isl = f.split("/")[-2]
        if isl.startswith("smoke"): continue
        P = json.load(open(f.replace("hof.pkl", "params.json")))["params"]
        for h in pickle.load(open(f, "rb")):
            hof.append(dict(island=isl, sel=h[0], ft=h[1], fv=h[2], g=h[3], info=h[4], P=P))
    hof.sort(key=lambda x: -x["sel"])
    Dh = C.Data(holdout=True)
    i0 = int(np.searchsorted(Dh.df.datetime.to_numpy(), np.datetime64(C.HOLDOUT_START.tz_convert(None))))
    cfg = cfg_for(Dh.F.shape[1] + 3)
    out = dict(pick=None, island_champs=[])
    picks = [hof[0]] + [next(h for h in hof if h["island"] == isl) for isl in sorted({h["island"] for h in hof}) if hof[0]["island"] != isl]
    for k, h in enumerate(picks):
        net = sim.flatten_net(h["g"], cfg)
        _, _, _, tr = Dh.run(net, h["P"], start=i0, log=True)
        _, _, _, tr2 = Dh.run(net, h["P"], start=i0, log=True, cost_mult=2.0)
        r = report(Dh, tr, f"{h['island']} sel={h['sel']:+.2f} train={h['ft']:+.2f} val={h['fv']:+.2f}")
        r["net_2x_cost"] = round(float(tr2.pnl.sum()), 2)
        r["pre_holdout"] = dict(train=h["info"]["it"], val=h["info"]["iv"], trades=h["info"]["ntr"])
        r["genome_size"] = dict(nodes=len(h["g"].nodes), conns=sum(1 for c in h["g"].connections.values() if c.enabled))
        if k == 0: out["pick"] = r
        else: out["island_champs"].append(r)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/workspace/neat_v2/holdout_v2.json", "w"), indent=1, default=str)
    for r in [out["pick"]] + out["island_champs"]:
        print(f"{r['label']}: HOLDOUT net ${r['net']:+.0f} (2x cost ${r['net_2x_cost']:+.0f}) trades {r['trades']} win {r['win_rate']} PF {r['pf']} DD ${r['max_dd']:.0f}")
        print("   monthly:", {m: v["net"] for m, v in r["monthly"].items()})
        print("   trend:", {m: v["net"] for m, v in r["trend"].items()}, "vol:", {m: v["net"] for m, v in r["vol"].items()}, "session:", {m: v["net"] for m, v in r["session"].items()})

if __name__ == "__main__":
    main()
