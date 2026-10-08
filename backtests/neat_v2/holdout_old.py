"""Holdout backtest of old continuous champions (cycle 78, 130). SIM ONLY. $3/RT, 1 contract, $500 daily loss cap."""
import json, sys
sys.path.insert(0, "/workspace/neat-plan/.pylib"); sys.path.insert(0, "/workspace/neat-plan")
import numpy as np, pandas as pd
import wf_common as W
from wf_common import Fold, load_data, build_features, fit_norm, apply_norm, HOLDOUT_START, WARMUP
from env.mnq_env import MnqEnv
from neat.config import Config
from neat.genome import Genome

SLIP, COMM = 0.25, 1.0   # 0.25pt/side slip ($0.50) + $1.00/side comm  => $3.00 round trip
df = load_data(final=True)
out = {}
for cyc in (78, 130):
    inc = json.load(open(f"/workspace/neat_plan_out/continuous/brain_cycle_{cyc:03d}.json"))
    g = Genome.from_dict(inc["genome"]); cfg = Config(num_inputs=inc["num_inputs"], num_outputs=2)
    f = build_features(df, inc["features"]); cols = f.attrs["feature_cols"]
    A = pd.Timestamp(inc["anchor"], tz="UTC")
    lo = max(int(df.datetime.searchsorted(A - pd.DateOffset(months=6))), WARMUP); hi = int(df.datetime.searchsorted(A))
    norm = fit_norm(f, lo, hi, cols)
    ia = int(df.datetime.searchsorted(HOLDOUT_START)) - 400
    sub = f.iloc[ia:].reset_index(drop=True); sub.attrs["feature_cols"] = cols
    X = apply_norm(sub, norm, cols)
    env = MnqEnv(bars=sub, features=X, long_only=True, cost_mult=1.0, sl_pts=10, tp_pts=20, slippage_pts=SLIP,
                 commission_per_side=COMM, point_value=2.0, initial_equity=10000, start_index=0, blowup_pct=0.999)
    r = env.run(g, cfg, log_trades=True, start=400, end=len(sub))
    # round trips: entry then close
    trades = []; cur = None
    for t in r.trades:
        if t.note == "open": cur = t
        elif cur is not None:
            trades.append(dict(t_in=sub.datetime.iloc[cur.bar], t_out=sub.datetime.iloc[t.bar], pnl=t.pnl)); cur = None
    T = pd.DataFrame(trades)
    T["day"] = T.t_in.dt.tz_convert("America/New_York").apply(lambda x: (x + pd.Timedelta(hours=6)).date())  # Globex day starts 18:00 ET
    keep = []; dsum = {}
    for _, row in T.iterrows():
        s = dsum.get(row.day, 0.0)
        if s <= -500: keep.append(False); continue
        keep.append(True); dsum[row.day] = s + row.pnl
    T = T[keep]
    T["month"] = T.t_out.dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    eq = T.pnl.cumsum(); dd = float((eq.cummax().clip(lower=0) - eq).max())
    gw = T.pnl[T.pnl > 0].sum(); gl = -T.pnl[T.pnl < 0].sum()
    m = T.groupby("month").pnl.agg(["sum", "count"])
    res = dict(cycle=cyc, net=float(T.pnl.sum()), trades=len(T), win_rate=float((T.pnl > 0).mean()), pf=float(gw / gl) if gl else None,
               max_dd=dd, monthly={k: [round(v["sum"], 2), int(v["count"])] for k, v in m.iterrows()})
    out[cyc] = res
    print(json.dumps(res, indent=1, default=str))
json.dump(out, open("/workspace/neat_v2/holdout_old.json", "w"), indent=1, default=str)
