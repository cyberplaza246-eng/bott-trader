"""Design check (pre-holdout only): WF island HOF genomes evaluated on the forward block 2025-12-01..2026-03-31 they never scored on."""
import sys, json, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/neat_v2")
import numpy as np, pandas as pd, common as C, sim
from eval_holdout import cfg_for
isl = sys.argv[1] if len(sys.argv) > 1 else "WF_br15_30_cut2512"
P = json.load(open(f"/workspace/neat_v2/runs/{isl}/params.json"))["params"]
hof = pickle.load(open(f"/workspace/neat_v2/runs/{isl}/hof.pkl", "rb"))
D = C.Data(holdout=False); cfg = cfg_for(D.F.shape[1] + 3)
fwd = (D.days.t0 >= pd.Timestamp(P["fwd_cut"], tz="UTC")).to_numpy() & D.days.full.to_numpy()
months = pd.to_datetime(D.days.t0).dt.strftime("%Y-%m").to_numpy()
for h in hof[:10]:
    dp, ntr, _, _ = D.run(sim.flatten_net(h[3], cfg), P)
    x = dp[fwd]; mm = {m: round(float(dp[fwd & (months == m)].sum())) for m in sorted(set(months[fwd]))}
    print(f"sel {h[0]:+6.2f} tr {h[1]:+6.2f} val {h[2]:+6.2f} | FWD net ${x.sum():+.0f} months {mm} gen {h[4]['gen']} r{h[4]['restart']}")
