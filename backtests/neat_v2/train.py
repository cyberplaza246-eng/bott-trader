#!/usr/bin/env python
"""NEAT v2 island trainer ('kid v2').  SIMULATION ONLY: no broker, no orders, never loads the holdout (>= 2026-04-01).
- 5m MNQ bars, 31 regime-aware causal features (+position, open-pnl, bars-in-trade), long/short/flat
- fitness on ALL pre-holdout regimes (train weeks), robust: 0.5*mean + 0.5*worst regime bucket - DD - too-few-trades + consistency
- internal validation = every 5th week (never used as fitness); hall of fame ranked by min(train, val)
- adaptive speciation threshold, immigrants on stagnation, full restart (new seed) on long stagnation; runs forever."""
import argparse, os, pickle, random, time, json, multiprocessing as mp
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/neat_v2")
import common as C
import numpy as np
import neat
import sim

D = None; P = None; CFG = None


def _eval(g):
    net = sim.flatten_net(g, CFG)
    dp, ntr, mdd, _ = D.run(net, P, cost_mult=P["cost_mult"])
    fwd = np.zeros(len(D.days), bool)
    if P.get("fwd_cut"):
        fwd = (D.days.t0 >= C.pd.Timestamp(P["fwd_cut"], tz="UTC")).to_numpy()
    tr = ~D.days.val.to_numpy() & ~fwd; va = D.days.val.to_numpy() & ~fwd
    ft, it = C.score(dp, ntr, D, tr)
    fv, iv = C.score(dp, ntr, D, va)
    return ft, fv, ntr, it, iv


def main():
    global D, P, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--island", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sl", type=float, default=1.5); ap.add_argument("--tp", type=float, default=3.0)
    ap.add_argument("--trail", type=float, default=0.0); ap.add_argument("--min_sl", type=float, default=8.0)
    ap.add_argument("--pop", type=int, default=200); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--cost_mult", type=float, default=1.25)
    ap.add_argument("--fwd_cut", default="", help="pseudo-holdout: ignore days >= this date in train/val scoring (design check only)")
    ap.add_argument("--immig_after", type=int, default=30); ap.add_argument("--restart_after", type=int, default=120)
    a = ap.parse_args()
    out = f"/workspace/neat_v2/runs/{a.island}"; os.makedirs(out, exist_ok=True)
    P = dict(sl=a.sl, tp=a.tp, trail=a.trail, min_sl=a.min_sl, cost_mult=a.cost_mult, fwd_cut=a.fwd_cut)
    json.dump(dict(vars(a), params=P), open(out + "/params.json", "w"))
    D = C.Data(holdout=False)
    assert D.df.datetime.max() < C.HOLDOUT_START
    CFG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      "/workspace/neat_v2/neat.cfg")
    CFG.pop_size = a.pop
    CFG.genome_config.num_inputs = D.F.shape[1] + 3
    CFG.genome_config.input_keys = [-i - 1 for i in range(CFG.genome_config.num_inputs)]
    hof = []  # (sel, ft, fv, genome, info)
    if os.path.exists(out + "/hof.pkl"):
        hof = pickle.load(open(out + "/hof.pkl", "rb"))
    logf = open(out + "/train.log", "a")
    def log(s):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"; logf.write(line + "\n"); logf.flush()
    restart = 0
    pool = mp.get_context("fork").Pool(a.procs)
    seed = a.seed
    while True:
        random.seed(seed * 1000 + restart); np.random.seed(seed * 1000 + restart)
        CFG.species_set_config.compatibility_threshold = 2.5
        pop = neat.Population(CFG, seed=seed * 1000 + restart)
        # seed restarts with a few hall-of-fame genomes (keeps the best structure alive)
        if restart > 0 and hof:
            keys = list(pop.population.keys())[: min(5, len(hof))]
            for k, h in zip(keys, hof[:5]):
                g = pickle.loads(pickle.dumps(h[3])); g.key = k; pop.population[k] = g
            pop.species.speciate(CFG, pop.population, pop.generation)
        best_sel = max([h[0] for h in hof], default=-1e9); last_imp = 0; gen = 0
        log(f"== start restart={restart} seed={seed*1000+restart} params={P} pop={a.pop}")
        while True:
            t0 = time.time()
            evald = []
            def fit_fn(genomes, config):
                res = pool.map(_eval, [g for _, g in genomes], chunksize=4)
                for (gid, g), r in zip(genomes, res):
                    g.fitness = r[0]; g._v = r; evald.append(g)
            try:
                pop.run(fit_fn, 1)
            except RuntimeError as e:
                if "min_species_size" not in str(e) and "Cannot satisfy" not in str(e):
                    raise
                CFG.species_set_config.compatibility_threshold += 0.5
                pop.species.speciate(CFG, pop.population, pop.generation)
                log(f"species overflow -> threshold {CFG.species_set_config.compatibility_threshold:.2f}, re-speciated")
                continue
            gen += 1
            improved = False
            for g in sorted(evald, key=lambda g: -min(g._v[0], g._v[1]))[:5]:
                ft, fv, ntr, it, iv = g._v
                sel = min(ft, fv)
                if len(hof) < 30 or sel > hof[-1][0]:
                    if not any(abs(h[0] - sel) < 1e-9 for h in hof):
                        hof.append((sel, ft, fv, pickle.loads(pickle.dumps(g)), dict(it=it, iv=iv, ntr=ntr, gen=gen, restart=restart)))
                        hof.sort(key=lambda x: -x[0]); hof = hof[:30]
                        if sel > best_sel + 1e-6:
                            best_sel = sel; improved = True
            if improved:
                last_imp = gen
            ns = len(pop.species.species)
            thr = CFG.species_set_config.compatibility_threshold
            if ns < 6: CFG.species_set_config.compatibility_threshold = max(0.8, thr - 0.1)
            elif ns > 15: CFG.species_set_config.compatibility_threshold = thr + 0.25
            if gen % 5 == 0 or improved:
                b = max(evald, key=lambda g: g._v[0]) if evald else None
                h = hof[0] if hof else None
                log(f"r{restart} gen {gen:4d} species {ns:2d} thr {thr:.2f} | gen-best train {b._v[0]:+7.2f} val {b._v[1]:+7.2f} tr {b._v[2]} | "
                    f"HOF sel {h[0]:+7.2f} (train {h[1]:+.2f} val {h[2]:+.2f} net_tr ${h[4]['it'].get('net',0):.0f} net_val ${h[4]['iv'].get('net',0):.0f} trades {h[4]['ntr']}) "
                    f"{'*IMPROVED*' if improved else ''} {time.time()-t0:.1f}s")
                pickle.dump(hof, open(out + "/hof.tmp", "wb")); os.replace(out + "/hof.tmp", out + "/hof.pkl")
            stale = gen - last_imp
            if stale >= a.restart_after:
                log(f"stagnant {stale} gens -> FULL RESTART with new seed"); break
            if stale > 0 and stale % a.immig_after == 0:
                n_imm = int(0.35 * a.pop)
                keys = list(pop.population.keys()); random.shuffle(keys)
                for k in keys[:n_imm]:
                    del pop.population[k]
                new = pop.reproduction.create_new(CFG.genome_type, CFG.genome_config, n_imm)
                pop.population.update(new)
                pop.species.speciate(CFG, pop.population, pop.generation)
                log(f"stagnant {stale} gens -> injected {n_imm} immigrants")
        restart += 1


if __name__ == "__main__":
    main()
