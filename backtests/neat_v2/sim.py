"""Numba MNQ 5m simulator for neat-python feed-forward genomes.  SIMULATION ONLY (no broker, no network).
Decision at close of bar t -> fill at OPEN of bar t+1 (slippage each side + commission per side).
Stops: ATR-scaled stop (min floor), optional ATR take-profit, optional ATR trailing stop; intrabar via H/L, stop first,
gap-through fills at the open.  Flat on/before no_trade bars (maintenance break, gaps, rolls).
Daily loss cap: once realised+open day P&L <= -cap, flatten at next open and no new trades until next Globex day.
Re-entry lock: after a stop/TP exit no same-direction re-entry until the desired position changes."""
import sys
sys.path.insert(0, "/workspace/neat-plan/.pylib"); sys.path.insert(0, "/workspace/neat_v2/pylib")
import math
import numpy as np
from numba import njit

PV = 2.0          # MNQ $/pt
SLIP = 0.25       # pts per side ($0.50)
COMM = 1.00       # $ per side  -> $3.00 per round trip incl. slippage


def flatten_net(genome, config):
    """neat-python genome -> arrays: node order (topological), bias, response, link ptr/src/w, output idx."""
    from neat.graphs import feed_forward_layers
    gc = config.genome_config
    ins = list(gc.input_keys); outs = list(gc.output_keys)
    conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers, required = feed_forward_layers(ins, outs, conns)[:2]
    idx = {k: i for i, k in enumerate(ins)}
    order = []
    for layer in layers:
        for n in layer:
            idx[n] = len(ins) + len(order); order.append(n)
    for o in outs:                              # outputs with no path still need a slot
        if o not in idx:
            idx[o] = len(ins) + len(order); order.append(o)
    req = set(required) | set(ins)
    bias = np.zeros(len(order)); resp = np.ones(len(order))
    ptr = np.zeros(len(order) + 1, np.int64); src = []; w = []
    incoming = {}
    for (a, b) in conns:
        if a in req and b in idx and a in idx:
            incoming.setdefault(b, []).append((idx[a], genome.connections[(a, b)].weight))
    for j, n in enumerate(order):
        ng = genome.nodes[n]
        bias[j] = ng.bias; resp[j] = ng.response
        for s, ww in incoming.get(n, []):
            src.append(s); w.append(ww)
        ptr[j + 1] = len(src)
    out_idx = np.array([idx[o] for o in outs], np.int64)
    return (ptr, np.array(src, np.int64), np.array(w, np.float64), bias, resp, out_idx, len(ins), len(ins) + len(order))


@njit(cache=True)
def _eval(vals, ptr, src, w, bias, resp, n_in):
    m = len(ptr) - 1
    for j in range(m):
        s = 0.0
        for e in range(ptr[j], ptr[j + 1]):
            s += w[e] * vals[src[e]]
        vals[n_in + j] = math.tanh(bias[j] + resp[j] * s)


@njit(cache=True)
def run(o, h, l, c, atr, nt, dayid, F, start, end, ptr, src, w, bias, resp, out_idx, n_in, n_tot,
        sl_atr, tp_atr, trail_atr, min_sl, slip, comm, pv, cap, n_days, long_only, cost_mult,
        log, t_in, t_out, t_pnl, t_dir, pos_out):
    """Returns (day_pnl[n_days], n_trades, max_dd, n_logged). day_pnl indexed by dayid (pnl booked at exit day)."""
    day_pnl = np.zeros(n_days)
    vals = np.zeros(n_tot)
    nf = F.shape[1]
    slp = slip * cost_mult; cm = comm * cost_mult
    pos = 0; entry = 0.0; stop = 0.0; tgt = 0.0; best = 0.0; e_bar = 0; e_atr = 0.0
    pend = 0; lock = 0; ntr = 0; nlog = 0
    eq = 0.0; peak = 0.0; maxdd = 0.0
    cur_day = -1; day_real = 0.0; halted = False
    for t in range(start, end):
        d = dayid[t]
        if d != cur_day:
            cur_day = d; day_real = 0.0; halted = False
        if t > start:
            xp = 0.0; do_exit = False; was_stop = False
            if pos != 0:
                # gap through stop / target at open
                if (pos > 0 and o[t] <= stop) or (pos < 0 and o[t] >= stop):
                    do_exit = True; xp = o[t]; was_stop = True
                elif tgt > 0 and ((pos > 0 and o[t] >= tgt) or (pos < 0 and o[t] <= tgt)):
                    do_exit = True; xp = o[t]; was_stop = True
                elif nt[t] or halted or pend != pos:
                    do_exit = True; xp = o[t]
                if do_exit:
                    px = xp - slp * pos
                    pnl = (px - entry) * pos * pv - 2 * cm
                    day_pnl[d] += pnl; day_real += pnl; eq += pnl; ntr += 1
                    if log and nlog < len(t_in):
                        t_in[nlog] = e_bar; t_out[nlog] = t; t_pnl[nlog] = pnl; t_dir[nlog] = pos; nlog += 1
                    if was_stop: lock = pos
                    pos = 0
            want = pend
            if nt[t] or nt[t - 1] or halted or (want != 0 and want == lock):
                want = 0
            if pos == 0 and want != 0:
                pos = want; entry = o[t] + slp * pos; e_bar = t; e_atr = atr[t - 1]
                sd = max(sl_atr * e_atr, min_sl)
                stop = entry - pos * sd
                tgt = entry + pos * tp_atr * e_atr if tp_atr > 0 else 0.0
                best = entry
            # intrabar
            if pos != 0:
                hit = False; xp = 0.0
                if pos > 0:
                    if l[t] <= stop: hit = True; xp = stop
                    elif tgt > 0 and h[t] >= tgt: hit = True; xp = tgt
                else:
                    if h[t] >= stop: hit = True; xp = stop
                    elif tgt > 0 and l[t] <= tgt: hit = True; xp = tgt
                if hit:
                    px = xp - slp * pos
                    pnl = (px - entry) * pos * pv - 2 * cm
                    day_pnl[d] += pnl; day_real += pnl; eq += pnl; ntr += 1
                    if log and nlog < len(t_in):
                        t_in[nlog] = e_bar; t_out[nlog] = t; t_pnl[nlog] = pnl; t_dir[nlog] = pos; nlog += 1
                    lock = pos; pos = 0
                elif trail_atr > 0:
                    if pos > 0:
                        if c[t] > best: best = c[t]
                        ns = best - trail_atr * e_atr
                        if ns > stop: stop = ns
                    else:
                        if c[t] < best: best = c[t]
                        ns = best + trail_atr * e_atr
                        if ns < stop: stop = ns
        upnl = (c[t] - entry) * pos * pv if pos != 0 else 0.0
        if day_real + upnl <= -cap:
            halted = True
        e = eq + upnl
        if e > peak: peak = e
        if peak - e > maxdd: maxdd = peak - e
        pos_out[t] = pos
        # decision
        if t < end - 1:
            for k in range(nf):
                vals[k] = F[t, k]
            vals[nf] = pos
            vals[nf + 1] = 0.0 if pos == 0 else max(-3.0, min(3.0, (c[t] - entry) * pos / max(e_atr, 1e-6)))
            vals[nf + 2] = 0.0 if pos == 0 else min(1.0, (t - e_bar) / 48.0)
            _eval(vals, ptr, src, w, bias, resp, n_in)
            a0 = vals[out_idx[0]]; a1 = vals[out_idx[1]]; a2 = vals[out_idx[2]]
            if a1 > a0 and a1 >= a2:
                pend = 1
            elif a2 > a0 and a2 > a1 and not long_only:
                pend = -1
            else:
                pend = 0
            if pend != lock:
                lock = 0
    if pos != 0:
        t = end - 1
        px = c[t] - slp * pos
        pnl = (px - entry) * pos * pv - 2 * cm
        day_pnl[dayid[t]] += pnl; eq += pnl; ntr += 1
        if log and nlog < len(t_in):
            t_in[nlog] = e_bar; t_out[nlog] = t; t_pnl[nlog] = pnl; t_dir[nlog] = pos; nlog += 1
    return day_pnl, ntr, maxdd, nlog
