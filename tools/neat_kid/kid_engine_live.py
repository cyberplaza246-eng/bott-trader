"""W5 shadow engine. Same features as neat_v2/data5m.py and the same fills as sim.py.
No broker, no network. Simulation only.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PV, SLIP, COMM, CAP = 2.0, 0.25, 1.0, 500.0
DROP = ("tod", "vlog", "vtod", "r1", "r3")
HERE = Path(__file__).resolve().parent


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def load_norm():
    import json
    spec = json.loads((HERE / "norm.json").read_text(encoding="utf-8"))
    return spec["cols"], spec["norm"]


def maintenance_mask(et: pd.Series) -> np.ndarray:
    """16:45 <= America/New_York < 18:15, the daily break plus 15 minutes each side."""
    mins = et.dt.hour * 60 + et.dt.minute
    return ((mins >= 1005) & (mins < 1095)).to_numpy()


def gap_mask(ts: pd.Series, bars: int = 3, gap_min: int = 30) -> np.ndarray:  # 15 min each side on 5m bars, like training
    dt = ts.diff().dt.total_seconds().to_numpy()
    flag = np.zeros(len(ts), dtype=bool)
    for i in np.where(dt >= gap_min * 60)[0]:
        lo = max(0, i - bars)
        hi = min(len(ts), i + bars)
        flag[lo:hi] = True
    return flag


def build_features(df: pd.DataFrame, cols: list[str], norm: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """df columns: datetime (UTC), open, high, low, close, volume, optional no_trade.
    Returns (frame, feature matrix with W5 drops zeroed, atr)."""
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    et = d["datetime"].dt.tz_convert("America/New_York")
    d["day"] = (et + pd.Timedelta(hours=6)).dt.date
    d["dayid"] = pd.factorize(d["day"])[0].astype(np.int64)
    c, h, l, v = d.close, d.high, d.low, d.volume.astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    F = pd.DataFrame(index=d.index)
    for k in (1, 3, 6, 12, 48, 288):
        F[f"r{k}"] = (c - c.shift(k)) / atr
    diff = c.diff()
    up = wilder(diff.clip(lower=0), 14)
    dn = wilder((-diff).clip(lower=0), 14)
    F["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50) / 50
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200, 1200)}
    for n in (20, 50, 200, 1200):
        F[f"ema{n}_d"] = (c - ema[n]) / atr
    F["ema20_slope"] = (ema[20] - ema[20].shift(6)) / atr
    F["ema200_slope"] = (ema[200] - ema[200].shift(24)) / atr
    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0)
    mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * wilder(pdm, 14) / atr
    mdi = 100 * wilder(mdm, 14) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    F["adx"] = wilder(dx, 14) / 50 - 0.5
    F["di_diff"] = (pdi - mdi) / 50
    F["atr_ratio"] = np.log(atr / atr.rolling(2760, min_periods=500).median())
    r1 = np.log(c).diff()
    F["vol_ratio"] = np.log(r1.rolling(12).std() / r1.rolling(72).std())
    sma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    F["bb"] = (c - sma) / (2 * sd)
    for n in (20, 96):
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        F[f"don{n}"] = (c - ll) / (hh - ll).replace(0, np.nan) * 2 - 1
    mins = et.dt.hour * 60 + et.dt.minute
    F["vlog"] = np.log((v + 1) / (v.rolling(48).mean() + 1))
    tod = mins // 5
    F["vtod"] = np.log((v + 1) / (v.groupby(tod).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean()) + 1))
    F["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    F["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    F["rth"] = ((mins >= 570) & (mins < 960)).astype(float)
    dg = d.groupby("dayid")
    day_open = dg.open.transform("first")
    day_hi = dg.high.cummax()
    day_lo = dg.low.cummin()
    dh, dl, dc = dg.high.max(), dg.low.min(), dg.close.last()
    prev_hi = d.dayid.map(dh.shift(1))
    prev_lo = d.dayid.map(dl.shift(1))
    prev_c = d.dayid.map(dc.shift(1))
    F["d_open"] = (c - day_open) / atr
    F["d_hi"] = (c - day_hi) / atr
    F["d_lo"] = (c - day_lo) / atr
    F["p_hi"] = (c - prev_hi) / atr
    F["p_lo"] = (c - prev_lo) / atr
    F["p_c"] = (c - prev_c) / atr
    F = F.replace([np.inf, -np.inf], np.nan)
    raw = {"tod_sin", "tod_cos", "rth"}
    for col in cols:
        if col in raw:
            continue
        st = norm[col]
        F[col] = ((F[col].clip(st["lo"], st["hi"]) - st["mu"]) / st["sd"]).clip(-4, 4)
    F = F.fillna(0.0)[cols]
    for col in cols:
        if any(col.startswith(p) for p in DROP):
            F[col] = 0.0
    nt = d["no_trade"].fillna(False).astype(bool).to_numpy() if "no_trade" in d.columns else np.zeros(len(d), bool)
    nt = nt | maintenance_mask(et) | gap_mask(d["datetime"])
    nt = nt.copy()
    nt[:1500] = True
    d["atr"] = atr.to_numpy()
    d["no_trade"] = nt
    return d, np.ascontiguousarray(F.to_numpy(np.float64)), atr.to_numpy(np.float64)


def _eval(vals, ptr, src, w, bias, resp, n_in):
    m = len(ptr) - 1
    for j in range(m):
        s = 0.0
        for e in range(int(ptr[j]), int(ptr[j + 1])):
            s += w[e] * vals[int(src[e])]
        vals[n_in + j] = math.tanh(bias[j] + resp[j] * s)


def _decide(vals, out_idx):
    a0 = vals[int(out_idx[0])]
    a1 = vals[int(out_idx[1])]
    a2 = vals[int(out_idx[2])]
    if a1 > a0 and a1 >= a2:
        return 1
    if a2 > a0 and a2 > a1:
        return -1
    return 0


def load_net(path: Path | None = None):
    z = np.load(path or (HERE / "w5_net.npz"))
    return {k: z[k] for k in ("ptr", "src", "w", "bias", "resp", "out_idx")} | {
        "n_in": int(z["n_in"]), "n_tot": int(z["n_tot"])
    }


def simulate(frame, F, net, p, *, cost_mult=1.0, start=1600, end=None, liquidate_end=True):
    """Match sim.run. liquidate_end=False keeps an open trade (live tail)."""
    o = frame.open.to_numpy(np.float64)
    h = frame.high.to_numpy(np.float64)
    l = frame.low.to_numpy(np.float64)
    c = frame.close.to_numpy(np.float64)
    atr = frame.atr.to_numpy(np.float64)
    nt = frame.no_trade.to_numpy(np.bool_)
    dayid = frame.dayid.to_numpy(np.int64)
    ts = frame.datetime
    end = len(c) if end is None else end
    slp = SLIP * cost_mult
    cm = COMM * cost_mult
    sl_atr, tp_atr, trail_atr = float(p["sl"]), float(p["tp"]), float(p["trail"])
    min_sl, every = float(p["min_sl"]), int(p["every"])
    ptr, src, w = net["ptr"], net["src"], net["w"]
    bias, resp, out_idx = net["bias"], net["resp"], net["out_idx"]
    n_in, n_tot = net["n_in"], net["n_tot"]
    nf = F.shape[1]
    vals = np.zeros(n_tot)
    pos = 0
    entry = stop = tgt = best = e_atr = 0.0
    e_bar = 0
    pend = lock = 0
    cur_day = -1
    day_real = 0.0
    halted = False
    trades = []

    def close(t, xp, reason, set_lock):
        nonlocal pos, day_real, lock
        px = xp - slp * pos
        pnl = (px - entry) * pos * PV - 2 * cm
        day_real += pnl
        side = "LONG" if pos > 0 else "SHORT"
        trades.append(dict(
            entry_time=ts.iloc[e_bar], side=side, entry=float(entry), sl=float(stop), tp=float(tgt) if tgt else "",
            exit_time=ts.iloc[t], exit=float(px), reason=reason, pnl=float(pnl),
        ))
        if set_lock:
            lock = pos
        pos = 0

    for t in range(start, end):
        d = int(dayid[t])
        if d != cur_day:
            cur_day = d
            day_real = 0.0
            halted = False
        if t > start and pos != 0:
            do_exit = False
            xp = 0.0
            reason = ""
            if (pos > 0 and o[t] <= stop) or (pos < 0 and o[t] >= stop):
                do_exit, xp, reason = True, o[t], "stop"
            elif tgt > 0 and ((pos > 0 and o[t] >= tgt) or (pos < 0 and o[t] <= tgt)):
                do_exit, xp, reason = True, o[t], "target"
            elif nt[t] or halted or pend != pos:
                do_exit, xp, reason = True, o[t], "flat"
            if do_exit:
                close(t, xp, reason, reason in ("stop", "target"))
        want = pend
        if nt[t] or nt[t - 1] or halted or (want != 0 and want == lock):
            want = 0
        if pos == 0 and want != 0:
            pos = want
            entry = o[t] + slp * pos
            e_bar = t
            e_atr = atr[t - 1]
            sd = max(sl_atr * e_atr, min_sl)
            stop = entry - pos * sd
            tgt = entry + pos * tp_atr * e_atr if tp_atr > 0 else 0.0
            best = entry
        if pos != 0 and t > start:
            hit = False
            xp = 0.0
            reason = ""
            if pos > 0:
                if l[t] <= stop:
                    hit, xp, reason = True, stop, "stop"
                elif tgt > 0 and h[t] >= tgt:
                    hit, xp, reason = True, tgt, "target"
            else:
                if h[t] >= stop:
                    hit, xp, reason = True, stop, "stop"
                elif tgt > 0 and l[t] <= tgt:
                    hit, xp, reason = True, tgt, "target"
            if hit:
                close(t, xp, reason, True)
            elif trail_atr > 0:
                if pos > 0:
                    if c[t] > best:
                        best = c[t]
                    ns = best - trail_atr * e_atr
                    if ns > stop:
                        stop = ns
                else:
                    if c[t] < best:
                        best = c[t]
                    ns = best + trail_atr * e_atr
                    if ns < stop:
                        stop = ns
        upnl = (c[t] - entry) * pos * PV if pos != 0 else 0.0
        if day_real + upnl <= -CAP:
            halted = True
        if t < end - 1 and t % every == 0:
            vals[:nf] = F[t]
            vals[nf] = pos
            vals[nf + 1] = 0.0 if pos == 0 else max(-3.0, min(3.0, (c[t] - entry) * pos / max(e_atr, 1e-6)))
            vals[nf + 2] = 0.0 if pos == 0 else min(1.0, (t - e_bar) / 48.0)
            _eval(vals, ptr, src, w, bias, resp, n_in)
            pend = _decide(vals, out_idx)
            if pend != lock:
                lock = 0
    if liquidate_end and pos != 0:
        t = end - 1
        close(t, c[t], "end", False)
    simulate.last_open = None
    if pos != 0:
        simulate.last_open = dict(side="LONG" if pos > 0 else "SHORT", entry=float(entry), sl=float(stop),
                                  tp=float(tgt) if tgt else None, since=ts.iloc[e_bar])
    return trades
