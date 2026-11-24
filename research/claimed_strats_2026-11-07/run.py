"""Enough of a test of the four claimed MNQ rules, frozen before the run.

Rules are chosen on 2025 only. A family is a candidate only if that chosen rule is
positive in 2026 H1, positive from Jul 1 to Sep 21, positive in at least 4 of 6
walk-forward folds, positive in total out of sample, and still positive in both
2026 windows at 3 ticks of slippage. Sep 22-Nov 6 is reported and not used to pick.
Long-only IBS must also beat being long every night. Costs: $4 round trip, 1 MNQ.

VWAP fade: extension of k x ATR (or k x session stdev) and a bar back toward VWAP,
target the VWAP, stop beyond the signal bar. RTH k = 1, 1.5, 2, 2.5. London overlap
(VWAP from 03:00 ET, entries 08:00-11:25) k = 1.5, 2. Stadev bands k = 1.5, 2, 2.5.
VWAP pullback: 09:30-10:00 direction, first touch of VWAP that closes back with it,
target 1.5R or 2R.

ORB: 5/15/30 min range, first 5-min close outside, stop at the other side.
Exit either 11:30 or 2R (flat by 15:55).

IBS: RTH (close-low)/(high-low) <= 0.15/0.20/0.30, long 16:05 to 09:25, 300-pt stop.
Also IBS 0.20 held to the next 15:55, and a short when IBS >= 0.80.

Sweep: wick through the prior RTH high or low and close back inside, 2R, one per side.
FVG: 5-min three-bar gap, enter on the first trade back into it, stop beyond the gap, 2R.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import backtest_revamp_live_stack as bt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ET = "America/New_York"
PV, COMM, TICK = 2.0, 4.0, 0.25
TRAIN, VAL, TEST, HOLD = ("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-07-01"), ("2026-07-01", "2026-09-22"), ("2026-09-22", "2026-11-07")
FOLDS = [
    ("2025-01-02", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("2025-01-02", "2025-07-01", "2025-07-01", "2025-10-01"),
    ("2025-04-01", "2025-10-01", "2025-10-01", "2026-01-01"),
    ("2025-07-01", "2026-01-01", "2026-01-01", "2026-04-01"),
    ("2025-10-01", "2026-04-01", "2026-04-01", "2026-07-01"),
    ("2026-01-01", "2026-07-01", "2026-07-01", "2026-09-22"),
]


def load_5m() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"), usecols=["datetime", "open", "high", "low", "close", "volume"])
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index().loc["2024-11-01":]
    m = m[~m.index.duplicated(keep="last")].astype(float)
    m, dropped = bt.clean_1m(m)
    x = m.tz_convert(ET)
    g = x.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna(subset=["open"])
    y = pd.read_csv(os.path.join(ROOT, "research", "live_audit_2026-11-07", "NQ_5m_yahoo.csv"))
    y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
    y = y.set_index("datetime").sort_index()
    y = y.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
    y = y[~y.index.duplicated(keep="last")]
    y, _ = bt.clean_1m(y)
    y = y[y.index > m.index.max()].tz_convert(ET)
    g = pd.concat([g, y]).sort_index()
    g = g[~g.index.duplicated(keep="first")]
    g["mins"] = g.index.hour * 60 + g.index.minute
    g["day"] = g.index.strftime("%Y-%m-%d")
    print(f"5m {g.index.min()} .. {g.index.max()} n={len(g):,} 1m_dropped={dropped}")
    return g


def add_features(g: pd.DataFrame) -> pd.DataFrame:
    h, l, c = g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy()
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    g["atr"] = pd.Series(tr, index=g.index).ewm(alpha=1 / 14, adjust=False).mean()
    tp = (g.high + g.low + g.close) / 3.0
    vol = g.volume.clip(lower=0).to_numpy()
    vol = np.where(vol > 0, vol, 1.0)
    # RTH VWAP from 09:30, London VWAP from 03:00, each reset by the anchor day
    rth = (g.mins >= 570) & (g.mins < 960)
    lon = (g.mins >= 180) & (g.mins < 690)
    g["vwap"] = _anchored(tp.to_numpy(), vol, rth.to_numpy(), g.day.to_numpy())
    g["lvwap"] = _anchored(tp.to_numpy(), vol, lon.to_numpy(), g.day.to_numpy())
    g["sd"] = _session_std(tp.to_numpy(), rth.to_numpy(), g.day.to_numpy())
    # prior RTH high/low
    rh = g.high.where(rth).groupby(g.day).max()
    rl = g.low.where(rth).groupby(g.day).min()
    rc = g.close.where(g.mins == 955).groupby(g.day).last()
    days = g.day.to_numpy()
    g["pdh"] = g.day.map(rh.shift(1))
    g["pdl"] = g.day.map(rl.shift(1))
    g["ibs"] = g.day.map(((rc - rl) / (rh - rl).replace(0, np.nan)).shift(0))
    # ibs of THIS day is only known at 15:55; keep a column of completed prior ibs via shift
    g["ibs_done"] = g.day.map(((rc - rl) / (rh - rl).replace(0, np.nan)).shift(1))
    g.attrs["rth_close"] = rc
    g.attrs["rth_hi"] = rh
    g.attrs["rth_lo"] = rl
    return g


def _anchored(tp, vol, mask, day) -> np.ndarray:
    out = np.full(len(tp), np.nan)
    cv = cp = 0.0
    prev = None
    for i in range(len(tp)):
        if not mask[i]:
            prev = None
            continue
        if day[i] != prev:
            cv = cp = 0.0
            prev = day[i]
        cv += tp[i] * vol[i]
        cp += vol[i]
        out[i] = cv / cp
    return out


def _session_std(tp, mask, day) -> np.ndarray:
    out = np.full(len(tp), np.nan)
    buf: list[float] = []
    prev = None
    for i in range(len(tp)):
        if not mask[i]:
            prev = None
            continue
        if day[i] != prev:
            buf = []
            prev = day[i]
        buf.append(tp[i])
        if len(buf) >= 6:
            out[i] = float(np.std(buf))
    return out


def simulate(g: pd.DataFrame, sig_i: np.ndarray, sig_d: np.ndarray, stop: np.ndarray,
             target: np.ndarray, exit_min: int, slip: int, max_trades: int = 2) -> pd.DataFrame:
    """sig_i is the signal bar. Fill is the next bar's open. stop/target are absolute, set at the signal."""
    o, h, l = g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy()
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    n = len(g)
    slip_px = slip * TICK
    trades = []
    i = 0
    while i < n - 1:
        if sig_d[i] == 0 or not np.isfinite(stop[i]):
            i += 1
            continue
        d = int(sig_d[i])
        ent = i + 1
        if day[ent] != day[i] and exit_min < 20 * 60:
            i += 1
            continue
        if mins[ent] >= exit_min and exit_min <= 960:
            i += 1
            continue
        entry = o[ent] + d * slip_px
        stp, tgt = float(stop[i]), float(target[i]) if np.isfinite(target[i]) else np.nan
        if (d > 0 and stp >= entry) or (d < 0 and stp <= entry):
            i += 1
            continue
        reason, exit_px, exit_at = "eod", None, None
        j = ent
        while j < n:
            if mins[j] >= exit_min or (j > ent and day[j] != day[ent] and exit_min <= 960):
                reason, exit_px, exit_at = "time", o[j] - d * slip_px, j
                break
            hit_s = l[j] <= stp if d > 0 else h[j] >= stp
            hit_t = np.isfinite(tgt) and (h[j] >= tgt if d > 0 else l[j] <= tgt)
            if hit_s:
                raw = o[j] if (d > 0 and o[j] < stp) or (d < 0 and o[j] > stp) else stp
                reason, exit_px, exit_at = "stop", raw - d * slip_px, j
                break
            if hit_t:
                raw = o[j] if (d > 0 and o[j] > tgt) or (d < 0 and o[j] < tgt) else tgt
                reason, exit_px, exit_at = "target", raw - d * slip_px, j
                break
            j += 1
        if exit_px is None:
            break
        trades.append({"day": day[ent], "pnl": d * (exit_px - entry) * PV - COMM, "reason": reason, "dir": d})
        # next signal after the exit, and respect max trades per day by skipping same day once count hits
        i = exit_at + 1
    if max_trades < 99 and trades:
        df = pd.DataFrame(trades)
        df = df.groupby("day", sort=False).head(max_trades)
        return df.reset_index(drop=True)
    return pd.DataFrame(trades)


def blank(n):
    return np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)


def vwap_fade(g, k, kind: str, slip: int) -> pd.DataFrame:
    c, h, l = g.close.to_numpy(), g.high.to_numpy(), g.low.to_numpy()
    atr, mins, day = g.atr.to_numpy(), g.mins.to_numpy(), g.day.to_numpy()
    if kind == "london":
        vw, lo, hi = g.lvwap.to_numpy(), 480, 685
        band = k * atr
    elif kind == "sd":
        vw, lo, hi = g.vwap.to_numpy(), 600, 930
        band = k * g.sd.to_numpy()
    else:
        vw, lo, hi = g.vwap.to_numpy(), 600, 930
        band = k * atr
    n = len(g)
    d, stop, tgt = blank(n)
    prev_c = np.r_[np.nan, c[:-1]]
    ok = (mins >= lo) & (mins <= hi) & np.isfinite(vw) & np.isfinite(band) & (band > 0)
    short = ok & (prev_c - vw >= band) & (c < prev_c)
    long = ok & (vw - prev_c >= band) & (c > prev_c)
    # use the signal bar, not the previous, for the extension: close vs vwap
    short = ok & (c - vw >= band) & (c < prev_c)
    long = ok & (vw - c >= band) & (c > prev_c)
    d[short] = -1
    d[long] = 1
    both = short & long
    d[both] = 0
    stop = np.where(short, h + 0.25 * atr, np.where(long, l - 0.25 * atr, np.nan))
    tgt = np.where(d != 0, vw, np.nan)
    # one signal per bar already; cap 2/day inside simulate by walking forward only
    return _walk(g, d, stop, tgt, 955 if kind != "london" else 690, slip, 2)


def _walk(g, d, stop, tgt, exit_min, slip, max_trades) -> pd.DataFrame:
    """Sequential so a second trade cannot overlap the first. max_trades per day."""
    o, h, l = g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy()
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    n = len(g)
    sp = slip * TICK
    trades = []
    taken: dict[str, int] = {}
    i = 1
    while i < n - 1:
        if d[i] == 0 or not np.isfinite(stop[i]):
            i += 1
            continue
        if taken.get(day[i], 0) >= max_trades:
            i += 1
            continue
        side = int(d[i])
        ent = i + 1
        if exit_min <= 960 and (mins[ent] >= exit_min or day[ent] != day[i]):
            i += 1
            continue
        entry = o[ent] + side * sp
        stp = float(stop[i])
        tg = float(tgt[i]) if np.isfinite(tgt[i]) else np.nan
        if (side > 0 and stp >= entry) or (side < 0 and stp <= entry):
            i += 1
            continue
        reason, exit_px, exit_at = None, None, None
        j = ent
        while j < n:
            timed = mins[j] >= exit_min or (j > ent and day[j] != day[ent] and exit_min <= 960)
            if timed:
                reason, exit_px, exit_at = "time", o[j] - side * sp, j
                break
            hit_s = l[j] <= stp if side > 0 else h[j] >= stp
            hit_t = np.isfinite(tg) and (h[j] >= tg if side > 0 else l[j] <= tg)
            if hit_s:
                raw = o[j] if (side > 0 and o[j] < stp) or (side < 0 and o[j] > stp) else stp
                reason, exit_px, exit_at = "stop", raw - side * sp, j
                break
            if hit_t:
                raw = o[j] if (side > 0 and o[j] > tg) or (side < 0 and o[j] < tg) else tg
                reason, exit_px, exit_at = "target", raw - side * sp, j
                break
            j += 1
        if exit_px is None:
            break
        trades.append({"day": str(day[ent]), "pnl": side * (exit_px - entry) * PV - COMM, "reason": reason, "dir": side})
        taken[day[i]] = taken.get(day[i], 0) + 1
        i = exit_at + 1
    return pd.DataFrame(trades)


def vwap_pull(g, target_r: float, slip: int) -> pd.DataFrame:
    c, o, h, l = (g[k].to_numpy() for k in ("close", "open", "high", "low"))
    vw, atr, mins, day = g.vwap.to_numpy(), g.atr.to_numpy(), g.mins.to_numpy(), g.day.to_numpy()
    # 09:30 open and 09:55 close (bar that ends 10:00)
    open930 = {}
    close1000 = {}
    for i in range(len(g)):
        if mins[i] == 570:
            open930[day[i]] = o[i]
        if mins[i] == 595:
            close1000[day[i]] = c[i]
    n = len(g)
    d, stop, tgt = blank(n)
    seen = set()
    for i in range(len(g)):
        if not (600 <= mins[i] <= 840) or day[i] in seen:
            continue
        if day[i] not in open930 or day[i] not in close1000 or not np.isfinite(vw[i]):
            continue
        mom = close1000[day[i]] - open930[day[i]]
        if mom > 0 and l[i] <= vw[i] and c[i] > vw[i] and c[i] > o[i]:
            d[i] = 1
            stop[i] = min(l[i], vw[i]) - 0.25 * atr[i]
            risk = c[i] - stop[i]
            tgt[i] = c[i] + target_r * risk
            seen.add(day[i])
        elif mom < 0 and h[i] >= vw[i] and c[i] < vw[i] and c[i] < o[i]:
            d[i] = -1
            stop[i] = max(h[i], vw[i]) + 0.25 * atr[i]
            risk = stop[i] - c[i]
            tgt[i] = c[i] - target_r * risk
            seen.add(day[i])
    return _walk(g, d, stop, tgt, 955, slip, 1)


def orb(g, range_min: int, mode: str, slip: int) -> pd.DataFrame:
    h, l, c = g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy()
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    n = len(g)
    d, stop, tgt = blank(n)
    end = 570 + range_min
    i = 0
    while i < n:
        if mins[i] != 570:
            i += 1
            continue
        dd = day[i]
        rh = rl = None
        k = i
        while k < n and day[k] == dd and mins[k] < end:
            rh = h[k] if rh is None else max(rh, h[k])
            rl = l[k] if rl is None else min(rl, l[k])
            k += 1
        if rh is None or k >= n or rh <= rl:
            i = k
            continue
        while k < n and day[k] == dd and mins[k] < 690:
            if c[k] > rh:
                d[k] = 1
                stop[k] = rl
                if mode == "2r":
                    tgt[k] = c[k] + 2 * (c[k] - rl)
                break
            if c[k] < rl:
                d[k] = -1
                stop[k] = rh
                if mode == "2r":
                    tgt[k] = c[k] - 2 * (rh - c[k])
                break
            k += 1
        i = k + 1
    exit_min = 955 if mode == "2r" else 690
    return _walk(g, d, stop, tgt, exit_min, slip, 1)


def ibs_trades(g, thr: float, side: int, exit_at: str, slip: int) -> pd.DataFrame:
    """Overnight from the 16:05 bar. IBS of the day is known at 15:55."""
    rc = g.attrs["rth_close"]
    rh = g.attrs["rth_hi"]
    rl = g.attrs["rth_lo"]
    ibs = ((rc - rl) / (rh - rl).replace(0, np.nan)).dropna()
    days = list(ibs.index)
    nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    loc = {(day, minute): i for i, (day, minute) in enumerate(zip(g.day, g.mins))}
    o, h, l = g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy()
    sp = slip * TICK
    out = []
    for d0, val in ibs.items():
        if side > 0 and val > thr:
            continue
        if side < 0 and val < thr:
            continue
        d1 = nxt.get(d0)
        if d1 is None:
            continue
        ent = loc.get((d0, 965))
        exm = 565 if exit_at == "open" else 955
        ex = loc.get((d1, exm))
        if ent is None or ex is None or ex <= ent:
            continue
        entry = o[ent] + side * sp
        stp = entry - side * 300.0
        reason, exit_px, exit_at_i = "time", o[ex] - side * sp, ex
        for j in range(ent, ex + 1):
            hit = l[j] <= stp if side > 0 else h[j] >= stp
            if hit:
                raw = o[j] if (side > 0 and o[j] < stp) or (side < 0 and o[j] > stp) else stp
                reason, exit_px, exit_at_i = "stop", raw - side * sp, j
                break
        out.append({"day": d0, "pnl": side * (exit_px - entry) * PV - COMM, "reason": reason, "dir": side})
    return pd.DataFrame(out)


def sweep(g, slip: int) -> pd.DataFrame:
    h, l, c, o = (g[k].to_numpy() for k in ("high", "low", "close", "open"))
    pdh, pdl = g.pdh.to_numpy(), g.pdl.to_numpy()
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    n = len(g)
    d, stop, tgt = blank(n)
    seen_l, seen_s = set(), set()
    for i in range(n):
        if not (575 <= mins[i] <= 900) or not np.isfinite(pdh[i]):
            continue
        if day[i] not in seen_s and h[i] > pdh[i] and c[i] < pdh[i]:
            d[i] = -1
            stop[i] = h[i] + TICK
            tgt[i] = c[i] - 2 * (stop[i] - c[i])
            seen_s.add(day[i])
        elif day[i] not in seen_l and l[i] < pdl[i] and c[i] > pdl[i]:
            d[i] = 1
            stop[i] = l[i] - TICK
            tgt[i] = c[i] + 2 * (c[i] - stop[i])
            seen_l.add(day[i])
    return _walk(g, d, stop, tgt, 955, slip, 2)


def fvg(g, slip: int) -> pd.DataFrame:
    h, l, c, o = (g[k].to_numpy() for k in ("high", "low", "close", "open"))
    mins, day = g.mins.to_numpy(), g.day.to_numpy()
    n = len(g)
    d, stop, tgt = blank(n)
    taken: dict[str, int] = {}
    i = 2
    while i < n - 2:
        if not (570 <= mins[i] <= 900) or taken.get(day[i], 0) >= 2:
            i += 1
            continue
        bull = l[i] > h[i - 2] and c[i - 1] > o[i - 1]
        bear = h[i] < l[i - 2] and c[i - 1] < o[i - 1]
        if not bull and not bear:
            i += 1
            continue
        gap_top, gap_bot = (l[i], h[i - 2]) if bull else (l[i - 2], h[i])
        side = 1 if bull else -1
        fired = False
        for j in range(i + 1, min(n, i + 13)):
            if day[j] != day[i] or mins[j] > 930:
                break
            touched = l[j] <= gap_top if bull else h[j] >= gap_bot
            if not touched:
                continue
            d[j] = side
            stop[j] = (gap_bot - TICK) if bull else (gap_top + TICK)
            risk = (c[j] - stop[j]) if bull else (stop[j] - c[j])
            if risk <= 0:
                d[j] = 0
                break
            tgt[j] = c[j] + side * 2 * risk
            taken[day[i]] = taken.get(day[i], 0) + 1
            fired = True
            i = j + 1
            break
        if not fired:
            i += 1
    return _walk(g, d, stop, tgt, 955, slip, 2)


def pack(tr: pd.DataFrame, a: str, b: str) -> dict:
    if tr is None or tr.empty:
        return {"net": 0.0, "n": 0, "pf": 0.0, "win": 0.0}
    t = tr[(tr.day >= a) & (tr.day < b)]
    if t.empty:
        return {"net": 0.0, "n": 0, "pf": 0.0, "win": 0.0}
    pnl = t.pnl.to_numpy()
    gl = -pnl[pnl < 0].sum()
    # closed-trade drawdown
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    return {"net": round(float(pnl.sum()), 1), "n": int(len(t)),
            "pf": round(float(pnl[pnl > 0].sum() / gl), 3) if gl else 0.0,
            "dd": round(dd, 1), "win": round(float((pnl > 0).mean()), 3)}


def main() -> None:
    g = add_features(load_5m())
    builders = {
        "VWAP_k1": lambda s: vwap_fade(g, 1.0, "atr", s),
        "VWAP_k1.5": lambda s: vwap_fade(g, 1.5, "atr", s),
        "VWAP_k2": lambda s: vwap_fade(g, 2.0, "atr", s),
        "VWAP_k2.5": lambda s: vwap_fade(g, 2.5, "atr", s),
        "VWAP_LON_k1.5": lambda s: vwap_fade(g, 1.5, "london", s),
        "VWAP_LON_k2": lambda s: vwap_fade(g, 2.0, "london", s),
        "VWAP_SD_k1.5": lambda s: vwap_fade(g, 1.5, "sd", s),
        "VWAP_SD_k2": lambda s: vwap_fade(g, 2.0, "sd", s),
        "VWAP_SD_k2.5": lambda s: vwap_fade(g, 2.5, "sd", s),
        "VWAP_PULL_1.5": lambda s: vwap_pull(g, 1.5, s),
        "VWAP_PULL_2": lambda s: vwap_pull(g, 2.0, s),
        "ORB5_1130": lambda s: orb(g, 5, "time", s),
        "ORB15_1130": lambda s: orb(g, 15, "time", s),
        "ORB30_1130": lambda s: orb(g, 30, "time", s),
        "ORB5_2R": lambda s: orb(g, 5, "2r", s),
        "ORB15_2R": lambda s: orb(g, 15, "2r", s),
        "ORB30_2R": lambda s: orb(g, 30, "2r", s),
        "IBS15": lambda s: ibs_trades(g, 0.15, 1, "open", s),
        "IBS20": lambda s: ibs_trades(g, 0.20, 1, "open", s),
        "IBS30": lambda s: ibs_trades(g, 0.30, 1, "open", s),
        "IBS20_DAY": lambda s: ibs_trades(g, 0.20, 1, "close", s),
        "IBS80_SHORT": lambda s: ibs_trades(g, 0.80, -1, "open", s),
        "ON_ALL": lambda s: ibs_trades(g, 1.01, 1, "open", s),
        "SWEEP": lambda s: sweep(g, s),
        "FVG": lambda s: fvg(g, s),
    }
    print("running", len(builders), "cells")
    cells = {k: fn(1) for k, fn in builders.items()}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for name, tr in cells.items():
        rec = {"cell": name}
        for lab, sp in ("train", TRAIN), ("val", VAL), ("test", TEST), ("hold", HOLD):
            s = pack(tr, *sp)
            rec[f"{lab}_net"] = s["net"]
            rec[f"{lab}_n"] = s["n"]
            rec[f"{lab}_pf"] = s["pf"]
            rec[f"{lab}_dd"] = s.get("dd", 0.0)
            rec[f"{lab}_win"] = s["win"]
        rows.append(rec)
    tab = pd.DataFrame(rows).sort_values("cell")
    tab.to_csv(os.path.join(HERE, "results", "cells.csv"), index=False)
    cols = ["cell", "train_net", "train_n", "train_pf", "val_net", "val_n", "test_net", "hold_net", "train_dd", "val_dd"]
    print(tab[cols].to_string(index=False))

    families = {
        "VWAP_FADE": ["VWAP_k1", "VWAP_k1.5", "VWAP_k2", "VWAP_k2.5"],
        "VWAP_LONDON": ["VWAP_LON_k1.5", "VWAP_LON_k2"],
        "VWAP_STDEV": ["VWAP_SD_k1.5", "VWAP_SD_k2", "VWAP_SD_k2.5"],
        "VWAP_PULL": ["VWAP_PULL_1.5", "VWAP_PULL_2"],
        "ORB": ["ORB5_1130", "ORB15_1130", "ORB30_1130", "ORB5_2R", "ORB15_2R", "ORB30_2R"],
        "IBS": ["IBS15", "IBS20", "IBS30"],
        "SWEEP": ["SWEEP"],
        "FVG": ["FVG"],
    }
    print("\nFAMILIES")
    for fam, members in families.items():
        fold_nets = []
        for ta, tb, va, vb in FOLDS:
            pick = max(members, key=lambda m: pack(cells[m], ta, tb)["net"])
            fold_nets.append(pack(cells[pick], va, vb)["net"])
        chosen = max(members, key=lambda m: float(tab.loc[tab.cell == m, "train_net"].iloc[0]))
        row = tab[tab.cell == chosen].iloc[0]
        s3 = builders[chosen](3)
        v3, t3 = pack(s3, *VAL)["net"], pack(s3, *TEST)["net"]
        pos = sum(x > 0 for x in fold_nets)
        ok = row.val_net > 0 and row.test_net > 0 and sum(fold_nets) > 0 and pos >= 4 and v3 > 0 and t3 > 0
        if fam == "IBS":
            vt = pack(cells[chosen], VAL[0], TEST[1])["net"]
            on = pack(cells["ON_ALL"], VAL[0], TEST[1])["net"]
            ok = ok and vt > on
            extra = f" val+test {vt} vs always-long {on}"
        else:
            extra = ""
        print(f"{fam:12} {chosen:14} train {row.train_net:8} val {row.val_net:8} test {row.test_net:8} "
              f"wf {sum(fold_nets):8.0f} {pos}/6 hold {row.hold_net:8} 3t {v3}/{t3} candidate {ok}{extra}")


if __name__ == "__main__":
    main()
