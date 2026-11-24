"""Backtest the LIVE revamp stack (start_mnq_live_revamp.bat) on 1m MNQ bars, then hunt add-on entries.

Mirrors start_live_rithmic.py with REVAMP_EXITS=1, overnight on:
  - research 5m ADX ensemble (trend / MR / breakout), revamp targets
    (trend 1.5R, breakout 2R, MR far band), overnight = trend sleeve only
  - add-ons ONL reclaim + E1 6 PM long (flat only, 2R, 24-bar time stop, no $ lock)
  - $ lock: +$100 peak -> SL +$20; from +$150 peak keep 50% of peak (4-tick buffer)
  - max 3 open; extra trade needs same direction + every open trade in profit
  - cooldown 1 bar after a win / 2 after a loss, 10 min after a loss,
    3 losses in a row -> 30 min pause
  - session limits: day -$500, overnight -$300, daily +$2,000; $500 max stop risk

Not modelled: re-entry arms, adopt/manual trades, broker rejects. Fills: market entry at the next
1m open + 1 tick, stops +1 tick slippage, TP limits at price; $3 round-trip commission.
Inside a 1m bar that touches both SL and TP the stop is assumed first.

Usage:
  python scripts/backtest_revamp_live_stack.py            # baseline + candidate hunt
  python scripts/backtest_revamp_live_stack.py --baseline # baseline only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core import addon_setups as ax  # noqa: E402
from src.research_ensemble.ensemble import EnsembleStrategy  # noqa: E402
from src.research_ensemble.indicators import adx, atr as ens_atr  # noqa: E402

ET = "America/New_York"
TICK = 0.25
PV = 2.0
COMM_RT = 3.0
SLIP = TICK

LOCK_ACTIVATE, LOCK_KEEP, LOCK_TRAIL_START, LOCK_KEEP_PCT, LOCK_BUF_TICKS = 100.0, 20.0, 150.0, 0.5, 4
TREND_TP_R, BREAKOUT_TP_R = 1.5, 2.0
MAX_OPEN = 3
MAX_RISK_USD = 500.0
DAY_LOSS, OVN_LOSS, DAILY_TARGET = 500.0, 300.0, 2000.0
CD_WIN_BARS, CD_LOSS_BARS = 1, 2
LOSS_CD_MIN, CONSEC_MAX, CONSEC_PAUSE_MIN = 10, 3, 30

PERIODS = [
    ("P1 Nov24-Apr25", "2024-11-01", "2025-05-01"),
    ("P2 May25-Oct25", "2025-05-01", "2025-11-01"),
    ("P3 Nov25-Mar26", "2025-11-01", "2026-04-01"),
    ("P4 Apr26-Sep26", "2026-04-01", "2026-10-01"),
]


# ----------------------------------------------------------------------------- data

def clean_1m(m1: pd.DataFrame, min_dev: float = 60.0, vol_mult: float = 8.0,
             persist_bars: int = 20, gap_min: float = 30.0) -> tuple[pd.DataFrame, int]:
    """Keep one contract's price track. MNQ_1m.csv interleaves the front and next contract
    around rolls (runs of minutes ~150-350 pts apart). A bar that jumps more than
    max(min_dev, vol_mult x median 1m move) from the track is held; held bars become the new
    track only if they persist `persist_bars` consistent minutes (real gaps / the roll itself),
    otherwise they are dropped when price returns. Any bar after a >gap_min minute gap resets
    the track (session reopen). Wicks/opens that leave the track are clipped to the body."""
    t = m1.index.values
    o, h, l, c = (m1[k].to_numpy(dtype=float).copy() for k in ("open", "high", "low", "close"))
    # bar range, not close-to-close: alternating contracts inflate close diffs to ~the spread
    vol = (m1["high"] - m1["low"]).rolling(61, center=True, min_periods=10).median().fillna(5.0).to_numpy()
    thr = np.maximum(min_dev, vol_mult * vol)
    keep = np.zeros(len(c), dtype=bool)
    gap = np.diff(t, prepend=t[0]) > np.timedelta64(int(gap_min * 60), "s")
    ref = c[0]
    pending: list[int] = []
    for i in range(len(c)):
        if gap[i] or i == 0:
            pending = []
            ref = c[i]
            keep[i] = True
            continue
        if abs(c[i] - ref) <= thr[i]:
            pending = []
            ref = c[i]
            keep[i] = True
            continue
        if pending and abs(c[i] - c[pending[-1]]) > thr[i]:
            pending = []
        pending.append(i)
        if len(pending) >= persist_bars:
            keep[pending] = True
            ref = c[i]
            pending = []
    out = m1[keep].copy()
    thr_k = thr[keep]
    prev_close = out["close"].shift().fillna(out["open"]).to_numpy()
    oc, cc = out["open"].to_numpy(), out["close"].to_numpy()
    bad_open = (np.abs(oc - cc) > thr_k) & (np.abs(prev_close - cc) <= thr_k)
    out.loc[bad_open, "open"] = prev_close[bad_open]
    body_hi = np.maximum(out["open"], out["close"])
    body_lo = np.minimum(out["open"], out["close"])
    out["high"] = out["high"].where(out["high"] - body_hi <= thr_k, body_hi)
    out["low"] = out["low"].where(body_lo - out["low"] <= thr_k, body_lo)
    return out, int((~keep).sum())


def load_bars(path: str, clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = pd.read_csv(path)
    m1["datetime"] = pd.to_datetime(m1["datetime"], utc=True)
    m1 = m1.set_index("datetime").sort_index()
    m1 = m1[~m1.index.duplicated(keep="last")][["open", "high", "low", "close", "volume"]].astype(float)
    if clean:
        m1, dropped = clean_1m(m1)
        print(f"cleaned 1m: dropped {dropped:,} off-contract / spike bars")
    m5 = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return m1, m5


def session_of(ts_utc: pd.Timestamp) -> tuple[str, pd.Timestamp]:
    t = ts_utc.tz_convert(ET)
    d = t.normalize().tz_localize(None)
    mins = t.hour * 60 + t.minute
    wd = t.weekday()
    if wd < 5 and 570 <= mins < 960:
        return "day", d
    nd = d
    if wd >= 5 or mins >= 960:
        nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:
        nd += pd.Timedelta(days=1)
    return "overnight", nd


# ----------------------------------------------------------------------------- signals

def ensemble_signals(m5: pd.DataFrame) -> pd.DataFrame:
    """Per 5m bar: dir, sl, tp, sleeve for the live revamp router (overnight entries allowed)."""
    strat = EnsembleStrategy(rth_only=False, revamp_exits=True, breakout_target_r=BREAKOUT_TP_R, mr_target="bb_far")
    sig = strat.generate_signals(m5)
    a = adx(m5["high"], m5["low"], m5["close"], 14)
    bo = strat._breakout.generate_signals(m5).entries
    ranging = a < strat.adx_range_max
    trending = a >= strat.adx_trend_min
    bo_fires = (bo != 0) & ~ranging
    sleeve = np.where(bo_fires, "breakout", np.where(trending, "trend", np.where(ranging, "mean_reversion", "trend")))
    out = pd.DataFrame({"dir": sig.entries.astype(int), "sl": sig.stop_price, "tp": sig.target_price,
                        "sleeve": sleeve, "adx": a, "atr": ens_atr(m5["high"], m5["low"], m5["close"], 14)},
                       index=m5.index)
    c = m5["close"]
    trend = (out.sleeve == "trend") & (out.dir != 0)
    out.loc[trend, "tp"] = c[trend] + out.dir[trend] * TREND_TP_R * (c[trend] - out.sl[trend]).abs()
    out = out[(out.dir != 0) & np.isfinite(out.sl)]
    out["tag"] = "ens_" + out["sleeve"]
    out["lock"] = True
    out["flat_only"] = False
    out["max_bars"] = 0
    return out


def featured(m5: pd.DataFrame) -> pd.DataFrame:
    b = m5[["open", "high", "low", "close"]].copy()
    b.index = b.index.tz_convert(ET)
    b = ax.add_features(b)
    pre = ((~b.rth) & (b.mins < 570)) | (b.mins >= 1080)
    onh = b.high.where(pre).groupby(b.tday).cummax()
    b["onh"] = onh.groupby(b.tday).ffill()
    rth_hi = b.high.where(b.rth).groupby(b.tday).max()
    rth_lo = b.low.where(b.rth).groupby(b.tday).min()
    b["pdh"] = b.tday.map(rth_hi.shift(1))
    b["pdl"] = b.tday.map(rth_lo.shift(1))
    ib = b.rth & (b.mins < 630)
    b["ibh"] = b.tday.map(b.high.where(ib).groupby(b.tday).max())
    b["ibl"] = b.tday.map(b.low.where(ib).groupby(b.tday).min())
    b["ema50"] = b.close.ewm(span=50, adjust=False).mean()
    b["ema200"] = b.close.ewm(span=200, adjust=False).mean()
    return b


def bracket_rows(b: pd.DataFrame, mask: pd.Series, direction: int, tag: str, first_per_day: bool = False,
                 stop_atr: float = ax.STOP_ATR, target_r: float = ax.TARGET_R, max_bars: int = ax.MAX_BARS) -> pd.DataFrame:
    m = mask.fillna(False) & np.isfinite(b.atr) & (b.atr > 0)
    if first_per_day:
        m = m & (m.groupby(b.tday).cumsum() == 1)
    rows = b[m]
    sd = np.maximum(stop_atr * rows.atr, ax.STOP_MIN_PTS)
    ref = (rows.close / TICK).round() * TICK
    out = pd.DataFrame({"dir": direction, "sl": ((ref - direction * sd) / TICK).round() * TICK,
                        "tp": ((ref + direction * target_r * sd) / TICK).round() * TICK,
                        "sleeve": "addon", "tag": tag, "lock": False, "flat_only": True, "max_bars": max_bars},
                       index=rows.index.tz_convert("UTC"))
    return out


def addon_rows(b: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    onl_mask = b.onl_hit & b.onl_complete
    return {
        "addon_onl": bracket_rows(b, onl_mask, 1, "addon_onl", first_per_day=True),
        "addon_e1": bracket_rows(b, b.e1_hit, 1, "addon_e1"),
    }


def candidate_rows(b: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """New entry ideas, all on the frozen add-on bracket (1.25 ATR stop, 2R, 24 bars, flat-only)."""
    am = (b.mins >= ax.ONL_FIRST_MIN) & (b.mins <= ax.ONL_LAST_MIN) & b.rth
    pre = ((~b.rth) & (b.mins < 570)) | (b.mins >= 1080)
    c: Dict[str, pd.DataFrame] = {}
    c["onh_reject_short"] = bracket_rows(b, am & b.onl_complete & (b.high > b.onh) & (b.close < b.onh), -1,
                                         "onh_reject_short", first_per_day=True)
    c["e1_short_6pm"] = bracket_rows(b, (~b.rth) & (b.mins >= ax.E1_FIRST_MIN) & (b.mins <= ax.E1_LAST_MIN)
                                     & (b.rsi <= 100 - ax.E1_RSI_MIN) & (b.ret12 <= -ax.E1_MOVE_ATR_MIN), -1, "e1_short_6pm")
    day = b.rth & (b.mins < 900)
    c["pdl_reclaim_long"] = bracket_rows(b, day & (b.low < b.pdl) & (b.close > b.pdl), 1, "pdl_reclaim_long", True)
    c["pdh_reject_short"] = bracket_rows(b, day & (b.high > b.pdh) & (b.close < b.pdh), -1, "pdh_reject_short", True)
    late = b.rth & (b.mins >= 630) & (b.mins < 900)
    c["ib_break_long_trend"] = bracket_rows(b, late & (b.close > b.ibh) & (b.close.shift() <= b.ibh)
                                            & (b.close > b.ema200), 1, "ib_break_long_trend", True)
    c["ib_break_short_trend"] = bracket_rows(b, late & (b.close < b.ibl) & (b.close.shift() >= b.ibl)
                                             & (b.close < b.ema200), -1, "ib_break_short_trend", True)
    c["onl_reclaim_ovn_long"] = bracket_rows(b, pre & (b.mins < 540) & (b.mins >= 120) & (b.low < b.onl.shift())
                                             & (b.close > b.onl.shift()), 1, "onl_reclaim_ovn_long", True)
    c["e1_long_9pm"] = bracket_rows(b, (~b.rth) & (b.mins >= 1260) & (b.mins <= 1315)
                                    & (b.rsi >= ax.E1_RSI_MIN) & (b.ret12 >= ax.E1_MOVE_ATR_MIN), 1, "e1_long_9pm")
    c["open_drive_long"] = bracket_rows(b, b.rth & (b.mins >= 570) & (b.mins <= 600) & (b.ret12 >= 1.48)
                                        & (b.rsi >= 59.5) & (b.close > b.ema200), 1, "open_drive_long", True)
    c["open_drive_short"] = bracket_rows(b, b.rth & (b.mins >= 570) & (b.mins <= 600) & (b.ret12 <= -1.48)
                                         & (b.rsi <= 40.5) & (b.close < b.ema200), -1, "open_drive_short", True)
    return c


# ----------------------------------------------------------------------------- simulator

@dataclass
class Pos:
    tag: str
    dir: int
    entry: float
    sl: float
    tp: float
    lock: bool
    entry_time: pd.Timestamp
    exit_after: Optional[pd.Timestamp]
    session: str
    tday: pd.Timestamp
    slip: float = SLIP
    peak_pts: float = 0.0


@dataclass
class Trade:
    tag: str
    dir: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    pnl: float
    reason: str
    session: str


@dataclass
class Rules:
    overnight_sleeves: str = "trend"     # 'trend' | 'all' | 'none' | comma list e.g. 'trend,breakout'
    rth_main_only: bool = False
    disabled_tags: tuple = ()
    max_open: int = MAX_OPEN
    ovn_extra_slip_ticks: int = 0        # extra slippage per side on overnight entries/stops

    def overnight_ok(self, sleeve: str) -> bool:
        if self.rth_main_only or self.overnight_sleeves == "none":
            return False
        if self.overnight_sleeves == "all":
            return True
        return sleeve in self.overnight_sleeves.split(",")


def _lock_stop(p: Pos, live: float) -> Optional[float]:
    live_pts = p.dir * (live - p.entry)
    if live_pts <= 0:
        return None
    peak = max(p.peak_pts, live_pts)
    peak_usd = peak * PV
    if peak_usd + 1e-9 < LOCK_ACTIVATE:
        return None
    lock_usd = LOCK_KEEP
    if peak_usd + 1e-9 >= LOCK_TRAIL_START:
        lock_usd = max(lock_usd, LOCK_KEEP_PCT * peak_usd)
    keep = math.floor((lock_usd / PV) / TICK + 1e-9) * TICK
    if keep <= 0:
        return None
    new_sl = p.entry + p.dir * keep
    buf = TICK * LOCK_BUF_TICKS
    if p.dir > 0 and new_sl > live - buf + 1e-9:
        return None
    if p.dir < 0 and new_sl < live + buf - 1e-9:
        return None
    if (p.dir > 0 and new_sl <= p.sl) or (p.dir < 0 and new_sl >= p.sl):
        return None
    return new_sl


def simulate(m1: pd.DataFrame, m5_index: pd.DatetimeIndex, signals: pd.DataFrame, rules: Rules) -> List[Trade]:
    sig = signals.sort_index()
    sig = sig[~sig.tag.isin(rules.disabled_tags)]
    by_bar: Dict[pd.Timestamp, pd.DataFrame] = {t: g for t, g in sig.groupby(level=0)}
    t1 = m1.index.values
    o1, h1, l1, c1 = (m1[k].values for k in ("open", "high", "low", "close"))
    sess_cache: Dict[pd.Timestamp, tuple] = {}

    def sess(ts):
        k = ts.floor("5min")
        if k not in sess_cache:
            sess_cache[k] = session_of(k)
        return sess_cache[k]

    open_pos: List[Pos] = []
    trades: List[Trade] = []
    cooldown = 0
    loss_cd_until = pd.Timestamp.min.tz_localize("UTC")
    pause_until = loss_cd_until
    consec = 0
    sess_pnl: Dict[tuple, float] = {}
    day_pnl: Dict[pd.Timestamp, float] = {}

    def close(p: Pos, ts, px, reason):
        nonlocal cooldown, consec, loss_cd_until, pause_until
        pnl = p.dir * (px - p.entry) * PV - COMM_RT
        trades.append(Trade(p.tag, p.dir, p.entry_time, ts, p.entry, px, pnl, reason, p.session))
        sess_pnl[(p.session, p.tday)] = sess_pnl.get((p.session, p.tday), 0.0) + pnl
        day_pnl[p.tday] = day_pnl.get(p.tday, 0.0) + pnl
        cooldown = CD_WIN_BARS if pnl > 0 else CD_LOSS_BARS
        if pnl < 0:
            consec += 1
            loss_cd_until = ts + pd.Timedelta(minutes=LOSS_CD_MIN)
            if consec >= CONSEC_MAX:
                pause_until = ts + pd.Timedelta(minutes=CONSEC_PAUSE_MIN)
        elif pnl > 0:
            consec = 0

    idx1 = 0
    n1 = len(t1)
    bars = m5_index
    for bi, bt in enumerate(bars):
        bend = bt + pd.Timedelta(minutes=5)
        bt64, bend64 = bt.to_datetime64(), bend.to_datetime64()
        while idx1 < n1 and t1[idx1] < bt64:
            idx1 += 1
        j = idx1
        while j < n1 and t1[j] < bend64:
            ts = pd.Timestamp(t1[j]).tz_localize("UTC")
            for p in list(open_pos):
                if p.entry_time > ts:
                    continue
                hit_sl = (l1[j] <= p.sl) if p.dir > 0 else (h1[j] >= p.sl)
                hit_tp = (h1[j] >= p.tp) if p.dir > 0 else (l1[j] <= p.tp)
                if hit_sl:
                    gap = (o1[j] < p.sl) if p.dir > 0 else (o1[j] > p.sl)
                    px = (o1[j] if gap else p.sl) - p.dir * p.slip
                    open_pos.remove(p)
                    close(p, ts, px, "STOP_LOSS")
                    continue
                if hit_tp:
                    open_pos.remove(p)
                    close(p, ts, p.tp, "TAKE_PROFIT")
                    continue
                if p.exit_after is not None and ts + pd.Timedelta(minutes=1) >= p.exit_after:
                    open_pos.remove(p)
                    close(p, ts, c1[j] - p.dir * p.slip, "TIME")
                    continue
                if p.lock:
                    best = h1[j] if p.dir > 0 else l1[j]
                    p.peak_pts = max(p.peak_pts, p.dir * (best - p.entry))
                    nsl = _lock_stop(p, c1[j])
                    if nsl is not None:
                        p.sl = nsl
            j += 1
        idx1 = j

        # --- bar closed: entry decision for signals on bar bt
        if cooldown > 0:
            cd_block = True
            cooldown -= 1
        else:
            cd_block = False
        g = by_bar.get(bt)
        if g is None or j >= n1:
            continue
        if bend < loss_cd_until or bend < pause_until:
            continue
        kind, tday = sess(bend)
        lim = DAY_LOSS if kind == "day" else OVN_LOSS
        if lim and sess_pnl.get((kind, tday), 0.0) <= -lim:
            continue
        if DAILY_TARGET and day_pnl.get(tday, 0.0) >= DAILY_TARGET:
            continue
        main = g[~g.flat_only]
        addons = g[g.flat_only]
        chosen = None
        if len(main) and not cd_block and len(open_pos) < rules.max_open:
            r = main.iloc[0]
            ok = kind != "overnight" or rules.overnight_ok(r.sleeve)
            if ok and open_pos:
                px = c1[j - 1]
                ok = all(p.dir == r.dir for p in open_pos) and all(p.dir * (px - p.entry) > 0 for p in open_pos)
            if ok:
                chosen = r
        if chosen is None and len(addons) and not open_pos:
            chosen = addons.iloc[0]
        if chosen is None:
            continue
        ref = c1[j - 1]
        if abs(ref - chosen.sl) * PV > MAX_RISK_USD:
            continue
        slip = SLIP + (TICK * rules.ovn_extra_slip_ticks if kind == "overnight" else 0.0)
        fill = o1[j] + chosen.dir * slip
        tp = float(chosen.tp) if np.isfinite(chosen.tp) else fill + chosen.dir * 1e6
        if (chosen.dir > 0 and (fill <= chosen.sl or fill >= tp)) or (chosen.dir < 0 and (fill >= chosen.sl or fill <= tp)):
            continue
        et = pd.Timestamp(t1[j]).tz_localize("UTC")
        exit_after = bend + pd.Timedelta(minutes=5 * int(chosen.max_bars)) if chosen.max_bars else None
        open_pos.append(Pos(chosen.tag, int(chosen.dir), fill, float(chosen.sl), tp, bool(chosen.lock),
                            et, exit_after, kind, tday, slip))
    return trades


# ----------------------------------------------------------------------------- stats

def stats(tr: List[Trade]) -> dict:
    if not tr:
        return {"n": 0, "net": 0.0, "pf": 0.0, "wr": 0.0, "avg": 0.0, "dd": 0.0}
    p = np.array([t.pnl for t in tr])
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": int(len(p)), "net": round(float(p.sum()), 1), "pf": round(float(g / l), 2) if l > 0 else 99.0,
            "wr": round(100 * float((p > 0).mean()), 1), "avg": round(float(p.mean()), 2), "dd": round(dd, 1)}


def by_period(tr: List[Trade]) -> Dict[str, dict]:
    out = {}
    for name, a, b in PERIODS:
        lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        out[name] = stats([t for t in tr if lo <= t.entry_time < hi])
    return out


def by_key(tr: List[Trade], key: Callable[[Trade], str]) -> Dict[str, dict]:
    keys = sorted({key(t) for t in tr})
    return {k: stats([t for t in tr if key(t) == k]) for k in keys}


def in_window(tr: List[Trade]) -> List[Trade]:
    lo, hi = pd.Timestamp(PERIODS[0][1], tz="UTC"), pd.Timestamp(PERIODS[-1][2], tz="UTC")
    return [t for t in tr if lo <= t.entry_time < hi]


def fmt(s: dict) -> str:
    return f"n={s['n']:4d} net=${s['net']:9.1f} PF={s['pf']:5.2f} WR={s['wr']:5.1f}% avg=${s['avg']:7.2f} DD=${s['dd']:7.1f}"


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "MNQ_1m.csv"))
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "revamp_live_stack_backtest.json"))
    args = ap.parse_args()

    m1, m5 = load_bars(args.data)
    print(f"1m bars {len(m1):,}  {m1.index[0]} -> {m1.index[-1]}; 5m bars {len(m5):,}")
    ens = ensemble_signals(m5)
    b = featured(m5)
    adds = addon_rows(b)
    base_sig = pd.concat([ens] + list(adds.values()))
    print(f"signals: ensemble {len(ens):,}, ONL {len(adds['addon_onl'])}, E1 {len(adds['addon_e1'])}")

    report: dict = {"data": [str(m1.index[0]), str(m1.index[-1])], "variants": {}}

    def run(name: str, sig: pd.DataFrame, rules: Rules) -> List[Trade]:
        tr = in_window(simulate(m1, m5.index, sig, rules))
        tot, per = stats(tr), by_period(tr)
        report["variants"][name] = {"total": tot, "periods": per,
                                    "by_tag": by_key(tr, lambda t: t.tag),
                                    "by_session": by_key(tr, lambda t: t.session)}
        print(f"\n=== {name}\n  TOTAL {fmt(tot)}")
        for k, v in per.items():
            print(f"  {k:15s} {fmt(v)}")
        return tr

    base = run("baseline_live", base_sig, Rules())
    print("  by tag:")
    for k, v in by_key(base, lambda t: t.tag).items():
        print(f"    {k:22s} {fmt(v)}")
    print("  by session:")
    for k, v in by_key(base, lambda t: t.session).items():
        print(f"    {k:22s} {fmt(v)}")
    print("  by ET hour (entry):")
    for k, v in by_key(base, lambda t: f"{t.entry_time.tz_convert(ET).hour:02d}").items():
        print(f"    {k:22s} {fmt(v)}")

    if not args.baseline:
        base_per = report["variants"]["baseline_live"]["periods"]
        hunt: dict = {}

        def compare(name: str, tr: List[Trade]) -> None:
            per = by_period(tr)
            d = {k: round(per[k]["net"] - base_per[k]["net"], 1) for k in per}
            wins = sum(1 for v in d.values() if v > 0)
            tot = stats(tr)
            hunt[name] = {"delta_by_period": d, "periods_better": wins,
                          "delta_total": round(tot["net"] - report["variants"]["baseline_live"]["total"]["net"], 1),
                          "delta_trades": tot["n"] - report["variants"]["baseline_live"]["total"]["n"]}
            print(f"  -> vs baseline: {hunt[name]['delta_total']:+.1f} $, {hunt[name]['delta_trades']:+d} trades, "
                  f"better in {wins}/4 periods {d}")

        compare("overnight_all_sleeves", run("overnight_all_sleeves", base_sig, Rules(overnight_sleeves="all")))
        compare("rth_main_only", run("rth_main_only", base_sig, Rules(rth_main_only=True)))
        for tag in ("ens_mean_reversion", "ens_breakout", "ens_trend", "addon_onl", "addon_e1"):
            compare(f"drop_{tag}", run(f"drop_{tag}", base_sig, Rules(disabled_tags=(tag,))))
        for name, rows in candidate_rows(b).items():
            print(f"\n[{name}] {len(rows)} raw signals")
            alone = in_window(simulate(m1, m5.index, rows, Rules()))
            report["variants"][f"alone_{name}"] = {"total": stats(alone), "periods": by_period(alone)}
            print(f"  ALONE {fmt(stats(alone))}  " + " | ".join(f"{k[:2]} {v['net']:+.0f}" for k, v in by_period(alone).items()))
            compare(f"add_{name}", run(f"add_{name}", pd.concat([base_sig, rows]), Rules()))
        report["hunt"] = hunt

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
