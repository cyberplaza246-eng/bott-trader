"""Intraday MNQ backtest engine (research only, no broker / live code).

Locked conventions
- Inputs are UTC bar-START labelled OHLCV bars. Everything is converted to America/New_York.
- Regular session only: bars whose ET start is in [09:30, 16:00). Indicators are computed on
  the RTH-only series (continuous across days) and use only completed bars.
- A signal is evaluated at the CLOSE of bar i; entry is the OPEN of bar i+1 +/- slippage.
- Stops and time exits are market fills (slippage per side); limit targets fill at the limit
  only if price trades 1 tick through it. If a bar touches stop and target, stop is assumed.
- Risk: one position, max trades/day, daily realized loss stop, 3 consecutive losers/day,
  stop distance capped at 60 pts, flat by the strategy's time exit (<= 15:55 ET).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REPO = r"C:\Users\shawa\Projects\bott-trader"
ET = "America/New_York"

POINT_VALUE = 2.0
TICK = 0.25
COMMISSION_RT = 4.0
MAX_STOP_PTS = 60.0
DAILY_LOSS_STOP = -400.0
MAX_CONSEC_LOSSES = 3
DD_LIMIT = 2000.0

# ----------------------------------------------------------------------------- data


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    for p in (REPO, REPO + r"\scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import backtest_revamp_live_stack as bt  # noqa: E402

    return bt.clean_1m(df)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def load_all(main_csv: str, hold5_csv: str, hold1_csv: str, holdout_start: str) -> dict:
    """Returns cleaned bars and a data log. Main history is cut to >= 2025-01-01 before cleaning."""
    log = {}
    raw = _read(main_csv)
    log["main_raw_rows"] = len(raw)
    raw = raw[raw.index >= pd.Timestamp("2025-01-01", tz="UTC")]
    log["main_rows_from_2025"] = len(raw)
    m1, dropped = _clean(raw)
    log["main_clean_dropped"] = dropped
    log["main_first"], log["main_last"] = str(m1.index[0]), str(m1.index[-1])

    h5raw = _read(hold5_csv)
    h5, d5 = _clean(h5raw)
    h1raw = _read(hold1_csv)
    h1, d1 = _clean(h1raw)
    log.update(hold5_rows=len(h5raw), hold5_dropped=d5, hold1_rows=len(h1raw), hold1_dropped=d1,
               hold5_first=str(h5.index[0]), hold5_last=str(h5.index[-1]),
               hold1_first=str(h1.index[0]), hold1_last=str(h1.index[-1]))

    hs = pd.Timestamp(holdout_start, tz=ET).tz_convert("UTC")
    m5_main = resample(m1, "5min")
    # 5m series for the 5m strategies: main history before the holdout + Yahoo 5m from it
    m5_comb = pd.concat([m5_main[m5_main.index < hs], h5[h5.index >= hs]])
    return {"m1": m1, "m5": m5_main, "m5_comb": m5_comb, "h1": h1, "log": log}


def rth(df: pd.DataFrame) -> pd.DataFrame:
    """ET-indexed regular-session bars with helper columns."""
    x = df.copy()
    x.index = x.index.tz_convert(ET)
    mins = x.index.hour * 60 + x.index.minute
    x = x[(mins >= 570) & (mins < 960) & (x.index.weekday < 5)].copy()
    x["mins"] = x.index.hour * 60 + x.index.minute
    x["day"] = x.index.tz_localize(None).normalize()
    return x


# ----------------------------------------------------------------------------- indicators


def atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["close"].shift()
    tr = pd.concat([x["high"] - x["low"], (x["high"] - pc).abs(), (x["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def adx(x: pd.DataFrame, n: int = 14) -> pd.Series:
    up = x["high"].diff()
    dn = -x["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = x["close"].shift()
    tr = pd.concat([x["high"] - x["low"], (x["high"] - pc).abs(), (x["low"] - pc).abs()], axis=1).max(axis=1)
    a = 1.0 / n
    atr_w = tr.ewm(alpha=a, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(pdm, index=x.index).ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_w
    ndi = 100 * pd.Series(ndm, index=x.index).ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_w
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=n).mean()


def rth_resample(x: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample RTH bars into session-aligned buckets (09:30 anchor); index = bucket END (ET)."""
    b = ((x["mins"] - 570) // minutes).astype(int)
    g = x.groupby([x["day"], b])
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
    days = out.index.get_level_values(0)
    buckets = out.index.get_level_values(1)
    end = pd.DatetimeIndex(days) + pd.to_timedelta(570 + (buckets + 1) * minutes, unit="min")
    out.index = end.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
    return out[out.index.notna()]


def asof_completed(x: pd.DataFrame, higher: pd.DataFrame, cols: list[str], bar_minutes: int) -> pd.DataFrame:
    """For each bar of x, the latest higher-TF row whose END <= this bar's END (no look-ahead).
    Only rows from the same session day are used for bucket completion; indicator values carry
    history across days because they were computed on the continuous RTH series."""
    left = pd.DataFrame({"end": x.index + pd.Timedelta(minutes=bar_minutes)}, index=x.index)
    right = higher[cols].copy()
    right["hend"] = right.index
    right = right.sort_values("hend")
    m = pd.merge_asof(left.sort_values("end"), right, left_on="end", right_on="hend", direction="backward")
    m.index = left.sort_values("end").index
    return m.reindex(x.index)[cols]


# ----------------------------------------------------------------------------- signals


@dataclass
class Signals:
    """Per-bar signal arrays aligned to the execution bars."""
    dir: np.ndarray            # +1 / -1 / 0, evaluated at bar close
    stop: np.ndarray           # absolute stop price (NaN -> use stop_dist)
    stop_dist: np.ndarray      # stop distance from fill (NaN -> use stop)
    target: np.ndarray         # absolute limit target (NaN -> use target_r or none)
    target_r: float | None     # R-multiple target measured from the actual fill
    exit_min: int              # time exit, ET minutes of day (bar start >= this -> exit at open)
    entry_cutoff_min: int      # entry fill bar must start before this
    max_trades: int


def sig_orb(x1: pd.DataFrame, range_min: int, target_r: float) -> Signals:
    """1) Opening-range breakout on 1m bars."""
    n = len(x1)
    d = np.zeros(n)
    stop = np.full(n, np.nan)
    end = 570 + range_min
    for day, g in x1.groupby("day", sort=False):
        rng = g[(g["mins"] >= 570) & (g["mins"] < end)]
        if len(rng) < 0.8 * range_min or rng["mins"].iloc[0] != 570:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        win = g[(g["mins"] >= end) & (g["mins"] < end + 90)]
        up = win.index[win["close"] > hi]
        dn = win.index[win["close"] < lo]
        first_up = up[0] if len(up) else None
        first_dn = dn[0] if len(dn) else None
        if first_up is None and first_dn is None:
            continue
        if first_dn is None or (first_up is not None and first_up < first_dn):
            i = x1.index.get_loc(first_up)
            d[i], stop[i] = 1, lo
        else:
            i = x1.index.get_loc(first_dn)
            d[i], stop[i] = -1, hi
    return Signals(d, stop, np.full(n, np.nan), np.full(n, np.nan), target_r,
                   exit_min=12 * 60, entry_cutoff_min=12 * 60, max_trades=1)


def features_5m(x5: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=x5.index)
    f["atr"] = atr(x5, 14)
    f["ema5"] = x5["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    tp = (x5["high"] + x5["low"] + x5["close"]) / 3
    pv = (tp * x5["volume"]).groupby(x5["day"]).cumsum()
    cv = x5["volume"].groupby(x5["day"]).cumsum()
    f["vwap"] = (pv / cv.replace(0, np.nan)).fillna(tp)

    m15 = rth_resample(x5, 15)
    m15["ema"] = m15["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    m15["slope"] = m15["ema"] - m15["ema"].shift(3)
    m15["trend"] = np.where((m15["close"] > m15["ema"]) & (m15["slope"] > 0), 1,
                            np.where((m15["close"] < m15["ema"]) & (m15["slope"] < 0), -1, 0))
    t15 = asof_completed(x5, m15, ["trend"], 5)
    f["trend15"] = t15["trend"].fillna(0).to_numpy()
    # 15m bucket must belong to the same session day: first 15m bucket ends 09:45
    f.loc[x5["mins"] + 5 < 585, "trend15"] = 0

    m30 = rth_resample(x5, 30)
    m30["adx"] = adx(m30, 14)
    a30 = asof_completed(x5, m30, ["adx"], 5)
    f["adx30"] = a30["adx"].to_numpy()

    # day-level: opening 30-min range and full RTH range, both against PRIOR days only
    day = x5.groupby("day").agg(hi=("high", "max"), lo=("low", "min"))
    op = x5[x5["mins"] < 600].groupby("day").agg(ohi=("high", "max"), olo=("low", "min"), n=("open", "size"))
    day = day.join(op)
    day["orng"] = day["ohi"] - day["olo"]
    day["orng_med20_prior"] = day["orng"].shift(1).rolling(20, min_periods=20).median()
    day["adr20_prior"] = (day["hi"] - day["lo"]).shift(1).rolling(20, min_periods=20).mean()
    dd = day.reindex(x5["day"].to_numpy())
    for c in ("orng", "orng_med20_prior", "adr20_prior"):
        f[c] = dd[c].to_numpy()
    return f


def _same_day_prev(x5: pd.DataFrame) -> np.ndarray:
    return np.r_[False, x5["day"].to_numpy()[1:] == x5["day"].to_numpy()[:-1]]


def sig_vwap_mr(x5: pd.DataFrame, f: pd.DataFrame, k: float, regime: str | None = None) -> Signals:
    """2) VWAP mean reversion with opening-range trend-day filter (optional range regime gate)."""
    n = len(x5)
    c, h, l = (x5[k_].to_numpy() for k_ in ("close", "high", "low"))
    a, vw = f["atr"].to_numpy(), f["vwap"].to_numpy()
    sd = _same_day_prev(x5)
    pc, ph, pl = np.r_[np.nan, c[:-1]], np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    pa, pvw = np.r_[np.nan, a[:-1]], np.r_[np.nan, vw[:-1]]
    ok = sd & (x5["mins"].to_numpy() >= 600) & np.isfinite(a) & np.isfinite(pa)
    ok &= np.isfinite(f["orng_med20_prior"].to_numpy()) & (f["orng"].to_numpy() <= f["orng_med20_prior"].to_numpy())
    if regime == "range":
        ok &= f["adx30"].to_numpy() < 25
    short = ok & (pc - pvw >= k * pa) & (c < pc)
    long_ = ok & (pvw - pc >= k * pa) & (c > pc)
    d = np.where(short, -1, np.where(long_, 1, 0)).astype(float)
    stop = np.where(short, np.maximum(h, ph) + 0.75 * a, np.where(long_, np.minimum(l, pl) - 0.75 * a, np.nan))
    tgt = np.where(d != 0, vw, np.nan)
    return Signals(d, stop, np.full(n, np.nan), tgt, None, exit_min=955, entry_cutoff_min=930, max_trades=2)


def sig_pullback(x5: pd.DataFrame, f: pd.DataFrame, target_r: float, regime: str | None = None) -> Signals:
    """3) 15m-trend pullback to 5m EMA20, continuation close (optional trend regime gate)."""
    n = len(x5)
    o, c, h, l = (x5[k_].to_numpy() for k_ in ("open", "close", "high", "low"))
    e = f["ema5"].to_numpy()
    tr = f["trend15"].to_numpy()
    sd = _same_day_prev(x5)
    pe, ph, pl = np.r_[np.nan, e[:-1]], np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    ok = sd & np.isfinite(e) & np.isfinite(pe)
    if regime == "trend":
        ok &= f["adx30"].to_numpy() >= 25
    long_ = ok & (tr == 1) & (pl <= pe) & (c > o) & (c > e)
    short = ok & (tr == -1) & (ph >= pe) & (c < o) & (c < e)
    d = np.where(long_, 1, np.where(short, -1, 0)).astype(float)
    stop = np.where(long_, np.minimum(l, pl) - TICK, np.where(short, np.maximum(h, ph) + TICK, np.nan))
    return Signals(d, stop, np.full(n, np.nan), np.full(n, np.nan), target_r,
                   exit_min=955, entry_cutoff_min=930, max_trades=2)


def sig_momentum(x5: pd.DataFrame, f: pd.DataFrame, m: float, regime: str | None = None) -> Signals:
    """4) 09:30-10:30 move >= m x prior-20-day ADR -> enter at 10:30 in that direction."""
    n = len(x5)
    d = np.zeros(n)
    sdist = np.full(n, np.nan)
    mins = x5["mins"].to_numpy()
    adr = f["adr20_prior"].to_numpy()
    adxv = f["adx30"].to_numpy()
    for day, idx in x5.groupby("day", sort=False).indices.items():
        g_m = mins[idx]
        if g_m[0] != 570:
            continue
        sig = np.where(g_m == 625)[0]  # 10:25 bar closes at 10:30
        if not len(sig):
            continue
        i = idx[sig[0]]
        move = x5["close"].iat[i] - x5["open"].iat[idx[0]]
        if not np.isfinite(adr[i]) or abs(move) < m * adr[i] or move == 0:
            continue
        if regime == "trend" and not (adxv[i] >= 25):
            continue
        d[i] = np.sign(move)
        sdist[i] = 0.5 * abs(move)
    return Signals(d, np.full(n, np.nan), sdist, np.full(n, np.nan), None,
                   exit_min=955, entry_cutoff_min=640, max_trades=1)


# ----------------------------------------------------------------------------- simulator


@dataclass
class RunResult:
    trades: pd.DataFrame
    bars: pd.DataFrame  # per-bar equity: eq_close, eq_worst (MTM at adverse extreme)
    days: pd.Series     # daily net P&L (all session days, zeros included)


def simulate(x: pd.DataFrame, s: Signals, slip_ticks: int = 1) -> RunResult:
    slip = slip_ticks * TICK
    o, h, l, c = (x[k].to_numpy() for k in ("open", "high", "low", "close"))
    mins = x["mins"].to_numpy()
    day = x["day"].to_numpy()
    ts = x.index
    n = len(x)
    eq_close = np.zeros(n)
    eq_worst = np.zeros(n)
    trades = []
    realized = 0.0
    pos = 0
    entry = stp = tgt = np.nan
    entry_i = -1
    pending = None
    d_trades = d_pnl = d_consec = 0
    cur_day = None
    day_pnl = {}

    def close_trade(i, px, reason):
        nonlocal realized, pos, d_pnl, d_consec
        pnl = (px - entry) * pos * POINT_VALUE - COMMISSION_RT
        realized += pnl
        d_pnl += pnl
        d_consec = d_consec + 1 if pnl < 0 else 0
        trades.append(dict(entry_time=ts[entry_i], exit_time=ts[i], dir=pos, entry=entry, exit=px,
                           stop=stp, target=tgt, risk_pts=abs(entry - stp), pnl=pnl, reason=reason,
                           day=day[i]))
        pos = 0

    for i in range(n):
        if day[i] != cur_day:
            if pos != 0:  # safety: should not happen (forced exit below)
                close_trade(i - 1, c[i - 1] - pos * slip, "eod_force")
            if cur_day is not None:
                day_pnl[cur_day] = d_pnl
            cur_day = day[i]
            d_trades = d_pnl = d_consec = 0
            pending = None
        last_of_day = i == n - 1 or day[i + 1] != day[i]

        # 1) pending entry at this bar's open
        if pending is not None and pos == 0:
            pdir, pstop, pdist, ptgt = pending
            pending = None
            if mins[i] < s.entry_cutoff_min and mins[i] < s.exit_min:
                fill = o[i] + pdir * slip
                st = fill - pdir * pdist if np.isfinite(pdist) else pstop
                if pdir * (fill - st) > MAX_STOP_PTS:
                    st = fill - pdir * MAX_STOP_PTS
                tg = ptgt
                if s.target_r is not None:
                    tg = fill + pdir * s.target_r * abs(fill - st)
                valid = pdir * (fill - st) > 0 and (not np.isfinite(tg) or pdir * (tg - fill) > 0)
                if valid:
                    pos, entry, stp, tgt, entry_i = int(pdir), fill, st, tg, i
                    d_trades += 1

        # 2) manage open position on this bar
        if pos != 0:
            if mins[i] >= s.exit_min:
                close_trade(i, o[i] - pos * slip, "time")
            else:
                if pos == 1:
                    if o[i] <= stp:
                        close_trade(i, o[i] - slip, "stop_gap")
                    elif l[i] <= stp:
                        close_trade(i, stp - slip, "stop")
                    elif np.isfinite(tgt) and h[i] >= tgt + TICK:
                        close_trade(i, tgt, "target")
                else:
                    if o[i] >= stp:
                        close_trade(i, o[i] + slip, "stop_gap")
                    elif h[i] >= stp:
                        close_trade(i, stp + slip, "stop")
                    elif np.isfinite(tgt) and l[i] <= tgt - TICK:
                        close_trade(i, tgt, "target")
            if pos != 0 and last_of_day:
                close_trade(i, c[i] - pos * slip, "eod")

        # 3) mark to market (adverse extreme includes exit slippage + commission)
        if pos != 0:
            adverse = l[i] if pos == 1 else h[i]
            eq_worst[i] = realized + ((adverse - entry) * pos - slip) * POINT_VALUE - COMMISSION_RT
            eq_close[i] = realized + ((c[i] - entry) * pos - slip) * POINT_VALUE - COMMISSION_RT
        else:
            eq_worst[i] = eq_close[i] = realized

        # 4) evaluate signal at this bar's close
        if (pos == 0 and pending is None and not last_of_day and s.dir[i] != 0
                and d_trades < s.max_trades and d_pnl > DAILY_LOSS_STOP and d_consec < MAX_CONSEC_LOSSES):
            pending = (s.dir[i], s.stop[i], s.stop_dist[i], s.target[i])

    if cur_day is not None:
        day_pnl[cur_day] = d_pnl
    tr = pd.DataFrame(trades)
    bars = pd.DataFrame({"eq_close": eq_close, "eq_worst": eq_worst, "day": day}, index=ts)
    return RunResult(tr, bars, pd.Series(day_pnl, dtype=float).sort_index())


# ----------------------------------------------------------------------------- statistics


def drawdowns(bars: pd.DataFrame, days: pd.Series, limit: float = DD_LIMIT) -> dict:
    """Max DD (intraday MTM and end-of-day) re-based to 0 at the slice start, plus breach episodes."""
    if len(bars) == 0:
        return dict(mdd_mtm=0.0, mdd_eod=0.0, breaches=[])
    eqc = bars["eq_close"].to_numpy()
    eqw = bars["eq_worst"].to_numpy()
    dayv = bars["day"].to_numpy()
    peak, mdd = 0.0, 0.0  # slices are re-based so equity before the first bar is 0
    breaches: list[dict] = []
    in_breach = False
    for k in range(len(eqc)):
        dd = peak - eqw[k]
        mdd = max(mdd, dd)
        if dd > limit:
            if not in_breach:
                breaches.append({"date": str(pd.Timestamp(dayv[k]).date()), "dd": float(dd)})
                in_breach = True
            else:
                breaches[-1]["dd"] = max(breaches[-1]["dd"], float(dd))
        if eqc[k] > peak:
            peak = eqc[k]
            in_breach = False
    de = days.cumsum()
    de = pd.concat([pd.Series([0.0]), de.reset_index(drop=True)])
    mdd_eod = float((de.cummax() - de).max())
    return dict(mdd_mtm=float(mdd), mdd_eod=mdd_eod, breaches=breaches)


def slice_run(rr: RunResult, d0: str, d1: str) -> RunResult:
    a, b = pd.Timestamp(d0), pd.Timestamp(d1)
    tr = rr.trades
    if len(tr):
        tr = tr[(tr["day"] >= a) & (tr["day"] <= b)]
    mask = (rr.bars["day"] >= a) & (rr.bars["day"] <= b)
    bars = rr.bars[mask].copy()
    first = np.argmax(mask.to_numpy()) if mask.any() else 0
    pre = 0.0
    if mask.any() and first > 0:
        pre = float(rr.bars["eq_close"].iloc[first - 1])  # always flat at prior session's end
    bars["eq_close"] -= pre
    bars["eq_worst"] -= pre
    days = rr.days[(rr.days.index >= a) & (rr.days.index <= b)]
    return RunResult(tr, bars, days)


def stats(rr: RunResult) -> dict:
    tr, days = rr.trades, rr.days
    dd = drawdowns(rr.bars, days)
    out = dict(trades=int(len(tr)), session_days=int(len(days)))
    if len(tr) == 0:
        out.update(net=0.0, pf=np.nan, win_rate=np.nan, avg_trade=np.nan, avg_win=np.nan, avg_loss=np.nan,
                   longest_losing_streak=0, worst_day=float(days.min()) if len(days) else 0.0,
                   trades_per_day=0.0, t_stat_daily=np.nan, pnl_by_entry_hour={}, **dd)
        out["breach_count"] = len(dd["breaches"])
        return out
    p = tr["pnl"].to_numpy()
    wins, losses = p[p > 0], p[p <= 0]
    streak = best = 0
    for v in p:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    sd = days.std(ddof=1)
    t = float(days.mean() / (sd / np.sqrt(len(days)))) if len(days) > 1 and sd > 0 else np.nan
    hours = pd.to_datetime(tr["entry_time"]).dt.hour
    out.update(
        net=float(p.sum()),
        pf=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        win_rate=float(len(wins) / len(p)),
        avg_trade=float(p.mean()),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        longest_losing_streak=int(best),
        worst_day=float(days.min()),
        trades_per_day=float(len(p) / max(len(days), 1)),
        t_stat_daily=t,
        pnl_by_entry_hour={int(k): round(float(v), 2) for k, v in tr.groupby(hours.to_numpy())["pnl"].sum().items()},
        **dd,
    )
    out["breach_count"] = len(dd["breaches"])
    return out
