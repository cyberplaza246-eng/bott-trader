"""Fresh rules. Not required to beat the bot. A daily rule is worth keeping only if
2010-2019, 2020-2022, and 2023-2024 are all positive. 2025-2026 is shown and not used to decide.

Daily, NQ=F, MNQ dollars ($2/point, $4, 1 tick):
  RSI2_5 / RSI2_10   RSI(2) under 5 or 10 and close above the 200-day average.
                     Buy the next open. Sell when RSI(2) closes back above 65, or after 5 days.
  RSI2_5_ANY         Same with a threshold of 5, no 200-day filter.
  DOWN3              Three lower closes in a row. Buy the next open, sell that close.
  BB                 Close under the lower Bollinger band (20, 2). Buy the next open.
                     Sell at the 20-day average or after 3 days.

Intraday, MNQ 5-minute, train 2025 / 2026 H1 / Jul-Sep 21:
  LUNCH    Fade the 09:30-12:00 move at 12:00. Out at 14:00. Stop 40 points.
  POWER    At 15:00, go with the day's direction if the move is at least a quarter of
           yesterday's range. Out at 15:55. Stop 30 points.
  COIL     If the 11:00-13:00 range is under 40% of the morning range, trade the first
           close outside it. Stop at the other side. Out at 15:55.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import backtest_revamp_live_stack as bt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PV, COMM, TICK = 2.0, 4.0, 0.25


def rsi2(close: np.ndarray) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    au = pd.Series(up).ewm(alpha=1 / 2, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1 / 2, adjust=False).mean().to_numpy()
    rs = np.divide(au, ad, out=np.full_like(au, np.inf), where=ad > 0)
    return 100 - 100 / (1 + rs)


def load_daily() -> pd.DataFrame:
    raw = yf.download("NQ=F", start="2009-01-01", interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna().astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    rng = df.High - df.Low
    bad = rng > 6 * rng.rolling(20, min_periods=10).median()
    df = df.loc[~bad.fillna(False)].copy()
    df["rsi"] = rsi2(df.Close.to_numpy())
    df["sma"] = df.Close.rolling(200).mean()
    df["mid"] = df.Close.rolling(20).mean()
    df["sd"] = df.Close.rolling(20).std()
    print(f"daily {df.index.min().date()} .. {df.index.max().date()} n={len(df)}")
    return df


def _row(day, entry, exit_px, side=1):
    return {"day": str(day)[:10], "pnl": side * (exit_px - entry) * PV - COMM,
            "ret": side * (exit_px - entry) / entry}


def run_rsi(df, thr, use_sma, max_hold=5):
    o, c, r, s = df.Open.to_numpy(), df.Close.to_numpy(), df.rsi.to_numpy(), df.sma.to_numpy()
    idx = df.index
    rows, i = [], 200
    while i < len(df) - 1:
        if r[i] < thr and (not use_sma or (np.isfinite(s[i]) and c[i] > s[i])):
            entry = o[i + 1] + TICK
            exit_i = min(i + 1 + max_hold, len(df) - 1)
            for j in range(i + 1, exit_i + 1):
                if r[j] > 65 or j == exit_i:
                    rows.append(_row(idx[i + 1], entry, c[j] - TICK))
                    i = j
                    break
        i += 1
    return pd.DataFrame(rows)


def run_down3(df):
    c, o = df.Close.to_numpy(), df.Open.to_numpy()
    rows = []
    for i in range(3, len(df) - 1):
        if c[i] < c[i - 1] < c[i - 2] < c[i - 3]:
            rows.append(_row(df.index[i + 1], o[i + 1] + TICK, c[i + 1] - TICK))
    return pd.DataFrame(rows)


def run_bb(df):
    o, c, mid, sd = df.Open.to_numpy(), df.Close.to_numpy(), df.mid.to_numpy(), df.sd.to_numpy()
    rows, i = [], 21
    while i < len(df) - 1:
        if np.isfinite(sd[i]) and c[i] < mid[i] - 2 * sd[i]:
            entry = o[i + 1] + TICK
            exit_i = min(i + 4, len(df) - 1)
            for j in range(i + 1, exit_i + 1):
                if c[j] >= mid[j] or j == exit_i:
                    rows.append(_row(df.index[i + 1], entry, c[j] - TICK))
                    i = j
                    break
        i += 1
    return pd.DataFrame(rows)


def pack(tr, a, b):
    if tr is None or tr.empty:
        return 0.0, 0, 0.0, 0.0
    t = tr[(tr.day >= a) & (tr.day < b)]
    if t.empty:
        return 0.0, 0, 0.0, 0.0
    pnl = t.pnl.to_numpy()
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    gl = -pnl[pnl < 0].sum()
    pf = float(pnl[pnl > 0].sum() / gl) if gl else 0.0
    return round(float(pnl.sum()), 0), int(len(t)), round(pf, 2), round(dd, 0)


def load_5m() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "data", "MNQ_1m.csv"), usecols=["datetime", "open", "high", "low", "close", "volume"])
    m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
    m = m.set_index("datetime").sort_index().loc["2024-11-01":]
    m = m[~m.index.duplicated(keep="last")].astype(float)
    m, _ = bt.clean_1m(m)
    x = m.tz_convert("America/New_York")
    g = x.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last")).dropna()
    y = pd.read_csv(os.path.join(ROOT, "research", "live_audit_2026-11-07", "NQ_5m_yahoo.csv"))
    y["datetime"] = pd.to_datetime(y["datetime"], utc=True)
    y = y.set_index("datetime").sort_index()[["open", "high", "low", "close"]].astype(float)
    y, _ = bt.clean_1m(y)
    y = y[y.index > m.index.max()].tz_convert("America/New_York")
    g = pd.concat([g, y]).sort_index()
    g = g[~g.index.duplicated(keep="first")]
    g["mins"] = g.index.hour * 60 + g.index.minute
    g["day"] = g.index.strftime("%Y-%m-%d")
    return g


def walk_day(g, day, ent_min, side, stop_pts, exit_min):
    d = g[g.day == day]
    ent = d[d.mins == ent_min]
    if ent.empty or side == 0:
        return None
    i0 = ent.index[0]
    entry = float(ent.open.iloc[0]) + side * TICK
    stop = entry - side * stop_pts
    after = d[d.index >= i0]
    for ts, bar in after.iterrows():
        if bar.mins >= exit_min and ts != i0:
            exit_px = float(bar.open) - side * TICK
            return {"day": day, "pnl": side * (exit_px - entry) * PV - COMM}
        hit = bar.low <= stop if side > 0 else bar.high >= stop
        if hit:
            raw = bar.open if (side > 0 and bar.open < stop) or (side < 0 and bar.open > stop) else stop
            return {"day": day, "pnl": side * (raw - side * TICK - entry) * PV - COMM}
    return None


def intraday(g):
    lunch, power, coil = [], [], []
    for day, d in g.groupby("day", sort=True):
        rth = d[(d.mins >= 570) & (d.mins <= 955)]
        if len(rth) < 20 or int(rth.mins.iloc[0]) != 570:
            continue
        # lunch fade
        noon = rth[rth.mins == 720]
        op = rth[rth.mins == 570]
        if len(noon) and len(op):
            side = -1 if float(noon.close.iloc[0]) > float(op.open.iloc[0]) else 1
            rec = walk_day(g, day, 725, side, 40, 840)
            if rec:
                lunch.append(rec)
        # power hour
        p = rth[rth.mins == 900]
        prev = g[(g.day < day) & (g.mins >= 570) & (g.mins <= 955)]
        if len(p) and len(op) and len(prev):
            yday = prev[prev.day == prev.day.iloc[-1]]
            yrange = float(yday.high.max() - yday.low.min()) if len(yday) else 0
            move = float(p.close.iloc[0]) - float(op.open.iloc[0])
            if yrange > 0 and abs(move) >= 0.25 * yrange:
                rec = walk_day(g, day, 905, 1 if move > 0 else -1, 30, 955)
                if rec:
                    power.append(rec)
        # coil
        morning = rth[(rth.mins >= 570) & (rth.mins < 660)]
        mid = rth[(rth.mins >= 660) & (rth.mins < 780)]
        if len(morning) > 5 and len(mid) > 5:
            mr = float(morning.high.max() - morning.low.min())
            cr_hi, cr_lo = float(mid.high.max()), float(mid.low.min())
            if mr > 0 and (cr_hi - cr_lo) < 0.4 * mr:
                later = rth[rth.mins >= 780]
                for ts, bar in later.iterrows():
                    if bar.close > cr_hi or bar.close < cr_lo:
                        side = 1 if bar.close > cr_hi else -1
                        # enter next 5-min bar
                        nxt = rth[rth.index > ts]
                        if nxt.empty:
                            break
                        ent_min = int(nxt.mins.iloc[0])
                        stop_pts = abs(float(nxt.open.iloc[0]) - (cr_lo if side > 0 else cr_hi))
                        if stop_pts < 2:
                            break
                        rec = walk_day(g, day, ent_min, side, stop_pts, 955)
                        if rec:
                            coil.append(rec)
                        break
    return {"LUNCH": pd.DataFrame(lunch), "POWER": pd.DataFrame(power), "COIL": pd.DataFrame(coil)}


def show(title, cells, windows):
    print("\n" + title)
    for name, tr in cells.items():
        bits = []
        flags = []
        for lab, a, b in windows:
            net, n, pf, dd = pack(tr, a, b)
            bits.append(f"{lab} {net:+.0f} n{n} pf{pf}")
            flags.append(net > 0)
        mark = "KEEP" if all(flags[:-1]) and flags[0] else "no"
        print(f"  {name:12} {mark:4}  " + " | ".join(bits))


def main():
    df = load_daily()
    daily = {
        "RSI2_5": run_rsi(df, 5, True),
        "RSI2_10": run_rsi(df, 10, True),
        "RSI2_5_ANY": run_rsi(df, 5, False),
        "DOWN3": run_down3(df),
        "BB": run_bb(df),
    }
    show("DAILY", daily, (
        ("2010s", "2010-01-01", "2020-01-01"),
        ("2020-22", "2020-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01"),
        ("2025-26", "2025-01-01", "2027-01-01"),
    ))
    intra = intraday(load_5m())
    show("INTRADAY", intra, (
        ("2025", "2025-01-01", "2026-01-01"),
        ("2026H1", "2026-01-01", "2026-07-01"),
        ("JulSep", "2026-07-01", "2026-09-22"),
        ("hold", "2026-09-22", "2027-01-01"),
    ))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, tr in {**daily, **intra}.items():
        if tr is not None and len(tr):
            tr.to_csv(os.path.join(HERE, "results", f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
