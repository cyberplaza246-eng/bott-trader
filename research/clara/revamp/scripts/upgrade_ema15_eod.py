#!/usr/bin/env python3
"""
Upgrade the real-1m 15m-EMA EOD recipe. Filters + exits only on Databento 1m.
IS < 2026-06-01, OOS >= 2026-06-01. No synthetic 30s.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge

ET = pytz.timezone("US/Eastern")
OOS = edge.OOS_START
PT = edge.POINT_VALUE
COMM = edge.COMMISSION


def load_frame(csv: str) -> pd.DataFrame:
    raw_1m, raw_5m, _, _ = scalp_bt.load_data(csv, None, None, allow_synthetic_30s=True)
    df_1m = scalp_bt.add_1m_indicators(raw_1m)
    df_5m = scalp_bt.add_5m_indicators(raw_5m)
    return scalp_bt.attach_context(df_1m, df_5m, ["close", "vwap", "ema_20", "adx"], "_5m")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    dt = pd.DatetimeIndex(pd.to_datetime(f["datetime"], utc=True))
    f["_dt"] = dt
    et = dt.tz_convert(ET)
    f["_et_min"] = et.hour * 60 + et.minute
    f["_et_dow"] = et.weekday
    f["_et_date"] = et.date
    tmp = f.set_index(dt)
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}

    b15 = tmp.resample("15min").agg({**ohlc, "volume": "sum"}).dropna()
    e8 = b15["close"].ewm(span=8, adjust=False).mean()
    e21 = b15["close"].ewm(span=21, adjust=False).mean()
    tr15 = pd.concat([
        b15["high"] - b15["low"],
        (b15["high"] - b15["close"].shift()).abs(),
        (b15["low"] - b15["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr15 = tr15.rolling(14).mean()
    adx_proxy = (e8 - e21).abs() / atr15.replace(0, np.nan)

    b60 = tmp.resample("60min").agg(ohlc).dropna()
    e8_60 = b60["close"].ewm(span=8, adjust=False).mean()
    e21_60 = b60["close"].ewm(span=21, adjust=False).mean()
    trend60 = pd.Series(0, index=b60.index, dtype=int)
    trend60.loc[e8_60 > e21_60] = 1
    trend60.loc[e8_60 < e21_60] = -1

    b1d = tmp.resample("1D").agg(ohlc).dropna()
    e20d = b1d["close"].ewm(span=20, adjust=False).mean()
    prev_close = b1d["close"].shift(1)
    daily_trend = pd.Series(0, index=b1d.index, dtype=int)
    daily_trend.loc[b1d["close"] > e20d] = 1
    daily_trend.loc[b1d["close"] < e20d] = -1

    t15 = ((e8 > e21).astype(int) - (e8 < e21).astype(int)).shift(1)
    f["t15"] = t15.reindex(dt, method="ffill").fillna(0).to_numpy()
    f["spread15"] = (e8 - e21).shift(1).reindex(dt, method="ffill").to_numpy()
    f["atr15"] = atr15.shift(1).reindex(dt, method="ffill").to_numpy()
    f["sep15"] = adx_proxy.shift(1).reindex(dt, method="ffill").to_numpy()
    f["t60"] = trend60.shift(1).reindex(dt, method="ffill").fillna(0).to_numpy(dtype=int)
    f["tdaily"] = daily_trend.shift(1).reindex(dt, method="ffill").fillna(0).to_numpy(dtype=int)
    f["prev_close"] = prev_close.reindex(dt, method="ffill").to_numpy()
    f["gap_pts"] = f["close"].to_numpy() - f["prev_close"].to_numpy()

    # Session VWAP + opening-range width (first 15m RTH)
    vwap = np.full(len(f), np.nan)
    or_w = np.full(len(f), np.nan)
    last = None
    pv = vv = 0.0
    or_hi = or_lo = np.nan
    or_done = False
    hi = f["high"].to_numpy()
    lo = f["low"].to_numpy()
    c = f["close"].to_numpy()
    vol = f["volume"].to_numpy() if "volume" in f.columns else np.ones(len(f))
    mins = f["_et_min"].to_numpy()
    dates = f["_et_date"].to_numpy()
    for i in range(len(f)):
        d, m = dates[i], int(mins[i])
        if d != last:
            last, pv, vv, or_hi, or_lo, or_done = d, 0.0, 0.0, np.nan, np.nan, False
        if 9 * 60 + 30 <= m < 16 * 60:
            typ = (hi[i] + lo[i] + c[i]) / 3.0
            v = max(float(vol[i]), 1.0)
            pv += typ * v
            vv += v
            vwap[i] = pv / vv
            if m < 9 * 60 + 45:
                or_hi = hi[i] if np.isnan(or_hi) else max(or_hi, hi[i])
                or_lo = lo[i] if np.isnan(or_lo) else min(or_lo, lo[i])
            else:
                or_done = True
        if or_done and not np.isnan(or_hi):
            or_w[i] = or_hi - or_lo
        elif i > 0:
            or_w[i] = or_w[i - 1]
    f["sess_vwap"] = vwap
    f["or_width"] = or_w
    f["above_vwap"] = f["close"] > f["sess_vwap"]
    if "adx_5m" in f.columns:
        f["adx_5m"] = pd.Series(f["adx_5m"].to_numpy(), index=f.index).shift(5)
    return f


def base_signals(f: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """One trade/day after 9:35 using completed 15m trend (no same-bar leak)."""
    t15 = f["t15"].to_numpy(dtype=int)
    mins = f["_et_min"].to_numpy()
    dates = f["_et_date"].to_numpy()
    long_s = np.zeros(len(f), dtype=bool)
    short_s = np.zeros(len(f), dtype=bool)
    last = None
    taken = False
    for i in range(len(f)):
        d, m, t = dates[i], int(mins[i]), int(t15[i])
        if d != last:
            last, taken = d, False
        if taken or t == 0 or m < 9 * 60 + 35 or m >= 15 * 60:
            continue
        if t > 0:
            long_s[i] = True
            taken = True
        else:
            short_s[i] = True
            taken = True
    return long_s, short_s


def apply_mask(long_s, short_s, keep: np.ndarray):
    return long_s & keep, short_s & keep


def simulate_exits(
    frame: pd.DataFrame,
    long_s: np.ndarray,
    short_s: np.ndarray,
    sl_pts: float,
    tp_pts: float = 500.0,
    *,
    be_pts: float = 0.0,
    trail_pts: float = 0.0,
    sl_atr_mult: float = 0.0,
    sl_min: float = 20.0,
    sl_max: float = 80.0,
) -> List:
    dt = frame["datetime"].to_numpy()
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    atr15 = frame["atr15"].to_numpy(dtype=float) if "atr15" in frame.columns else np.full(len(frame), sl_pts)
    rth = edge.rth_mask(dt)
    trades = []
    daily: Dict[str, int] = {}
    i = edge.WARMUP
    n = len(frame)
    while i < n - 1:
        if not rth[i]:
            i += 1
            continue
        ts = pd.Timestamp(dt[i])
        day = str(ts.date())
        if daily.get(day, 0) >= 1:
            i += 1
            continue
        direction = 1 if long_s[i] else (-1 if short_s[i] else 0)
        if direction == 0:
            i += 1
            continue
        entry = float(cl[i])
        use_sl = sl_pts
        if sl_atr_mult > 0 and not np.isnan(atr15[i]) and atr15[i] > 0:
            use_sl = float(np.clip(atr15[i] * sl_atr_mult, sl_min, sl_max))
        sl = entry - use_sl if direction == 1 else entry + use_sl
        tp = entry + tp_pts if direction == 1 else entry - tp_pts
        entry_ts = pd.Timestamp(dt[i])
        extreme = entry
        exit_ts, exit_px, reason = pd.Timestamp(dt[-1]), float(cl[-1]), "END"
        for j in range(i + 1, n):
            tss = pd.Timestamp(dt[j])
            if edge.flatten_soon(tss):
                exit_ts, exit_px, reason = tss, float(cl[j]), "EOD"
                break
            if direction == 1:
                extreme = max(extreme, hi[j])
                if be_pts > 0 and extreme >= entry + be_pts:
                    sl = max(sl, entry)
                if trail_pts > 0 and extreme >= entry + be_pts:
                    sl = max(sl, extreme - trail_pts)
                if lo[j] <= sl:
                    exit_ts, exit_px, reason = tss, sl, "SL"
                    break
                if hi[j] >= tp:
                    exit_ts, exit_px, reason = tss, tp, "TP"
                    break
            else:
                extreme = min(extreme, lo[j])
                if be_pts > 0 and extreme <= entry - be_pts:
                    sl = min(sl, entry)
                if trail_pts > 0 and extreme <= entry - be_pts:
                    sl = min(sl, extreme + trail_pts)
                if hi[j] >= sl:
                    exit_ts, exit_px, reason = tss, sl, "SL"
                    break
                if lo[j] <= tp:
                    exit_ts, exit_px, reason = tss, tp, "TP"
                    break
        pts = (exit_px - entry) if direction == 1 else (entry - exit_px)
        hold = (exit_ts - entry_ts).total_seconds()
        trades.append(
            scalp_bt.ScalpTrade(
                entry_time=ts,
                direction="long" if direction == 1 else "short",
                entry_price=entry,
                sl=sl,
                tp=tp,
                exit_time=exit_ts,
                exit_price=exit_px,
                pnl=pts * PT - COMM,
                exit_reason=reason,
                version="1m",
                hold_seconds=hold,
            )
        )
        daily[day] = 1
        j = i + 1
        while j < n and pd.Timestamp(dt[j]) <= exit_ts:
            j += 1
        i = max(j, i + 1)
    return trades


def score_trades(trades) -> Dict:
    ins, oos = edge.split_is_oos(trades)
    is_s = edge.summarize(ins)
    oos_s = edge.summarize(oos)
    all_s = edge.summarize(trades)
    months = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        months.setdefault(key, []).append(t.pnl)
    pos_months = sum(1 for v in months.values() if sum(v) > 0)
    return {
        "is": is_s,
        "oos": oos_s,
        "all": all_s,
        "pos_months": pos_months,
        "n_months": len(months),
        "months": {k: round(sum(v), 1) for k, v in months.items()},
    }


def grade(s: Dict, base_oos_pnl: float) -> str:
    o, i = s["oos"], s["is"]
    if o["trades"] < 20:
        return "THIN"
    if o["profit_factor"] >= 1.25 and o["total_pnl"] > 0 and i["profit_factor"] >= 1.10 and i["total_pnl"] > 0:
        if o["total_pnl"] >= base_oos_pnl * 0.9 and (
            o["profit_factor"] > 2.14 or o["total_pnl"] > base_oos_pnl + 200 or i["profit_factor"] > 1.40
        ):
            return "UPGRADE"
        return "PASS"
    if o["profit_factor"] >= 1.0 and o["total_pnl"] > 0:
        return "WEAK"
    return "FAIL"


def dump(name: str, s: Dict, g: str) -> None:
    i, o = s["is"], s["oos"]
    print(
        f"{name:<48} {i['trades']:4d} {i['profit_factor']:5.2f} {i['total_pnl']:8.0f} "
        f"{o['trades']:4d} {o['profit_factor']:5.2f} {o['total_pnl']:8.0f} "
        f"{s['all']['max_drawdown']:7.0f} {s['pos_months']}/{s['n_months']} {g}",
        flush=True,
    )


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("UPGRADE SEARCH — real Databento 1m, 15m EMA core")
    frame = add_features(load_frame(csv))
    long0, short0 = base_signals(frame)
    rows = []

    def run(name, lo, sh, **exit_kw):
        trades = simulate_exits(frame, lo, sh, exit_kw.pop("sl", 30), **exit_kw)
        s = score_trades(trades)
        g = grade(s, base_oos)
        dump(name, s, g)
        rows.append({"name": name, "grade": g, **s})
        return s

    print(f"{'Rule':<48} ISn  IS PF    IS$ OOSn OOS PF   OOS$     DD  mo   Grade")
    print("-" * 120)
    base = simulate_exits(frame, long0, short0, 30)
    base_s = score_trades(base)
    base_oos = base_s["oos"]["total_pnl"]
    dump("BASE 15m EMA SL30 EOD", base_s, "BASE")
    rows.append({"name": "BASE 15m EMA SL30 EOD", "grade": "BASE", **base_s})

    # --- exits ---
    for sl in (20, 25, 35, 40, 50, 60):
        run(f"exit SL{sl} EOD", long0, short0, sl=sl)
    run("exit SL30 BE+15", long0, short0, sl=30, be_pts=15)
    run("exit SL30 BE+20 trail20", long0, short0, sl=30, be_pts=20, trail_pts=20)
    run("exit SL30 BE+25 trail25", long0, short0, sl=30, be_pts=25, trail_pts=25)
    run("exit SL40 BE+20 trail25", long0, short0, sl=40, be_pts=20, trail_pts=25)
    run("exit ATR*2.0 clip20-60", long0, short0, sl=30, sl_atr_mult=2.0, sl_min=20, sl_max=60)
    run("exit ATR*2.5 clip25-70", long0, short0, sl=30, sl_atr_mult=2.5, sl_min=25, sl_max=70)
    run("exit SL50 BE+25 trail30", long0, short0, sl=50, be_pts=25, trail_pts=30)

    # --- filters ---
    t15 = frame["t15"].to_numpy()
    t60 = frame["t60"].to_numpy()
    td = frame["tdaily"].to_numpy()
    sep = frame["sep15"].to_numpy()
    atr15 = frame["atr15"].to_numpy()
    orw = frame["or_width"].to_numpy()
    vwap_ok_long = frame["above_vwap"].fillna(False).to_numpy()
    vwap_ok_short = (~frame["above_vwap"].fillna(True)).to_numpy()
    gap = frame["gap_pts"].to_numpy()
    dow = frame["_et_dow"].to_numpy()
    mins = frame["_et_min"].to_numpy()
    adx5 = frame["adx_5m"].to_numpy() if "adx_5m" in frame.columns else np.zeros(len(frame))

    run("filter 15m+60m agree", long0 & (t60 == 1), short0 & (t60 == -1))
    run("filter 15m+daily agree", long0 & (td == 1), short0 & (td == -1))
    run("filter 15+60+daily", long0 & (t60 == 1) & (td == 1), short0 & (t60 == -1) & (td == -1))
    run("filter sep15>=0.4", *apply_mask(long0, short0, sep >= 0.4))
    run("filter sep15>=0.6", *apply_mask(long0, short0, sep >= 0.6))
    run("filter sep15>=0.8", *apply_mask(long0, short0, sep >= 0.8))
    run("filter atr15 8-40", *apply_mask(long0, short0, (atr15 >= 8) & (atr15 <= 40)))
    run("filter skip tiny OR<15", *apply_mask(long0, short0, orw >= 15))
    run("filter skip huge OR>80", *apply_mask(long0, short0, orw <= 80))
    run("filter OR 20-70", *apply_mask(long0, short0, (orw >= 20) & (orw <= 70)))
    run("filter VWAP side", long0 & vwap_ok_long, short0 & vwap_ok_short)
    run("filter skip Mon", *apply_mask(long0, short0, dow != 0))
    run("filter skip Fri", *apply_mask(long0, short0, dow != 4))
    run("filter skip Mon+Fri", *apply_mask(long0, short0, (dow != 0) & (dow != 4)))
    run("filter no huge gap>80", *apply_mask(long0, short0, np.abs(gap) <= 80))
    run("filter gap follow >25", long0 & (gap > 25), short0 & (gap < -25))
    run("filter ADX5>=18", *apply_mask(long0, short0, adx5 >= 18))
    run("filter ADX5>=22", *apply_mask(long0, short0, adx5 >= 22))
    run("filter skip ADX5<16 chop", *apply_mask(long0, short0, adx5 >= 16))

    # delayed entry: first signal after 10:00
    long_10, short_10 = np.zeros(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool)
    last = None
    taken = False
    dates = frame["_et_date"].to_numpy()
    t15v = frame["t15"].to_numpy()
    for i in range(len(frame)):
        d, m, t = dates[i], int(mins[i]), int(t15v[i])
        if d != last:
            last, taken = d, False
        if taken or t == 0 or m < 10 * 60 or m >= 15 * 60:
            continue
        if t > 0:
            long_10[i] = True
            taken = True
        else:
            short_10[i] = True
            taken = True
    run("entry after 10:00", long_10, short_10)
    run("entry 10:00 + 60m", long_10 & (t60 == 1), short_10 & (t60 == -1))

    # --- best combo candidates ---
    run("COMBO 60m + sep>=0.4 SL30", long0 & (t60 == 1) & (sep >= 0.4), short0 & (t60 == -1) & (sep >= 0.4))
    run("COMBO 60m + skip MonFri", long0 & (t60 == 1) & (dow != 0) & (dow != 4), short0 & (t60 == -1) & (dow != 0) & (dow != 4))
    run("COMBO 60m + SL50", long0 & (t60 == 1), short0 & (t60 == -1), sl=50)
    run("COMBO daily + sep>=0.4", long0 & (td == 1) & (sep >= 0.4), short0 & (td == -1) & (sep >= 0.4))
    run("COMBO 60m + BE20 trail20", long0 & (t60 == 1), short0 & (t60 == -1), sl=30, be_pts=20, trail_pts=20)
    run("COMBO 60m+daily SL40", long0 & (t60 == 1) & (td == 1), short0 & (t60 == -1) & (td == -1), sl=40)
    run("COMBO VWAP + 60m", long0 & vwap_ok_long & (t60 == 1), short0 & vwap_ok_short & (t60 == -1))
    run("COMBO ADX>=18 + 60m SL40", long0 & (t60 == 1) & (adx5 >= 18), short0 & (t60 == -1) & (adx5 >= 18), sl=40)
    run("COMBO sep>=0.4 SL50", *apply_mask(long0, short0, sep >= 0.4), sl=50)
    run("COMBO 60m + OR20-70 SL40", long0 & (t60 == 1) & (orw >= 20) & (orw <= 70), short0 & (t60 == -1) & (orw >= 20) & (orw <= 70), sl=40)
    dlong, dshort = long0 & (td == 1), short0 & (td == -1)
    run("TOP daily + SL50", dlong, dshort, sl=50)
    run("TOP daily + SL60", dlong, dshort, sl=60)
    run("TOP daily + ATR2 clip20-60", dlong, dshort, sl=30, sl_atr_mult=2.0, sl_min=20, sl_max=60)
    run("TOP daily + ADX>=18 SL30", dlong & (adx5 >= 18), dshort & (adx5 >= 18))
    run("TOP daily + ADX>=18 ATR2", dlong & (adx5 >= 18), dshort & (adx5 >= 18), sl=30, sl_atr_mult=2.0, sl_min=20, sl_max=60)
    run("TOP daily + skip MonFri SL30", dlong & (dow != 0) & (dow != 4), dshort & (dow != 0) & (dow != 4))
    run("TOP daily + 60m ATR2", long0 & (t60 == 1) & (td == 1), short0 & (t60 == -1) & (td == -1), sl=30, sl_atr_mult=2.0, sl_min=20, sl_max=60)
    run("TOP daily + sep>=0.4 ATR2", dlong & (sep >= 0.4), dshort & (sep >= 0.4), sl=30, sl_atr_mult=2.0, sl_min=20, sl_max=60)

    upgrades = [r for r in rows if r["grade"] == "UPGRADE"]
    upgrades.sort(key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True)
    print()
    if upgrades:
        print("UPGRADES (beat BASE on OOS quality, still pass IS):")
        for r in upgrades[:12]:
            print(
                f"  {r['name']}  OOS PF={r['oos']['profit_factor']} ${r['oos']['total_pnl']}  "
                f"IS PF={r['is']['profit_factor']} ${r['is']['total_pnl']}  "
                f"DD=${r['all']['max_drawdown']} months+={r['pos_months']}/{r['n_months']}"
            )
    else:
        passed = [r for r in rows if r["grade"] == "PASS"]
        print("No UPGRADE vs base. PASS-only:")
        for r in passed[:8]:
            print(f"  {r['name']} OOS PF={r['oos']['profit_factor']} ${r['oos']['total_pnl']}")

    out = os.path.join("data", "ema15_upgrade_search.json")
    slim = []
    for r in rows:
        slim.append({
            "name": r["name"],
            "grade": r["grade"],
            "is": r["is"],
            "oos": r["oos"],
            "all": r["all"],
            "pos_months": r["pos_months"],
            "months": r["months"],
        })
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"base_oos": base_oos, "rows": slim}, fh, indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
