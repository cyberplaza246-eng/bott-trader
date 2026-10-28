"""(verbatim port of research_v2/levels_entries.py; changes: gen_signals(from_i=) to start late; build_features(causal=True) removes 2 research look-aheads, causal=False = research)
OG v2 entry module. Trend regime + key levels + pullback / breakout-retest entries.
All features use only bars that have CLOSED before the decision time (no lookahead)."""
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ETZ = ZoneInfo("America/New_York")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr_f(df, n=14):
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(), (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def adx_f(df, n=14):
    up = df.high.diff(); dn = -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr_f(df, n)
    pdi = 100*pd.Series(pdm, df.index).ewm(alpha=1/n, adjust=False).mean()/a
    ndi = 100*pd.Series(ndm, df.index).ewm(alpha=1/n, adjust=False).mean()/a
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()
def rs(d, rule):
    x = d.resample(rule, label="left", closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    x.index = x.index + pd.Timedelta(rule)   # index = CLOSE time = when the bar is known
    return x

def build_features(d1, causal=True):
    """d1: 1m OHLCV, UTC index (left-labelled). Returns 5m frame indexed by close time with all features."""
    f = rs(d1, "5min"); m15 = rs(d1, "15min"); h1 = rs(d1, "1h")
    f["atr"] = atr_f(f); f["ema20"] = ema(f.close, 20)
    h1["e20"] = ema(h1.close, 20); h1["e50"] = ema(h1.close, 50)
    m15["e50"] = ema(m15.close, 50); m15["slope"] = m15.e50.diff(4); m15["adx"] = adx_f(m15)
    # 15m swing pivots (k=3 each side), known 3 bars after the pivot bar closes
    k = 5; H, L = m15.high.to_numpy(), m15.low.to_numpy(); n = len(m15)
    ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    for i in range(k, n-k):
        if H[i] == H[i-k:i+k+1].max(): ph[i+k] = H[i]
        if L[i] == L[i-k:i+k+1].min(): pl[i+k] = L[i]
    m15["ph"] = ph; m15["pl"] = pl
    a = lambda s: s.reindex(f.index, method="ffill")
    f["h1_up"] = a(h1.e20 > h1.e50).astype(float); f["h1_e20"] = a(h1.e20)
    f["m15_slope"] = a(m15.slope); f["adx"] = a(m15.adx)
    # swing structure: last two confirmed pivot highs / lows
    phs = m15.ph.dropna(); pls = m15.pl.dropna()
    f["hh"] = a(phs > phs.shift()).astype(float); f["hl"] = a(pls > pls.shift()).astype(float)
    # ---- session bookkeeping (ET, trading day rolls at 18:00) ----
    et = (f.index - pd.Timedelta("5min")).tz_convert(ETZ)   # bar OPEN time in ET
    mins = et.hour*60 + et.minute
    td = pd.Series(et.date, f.index); roll = mins >= 18*60
    td[roll] = (pd.to_datetime(td[roll]) + pd.Timedelta(days=1)).dt.date
    td = pd.to_datetime(td); td = td + pd.to_timedelta(np.where(td.dt.weekday == 5, 2, np.where(td.dt.weekday == 6, 1, 0)), "D")
    f["td"] = td.values; f["mins"] = mins; f["et"] = et
    rth = (mins >= 570) & (mins < 960)
    # VWAP from 18:00 session start
    tpv = (f.high+f.low+f.close)/3*f.volume
    f["vwap"] = tpv.groupby(f.td).cumsum()/f.volume.groupby(f.td).cumsum().replace(0, np.nan)
    # prior RTH day H/L/C
    r = f[rth].groupby("td").agg(h=("high","max"), l=("low","min"), c=("close","last"))
    if causal:   # FIX (research look-ahead): research only had a prev-day row if the CURRENT day later had RTH bars
        j = np.searchsorted(r.index.values, f.td.values, side="left") - 1; ok = j >= 0; jj = np.maximum(j, 0)
        f["pdh"] = np.where(ok, r.h.values[jj], np.nan); f["pdl"] = np.where(ok, r.l.values[jj], np.nan); f["pdc"] = np.where(ok, r.c.values[jj], np.nan)
    else:
        prev = r.shift(); f["pdh"] = f.td.map(prev.h); f["pdl"] = f.td.map(prev.l); f["pdc"] = f.td.map(prev.c)
    # overnight H/L (18:00-9:30) usable from 9:30; opening range 9:30-9:45 usable from 9:45
    on = f[~rth & ((mins >= 1080) | (mins < 570))].groupby("td").agg(h=("high","max"), l=("low","min"))
    onv = (mins >= 570) & (mins < 1080) if causal else (mins >= 570)   # FIX: research used the not-yet-finished overnight range on 18:00-24:00 bars
    f["onh"] = np.where(onv, f.td.map(on.h), np.nan); f["onl"] = np.where(onv, f.td.map(on.l), np.nan)
    orr = f[(mins >= 570) & (mins < 585)].groupby("td").agg(h=("high","max"), l=("low","min"))
    f["orh"] = np.where((mins >= 585) & (mins < 1020), f.td.map(orr.h), np.nan); f["orl"] = np.where((mins >= 585) & (mins < 1020), f.td.map(orr.l), np.nan)
    # prior-week H/L
    wk = f.td.dt.to_period("W-FRI"); w = f.groupby(wk).agg(h=("high","max"), l=("low","min")).shift()
    f["pwh"] = wk.map(w.h).values; f["pwl"] = wk.map(w.l).values
    # S/R zones from 15m pivots of the last 5 days, clustered with touch counts
    piv = pd.concat([phs, pls]).sort_index(); pt = piv.index.as_unit('ns').asi8; pv = piv.to_numpy()
    zones = []; fi = f.index.as_unit('ns').asi8; atr = f.atr.to_numpy(); last = -1; cur = []
    for i in range(len(f)):
        j = np.searchsorted(pt, fi[i], side="right")
        if j != last or i % 12 == 0:
            lo = np.searchsorted(pt, fi[i] - 3*86400*10**9); p = np.sort(pv[lo:j]); tol = 0.5*atr[i] if atr[i] == atr[i] else 5
            cur = []; s = 0
            while s < len(p):
                e = s
                while e+1 < len(p) and p[e+1]-p[s] <= tol: e += 1
                if e-s+1 >= 3: cur.append((float(p[s:e+1].mean()), e-s+1))
                s = e+1
            cur = [z for z in sorted(cur, key=lambda z: -z[1]) if True]
            keep = []
            for z in cur:
                if all(abs(z[0]-y[0]) > atr[i] for y in keep): keep.append(z)
            cur = keep; last = j
        zones.append(cur)
    f["zones"] = zones
    return f

KEY = ["pdh","pdl","pdc","onh","onl","orh","orl","pwh","pwl"]
def levels_at(row):
    lv = [(row[c], c) for c in KEY if row[c] == row[c]]
    if row.vwap == row.vwap: lv.append((row.vwap, "vwap"))
    lv.append((row.ema20, "ema20"))
    lv += [(z, f"zone{t}") for z, t in row.zones]
    return lv

def trend_label(row, adx_min, need_struct):
    if row.adx != row.adx or row.adx < adx_min: return 0
    up = row.h1_up == 1 and row.m15_slope > 0 and (not need_struct or (row.hh == 1 and row.hl == 1))
    dn = row.h1_up == 0 and row.m15_slope < 0 and (not need_struct or (row.hh == 0 and row.hl == 0))
    return 1 if up else (-1 if dn else 0)

def gen_signals(f, adx_min=20, need_struct=True, buf=0.5, touch=0.25, min_room=1.0, block=((240,540),(900,1080)), from_i=0):
    """Yields dicts: T (ns, decision=5m close), dir, src, level, sl, tp."""
    out = []; recent_break = {}   # (name)->(dir, level, bars_ago)
    prev = None
    for T, r in (f.iloc[from_i:] if from_i else f).iterrows():
        if prev is None or r.atr != r.atr: prev = r; continue
        m = r.mins; a = r.atr
        tr = trend_label(r, adx_min, need_struct)
        lv = levels_at(r); bull = r.close > r.open and r.close > prev.high; bear = r.close < r.open and r.close < prev.low
        # track breakouts of key levels (close through by 0.1 ATR) for retest entries
        for c in KEY:
            L = r[c]
            if L != L: continue
            if prev.close <= L and r.close > L + 0.1*a: recent_break[c] = (1, L, 0)
            elif prev.close >= L and r.close < L - 0.1*a: recent_break[c] = (-1, L, 0)
        for c in list(recent_break):
            dd, L, ago = recent_break[c]; recent_break[c] = (dd, L, ago+1)
            if ago > 24: del recent_break[c]
        blocked = any(lo <= m < hi for lo, hi in block)
        if not blocked and tr != 0:
            cand = []
            if tr == 1 and bull:
                for L, nm in lv:
                    if r.low <= L + touch*a and r.close > L and r.close - L < 1.0*a: cand.append(("pb_"+nm.rstrip("0123456789"), L))
                for c, (dd, L, ago) in recent_break.items():
                    if dd == 1 and 1 <= ago and r.low <= L + touch*a and r.close > L: cand.append(("bo_retest", L))
            if tr == -1 and bear:
                for L, nm in lv:
                    if r.high >= L - touch*a and r.close < L and L - r.close < 1.0*a: cand.append(("pb_"+nm.rstrip("0123456789"), L))
                for c, (dd, L, ago) in recent_break.items():
                    if dd == -1 and 1 <= ago and r.high >= L - touch*a and r.close < L: cand.append(("bo_retest", L))
            if cand:
                src, L = sorted(cand, key=lambda x: (x[0] != "bo_retest", abs(r.close-x[1])))[0]
                d = tr; e = r.close
                sl = (min(L, r.low) - buf*a) if d == 1 else (max(L, r.high) + buf*a)
                R = abs(e - sl)
                opp = sorted([x for x, _ in lv if d*(x - e) > 0.25*a], key=lambda x: d*(x-e))
                room = d*(opp[0]-e) if opp else 9e9
                if R > 0 and room >= min_room*R:               # don't buy into resistance / sell into support
                    tp = e + d*min(room, 2*R)
                    out.append(dict(T=T.value, dir=d, src=src, ref=e, sl=sl, tp=tp, R=R))
        prev = r
    return pd.DataFrame(out)
