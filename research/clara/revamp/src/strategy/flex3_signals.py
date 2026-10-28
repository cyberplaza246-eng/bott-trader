"""flex3 live signal generators (no broker calls). Exact ports of the box research:
  v2     research_v2/levels_entries.py build_features + gen_signals, params P=(adx 25, no struct, buf 0.6), plus sim_v2.simulate()
         entry filters (1 open v2 at a time, <=10 v2+ONL/E1 entries per trading day, skip risk > $300/contract,
         v2 session loss stops -$500 day / -$300 overnight per contract). TP distance = |tp-ref| from fill. Lock +$400 -> +$50 /contract.
  ONL/E1 og50k/og50k_sim.py add-on formulas (5m left-labelled bars): long, SL = ref - max(1.25*ATR5, 3), TP = 2R, 2 h time exit, no lock.
         ONL: first 5m bar 09:30-11:25 ET with low < overnight low and close > it.  E1: 18:00-18:55 ET bar, Mon-Sun(wd<=6),
         RSI14 >= 59.5 and 12-bar return >= 1.48 ATR.
  6pm    long at 18:00 ET open, SL clip(1.5*ATR15, 12, 40) pts, no TP, out 09:25 ET bar close, lock +$400 -> +$50 /contract.
Input: 1m OHLCV DataFrame, UTC DatetimeIndex labelled at bar OPEN, only CLOSED bars. Needs >= 21 calendar days of history."""
from datetime import timedelta
import os
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from . import _flex3_levels as L
ETZ = L.ETZ; ONE = 60*10**9; PV = 2.0; TICK = 0.25
V2_PARAMS = dict(adx_min=25, need_struct=False, buf=0.6)
HIST_DAYS = 21
# Tokyo-open filter (PAPER test): block NEW v2 entries from 09:00 Asia/Tokyo (20:00 ET in US DST, 19:00 ET in US standard
# time) up to +FLEX3_TOKYO_BLOCK_MIN minutes. 0/absent = off. 6 PM long, ONL/E1 and OG trend are never blocked.
TOKYO_BLOCK_MIN = int(float(os.getenv("FLEX3_TOKYO_BLOCK_MIN", "0") or 0))
TKZ = ZoneInfo("Asia/Tokyo")
def tokyo_open_et(ts_utc):
    """Most recent Tokyo cash open (09:00 JST) at or before ts, as an ET timestamp."""
    t = pd.Timestamp(ts_utc).tz_convert(TKZ); o = t.normalize() + pd.Timedelta(hours=9)
    if o > t: o -= pd.Timedelta(days=1)
    return o.tz_convert(ETZ)
def tokyo_block_until(ts_utc, minutes=None):
    """ET end of the Tokyo-open block if ts is inside it, else None."""
    m = TOKYO_BLOCK_MIN if minutes is None else minutes
    if m <= 0: return None
    t = pd.Timestamp(ts_utc); o = tokyo_open_et(t); end = o + pd.Timedelta(minutes=m)
    return end if o <= t.tz_convert(ETZ) < end else None
rt = lambda x: float(np.round(float(x)/TICK)*TICK)
# ---- 6 PM long tweaks (2026-10-26 research /workspace/sixpm b.py + t.py; shared by EVAL PASSER and FUNDED 12K) ----
# SIXPM_SKIP_MONDAY=1: no 6 PM long when the 18:00 ET entry falls on a Monday (Mondays lost every year tested).
# SIXPM_TRAIL=1: chandelier trail, dist = SIXPM_TRAIL_ATR_MULT x ATR15 at entry, armed once high since entry >= entry +
#   SIXPM_TRAIL_START_PTS; the +$400 -> +$50 lock is off for 6pm while it is on. SIXPM_TRAIL_KEEP_TP=0: the far TP bracket
#   is pushed out to SIXPM_NO_TP_PTS (default 2000 pts = never hit, a TP leg stays on the broker as a safety bracket).
SIXPM_SKIP_MONDAY = os.getenv("SIXPM_SKIP_MONDAY", "1") == "1"
SIXPM_TRAIL = os.getenv("SIXPM_TRAIL", "1") == "1"
SIXPM_TRAIL_ATR_MULT = float(os.getenv("SIXPM_TRAIL_ATR_MULT", "3"))
SIXPM_TRAIL_START_PTS = float(os.getenv("SIXPM_TRAIL_START_PTS", "100"))
SIXPM_TRAIL_KEEP_TP = os.getenv("SIXPM_TRAIL_KEEP_TP", "0") == "1"
SIXPM_NO_TP_PTS = float(os.getenv("SIXPM_NO_TP_PTS", "2000"))
def sixpm_skip_monday(et_ts): return SIXPM_SKIP_MONDAY and pd.Timestamp(et_ts).tz_convert(ETZ).weekday() == 0

def addon_frame(d1):
    df5 = d1.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h, l, c = df5.high, df5.low, df5.close
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1); atr = tr.ewm(alpha=1/14, adjust=False).mean()
    dl = c.diff(); up = dl.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100/(1 + up/dn); e5 = df5.index.tz_convert(ETZ); m5 = e5.hour*60 + e5.minute
    return pd.DataFrame({"l": l, "c": c, "atr": atr, "rsi": rsi, "ret12": (c - c.shift(12))/atr, "m": m5,
                         "td": (e5 + pd.Timedelta(hours=6)).date, "wd": e5.weekday}, index=df5.index)

def addon_sig(F, ts, src):
    r = F.loc[ts]; ref = rt(r.c); dist = max(1.25*r.atr, 3.0); sl = rt(ref - dist)
    return dict(T=int((ts + pd.Timedelta(minutes=5)).value), dir=1, src=src, ref=ref, sl=sl, tp=rt(ref + 2*(ref - sl)))

def addon_signals(F, t_lo=None):
    """All ONL/E1 signals in F (decision time T = 5m bar close). t_lo: only return T >= t_lo."""
    out = []
    for td_, g in F.groupby("td"):
        ovn = g[(g.m >= 1080) | (g.m < 570)]
        if not (ovn.m == 1080).any(): continue
        onl = ovn.l.min(); w = g[(g.m >= 570) & (g.m <= 685)]; hit = w[(w.l < onl) & (w.c > onl)]
        if len(hit): out.append(addon_sig(F, hit.index[0], "ONL"))
    for ts in F[(F.m >= 1080) & (F.m <= 1135) & (F.wd <= 6) & (F.rsi >= 59.5) & (F.ret12 >= 1.48)].index: out.append(addon_sig(F, ts, "E1"))
    out = [s for s in out if t_lo is None or s["T"] >= t_lo]
    return sorted(out, key=lambda s: s["T"])

def atr15_before(d1, t_ns):
    """ATR14 (Wilder) of 15m bars right-labelled, using only bars CLOSED by t (research asof(atr15, ts))."""
    m15 = d1.resample("15min", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr = pd.concat([m15.high-m15.low, (m15.high-m15.close.shift()).abs(), (m15.low-m15.close.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/14, adjust=False).mean(); i = a.index.searchsorted(pd.Timestamp(t_ns, tz="UTC"), side="right") - 1
    return float(a.iloc[i]) if i >= 0 else float("nan")

def sixpm_signal(d1, t_ns, research_lookahead=False):
    """6 PM long at 18:00 ET. research used the 15m bar labelled 18:00 which includes the 18:00 1m bar (1-minute lookahead);
    live (default) uses only bars closed before 18:00."""
    x = d1 if research_lookahead else d1[d1.index < pd.Timestamp(t_ns, tz="UTC")]
    a = atr15_before(x, t_ns); sl = float(np.clip(1.5*a, 12, 40))
    return dict(T=int(t_ns), dir=1, src="6pm_long", sl_pts=sl, atr15=a)

def v2_signals(d1, t_lo=None, causal=True):
    f = L.build_features(d1, causal=causal); i0 = 0
    if t_lo is not None:
        i0 = max(0, int(np.searchsorted(f.index.as_unit("ns").asi8, t_lo)) - 30)   # 30 bars seeds prev/recent_break (<=24 bars)
    S = L.gen_signals(f, from_i=i0, **V2_PARAMS)
    if len(S) and t_lo is not None: S = S[S["T"] >= t_lo]
    return [] if len(S) == 0 else S.to_dict("records")

def trading_day(ts_utc):
    x = pd.Timestamp(ts_utc).tz_convert(ETZ); return str((x + pd.Timedelta(hours=6)).date())

def session(ts_utc):
    x = pd.Timestamp(ts_utc).tz_convert(ETZ); m = x.hour*60 + x.minute; return "day" if 570 <= m < 960 else "ovn"

class Flex3Planner:
    """Stateful entry planner. Call on_bar(d1, now_utc) once per loop; it returns order intents for each newly CLOSED 5m bar
    (and the 18:00 ET 6 PM entry). The runner reports fills/closes back with on_entry / on_close."""
    def __init__(self, size=3, exclude_src=(), shared_busy=False, v2_sess_stops=True, max_per_day=10, og_trend=False):
        # defaults = EVAL PASSER research; FUNDED 12K (MID A): exclude pb_ema, v2+ONL/E1 share one slot, no session stops / day cap
        self.exclude_src = tuple(exclude_src); self.shared_busy = shared_busy; self.v2_sess_stops = v2_sess_stops
        self.max_per_day = max_per_day; self.addon_open = 0
        self.og_trend = bool(og_trend)   # EVAL PASSER: OG trend sleeve, only when the account is flat; nothing opposite an open OG trend
        self.six_open = None; self.six_skip_td = ""
        self.size = size; self.last_T = None; self.v2_open = 0; self.entries = {}; self.v2_sess = {}; self.six_done = set()
    def state(self): return dict(last_T=self.last_T, v2_open=self.v2_open, addon_open=self.addon_open, entries=self.entries, v2_sess={f"{k[0]}|{k[1]}": v for k, v in self.v2_sess.items()}, six_done=sorted(self.six_done), six_open=self.six_open, six_skip_td=self.six_skip_td)
    def load(self, s):
        self.last_T = s.get("last_T"); self.v2_open = int(s.get("v2_open", 0)); self.addon_open = int(s.get("addon_open", 0)); self.entries = dict(s.get("entries", {}))
        self.v2_sess = {tuple(k.split("|")): v for k, v in s.get("v2_sess", {}).items()}; self.six_done = set(s.get("six_done", []))
        self.six_open = s.get("six_open"); self.six_skip_td = s.get("six_skip_td", "")
    def on_entry(self, intent):
        td = intent["td"]
        if intent["flex3_stream"] in ("v2", "addon", "ogtrend"): self.entries[td] = self.entries.get(td, 0) + 1
        if intent["flex3_stream"] == "v2": self.v2_open += 1
        if intent["flex3_stream"] == "addon": self.addon_open += 1
    def on_close(self, stream, pnl_usd, size, entry_ts_utc):
        if stream == "addon": self.addon_open = max(0, self.addon_open - 1)
        if stream == "6pm": self.six_open = None                                   # trail state no longer needed
        if stream != "v2": return
        self.v2_open = max(0, self.v2_open - 1); k = (trading_day(entry_ts_utc), session(entry_ts_utc))
        self.v2_sess[k] = self.v2_sess.get(k, 0.0) + pnl_usd/max(1, size)          # per contract, like research
    # ---- filters identical to sim_v2.simulate() (v2) and og50k_sim (add-ons) ----
    def _v2_ok(self, s, td):
        if self.v2_open > 0 or (self.shared_busy and self.addon_open > 0): return False
        if self.max_per_day is not None and self.entries.get(td, 0) >= self.max_per_day: return False
        if s["src"] in self.exclude_src: return False
        if TOKYO_BLOCK_MIN > 0 and tokyo_block_until(pd.Timestamp(s["T"], tz="UTC")) is not None: return False
        R = abs(s["ref"] - s["sl"])                                                 # research: R vs actual fill; ref used live
        if R <= TICK or R*PV + 4.0 > 300: return False
        se = session(pd.Timestamp(s["T"], tz="UTC")); dn = self.v2_sess.get((td, se), 0.0)
        return not self.v2_sess_stops or not ((se == "day" and dn <= -500) or (se == "ovn" and dn <= -300))
    def _addon_ok(self, s, td):
        if self.max_per_day is not None and self.entries.get(td, 0) >= self.max_per_day: return False
        if self.addon_open > 0: return False                                          # one ONL/E1 at a time (backtest)
        if s.get("src") in self.exclude_src: return False
        if self.shared_busy and (self.v2_open > 0 or self.addon_open > 0): return False
        return s["sl"] < s["ref"] < s["tp"] and (s["ref"] - s["sl"])*PV + 4.0 <= 500
    def intents(self, d1, now_utc, open_pos=None):
        """d1: CLOSED 1m bars (index = bar open, UTC). now_utc: wall clock (pd.Timestamp tz=UTC).
        open_pos: [(flex3_stream, dir +1/-1), ...] of the bot's open positions (used by the OG trend rules)."""
        out = self._intents(d1, now_utc)
        if not self.og_trend: return out
        opn = list(open_pos or []); og_dirs = {d for st, d in opn if st == "ogtrend"}
        keep = [o for o in out if not any((1 if o["direction"] == "long" else -1) != d for d in og_dirs)]   # nothing opposite an open OG trend
        for o in out:                                                                 # 6 PM held back by an open OG trend: retry next loop (18:00-18:04)
            if o not in keep and o["flex3_stream"] == "6pm": self.six_done.discard(o["td"])
        out = keep
        Tc = getattr(self, "_Tc", None)
        if Tc is None or opn or out: return out                                       # OG trend only when flat (variant b)
        td = trading_day(Tc)
        if self.max_per_day is not None and self.entries.get(td, 0) >= self.max_per_day: return out
        x = d1[(d1.index >= Tc - pd.Timedelta(days=HIST_DAYS)) & (d1.index < Tc)]
        for s in ogtrend_signals(x, t_lo=Tc.value):
            if s["T"] == Tc.value: out.append(self._intent("ogtrend", s, None, td)); break
        return out
    def _intents(self, d1, now_utc):
        now = pd.Timestamp(now_utc).tz_convert("UTC"); out = []
        d1 = d1[d1.index >= now - pd.Timedelta(days=HIST_DAYS)]
        Tc = now.floor("5min")                                                         # latest 5m close
        if len(d1) == 0 or d1.index[-1] < Tc - pd.Timedelta(minutes=1): Tc = None      # bar for this 5m slot not complete
        et = now.tz_convert(ETZ); td_now = trading_day(now)
        if et.hour == 18 and et.minute < 5 and td_now not in self.six_done:            # 6 PM long
            t18 = pd.Timestamp(et.replace(minute=0, second=0, microsecond=0)).tz_convert("UTC")
            self.six_done.add(td_now)
            if sixpm_skip_monday(t18):
                if self.six_skip_td != td_now: print(f"   6PM long: skipped (Monday) - SIXPM_SKIP_MONDAY=1")
                self.six_skip_td = td_now
            else:
              s = sixpm_signal(d1, t18.value)
              out.append(self._intent("6pm", s, None, td_now, exit_at=(pd.Timestamp(et.date()) + pd.Timedelta(days=1)).tz_localize(ETZ) + pd.Timedelta(hours=9, minutes=26)))
        self._Tc = None
        if Tc is None or (self.last_T is not None and Tc.value <= self.last_T): return out
        self._Tc = Tc
        self.last_T = Tc.value; x = d1[d1.index < Tc]
        for s in v2_signals(x, t_lo=Tc.value):
            if s["T"] != Tc.value: continue
            td = trading_day(Tc)
            if self._v2_ok(s, td): out.append(self._intent("v2", s, None, td))
        if self.shared_busy and any(o["flex3_stream"] == "v2" for o in out): return out     # one shared slot
        F = addon_frame(x)
        for s in addon_signals(F, t_lo=Tc.value):
            if s["T"] != Tc.value: continue
            td = trading_day(Tc)
            if self._addon_ok(s, td): out.append(self._intent("addon", s, None, td, exit_at=Tc + pd.Timedelta(minutes=120)))
        return out
    def _intent(self, stream, s, _, td, exit_at=None):
        d = int(s["dir"])
        trail_atr = trail_start = None
        if stream == "6pm":
            ref = None; sl_pts = s["sl_pts"]; tp_pts = None
            if SIXPM_TRAIL and s.get("atr15") == s.get("atr15") and s.get("atr15"):
                trail_atr = SIXPM_TRAIL_ATR_MULT*float(s["atr15"]); trail_start = SIXPM_TRAIL_START_PTS
                if not SIXPM_TRAIL_KEEP_TP: tp_pts = SIXPM_NO_TP_PTS
        elif stream == "ogtrend":
            ref = float(s["ref"]); sl_pts = abs(ref - float(s["sl"])); tp_pts = None
        else:
            ref = float(s["ref"]); sl_pts = abs(ref - float(s["sl"])); tp_pts = abs(float(s["tp"]) - ref)
        return dict(symbol="MNQ", direction="long" if d > 0 else "short", flex3_stream=stream, source=s["src"], td=td,
                    T=int(s["T"]), ref=ref, sl_pts=sl_pts, tp_pts=tp_pts, flex3_exit_at=exit_at,
                    confidence=1.0, trailing_enabled=False,
                    trail_atr=(OGT["trail_mult"]*float(s["atr"]) if stream == "ogtrend" else trail_atr), trail_start_pts=trail_start)

# ---------------- OG TREND sleeve (Eval Passer, FLEX3_OG_TREND) ----------------
# Exact port of og_v2 src/research_ensemble TrendFollowingStrategy on CLOSED 5m bars (left-labelled, built from 1m):
# EMA20/EMA50 cross confirmed by close vs EMA200, ADX14 >= 25, RTH bar start 09:30-16:00 ET. Entry at the 5m close,
# SL = close -/+ 2*ATR14, no TP, REVAMP exit: trail stop to best price -/+ 4*ATR14 (ATR at entry), flat 16:44 ET.
OGT = dict(fast=20, slow=50, trend=200, atr_n=14, stop_mult=2.0, adx_min=25.0, trail_mult=4.0)
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _tr(h, l, c):
    pc = c.shift(1); return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
def _atr(h, l, c, n=14): return _tr(h, l, c).ewm(alpha=1/n, adjust=False).mean()
def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = _atr(h, l, c, n); pdi = 100*pd.Series(pdm, index=h.index).ewm(alpha=1/n, adjust=False).mean()/a.replace(0, np.nan)
    mdi = 100*pd.Series(mdm, index=h.index).ewm(alpha=1/n, adjust=False).mean()/a.replace(0, np.nan); ds = pdi + mdi
    dx = pd.Series(np.where(ds > 1e-6, 100*(pdi - mdi).abs()/ds.replace(0, np.nan), 0.0), index=h.index)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)
def ogtrend_signals(d1, t_lo=None):
    """d1: CLOSED 1m bars. Returns OG trend entries with decision time T = 5m bar close (ns), dir, ref (=5m close), sl, atr."""
    f5 = d1.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(f5) < 60: return []
    c = f5.close; fe, se, te = _ema(c, OGT["fast"]), _ema(c, OGT["slow"]), _ema(c, OGT["trend"]); a = _atr(f5.high, f5.low, c, OGT["atr_n"])
    ad = _adx(f5.high, f5.low, c, 14); e5 = f5.index.tz_convert(ETZ); m = e5.hour*60 + e5.minute; rth = (m >= 570) & (m < 960)
    up = (fe > se) & (fe.shift(1) <= se.shift(1)) & (c > te); dn = (fe < se) & (fe.shift(1) >= se.shift(1)) & (c < te)
    d = np.where(up, 1, np.where(dn, -1, 0)) * (rth & (ad >= OGT["adx_min"]).to_numpy())
    T = (f5.index + pd.Timedelta(minutes=5)).as_unit("ns").asi8; out = []
    for i in np.where(d != 0)[0]:
        if t_lo is not None and T[i] < t_lo: continue
        dd = int(d[i]); ref = float(c.iloc[i]); sl = ref - dd*OGT["stop_mult"]*float(a.iloc[i])
        if not np.isfinite(sl) or dd*(ref - sl) <= 0: continue
        out.append(dict(T=int(T[i]), dir=dd, src="og_trend", ref=ref, sl=sl, atr=float(a.iloc[i])))
    return out
