"""EVAL PASSER read-only status readout (FLEX3_STATUS=1). Printed after each newly closed 5m bar.
Uses the SAME feature/signal functions as flex3_signals / _flex3_levels on the SAME closed bars the planner just used.
Never mutates the planner or guard; never places orders. Any error is swallowed by the caller."""
import json, os
import numpy as np, pandas as pd
from . import _flex3_levels as L
from . import flex3_signals as S
ETZ = L.ETZ
_WR = None
def _wr():
    global _WR
    if _WR is None:
        try: _WR = json.load(open(os.path.join(os.path.dirname(__file__), "flex3_winrates.json")))["stats"]
        except Exception: _WR = {}
    return _WR
def wr(src, d):
    s = _wr().get(f"{src}|{int(d)}")
    if not s: return "hist n/a"
    aw = f"+${s['avg_win']:.0f}" if s.get("avg_win") is not None else "-"; al = f"-${abs(s['avg_loss']):.0f}" if s.get("avg_loss") is not None else "-"
    return f"hist win {100*s['win']:.0f}% (n={s['n']}{', SMALL SAMPLE' if s['n'] < 20 else ''}, avg win {aw} / loss {al}, exp ${s['exp']:+.0f}/trade)"
def _ok(b): return "ok" if b else "NO"
LVNAME = {"pdh": "prior-day high", "pdl": "prior-day low", "pdc": "prior-day close", "onh": "overnight high", "onl": "overnight low",
          "orh": "open-range high", "orl": "open-range low", "pwh": "prior-week high", "pwl": "prior-week low", "vwap": "VWAP", "ema20": "EMA20"}

def lines(planner, guard, d1, Tc, now_utc, account, open_pos=(), taken=()):
    """planner/guard: live objects (read only). d1: closed 1m bars. Tc: 5m close the planner just evaluated (UTC Timestamp)."""
    out = []; P = S.V2_PARAMS
    x = d1[(d1.index >= Tc - pd.Timedelta(days=S.HIST_DAYS)) & (d1.index < Tc)]
    f = L.build_features(x, causal=True)
    if len(f) < 3 or f.index[-1] != Tc: return [f"STATUS: no 5m bar closing {Tc.tz_convert(ETZ):%H:%M} ET yet"]
    r, pv = f.iloc[-1], f.iloc[-2]; a = float(r.atr); c = float(r.close)
    h1 = L.rs(x, "1h"); e20 = float(L.ema(h1.close, 20).iloc[-1]); e50 = float(L.ema(h1.close, 50).iloc[-1])
    tr = L.trend_label(r, P["adx_min"], P["need_struct"]); bias = {1: "LONG", -1: "SHORT", 0: "NONE (flat - no v2 trades)"}[tr]
    et = Tc.tz_convert(ETZ); td = S.trading_day(Tc)
    out.append(f"===== EVAL PASSER STATUS {et:%a %Y-%m-%d %H:%M} ET (5m bar closed) | MNQ {c:.2f} | ATR5 {a:.1f} =====")
    out.append(f"BIAS: {bias} | 1h EMA20 {e20:.2f} {'>' if e20 > e50 else '<='} EMA50 {e50:.2f} ({'up' if r.h1_up == 1 else 'down'}) | "
               f"15m ADX {r.adx:.1f} (>= {P['adx_min']} {_ok(r.adx >= P['adx_min'])}) | 15m EMA50 slope {r.m15_slope:+.2f} ({'up' if r.m15_slope > 0 else 'down'})")
    m = int(r.mins); blocked = any(lo <= m < hi for lo, hi in ((240, 540), (900, 1080)))
    bull = r.close > r.open and r.close > pv.high; bear = r.close < r.open and r.close < pv.low
    excl = set(getattr(planner, "exclude_src", ()))
    lv = [(float(Lv), nm) for Lv, nm in L.levels_at(r) if ("pb_" + nm.rstrip("0123456789")) not in excl]
    def nice(nm): return LVNAME.get(nm, "S/R zone" if nm.startswith("zone") else nm)
    if tr == 0:
        out.append("WATCHING v2: nothing - needs 1h EMA20/50 + 15m slope to agree and ADX >= 25")
    else:
        side = "long" if tr == 1 else "short"; d = tr
        cands = sorted([(Lv, nm) for Lv, nm in lv if d*(c - Lv) > 0], key=lambda z: abs(c - z[0]))[:3]
        trig = (f"5m bar closing green above prior high {pv.high:.2f}" if d == 1 else f"5m bar closing red below prior low {pv.low:.2f}")
        out.append(f"WATCHING v2 {side} pullback (needs bar to tag level within {0.25*a:.1f} pts, close back {'above' if d == 1 else 'below'} it within 1 ATR, {trig}; this bar {_ok(bull if d == 1 else bear)}):")
        for Lv, nm in cands:
            src = "pb_" + nm.rstrip("0123456789")
            out.append(f"   - {nice(nm)} {Lv:.2f} ({abs(c - Lv):.1f} pts away) - {wr(src, d)}")
        if not cands: out.append(f"   - no level {'below' if d == 1 else 'above'} price")
        # breakouts the planner would retest (same rule as gen_signals: close through level by 0.1 ATR within last 24 bars)
        rb = {}
        g = f.iloc[-27:]
        for i in range(1, len(g)):
            rr, pp = g.iloc[i], g.iloc[i-1]
            for k in L.KEY:
                Lk = rr[k]
                if Lk != Lk: continue
                if pp.close <= Lk and rr.close > Lk + 0.1*rr.atr: rb[k] = (1, Lk, len(g)-1-i)
                elif pp.close >= Lk and rr.close < Lk - 0.1*rr.atr: rb[k] = (-1, Lk, len(g)-1-i)
        rb = {k: v for k, v in rb.items() if v[0] == d and v[2] <= 24}
        for k, (dd, Lk, ago) in rb.items():
            out.append(f"   - breakout-retest of {nice(k)} {Lk:.2f} (broke {ago} bars ago, {abs(c - Lk):.1f} pts away) - {wr('bo_retest', d)}")
        out.append(f"   opposite side ({'short' if d == 1 else 'long'}): n/a - against bias")
    fired = [s for s in S.v2_signals(x, t_lo=Tc.value) if s["T"] == Tc.value]
    for s in fired:
        tk = any(o.get("flex3_stream") == "v2" and int(o.get("T", -1)) == s["T"] for o in taken)
        why = "TAKEN -> order sent" if tk else ("skipped: setup type turned off live" if s["src"] in excl else "blocked (1 v2 open / 10 per day / risk>$300 / session stop)")
        out.append(f"   >> v2 SIGNAL this bar: {s['src']} {'long' if s['dir'] > 0 else 'short'} ref {s['ref']:.2f} sl {s['sl']:.2f} tp {s['tp']:.2f} "
                   f"(risk ${abs(s['ref']-s['sl'])*S.PV+4:.0f}) -> {why} - {wr(s['src'], s['dir'])}")
    # ---- ONL / E1 add-ons ----
    F = S.addon_frame(x); g = F[F.td == F.td.iloc[-1]]; ovn = g[(g.m >= 1080) | (g.m < 570)]; fm = int(F.m.iloc[-1])
    if (ovn.m == 1080).any():
        onl = float(ovn.l.min()); w = g[(g.m >= 570) & (g.m <= 685)]; hit = w[(w.l < onl) & (w.c > onl)]
        st = ("already fired today" if len(hit) else ("window open" if 570 <= fm <= 685 else ("window closed" if fm > 685 and fm < 1080 else "waits for 09:30")))
        out.append(f"ONL long: overnight low{' so far' if (fm >= 1080 or fm < 570) else ''} {onl:.2f} ({c - onl:+.1f} pts) | window 09:30-11:25 ET: {st} | needs 5m dip below + close back above - {wr('ONL', 1)}")
    else:
        out.append(f"ONL long: no overnight range yet (session started without an 18:00 bar) - {wr('ONL', 1)}")
    fr = F.iloc[-1]; e1win = 1080 <= fm <= 1135
    out.append(f"E1 long: window 18:00-18:55 ET {'OPEN' if e1win else 'closed'} | RSI {fr.rsi:.1f}/59.5 {_ok(fr.rsi >= 59.5)} | ret12 {fr.ret12:.2f}/1.48 ATR {_ok(fr.ret12 >= 1.48)} - {wr('E1', 1)}")
    # ---- 6 PM long ----
    dd = guard.eval_hwm - guard.eval_pnl; gate = float(os.getenv("FLEX3_6PM_DD_GATE", "500"))
    six_td = S.trading_day(pd.Timestamp(now_utc)); done = six_td in getattr(planner, "six_done", set())
    nxt = "already handled this session" if done else "next 18:00 ET"
    out.append(f"6PM long: {nxt} | DD from high ${dd:.0f} (< ${gate:.0f} {_ok(dd < gate)} -> {'ARMED' if dd < gate else 'BLOCKED by DD gate'}) | "
               f"stop clip(1.5xATR15,12,40) ~{np.clip(1.5*S.atr15_before(x, Tc.value), 12, 40):.1f} pts - {wr('6pm_long', 1)}")
    # ---- OG trend sleeve ----
    if getattr(planner, "og_trend", False):
        f5 = x.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if len(f5) >= 60:
            cc = f5.close; fe, se, te = S._ema(cc, 20).iloc[-1], S._ema(cc, 50).iloc[-1], S._ema(cc, 200).iloc[-1]; ad = S._adx(f5.high, f5.low, cc, 14).iloc[-1]
            rth = 570 <= m < 960; up = fe > se
            want = "short" if up else "long"
            out.append(f"OG trend: 5m EMA20 {fe:.2f} {'>' if up else '<'} EMA50 {se:.2f}, EMA200 {te:.2f} | ADX {ad:.1f} (>=25 {_ok(ad >= 25)}) | RTH {_ok(rth)} | "
                       f"next trigger = EMA20 crossing {'below' if up else 'above'} EMA50 with close {'<' if up else '>'} EMA200 ({want}) - {wr('og_trend', -1 if up else 1)}")
            ogs = [s for s in S.ogtrend_signals(x, t_lo=Tc.value) if s["T"] == Tc.value]
            for s in ogs: out.append(f"   >> OG trend SIGNAL this bar: {'long' if s['dir'] > 0 else 'short'} ref {s['ref']:.2f} (only taken when flat)")
    # ---- blockers ----
    bl = []
    ok, why = guard.allow_entry("v2", et.time(), account)
    if not ok: bl.append(why)
    _tb = S.tokyo_block_until(pd.Timestamp(Tc).tz_convert("UTC")) if getattr(S, "TOKYO_BLOCK_MIN", 0) > 0 else None
    if _tb is not None: bl.append(f"Tokyo open window (until {_tb:%H:%M} ET)")
    if blocked: bl.append("v2 time block (04:00-09:00 / 15:00-18:00 ET)")
    if tr == 0: bl.append(f"no trend (ADX {r.adx:.1f}<25)" if r.adx < 25 else "no trend (1h and 15m disagree)")
    n_td = planner.entries.get(td, 0)
    if n_td >= (planner.max_per_day or 10**9): bl.append(f"trade cap {n_td}/{planner.max_per_day}")
    if planner.v2_open: bl.append("v2 trade already open")
    if planner.addon_open: bl.append("ONL/E1 already open")
    if open_pos: bl.append(f"{len(open_pos)} position(s) open (OG trend needs flat)")
    out.append(f"DAY: P&L ${guard.day_pnl:.0f} (stop -${float(os.getenv('FLEX3_DAILY_LOSS_STOP','1500')):.0f} / cap +${float(os.getenv('FLEX3_DAY_PROFIT_CAP','1350')):.0f}) | eval ${guard.eval_pnl:.0f} | entries today {n_td}/{planner.max_per_day}")
    for o in taken:
        out.append(f"ORDER THIS BAR: {o['flex3_stream']} {o['source']} {o['direction']} stop {o['sl_pts']:.1f} pts")
    out.append("BLOCKERS: " + ("; ".join(bl) if bl else "none"))
    out.append("(hist = every signal of that type, 2024-10..2026-10 backtest, 1 MNQ, $4 costs incl.)")
    return out
