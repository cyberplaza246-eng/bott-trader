"""1m-managed backtest engine for the REVAMP 5m ensemble: baseline (5m market entry) vs 1m-timed entries.

SIMULATION ONLY. No broker, no credentials, no network.

Exits (both variants, managed on 1m bars, ordering = live loop: check SL/TP first, then tighten):
  trend    -> no TP; live trailing logic (activation 0.30R, SL -> max(entry+1tick, high-4.0*ATR5m_SMA14))
  breakout -> fixed 2R TP from the 5m signal (absolute level), no trail
  MR       -> TP = far Bollinger band at the signal bar (absolute level), no trail
  all      -> profit locks: +$100 uPnL (bar high/low) -> SL entry+$15; +$150 -> SL entry+$50
  SL/TP absolute prices are identical for both variants (only the entry fill differs).
Risk gates (live start_live_rithmic.py): max 2 concurrent (pending 1m orders occupy a slot),
  5-bar (25 min) cooldown after any exit, 3 consecutive losses -> 30 min pause, $200 daily realized
  loss stop (ET calendar day), 20 trades/day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import TICK, rtick

PV = 2.0                # MNQ $/point
LOCK1_TRIG, LOCK1 = 100.0, 15.0
LOCK2_TRIG, LOCK2 = 150.0, 50.0
TRAIL_MULT = 4.0
ACT_R = 0.30
MAX_CONC = 2
COOLDOWN_MIN = 25
CONSEC_LOSSES, CONSEC_PAUSE_MIN = 3, 30
DAILY_LOSS = 200.0
LOCK_MODE = "usd_old"
USD_ACT, USD_KEEP, USD_TRAIL, USD_PCT = 100.0, 20.0, 150.0, 0.5
R1, R1_KEEP, R2, R2_KEEP, TRAIL_KEEP = 0.5, 0.1, 1.0, 0.5, 0.5
MAX_TRADES_DAY = 20
WINDOW_MIN = 15         # 1m entry window = next 3 x 5m bars (pre-declared, not tuned)
LIMIT_ATR = 0.25        # R2 limit offset in 5m ATR (pre-declared, not tuned)
ONE_MIN = np.int64(60_000_000_000)


class Bars:
    def __init__(self, d1: pd.DataFrame, df5: pd.DataFrame, atr5: pd.Series, rsi5: pd.Series = None):
        self.ts = d1.index.as_unit("ns").asi8
        self.o = d1["open"].to_numpy(float)
        self.h = d1["high"].to_numpy(float)
        self.l = d1["low"].to_numpy(float)
        self.c = d1["close"].to_numpy(float)
        et = d1.index.tz_convert("America/New_York")
        mins = et.hour * 60 + et.minute
        self.rth = (mins >= 570) & (mins < 960)
        nxt = np.r_[self.rth[1:], False]
        same_day_next = np.r_[et.date[1:] == et.date[:-1], False]
        self.last_rth = self.rth & ~(nxt & same_day_next)
        self.entry_ok = self.rth & (mins <= 955) & ~d1["no_trade"].to_numpy(bool)
        self.etdate = np.asarray(et.date)
        # ATR of the last COMPLETED 5m bar at the end of each 1m bar: 5m start s complete if s <= t-4min
        s5 = atr5.index.as_unit("ns").asi8
        idx = np.searchsorted(s5, self.ts - 4 * ONE_MIN, side="right") - 1
        a = atr5.to_numpy(float)
        self.atr = np.where(idx >= 0, a[np.clip(idx, 0, None)], np.nan)
        if rsi5 is not None:
            rv = rsi5.reindex(atr5.index).to_numpy(float)
            self.rsi = np.where(idx >= 0, rv[np.clip(idx, 0, None)], np.nan)
        else:
            self.rsi = None
        self.n = len(self.ts)


def _valid(direction, px, sl, tp):
    if direction * (px - sl) <= 0:
        return False
    if np.isfinite(tp) and direction * (tp - px) <= 0:
        return False
    return True


def find_entry(B: Bars, s: dict, rule: str):
    """Return dict(status, k_fill, fill, partial, slot_end_ns)."""
    d, ref, sl, tp = s["dir"], s["ref"], s["sl"], s["tp"]
    T = s["T"]
    k0 = int(np.searchsorted(B.ts, T))
    if k0 >= B.n or B.ts[k0] - T >= 5 * ONE_MIN or not B.entry_ok[k0]:
        return dict(status="no_bar", slot_end=T)
    base_px = B.o[k0]
    res = dict(base_px=base_px)
    if rule == "base":
        if not _valid(d, base_px, sl, tp):
            return dict(res, status="invalid_at_fill", slot_end=T)
        return dict(res, status="filled", k=k0, fill=base_px, partial=False)

    wend = T + WINDOW_MIN * ONE_MIN
    if rule == "R1":   # pullback (1m close through the 5m signal close) then 1m close beyond prior 1m high/low
        pb = False
        j = k0
        while j < B.n and B.ts[j] < wend and B.entry_ok[j]:
            if d * (B.l[j] if d > 0 else B.h[j]) <= d * sl:
                return dict(res, status="stop_before_entry", slot_end=B.ts[j] + ONE_MIN)
            if np.isfinite(tp) and d * (B.h[j] if d > 0 else B.l[j]) >= d * tp:
                return dict(res, status="target_before_entry", slot_end=B.ts[j] + ONE_MIN)
            if not pb:
                if d * (B.c[j] - ref) < 0:
                    pb = True
            else:
                prior = B.h[j - 1] if d > 0 else B.l[j - 1]
                if d * (B.c[j] - prior) > 0:
                    kf = j + 1
                    if kf < B.n and B.ts[kf] - B.ts[j] == ONE_MIN and B.entry_ok[kf]:
                        px = B.o[kf]
                        if _valid(d, px, sl, tp):
                            return dict(res, status="filled", k=kf, fill=px, partial=False)
                    return dict(res, status="invalid_at_fill", slot_end=B.ts[j] + ONE_MIN)
            j += 1
        return dict(res, status="no_trigger", slot_end=wend)

    if rule == "R2":   # limit at ref -/+ 0.25*ATR for 15 min, else market at window end
        atr = s["atr_live"]
        L = float(rtick(ref - d * LIMIT_ATR * atr)) if np.isfinite(atr) else np.nan
        if not np.isfinite(L) or d * (L - sl) <= 0:
            if not _valid(d, base_px, sl, tp):
                return dict(res, status="invalid_at_fill", slot_end=T)
            return dict(res, status="filled", k=k0, fill=base_px, partial=False, how="market_now")
        j = k0
        while j < B.n and B.ts[j] < wend and B.entry_ok[j]:
            if d > 0:
                if B.o[j] <= L:
                    return dict(res, status="filled", k=j, fill=B.o[j], partial=False, how="limit_open")
                if B.l[j] <= L - TICK:
                    return dict(res, status="filled", k=j, fill=L, partial=True, how="limit")
            else:
                if B.o[j] >= L:
                    return dict(res, status="filled", k=j, fill=B.o[j], partial=False, how="limit_open")
                if B.h[j] >= L + TICK:
                    return dict(res, status="filled", k=j, fill=L, partial=True, how="limit")
            if np.isfinite(tp) and d * (B.h[j] if d > 0 else B.l[j]) >= d * tp:
                return dict(res, status="target_before_entry", slot_end=B.ts[j] + ONE_MIN)
            j += 1
        kw = int(np.searchsorted(B.ts, wend))
        if kw < B.n and B.ts[kw] - wend < 5 * ONE_MIN and B.entry_ok[kw]:
            px = B.o[kw]
            if _valid(d, px, sl, tp):
                return dict(res, status="filled", k=kw, fill=px, partial=False, how="market_end")
        return dict(res, status="invalid_at_fill", slot_end=wend)
    raise ValueError(rule)


def manage(B: Bars, s: dict, k: int, entry: float, partial: bool, overnight: bool):
    d, sl, tp, trailing = s["dir"], s["sl"], s["tp"], s["trailing"]
    has_tp = np.isfinite(tp)
    lock_lvl = 0
    mfe = 0.0
    risk0 = abs(entry - sl)
    xhi = s.get("xhi", np.nan)
    x50 = bool(s.get("x50", False))
    rsi_on = np.isfinite(xhi) and B.rsi is not None
    pend = False
    for b in range(k, B.n):
        o, h, l, c = B.o[b], B.h[b], B.l[b], B.c[b]
        ex = None
        if pend:
            ex = (o, "RSI")
        if ex is not None:
            pass
        elif d > 0:
            if partial and b == k:
                if l <= sl: ex = (sl, "SL")
            elif o <= sl: ex = (o, "SL")
            elif l <= sl: ex = (sl, "SL")
            elif has_tp and o >= tp: ex = (o, "TP")
            elif has_tp and h >= tp: ex = (tp, "TP")
        else:
            if partial and b == k:
                if h >= sl: ex = (sl, "SL")
            elif o >= sl: ex = (o, "SL")
            elif h >= sl: ex = (sl, "SL")
            elif has_tp and o <= tp: ex = (o, "TP")
            elif has_tp and l <= tp: ex = (tp, "TP")
        if ex is None and not overnight and B.last_rth[b]:
            ex = (c, "EOD")
        if ex is None and b == B.n - 1:
            ex = (c, "END")
        if ex is not None:
            return dict(exit_k=b, exit_time=B.ts[b] + ONE_MIN, exit_px=ex[0], exit_reason=ex[1],
                        pts=d * (ex[0] - entry), lock=lock_lvl, final_sl=sl)
        # ---- tighten (live order: profit locks then trail) ----
        fav = h if d > 0 else l
        if LOCK_MODE == "usd_new":  # live LOCK_MODE=usd: +$A -> lock $K; from +$T keep pct of peak
            mfe = max(mfe, d * (fav - entry))
            peak_usd = mfe * PV
            if peak_usd >= USD_ACT and d * (c - entry) > 0:
                lock_usd = USD_KEEP
                if peak_usd >= USD_TRAIL:
                    lock_usd = max(lock_usd, USD_PCT * peak_usd)
                keep = np.floor(lock_usd / PV / TICK + 1e-9) * TICK
                nsl = entry + d * keep
                if keep > 0 and d * (c - nsl) >= 4 * TICK - 1e-9 and d * (nsl - sl) > 1e-9:
                    sl = nsl
                    lock_lvl = max(lock_lvl, 2 if peak_usd >= USD_TRAIL else 1)
        elif LOCK_MODE == "usd_old":
            upnl = d * (fav - entry) * PV
            if upnl >= LOCK2_TRIG:
                lv, usd = 2, LOCK2
            elif upnl >= LOCK1_TRIG:
                lv, usd = 1, LOCK1
            else:
                lv = 0
            if lv:
                nsl = float(rtick(entry + d * usd / PV))
                if d * (nsl - sl) > 1e-9:
                    sl = nsl
                lock_lvl = max(lock_lvl, lv)
        else:  # live R-lock (start_live_rithmic.r_lock_stop)
            mfe = max(mfe, d * (fav - entry))
            live_pts = d * (c - entry)
            if risk0 > 0 and live_pts > 0:
                mr = mfe / risk0
                if mr >= R1:
                    keep = R1_KEEP * risk0
                    if mr >= R2:
                        keep = max(R2_KEEP * risk0, TRAIL_KEEP * mfe)
                    keep = np.floor(keep / TICK + 1e-9) * TICK
                    if keep > 0:
                        nsl = entry + d * keep
                        if d * (c - nsl) >= 4 * TICK - 1e-9 and d * (nsl - sl) > 1e-9:
                            sl = nsl
                            lock_lvl = max(lock_lvl, 2 if mr >= R2 else 1)
        if trailing:
            atr = B.atr[b]
            if np.isfinite(atr) and atr > 0:
                td = max(TICK, atr * TRAIL_MULT)
                risk = max(TICK, abs(entry - sl))
                if d * (fav - entry) >= risk * ACT_R:
                    if d > 0:
                        cand = float(rtick(max(entry + TICK, h - td)))
                        if cand >= l:
                            cand = float(rtick(l - TICK))
                        if cand > sl and cand - sl >= TICK:
                            sl = cand
                    else:
                        cand = float(rtick(min(entry - TICK, l + td)))
                        if cand <= h:
                            cand = float(rtick(h + TICK))
                        if cand < sl and sl - cand >= TICK:
                            sl = cand
        if rsi_on:
            rv = B.rsi[b]
            if np.isfinite(rv):
                if d > 0 and (rv >= xhi or (x50 and rv < 50)):
                    pend = True
                elif d < 0 and (rv <= 100 - xhi or (x50 and rv > 50)):
                    pend = True
    raise RuntimeError("unreachable")


def attempts(B: Bars, sigs: pd.DataFrame, rule: str, overnight: bool) -> pd.DataFrame:
    rows = []
    for t5, r in sigs.iterrows():
        s = dict(dir=int(r["dir"]), ref=float(r["ref"]), sl=float(r["sl"]), tp=float(r["tp"]),
                 trailing=bool(r["trailing"]), atr_live=float(r["atr_live"]),
                 xhi=float(r.get("xhi", np.nan)), x50=bool(r.get("x50", False)),
                 T=np.int64(pd.Timestamp(r["signal_time"]).as_unit("ns").value))
        e = find_entry(B, s, rule)
        row = dict(bar5=t5, T=s["T"], dir=s["dir"], sleeve=r["sleeve"], ref=s["ref"], sl=s["sl"], tp=s["tp"],
                   status=e["status"], base_px=e.get("base_px", np.nan), how=e.get("how", ""), adx=float(r["adx"]))
        if e["status"] == "filled":
            m = manage(B, s, e["k"], e["fill"], e["partial"], overnight)
            row.update(fill_time=B.ts[e["k"]], fill=e["fill"], slot_end=m["exit_time"], **m)
            row["improve_pts"] = s["dir"] * (row["base_px"] - e["fill"])
        else:
            row.update(fill_time=np.nan, fill=np.nan, slot_end=e["slot_end"], pts=np.nan, improve_pts=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def gate(att: pd.DataFrame, cost_rt: float, start_ns: int, end_ns: int) -> pd.DataFrame:
    """Chronological live-style risk gating. Returns att with 'taken' flag and net pnl."""
    att = att.sort_values("T").reset_index(drop=True)
    att["pnl"] = att["pts"] * PV - cost_rt
    et_day = pd.to_datetime(att["T"], utc=True).dt.tz_convert("America/New_York").dt.date
    att["day"] = et_day
    taken = np.zeros(len(att), bool)
    gate_reason = np.array([""] * len(att), dtype=object)
    acc = []   # (slot_start, slot_end, exit_time or None, pnl, day, filled)
    for i, r in att.iterrows():
        T = r["T"]
        if T < start_ns or T >= end_ns:
            gate_reason[i] = "out_of_period"
            continue
        day = r["day"]
        busy = sum(1 for a in acc if a[0] <= T < a[1])
        if busy >= MAX_CONC:
            gate_reason[i] = "max_concurrent"; continue
        exits_today = sorted([a for a in acc if a[5] and a[2] <= T and a[4] == day], key=lambda a: a[2])
        if any(T - COOLDOWN_MIN * ONE_MIN < a[2] <= T for a in acc if a[5]):
            gate_reason[i] = "cooldown"; continue
        if sum(a[3] for a in exits_today) <= -DAILY_LOSS:
            gate_reason[i] = "daily_loss"; continue
        if len(exits_today) >= MAX_TRADES_DAY:
            gate_reason[i] = "max_trades_day"; continue
        cnt, pause_until = 0, -1
        for a in exits_today:
            if pause_until >= 0 and a[2] >= pause_until:
                cnt, pause_until = 0, -1
            if a[3] < 0:
                cnt += 1
                if cnt >= CONSEC_LOSSES:
                    pause_until = a[2] + CONSEC_PAUSE_MIN * ONE_MIN
            elif a[3] > 0:
                cnt = 0
        if pause_until >= 0 and T < pause_until:
            gate_reason[i] = "consec_loss_pause"; continue
        filled = r["status"] == "filled"
        if filled:
            ft = r["fill_time"]
            # re-check daily loss at fill time (pending 1m order)
            if ft > T and sum(a[3] for a in acc if a[5] and a[2] <= ft and a[4] == day) <= -DAILY_LOSS:
                gate_reason[i] = "daily_loss_at_fill"
                acc.append((T, ft, None, 0.0, day, False))
                continue
            acc.append((T, r["slot_end"], r["exit_time"], r["pnl"], day, True))
        else:
            acc.append((T, r["slot_end"], None, 0.0, day, False))
        taken[i] = True
        gate_reason[i] = "taken"
    att["taken"] = taken
    att["gate"] = gate_reason
    return att
