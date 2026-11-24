"""Two backtested LONG add-on setups for both live bots (user 2026-10-10).

Frozen rules: docs/ADDON_SPECS_20261010.md. Box research: out/ideas/IDEA_LIST_RESULTS.md (#6)
and PATTERN_MINING_RESULTS.md (E1). Both use the research trade model "label A":
  stop   = max(1.25 x ATR(5m), 3.0 pts) below entry
  target = 2R (= 2.5 ATR) above entry
  time   = flat 24 x 5m bars (2h) after the signal bar closes if neither SL nor TP hit
  size   = 1 MNQ, only when the bot has NO open trade.

1) ONL reclaim (ADDON_ONL_RECLAIM, default 1): RTH 5m bar starting 09:30..11:25 ET whose low is
   below the overnight low (Globex 18:00 -> 09:25 bars of this trading day) and whose close is
   above it. Only the FIRST such bar of the day counts; if the bot is not flat then, no trade today.
2) E1 6 PM momentum (ADDON_E1_6PM, default 1): 5m bar starting 18:00..18:55 ET with
   RSI(14) >= 59.5 and (close - close 12 bars back) / ATR >= 1.48. Any such bar while flat.

All indicators: ATR = EWM(alpha 1/14) of true range, RSI = EWM(alpha 1/14) on 5m closes,
continuous across sessions (no reset at the 5-6 PM halt). Bars are keyed by START time (ET).
Pure functions + small helpers; no broker calls here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ET = "America/New_York"
BAR_MIN = 5

# ---- frozen backtest constants (do not tune) ----
STOP_ATR = 1.25
STOP_MIN_PTS = 3.0
TARGET_R = 2.0
MAX_BARS = 24                      # 24 x 5m = 2h
ONL_FIRST_MIN, ONL_LAST_MIN = 570, 685      # bar starts 09:30 .. 11:25 ET (mins < 690)
E1_FIRST_MIN, E1_LAST_MIN = 1080, 1135      # bar starts 18:00 .. 18:55 ET
E1_RSI_MIN = 59.5
E1_MOVE_ATR_MIN = 1.48
E1_LOOKBACK = 12
WARMUP_BARS = 150                  # EWM(1/14) warm-up before a reading is trusted
CANDLE_COUNT = 400
# backtest win rates (flat-only, Nov 2024 - Sep 2026), shown as the entry confidence
CONF = {"onl": 0.45, "e1": 0.51}
LABEL = {"onl": "ONL reclaim", "e1": "6PM long"}


def enabled(name: str) -> bool:
    env = {"onl": "ADDON_ONL_RECLAIM", "e1": "ADDON_E1_6PM"}[name]
    return str(os.getenv(env, "1")).strip().lower() not in ("0", "false", "no", "off")


def candle_count() -> int:
    try:
        return max(250, int(os.getenv("ADDON_CANDLE_COUNT", str(CANDLE_COUNT))))
    except ValueError:
        return CANDLE_COUNT


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def closed_bars(df: pd.DataFrame, now=None, stamp: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Normalize a live 5m frame to CLOSED bars indexed by bar START (ET).

    stamp: 'end' (Rithmic history: datetime = bar end, all bars finished) or 'start'.
    Default: df.attrs['bar_time'] if present, else 'start'. Start-stamped frames drop a
    still-forming newest bar.
    """
    if df is None or len(df) == 0:
        return None
    stamp = stamp or (getattr(df, "attrs", {}) or {}).get("bar_time") or "start"
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if "datetime" in d.columns:
        t = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
    elif "time" in d.columns:
        t = pd.to_datetime(d["time"], utc=True, errors="coerce")
    elif isinstance(d.index, pd.DatetimeIndex):
        t = pd.Series(d.index, index=d.index)
        t = pd.to_datetime(t, utc=True)
    else:
        return None
    d = d.assign(_t=t.values).dropna(subset=["_t"])
    if stamp == "end":
        d["_t"] = pd.to_datetime(d["_t"], utc=True) - pd.Timedelta(minutes=BAR_MIN)
    d["_t"] = pd.to_datetime(d["_t"], utc=True)
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d.sort_values("_t").drop_duplicates("_t", keep="last")
    now_ts = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    d = d[d["_t"] + pd.Timedelta(minutes=BAR_MIN) <= now_ts]     # finished bars only
    if d.empty:
        return None
    d.index = pd.DatetimeIndex(d["_t"]).tz_convert(ET)
    return d[["open", "high", "low", "close"]].astype(float)


def add_features(b: pd.DataFrame) -> pd.DataFrame:
    """Same formulas as pattern_mining/build.py (atr, rsi, ret12, mins, tday, rth, onl)."""
    b = b.copy()
    et = b.index
    b["mins"] = et.hour * 60 + et.minute
    b["tday"] = (et.tz_localize(None) + pd.Timedelta(hours=6)).normalize()
    b["rth"] = (b.mins >= 570) & (b.mins < 960)
    pc = b.close.shift()
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    dlt = b.close.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    b["rsi"] = 100 - 100 / (1 + up / dn)
    b["ret12"] = (b.close - b.close.shift(E1_LOOKBACK)) / b.atr
    pre = ((~b.rth) & (b.mins < 570)) | (b.mins >= 1080)
    onl = b.low.where(pre).groupby(b.tday).cummin()
    b["onl"] = onl.groupby(b.tday).ffill()
    # overnight low is complete only if this trading day's 18:00 bar is in the frame
    first_m = b.groupby("tday")["mins"].transform("first")
    pos = pd.Series(np.arange(len(b)), index=b.index)
    first_pos = pos.groupby(b.tday.values).transform("first").values
    # complete = this trading day's first bar is the 18:00 open (or 18:05-18:15 if the frame
    # also holds the previous session, i.e. we did not just cut the frame there)
    b["onl_complete"] = (first_m == 1080) | ((first_m > 1080) & (first_m <= 1095) & (first_pos > 0))
    b["onl_hit"] = b.rth & (b.mins >= ONL_FIRST_MIN) & (b.mins <= ONL_LAST_MIN) & (b.low < b.onl) & (b.close > b.onl)
    b["e1_hit"] = (~b.rth) & (b.mins >= E1_FIRST_MIN) & (b.mins <= E1_LAST_MIN) & \
                  (b.rsi >= E1_RSI_MIN) & (b.ret12 >= E1_MOVE_ATR_MIN)
    return b


def bracket(entry_ref: float, atr: float, tick: float = 0.25) -> Dict[str, float]:
    sd = max(STOP_ATR * float(atr), STOP_MIN_PTS)
    r = lambda p: round(round(p / tick) * tick, 8)
    return {"stop_dist": sd, "sl": r(entry_ref - sd), "tp": r(entry_ref + TARGET_R * sd)}


@dataclass
class Eval:
    kind: str
    signal: bool = False
    status: str = ""
    bar_start: Optional[pd.Timestamp] = None
    close: float = float("nan")
    atr: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)


def _fmt(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".") if np.isfinite(x) else "n/a"


def evaluate(b: pd.DataFrame, kind: str, now=None) -> Eval:
    """Evaluate the newest CLOSED bar of a featured frame (add_features) for one setup."""
    ev = Eval(kind=kind)
    lab = LABEL[kind]
    if not enabled(kind):
        ev.status = f"{lab}: off ({'ADDON_ONL_RECLAIM' if kind == 'onl' else 'ADDON_E1_6PM'}=0)"
        return ev
    if b is None or len(b) == 0:
        ev.status = f"{lab}: no 5m bars loaded"
        return ev
    last = b.iloc[-1]
    ev.bar_start, ev.close, ev.atr = b.index[-1], float(last.close), float(last.atr)
    now_ts = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    stale = (now_ts - (b.index[-1].tz_convert("UTC") + pd.Timedelta(minutes=BAR_MIN))) > pd.Timedelta(minutes=4)
    m = int(last.mins)
    if kind == "onl":
        today = b[b.tday == last.tday]
        onl = float(last.onl) if pd.notna(last.onl) else float("nan")
        if not bool(last.onl_complete) or not np.isfinite(onl):
            ev.status = f"{lab}: overnight low unknown (need bars back to 6 PM ET, have {len(b)})"
            return ev
        prior = today[(today.index < b.index[-1]) & today.onl_hit]
        if len(prior):
            ev.status = f"{lab}: overnight low {_fmt(onl)} - today's setup already came at {prior.index[0].strftime('%H:%M')} ET, done for today"
            return ev
        if not (ONL_FIRST_MIN <= m <= ONL_LAST_MIN) or not bool(last.rth):
            when = "window 9:30-11:30 ET" + (" (closed for today)" if m > ONL_LAST_MIN and bool(last.rth) else "")
            ev.status = f"{lab}: overnight low {_fmt(onl)}, price {_fmt(ev.close)}, {when}"
            return ev
        if bool(last.onl_hit):
            ev.signal = len(b) >= WARMUP_BARS and not stale and np.isfinite(ev.atr) and ev.atr > 0
            ev.status = (f"{lab}: dipped to {_fmt(float(last.low))} under overnight low {_fmt(onl)} and closed back above at {_fmt(ev.close)}"
                         + ("" if ev.signal else " (not traded: stale or too few bars)"))
            return ev
        low_today = float(today[today.rth].low.min()) if today.rth.any() else float("nan")
        dipped = np.isfinite(low_today) and low_today < onl
        ev.status = (f"{lab}: overnight low {_fmt(onl)}, price {_fmt(ev.close)}, "
                     + ("needs a 5m close back above it" if (dipped and ev.close < onl) else
                        f"needs dip under {_fmt(onl)} then close back above") + " (9:30-11:30)")
        ev.extra["onl"] = onl
        return ev
    # E1
    rsi, mv = float(last.rsi), float(last.ret12)
    in_win = (not bool(last.rth)) and E1_FIRST_MIN <= m <= E1_LAST_MIN
    core = f"up {mv:.2f} ATR, needs {E1_MOVE_ATR_MIN:g}; RSI {rsi:.0f}, needs {E1_RSI_MIN:g}"
    if not in_win:
        ev.status = f"{lab}: {core} (window 6:00-7:00 PM ET)"
        return ev
    if bool(last.e1_hit):
        ev.signal = len(b) >= WARMUP_BARS and not stale and np.isfinite(ev.atr) and ev.atr > 0
        ev.status = f"{lab}: {core} - setup" + ("" if ev.signal else " (not traded: stale or too few bars)")
    else:
        ev.status = f"{lab}: {core}"
    return ev


def make_signal(ev: Eval, symbol: str, tick: float = 0.25) -> Dict[str, Any]:
    br = bracket(ev.close, ev.atr, tick)
    exit_at = ev.bar_start.tz_convert("UTC") + pd.Timedelta(minutes=BAR_MIN * (1 + MAX_BARS))
    return {
        "symbol": symbol, "direction": "long", "entry": float(ev.close), "sl": br["sl"], "tp": br["tp"],
        "atr": float(ev.atr), "confidence": CONF[ev.kind], "ensemble_signal": "BUY",
        "model_signals": {"addon": ev.kind}, "regime": "addon", "trailing_enabled": False,
        "exit_sleeve": "addon", "addon": ev.kind, "fixed_contracts": 1,
        "addon_exit_at": exit_at.to_pydatetime(), "addon_label": LABEL[ev.kind],
    }


def in_display_window(now=None) -> List[str]:
    """Which setups to show/evaluate at this wall-clock time (ET)."""
    t = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    e = t.tz_convert(ET)
    m = e.hour * 60 + e.minute
    out = []
    if e.weekday() < 5 and 9 * 60 + 25 <= m <= 11 * 60 + 40:
        out.append("onl")
    if e.weekday() not in (4, 5) and 17 * 60 + 55 <= m <= 19 * 60 + 10:   # Sun-Thu evenings (no Fri/Sat open)
        out.append("e1")
    return out


def entry_line(signal: Dict[str, Any], contracts: int, fill: float) -> str:
    return (f"LONG {contracts} {signal['symbol']} at {fill:,.2f}. SL {signal['sl']:,.2f}, TP {signal['tp']:,.2f}. "
            f"({signal.get('addon_label', 'add-on')}, conf {signal['confidence']:.0%} = backtest win rate, "
            f"time exit {pd.Timestamp(signal['addon_exit_at']).tz_convert(ET).strftime('%H:%M')} ET)")
