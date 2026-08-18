"""
Overnight Globex PAPER overlay — break_settled_onh_onl only.

This is not the locked ema15_eod RTH recipe. Same 15m windows on Globex
already failed IS (PF ~0.97). Do not set OVERNIGHT_TRADING on live.

Paper recipe (1 MNQ, Lucid TEST/sim — not funded live):
  After ~60 overnight minutes, ONH/ONL must be 20+ bars old.
  First completed 1m close that holds beyond that settled extreme
  (break_onh long / break_onl short). Max 1 fill per Globex night.
  Stop = clip(completed 15m ATR14 × 1.5, 12–40). TP = 1R. Flatten 09:25 ET.
  Fade is OFF (IS PF ~0.52). Combined fade+break failed IS (PF 0.86).

Zones (journaled, not all traded): overnight high/low, prior RTH high/low,
nearest round number, completed 15m EMA21 distance.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from src.strategy.mnq_15m_ema_eod import (
    EMA_SLOW,
    ET,
    _as_utc_frame,
    _atr,
    _ohlc_resample,
    ema_series,
    sl_pts_from_atr as ema15_sl_pts,
    trend_15m_on_1m,
)
from src.utils.trading_session import is_globex_session_et

SESSION_NAME = "overnight_research"
PAPER_RECIPE = "break_settled_onh_onl"
POINT_VALUE = 2.0
COMMISSION = 1.24
QTY = 1
FLATTEN_MINUTE = 9 * 60 + 25  # 09:25 ET — before locked RTH 09:30
NO_ENTRY_MINUTE = 9 * 60 + 20
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60
GLOBEX_OPEN_MINUTE = 18 * 60
WARMUP_BARS = 60
SETTLE_BARS = 20
ATR_MULT = 1.5
SL_MIN = 12.0
SL_MAX = 40.0
TP_R = 1.0
ROUND_INCREMENT = 50.0
TOUCH_TICK = 0.25

CUE_FADE_ONH = "fade_onh"
CUE_FADE_ONL = "fade_onl"
CUE_BREAK_ONH = "break_onh"
CUE_BREAK_ONL = "break_onl"

ZONE_ONH = "overnight_high"
ZONE_ONL = "overnight_low"
ZONE_RTH_HIGH = "prior_rth_high"
ZONE_RTH_LOW = "prior_rth_low"
ZONE_ROUND = "round_number"
ZONE_EMA15 = "ema15_slow"


def overnight_flag_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_OVERNIGHT", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def paper_test_fill_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_TEST_FILL", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def enable_overnight_lucid_sim_orders(
    environ: Optional[MutableMapping[str, str]] = None,
    *,
    keep_test_fill: bool = False,
) -> MutableMapping[str, str]:
    """Overnight research submits Lucid TEST/sim orders. Not local fake fills."""
    env = environ if environ is not None else os.environ
    env["PAPER_OVERNIGHT"] = "true"
    env["PAPER_USE_RITHMIC"] = "true"
    env["PAPER_RITHMIC_BRACKETS"] = "true"
    env["OVERNIGHT_TRADING"] = "false"
    env["TRADING_MODE"] = "paper"
    env["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
    env["RITHMIC_ALLOW_SIMULATOR"] = "true"
    env["RITHMIC_SKIP_HISTORY_PNL"] = "true"
    env["RITHMIC_QUOTES_ONLY"] = "false"
    env["RITHMIC_ORDER_PLANT_ONLY"] = "false"
    if not keep_test_fill:
        env["PAPER_TEST_FILL"] = "false"
    return env


def skip_local_fake_fill(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when a fill must be booked on Lucid — never journaled locally."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_RITHMIC_BRACKETS", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def paper_test_fill_signal(
    df_1m: Optional[pd.DataFrame],
    last_price: float,
) -> Dict[str, Any]:
    """Immediate 1 MNQ paper fill for desk testing. Never a live order.

    Side follows completed 15m EMA8 vs EMA21 when known, else long.
    Stop = clip(ATR14×2, 20–60) when ATR is known, else 20 pts.
    """
    px = round(float(last_price), 2)
    side = "long"
    sl_pts = 20.0
    if df_1m is not None and not getattr(df_1m, "empty", True):
        try:
            trend = trend_15m_on_1m(df_1m, completed_only=True)
            last_t = int(trend.iloc[-1]) if trend is not None and not trend.empty else 0
            if last_t < 0:
                side = "short"
            elif last_t > 0:
                side = "long"
            tmp = _as_utc_frame(df_1m)
            b15 = _ohlc_resample(tmp, "15min")
            atr = _atr(b15, 14).shift(1).dropna()
            if not atr.empty:
                sl_pts = float(ema15_sl_pts(float(atr.iloc[-1]), fallback=20.0))
        except Exception:
            side = "long"
            sl_pts = 20.0
    sl_pts = float(np.clip(sl_pts, 20.0, 60.0))
    if side == "short":
        stop = round(px + sl_pts, 2)
        target = round(px - sl_pts, 2)
    else:
        stop = round(px - sl_pts, 2)
        target = round(px + sl_pts, 2)
    return {
        "session": SESSION_NAME,
        "cue": "test_fill",
        "side": side,
        "direction": side,
        "qty": QTY,
        "symbol": "MNQ",
        "entry_price": px,
        "stop": stop,
        "target": target,
        "atr_stop_pts": round(sl_pts, 2),
        "zone_name": "test_fill",
        "zone_price": px,
        "why": "test_fill — paper only, not sent to Lucid",
        "flatten_et": "09:25",
        "paper": True,
    }


def resolve_overnight_research(
    *,
    live: bool = False,
    flag: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Overnight paper never runs under --live. Locked RTH recipe stays RTH."""
    if live:
        return False
    return bool(flag) or overnight_flag_from_env(environ)


def _to_et(now) -> datetime:
    if now is None:
        return datetime.now(ET)
    t = pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).to_pydatetime()


def et_minutes(now) -> int:
    et = _to_et(now)
    return int(et.hour * 60 + et.minute)


def globex_session_start_et(now=None) -> datetime:
    """Last Globex open (18:00 ET Sun–Thu)."""
    et = _to_et(now)
    mins = et_minutes(et)
    start = et.replace(hour=18, minute=0, second=0, microsecond=0)
    if mins < GLOBEX_OPEN_MINUTE:
        start = start - timedelta(days=1)
    if start.weekday() == 5:  # Saturday — Sunday 18:00 is the open
        start = start - timedelta(days=1)
    return start


def in_rth_hard_idle(now=None) -> bool:
    """Weekday 09:30–16:00 ET: no overnight entries (including test_fill)."""
    et = _to_et(now)
    if et.weekday() >= 5:
        return False
    return RTH_OPEN_MINUTE <= et_minutes(et) < RTH_END_MINUTE


def is_overnight_research_hours(now=None) -> bool:
    """Globex after 18:00 ET, excluding the day-bot Lucid window (09:20–18:00)."""
    if not is_globex_session_et(now):
        return False
    if day_bot_owns_lucid(now):
        return False
    if in_rth_hard_idle(now):
        return False
    return True


def should_flatten_before_rth(now=None) -> bool:
    et = _to_et(now)
    if et.weekday() >= 5:
        return False
    return et_minutes(et) >= FLATTEN_MINUTE and et_minutes(et) < GLOBEX_OPEN_MINUTE


def day_bot_owns_lucid(now=None) -> bool:
    """Weekday 09:20–18:00 ET: overnight must not enter or hold the Lucid ticker."""
    et = _to_et(now)
    if et.weekday() >= 5:
        return False
    return NO_ENTRY_MINUTE <= et_minutes(et) < GLOBEX_OPEN_MINUTE


def overnight_entry_blocked_reason(now=None) -> Optional[str]:
    """Hard stop for paper fills (including test_fill) during the day session."""
    if in_rth_hard_idle(now):
        return (
            "RTH — locked live recipe owns 9:30–16:00; overnight paper will not enter"
        )
    if day_bot_owns_lucid(now):
        return (
            "Pre-Globex idle — overnight entries start after 18:00 ET "
            "(one Lucid session)"
        )
    if not allow_new_overnight_entry(now):
        return "Overnight session closed — no new entries"
    return None


def allow_new_overnight_entry(now=None) -> bool:
    if in_rth_hard_idle(now):
        return False
    if day_bot_owns_lucid(now):
        return False
    if not is_overnight_research_hours(now):
        return False
    return True


def _mask_rth(idx: pd.DatetimeIndex) -> np.ndarray:
    et = idx.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    return (et.weekday < 5) & (mins >= RTH_OPEN_MINUTE) & (mins < RTH_END_MINUTE)


def overnight_completed_bars(df_1m: pd.DataFrame, now=None) -> pd.DataFrame:
    """Completed 1m bars from this Globex open up to `now`, excluding an unfinished last minute."""
    if df_1m is None or getattr(df_1m, "empty", True):
        return pd.DataFrame()
    tmp = _as_utc_frame(df_1m).sort_index()
    if "datetime" in tmp.columns:
        tmp = tmp.drop(columns=["datetime"])
    start = globex_session_start_et(now)
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(_to_et(now)).tz_convert("UTC")
    out = tmp.loc[(tmp.index >= start_utc) & (tmp.index <= end)]
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if out.empty:
        return pd.DataFrame(columns=[c for c in cols if c == "datetime" or c in tmp.columns])
    last = out.index[-1]
    if (end - last) < pd.Timedelta(minutes=1) and len(out) > 1:
        out = out.iloc[:-1]
    out = out.copy()
    out.insert(0, "datetime", out.index.tz_convert("UTC"))
    return out.reset_index(drop=True)


def prior_rth_high_low(df_1m: pd.DataFrame, now=None) -> Tuple[Optional[float], Optional[float]]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None, None
    tmp = _as_utc_frame(df_1m)
    rth = tmp.loc[_mask_rth(tmp.index)]
    if rth.empty:
        return None, None
    start = globex_session_start_et(now)
    cutoff = pd.Timestamp(start).tz_convert("UTC")
    rth = rth.loc[rth.index < cutoff]
    if rth.empty:
        return None, None
    et_dates = rth.index.tz_convert(ET).date
    last_day = et_dates[-1]
    day = rth.loc[et_dates == last_day]
    if day.empty:
        return None, None
    return float(day["high"].max()), float(day["low"].min())


def settled_overnight_range(on_bars: pd.DataFrame) -> Dict[str, Any]:
    """ONH/ONL from completed overnight bars; settled if extreme is SETTLE_BARS old."""
    empty = {
        "onh": None,
        "onl": None,
        "onh_settled": False,
        "onl_settled": False,
        "bars": 0,
    }
    if on_bars is None or on_bars.empty or len(on_bars) < 2:
        return empty
    n = len(on_bars)
    prior = on_bars.iloc[:-1]
    onh = float(prior["high"].max())
    onl = float(prior["low"].min())
    hi_pos = int(prior["high"].values.argmax())
    lo_pos = int(prior["low"].values.argmin())
    age_hi = (len(prior) - 1) - hi_pos
    age_lo = (len(prior) - 1) - lo_pos
    return {
        "onh": onh,
        "onl": onl,
        "onh_settled": n >= WARMUP_BARS and age_hi >= SETTLE_BARS,
        "onl_settled": n >= WARMUP_BARS and age_lo >= SETTLE_BARS,
        "bars": n,
        "onh_age_bars": age_hi,
        "onl_age_bars": age_lo,
    }


def atr_stop_pts(df_1m: pd.DataFrame) -> float:
    if df_1m is None or getattr(df_1m, "empty", True):
        return 24.0
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    atr = _atr(b15, 14).shift(1).dropna()
    if atr.empty:
        return 24.0
    return float(np.clip(float(atr.iloc[-1]) * ATR_MULT, SL_MIN, SL_MAX))


def ema15_slow_price(df_1m: pd.DataFrame) -> Optional[float]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    slow = ema_series(b15["close"], EMA_SLOW).shift(1).dropna()
    if slow.empty:
        return None
    return float(slow.iloc[-1])


def nearest_round(price: float, increment: float = ROUND_INCREMENT) -> float:
    return float(round(price / increment) * increment)


def _zone(
    name: str,
    ztype: str,
    price: Optional[float],
    *,
    extra: Optional[Dict[str, Any]] = None,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev = prev or {}
    row = {
        "name": name,
        "type": ztype,
        "price": None if price is None else round(float(price), 2),
        "touches": int(prev.get("touches") or 0),
        "last_updated": datetime.now(pytz.UTC).isoformat(),
    }
    if extra:
        row.update(extra)
    return row


def compute_zones(
    df_1m: pd.DataFrame,
    now=None,
    *,
    last_price: Optional[float] = None,
    prev_zones: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prev_map = {str(z.get("name")): z for z in (prev_zones or []) if isinstance(z, dict)}
    rng = settled_overnight_range(overnight_completed_bars(df_1m, now))
    rth_hi, rth_lo = prior_rth_high_low(df_1m, now)
    ema = ema15_slow_price(df_1m)
    px = last_price
    if px is None and df_1m is not None and not getattr(df_1m, "empty", True):
        px = float(df_1m.iloc[-1]["close"])
    round_px = nearest_round(float(px)) if px else None
    dist = None if ema is None or px is None else round(float(px) - float(ema), 2)
    zones = [
        _zone(ZONE_ONH, ZONE_ONH, rng.get("onh"), extra={"settled": rng.get("onh_settled")}, prev=prev_map.get(ZONE_ONH)),
        _zone(ZONE_ONL, ZONE_ONL, rng.get("onl"), extra={"settled": rng.get("onl_settled")}, prev=prev_map.get(ZONE_ONL)),
        _zone(ZONE_RTH_HIGH, ZONE_RTH_HIGH, rth_hi, prev=prev_map.get(ZONE_RTH_HIGH)),
        _zone(ZONE_RTH_LOW, ZONE_RTH_LOW, rth_lo, prev=prev_map.get(ZONE_RTH_LOW)),
        _zone(ZONE_ROUND, ZONE_ROUND, round_px, extra={"increment": ROUND_INCREMENT}, prev=prev_map.get(ZONE_ROUND)),
        _zone(ZONE_EMA15, ZONE_EMA15, ema, extra={"distance_pts": dist}, prev=prev_map.get(ZONE_EMA15)),
    ]
    return {
        "updated_at": datetime.now(pytz.UTC).isoformat(),
        "updated_at_et": pd.Timestamp(_to_et(now)).isoformat(),
        "session": SESSION_NAME,
        "globex_start_et": globex_session_start_et(now).isoformat(),
        "price": None if px is None else round(float(px), 2),
        "overnight_bars": rng.get("bars") or 0,
        "zones": zones,
        "advisory_only": True,
        "writes_live_config": False,
    }


def increment_zone_touch(payload: Dict[str, Any], name: str) -> None:
    for z in payload.get("zones") or []:
        if z.get("name") == name:
            z["touches"] = int(z.get("touches") or 0) + 1
            z["last_updated"] = datetime.now(pytz.UTC).isoformat()
            return


def last_completed_bar(on_bars: pd.DataFrame) -> Optional[pd.Series]:
    if on_bars is None or on_bars.empty:
        return None
    return on_bars.iloc[-1]


def evaluate_overnight_cue(
    df_1m: pd.DataFrame,
    now=None,
    *,
    already_filled: bool = False,
    used_cues: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Paper signal or None. Break-and-hold only. One fill per Globex session.

    Fade is intentionally disabled — it lost in-sample (PF ~0.52).
    """
    if already_filled or not allow_new_overnight_entry(now):
        return None
    on_bars = overnight_completed_bars(df_1m, now)
    rng = settled_overnight_range(on_bars)
    bar = last_completed_bar(on_bars)
    if bar is None:
        return None
    used = used_cues or set()
    sl = atr_stop_pts(df_1m)
    close = float(bar["close"])
    ts = bar["datetime"] if "datetime" in bar.index else on_bars.iloc[-1].get("datetime")

    def _sig(cue: str, side: str, zone_name: str, zone_px: float, why: str) -> Dict[str, Any]:
        entry = close
        if side == "short":
            stop = entry + sl
            target = entry - sl * TP_R
        else:
            stop = entry - sl
            target = entry + sl * TP_R
        return {
            "session": SESSION_NAME,
            "recipe": PAPER_RECIPE,
            "cue": cue,
            "side": side,
            "direction": side,
            "qty": QTY,
            "symbol": "MNQ",
            "entry_price": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "atr_stop_pts": round(sl, 2),
            "zone_name": zone_name,
            "zone_price": round(float(zone_px), 2),
            "bar_ts": str(ts),
            "why": why,
            "flatten_et": "09:25",
        }

    onh, onl = rng.get("onh"), rng.get("onl")
    if rng.get("onh_settled") and onh is not None and CUE_BREAK_ONH not in used:
        if close > onh + TOUCH_TICK:
            return _sig(
                CUE_BREAK_ONH, "long", ZONE_ONH, onh,
                f"break-and-hold: 1m close {close:.2f} held above settled overnight high {onh:.2f}",
            )
    if rng.get("onl_settled") and onl is not None and CUE_BREAK_ONL not in used:
        if close < onl - TOUCH_TICK:
            return _sig(
                CUE_BREAK_ONL, "short", ZONE_ONL, onl,
                f"break-and-hold: 1m close {close:.2f} held below settled overnight low {onl:.2f}",
            )
    return None


def paper_pnl_usd(side: str, entry: float, exit_px: float, qty: int = QTY) -> Tuple[float, float]:
    si = 1 if str(side).lower() in ("long", "buy") else -1
    pts = (float(exit_px) - float(entry)) * si
    usd = pts * POINT_VALUE * int(qty) - COMMISSION
    return round(pts, 2), round(usd, 2)


def update_mae_mfe(
    side: str,
    entry: float,
    high: float,
    low: float,
    mae: float,
    mfe: float,
) -> Tuple[float, float]:
    si = 1 if str(side).lower() in ("long", "buy") else -1
    if si >= 0:
        mae = max(mae, max(0.0, entry - low))
        mfe = max(mfe, max(0.0, high - entry))
    else:
        mae = max(mae, max(0.0, high - entry))
        mfe = max(mfe, max(0.0, entry - low))
    return round(mae, 2), round(mfe, 2)


def check_paper_exit(
    position: Dict[str, Any],
    *,
    high: float,
    low: float,
    last: float,
    now=None,
) -> Optional[Dict[str, Any]]:
    side = str(position.get("side") or "long").lower()
    stop = float(position["stop"])
    target = float(position.get("target") or 0)
    if should_flatten_before_rth(now):
        return {"reason": "flatten_before_rth", "exit_price": last}
    if side in ("long", "buy"):
        if low <= stop:
            return {"reason": "SL", "exit_price": stop}
        if target and high >= target:
            return {"reason": "TP", "exit_price": target}
    else:
        if high >= stop:
            return {"reason": "SL", "exit_price": stop}
        if target and low <= target:
            return {"reason": "TP", "exit_price": target}
    return None
