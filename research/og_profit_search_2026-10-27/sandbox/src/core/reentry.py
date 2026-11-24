"""Re-entry after a profitable lock/trail stop exit (REVAMP live, user-approved 2026-10-08).

Backtest: /workspace/nq_bt/out/sr_targets/REENTRY_RESULTS.md (best variant
N12_max1_stop-swing_tp-new2R_none: holdout net +$4,422 / PF 1.42 vs +$2,556 / 1.29).

Rule
  trigger  a bot trade closes in profit via its STOP (lock/trail exit). Not TP, not
           EOD/flatten/broker-sync, not adopted/manual trades, not past REENTRY_MAX.
  arm      for REENTRY_WINDOW_BARS x 5m after the exit, watch for price 1 tick beyond
           the parent's peak (long: high, short: low), same direction.
  entry    bot-side MARKET entry when the live price crosses (no resting broker stop
           order -> nothing for the orphan sweep to find). Only 09:30-15:55 ET; the arm
           is dropped at the first check outside that window (like the backtest).
           The first cross consumes the arm even if a risk gate blocks it.
  stop     max(last 6 completed 5m bars' low/high -/+ 1 tick, 1.5 x ATR), capped so
           1 contract risks <= MAX_RISK_PER_TRADE_USD.
  target   new 2R from the entry price. Same $ lock as every trade.
  gates    bypasses the post-exit cooldown; respects daily loss cap, max trades/day,
           max concurrent trades, 3-loss pause, session filter, and the extra-trade
           confidence rule (re-entry confidence = parent's).
Env (default OFF; only start_mnq_live_revamp.bat turns it on):
  REENTRY_ENABLED=0|1  REENTRY_WINDOW_BARS=12  REENTRY_MAX=1
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import pandas as pd

ET = "America/New_York"
BAR_MINUTES = 5
ENTRY_START_MIN = 9 * 60 + 30   # 09:30 ET
ENTRY_END_MIN = 15 * 60 + 55    # 15:55 ET (inclusive, like the backtest's last entry bar)


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip() or default)
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip() or default)
    except (TypeError, ValueError):
        return default


REENTRY_ENABLED = _truthy(os.getenv("REENTRY_ENABLED", "0"))
REENTRY_WINDOW_BARS = max(1, _int("REENTRY_WINDOW_BARS", 12))
REENTRY_MAX = max(0, _int("REENTRY_MAX", 1))
REENTRY_SWING_BARS = max(1, _int("REENTRY_SWING_BARS", 6))
REENTRY_ATR_MULT = _float("REENTRY_ATR_MULT", 1.5)
REENTRY_TARGET_R = _float("REENTRY_TARGET_R", 2.0)


@dataclass
class ReentryArm:
    parent_id: str
    symbol: str
    direction: str          # 'long' | 'short'
    trigger: float          # enter when live price is at/through this
    peak: float             # parent's best price
    armed_at: datetime
    expires_at: datetime
    confidence: float
    sleeve: str = ""
    gen: int = 1            # generation of the re-entry this arm would open
    ensemble_signal: str = ""
    regime: str = ""

    @property
    def side(self) -> str:
        return "BUY" if self.direction == "long" else "SELL"

    def status_text(self, now: Optional[datetime] = None) -> str:
        now = now or datetime.now(timezone.utc)
        left = max(0, math.ceil((self.expires_at - now).total_seconds() / 60.0))
        rel = "above" if self.direction == "long" else "below"
        return f"Re-entry armed: {self.side} {rel} {self.trigger:.2f} for {left} min"


def round_tick(px: float, tick: float) -> float:
    return round(round(float(px) / tick) * tick, 10)


def in_entry_window_et(now_utc: datetime) -> bool:
    t = pd.Timestamp(now_utc)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert(ET)
    m = t.hour * 60 + t.minute
    return t.weekday() < 5 and ENTRY_START_MIN <= m <= ENTRY_END_MIN


def is_adopted(pos: Any) -> bool:
    oid = str(getattr(pos, "order_id", "") or "")
    return oid.startswith("bot_adopt") or str(getattr(pos, "regime", "") or "") == "broker_adopt"


def should_arm(pos: Any, exit_type: str, pnl: float, exit_price: float,
               enabled: Optional[bool] = None, max_reentries: Optional[int] = None) -> Tuple[bool, str]:
    """(arm?, reason). Profitable STOP exit of a bot trade that may still re-enter."""
    enabled = REENTRY_ENABLED if enabled is None else enabled
    max_reentries = REENTRY_MAX if max_reentries is None else max_reentries
    if not enabled:
        return False, "disabled"
    if str(exit_type).upper() != "STOP_LOSS":
        return False, f"exit {exit_type} (only profitable stop exits)"
    if not (pnl > 0):
        return False, "not profitable"
    if is_adopted(pos):
        return False, "adopted/manual trade"
    if int(getattr(pos, "reentry_gen", 0) or 0) >= int(max_reentries):
        return False, f"max re-entries ({max_reentries}) reached"
    d = 1 if pos.direction == "long" else -1
    try:
        if d * (float(exit_price) - float(pos.entry_price)) <= 0:
            return False, "stop not in profit"
    except (TypeError, ValueError):
        return False, "bad prices"
    return True, "ok"


def _bar_times(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ("datetime", "time"):
        if col in df.columns:
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            return s
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
        return pd.Series(idx.tz_convert("UTC"), index=df.index)
    return None


def completed_bars(df: Optional[pd.DataFrame], now_utc: datetime,
                   bar_minutes: int = BAR_MINUTES) -> Optional[pd.DataFrame]:
    """Rows whose 5m bar has fully closed by now (bar start + 5m <= now)."""
    if df is None or len(df) == 0:
        return None
    ts = _bar_times(df)
    if ts is None:
        return df.iloc[:-1]  # unknown timing: assume the last row is still forming
    cutoff = pd.Timestamp(now_utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    mask = (ts + pd.Timedelta(minutes=bar_minutes)) <= cutoff
    return df[mask.values]


def parent_peak_price(pos: Any, df: Optional[pd.DataFrame] = None,
                      exit_time: Optional[datetime] = None) -> float:
    """Best price the parent reached: tracked peak/MFE, refined with 5m bars after entry."""
    d = 1 if pos.direction == "long" else -1
    tracked = max(float(getattr(pos, "peak_pts", 0) or 0), float(getattr(pos, "mfe_pts", 0) or 0))
    peak = float(pos.entry_price) + d * tracked
    if df is not None and len(df) and getattr(pos, "entry_time", None) is not None:
        ts = _bar_times(df)
        if ts is not None:
            et = pd.Timestamp(pos.entry_time)
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            m = ts > et  # bars that started after the fill
            if exit_time is not None:
                xt = pd.Timestamp(exit_time)
                if xt.tzinfo is None:
                    xt = xt.tz_localize("UTC")
                m &= ts <= xt
            sub = df[m.values]
            if len(sub):
                px = float(sub["high"].max()) if d > 0 else float(sub["low"].min())
                peak = max(peak, px) if d > 0 else min(peak, px)
    return peak


def make_arm(pos: Any, peak: float, tick: float, now_utc: datetime, confidence: float,
             window_bars: Optional[int] = None) -> ReentryArm:
    window_bars = REENTRY_WINDOW_BARS if window_bars is None else window_bars
    d = 1 if pos.direction == "long" else -1
    return ReentryArm(
        parent_id=str(pos.order_id), symbol=str(pos.symbol), direction=pos.direction,
        trigger=round_tick(peak + d * tick, tick), peak=float(peak), armed_at=now_utc,
        expires_at=now_utc + timedelta(minutes=BAR_MINUTES * int(window_bars)),
        confidence=float(confidence or 0.0), sleeve=str(getattr(pos, "exit_sleeve", "") or ""),
        gen=int(getattr(pos, "reentry_gen", 0) or 0) + 1,
        ensemble_signal=("BUY" if pos.direction == "long" else "SELL"),
        regime=str(getattr(pos, "regime", "") or ""),
    )


def triggered(arm: ReentryArm, last: float) -> bool:
    if not last or last <= 0:
        return False
    return last >= arm.trigger - 1e-9 if arm.direction == "long" else last <= arm.trigger + 1e-9


def compute_levels(direction: str, entry: float, df_completed: Optional[pd.DataFrame], atr: float,
                   tick: float, point_value: float, max_risk_usd: float,
                   swing_bars: Optional[int] = None, atr_mult: Optional[float] = None,
                   target_r: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
    """(sl, tp, stop_pts) for a re-entry, or None if it cannot be built."""
    swing_bars = REENTRY_SWING_BARS if swing_bars is None else swing_bars
    atr_mult = REENTRY_ATR_MULT if atr_mult is None else atr_mult
    target_r = REENTRY_TARGET_R if target_r is None else target_r
    d = 1 if direction == "long" else -1
    dist = 0.0
    if df_completed is not None and len(df_completed) >= 1:
        last = df_completed.tail(swing_bars)
        if d > 0:
            dist = float(entry) - (float(last["low"].min()) - tick)
        else:
            dist = (float(last["high"].max()) + tick) - float(entry)
    try:
        a = float(atr)
    except (TypeError, ValueError):
        a = float("nan")
    if a == a and a > 0:
        dist = max(dist, atr_mult * a)
    if not (dist > 0):
        return None
    if max_risk_usd and max_risk_usd > 0 and point_value > 0:
        cap = math.floor((float(max_risk_usd) / float(point_value)) / tick + 1e-9) * tick
        dist = min(dist, cap)
    dist = math.ceil(dist / tick - 1e-9) * tick
    if max_risk_usd and max_risk_usd > 0 and point_value > 0:
        cap = math.floor((float(max_risk_usd) / float(point_value)) / tick + 1e-9) * tick
        dist = min(dist, cap)
    if dist <= 0:
        return None
    sl = round_tick(entry - d * dist, tick)
    tp = round_tick(entry + d * target_r * dist, tick)
    return sl, tp, float(dist)
