"""Daily dip: three lower Nasdaq closes and RSI(2) under 15.

Saved rule from research/build_dips_2026-11-07. Buy the next session open.
Sell when RSI(2) closes back above 65, or on the close of the 6th session
(the entry session plus five more), whichever comes first. One MNQ.

DIP_DOWN_RSI defaults off. The research trade has no stop. Live brackets still
need one, so the order carries a disaster stop (DIP_DOWN_RSI_STOP_PTS, default
2500 points, $5,000 on 1 MNQ). That distance was never reached from 2010
through 2026-11-06 (worst open-to-low heat was 1,834 points).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
RSI_ENTRY = 15.0
RSI_EXIT = 65.0
# Entry session is day 1. Exit on that close through the close 5 sessions later.
MAX_SESSIONS = 6
# Biggest open-to-high in the 2010-2026 trades was 1,251 points. 3,000 never hit.
FAR_TP_PTS = 3000.0
TICK = 0.25

Bar = Tuple[date, float]


def enabled() -> bool:
    return os.getenv("DIP_DOWN_RSI", "0").strip().lower() in ("1", "true", "yes", "on")


def stop_pts() -> float:
    try:
        pts = float(os.getenv("DIP_DOWN_RSI_STOP_PTS", "2500"))
    except (TypeError, ValueError):
        pts = 2500.0
    return pts if pts > 0 else 2500.0


def state_path() -> Path:
    return Path(os.getenv("DIP_DOWN_RSI_STATE", os.path.join("logs", "down_rsi_state.json")))


def cache_path() -> Path:
    return Path(os.getenv("DIP_DOWN_RSI_CACHE", os.path.join("logs", "down_rsi_daily.csv")))


def rsi2(close: Sequence[float]) -> np.ndarray:
    """Same RSI(2) as research/fresh_rules_2026-11-07/run.py."""
    c = np.asarray(close, dtype=float)
    if len(c) == 0:
        return c
    d = np.diff(c, prepend=c[0])
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    au = pd.Series(up).ewm(alpha=1 / 2, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1 / 2, adjust=False).mean().to_numpy()
    rs = np.divide(au, ad, out=np.full_like(au, np.inf), where=ad > 0)
    return 100 - 100 / (1 + rs)


def is_signal(closes: Sequence[float]) -> bool:
    """True on the last close when it is the third lower close and RSI(2) < 15."""
    c = np.asarray(closes, dtype=float)
    if len(c) < 4 or not np.isfinite(c[-4:]).all():
        return False
    down3 = bool(c[-1] < c[-2] < c[-3] < c[-4])
    if not down3:
        return False
    r = float(rsi2(c)[-1])
    return bool(np.isfinite(r) and r < RSI_ENTRY)


def round_tick(price: float, tick: float = TICK) -> float:
    return round(round(float(price) / tick) * tick, 2)


def _minutes(now: datetime) -> int:
    return now.hour * 60 + now.minute


def as_et(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=ET)
    return now.astimezone(ET)


def previous_weekday(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def add_weekdays(d: date, n: int) -> date:
    cur = d
    left = n
    while left > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            left -= 1
    return cur


def weekdays_inclusive(a: date, b: date) -> int:
    if b < a:
        return 0
    n = 0
    cur = a
    while cur <= b:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def completed_session(now: datetime) -> date:
    """Session date whose 17:00 ET close is already in."""
    now = as_et(now)
    d = now.date()
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d - timedelta(days=2)
    if _minutes(now) >= 17 * 60:
        return d
    return previous_weekday(d)


def entry_moment(signal_day: date) -> datetime:
    """18:00 ET open after that session's close. Friday waits until Sunday."""
    if signal_day.weekday() == 4:
        open_day = signal_day + timedelta(days=2)
    elif signal_day.weekday() == 5:
        open_day = signal_day + timedelta(days=1)
    else:
        open_day = signal_day
    return datetime(open_day.year, open_day.month, open_day.day, 18, 0, tzinfo=ET)


def entry_session(signal_day: date) -> date:
    d = signal_day + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def parse_day(text: str) -> Optional[date]:
    text = (text or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


@dataclass
class DipState:
    phase: str = "idle"  # idle | pending | open
    signal_day: str = ""
    entry_session: str = ""
    force_session: str = ""
    entry_price: float = 0.0
    stop: float = 0.0
    tp: float = 0.0
    order_id: str = ""
    entry_sent: bool = False
    last_exit_day: str = ""
    entry_time: str = ""

    def signal_date(self) -> Optional[date]:
        return parse_day(self.signal_day)

    def entry_date(self) -> Optional[date]:
        return parse_day(self.entry_session)

    def force_date(self) -> Optional[date]:
        return parse_day(self.force_session)


@dataclass
class DipPlan:
    action: str  # none | arm | enter | exit | expire
    state: DipState
    reason: str
    status: str


def load_state(path: Optional[Path] = None) -> DipState:
    path = path or state_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DipState()
    if not isinstance(raw, dict):
        return DipState()
    known = {k: raw.get(k, getattr(DipState(), k)) for k in asdict(DipState())}
    st = DipState(**known)
    if st.phase not in ("idle", "pending", "open"):
        st.phase = "idle"
    return st


def save_state(state: DipState, path: Optional[Path] = None) -> None:
    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _idle(state: DipState, reason: str, status: str) -> DipPlan:
    nxt = DipState(last_exit_day=state.last_exit_day)
    return DipPlan("expire", nxt, reason, status)


def plan(state: DipState, now: datetime, bars: Sequence[Bar], flat: bool) -> DipPlan:
    """Decide the next action. Does not place orders.

    `bars` is session date and close, oldest first. Include today's live price
    only once that session's close is being judged (16:55 ET onward).
    """
    now = as_et(now)
    bars = [(d, float(c)) for d, c in bars if d is not None and np.isfinite(c)]
    bars.sort(key=lambda x: x[0])
    closes = [c for _, c in bars]
    mins = _minutes(now)
    today = now.date()

    if state.phase == "pending":
        sig = state.signal_date()
        if sig is None:
            return _idle(state, "pending dip had no signal day", "daily dip: cleared a bad pending state")
        start = entry_moment(sig)
        end = start + timedelta(minutes=15)
        if now < start:
            return DipPlan("none", state, "", f"daily dip: armed {sig}, buy at {start.strftime('%a %H:%M')} ET. 5-minute entries paused")
        if now <= end:
            if not flat:
                return DipPlan("none", state, "", "daily dip: 6pm open, waiting for the bot to be flat. 5-minute entries paused")
            return DipPlan("enter", state, f"buy 1 at the open after {sig}", f"daily dip: buying 1 after {sig}")
        return _idle(state, "missed the 6pm open", f"daily dip: missed the open after {sig}")

    if state.phase == "open":
        entry = state.entry_date()
        if entry is None:
            return _idle(state, "open dip had no entry session", "daily dip: cleared a bad open state")
        in_exit = today.weekday() < 5 and (16 * 60 + 55) <= mins <= (17 * 60 + 10)
        if not in_exit or today < entry:
            held = weekdays_inclusive(entry, today) if today >= entry else 0
            return DipPlan("none", state, "", f"daily dip: long from {entry}, session {held} of {MAX_SESSIONS}. 5-minute entries paused")
        held = weekdays_inclusive(entry, today)
        r = float(rsi2(closes)[-1]) if closes else float("nan")
        if (np.isfinite(r) and r > RSI_EXIT) or held >= MAX_SESSIONS:
            why = f"RSI(2) {r:.1f}" if np.isfinite(r) and r > RSI_EXIT else f"session {held} of {MAX_SESSIONS}"
            return DipPlan("exit", state, why, f"daily dip: selling, {why}")
        return DipPlan("none", state, "", f"daily dip: long from {entry}, RSI(2) {r:.1f}, session {held} of {MAX_SESSIONS}. 5-minute entries paused")

    # idle
    done = completed_session(now)
    if state.last_exit_day[:10] == done.isoformat():
        return DipPlan("none", state, "", "daily dip: off, already exited this session")
    if not bars or bars[-1][0] != done:
        return DipPlan("none", state, "", "daily dip: off, waiting for the daily close")
    if not is_signal(closes):
        r = float(rsi2(closes)[-1])
        return DipPlan("none", state, "", f"daily dip: off, last close {done} RSI(2) {r:.1f}")
    if now > entry_moment(done) + timedelta(minutes=15):
        return DipPlan("none", state, "", f"daily dip: off, {done} was a signal and the open was missed")
    nxt = DipState(
        phase="pending",
        signal_day=done.isoformat(),
        entry_session=entry_session(done).isoformat(),
        force_session=add_weekdays(entry_session(done), MAX_SESSIONS - 1).isoformat(),
        last_exit_day=state.last_exit_day,
    )
    start = entry_moment(done)
    return DipPlan("arm", nxt, f"signal {done}", f"daily dip: signal {done}, buy at {start.strftime('%a %H:%M')} ET")


def with_live_close(bars: Sequence[Bar], day: date, price: float) -> List[Bar]:
    out = [(d, float(c)) for d, c in bars if d != day]
    if price > 0 and np.isfinite(price):
        out.append((day, float(price)))
    out.sort(key=lambda x: x[0])
    return out


def use_live_close(now: datetime) -> bool:
    now = as_et(now)
    return now.weekday() < 5 and _minutes(now) >= (16 * 60 + 55)


def filter_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the same range spikes the research daily loader drops."""
    rng = df["High"] - df["Low"]
    bad = rng > 6 * rng.rolling(20, min_periods=10).median()
    return df.loc[~bad.fillna(False)].copy()


def closes_from_ohlc(df: pd.DataFrame) -> List[Bar]:
    if df is None or df.empty:
        return []
    out = []
    for ts, row in df.iterrows():
        d = pd.Timestamp(ts).date()
        out.append((d, float(row["Close"])))
    return out


def fetch_yahoo(start: str = "2009-01-01") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download("NQ=F", start=start, interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna().astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_bars(path: Optional[Path] = None, fetch=None, refresh: bool = True) -> List[Bar]:
    """Cached NQ=F daily closes, spike-filtered the same way as the research."""
    path = path or cache_path()
    df = None
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        except Exception:
            df = None
    stale = df is None or df.empty
    if not stale and refresh:
        last = pd.Timestamp(df.index.max()).date()
        stale = (date.today() - last).days > 3
    if stale and fetch is not False:
        try:
            got = fetch() if fetch is not None else fetch_yahoo()
            if got is not None and not got.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                save = got[["Open", "High", "Low", "Close"]].copy()
                save.index.name = "date"
                save.to_csv(path)
                df = save
        except Exception:
            if df is None:
                return []
    if df is None or df.empty:
        return []
    return closes_from_ohlc(filter_spikes(df))


def make_order(symbol: str, price: float, force_at: datetime, tick: float = TICK) -> dict:
    """Bracket the live bot can send. Target is far so only the daily rule exits."""
    pts = stop_pts()
    entry = round_tick(price, tick)
    sl = round_tick(entry - pts, tick)
    tp = round_tick(entry + FAR_TP_PTS, tick)
    return {
        "symbol": symbol,
        "direction": "long",
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": pts,
        "confidence": 0.5,
        "ensemble_signal": "BUY",
        "model_signals": {"dip": "down_rsi"},
        "regime": "down_rsi",
        "trailing_enabled": False,
        "exit_sleeve": "down_rsi",
        "addon": "dip",
        "addon_label": "3 down days",
        "fixed_contracts": 1,
        "addon_exit_at": force_at,
        "ignore_risk_cap": True,
    }
