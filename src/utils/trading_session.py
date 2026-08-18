"""
CME / US/Eastern trading session helpers for live MTF bot and backtests.

Modes:
  rth       — Mon–Fri 9:30 AM – 4:30 PM ET (default)
  extended  — CME Globex equity index window: Sun 6 PM ET – Fri 5 PM ET,
              with daily maintenance halt 5:00–6:00 PM ET Mon–Thu

Optional SCALP_SESSIONS — intraday windows inside the outer session:
  morning=09:30-12:00;afternoon=13:30-16:00
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

try:
    import pytz

    ET = pytz.timezone("US/Eastern")
    UTC = pytz.UTC
except ImportError:
    ET = None
    UTC = None

SESSION_RTH = "rth"
SESSION_EXTENDED = "extended"

DatetimeLike = Union[datetime, None]


@dataclass(frozen=True)
class SessionWindow:
    """Named HH:MM–HH:MM window in US/Eastern."""

    label: str
    start: str
    end: str


# Profit-mode default: skip Globex overnight + lunch chop; trade liquid RTH.
DEFAULT_SCALP_SESSIONS = (
    SessionWindow("morning", "09:30", "12:00"),
    SessionWindow("afternoon", "13:30", "16:00"),
)


def coerce_session_mode(mode: Optional[str]) -> str:
    if not mode:
        return SESSION_RTH
    m = str(mode).strip().lower()
    if m in (SESSION_EXTENDED, "overnight", "globex", "extended_hours", "24h"):
        return SESSION_EXTENDED
    return SESSION_RTH


def resolve_session_mode_from_env() -> str:
    """SESSION_MODE=extended|rth, or OVERNIGHT_TRADING=true → extended. Default rth."""
    explicit = os.getenv("SESSION_MODE", "").strip()
    if explicit:
        return coerce_session_mode(explicit)
    overnight = os.getenv("OVERNIGHT_TRADING", "false").strip().lower()
    if overnight in ("1", "true", "yes", "on"):
        return SESSION_EXTENDED
    return SESSION_RTH


def _to_et(now: DatetimeLike) -> datetime:
    if ET is None:
        raise ImportError("pytz is required for trading session checks")
    if now is None:
        return datetime.now(ET)
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    if now.tzinfo is None:
        now = UTC.localize(now)
    return now.astimezone(ET)


def is_rth_session_et(now: DatetimeLike = None) -> bool:
    """Mon–Fri 9:30 AM – 4:30 PM US/Eastern."""
    et = _to_et(now)
    if et.weekday() >= 5:
        return False
    mins = et.hour * 60 + et.minute
    return (9 * 60 + 30) <= mins < (16 * 60 + 30)


def is_globex_session_et(now: DatetimeLike = None) -> bool:
    """CME Globex MNQ/NQ: Sun 6 PM – Fri 5 PM ET; daily halt 5–6 PM Mon–Thu."""
    et = _to_et(now)
    wd = et.weekday()
    mins = et.hour * 60 + et.minute

    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday — opens 6 PM
        return mins >= 18 * 60
    if wd == 4 and mins >= 17 * 60:  # Friday — closes 5 PM
        return False
    if wd <= 3 and 17 * 60 <= mins < 18 * 60:  # Mon–Thu maintenance
        return False
    return True


def is_session_open_et(now: DatetimeLike = None, mode: str = SESSION_RTH) -> bool:
    if coerce_session_mode(mode) == SESSION_EXTENDED:
        return is_globex_session_et(now)
    return is_rth_session_et(now)


def session_mode_label(mode: str = SESSION_RTH) -> str:
    if coerce_session_mode(mode) == SESSION_EXTENDED:
        return "EXTENDED — Sun 6:00 PM – Fri 5:00 PM ET (halt 5–6 PM Mon–Thu)"
    return "RTH — Mon–Fri 9:30 AM – 4:30 PM ET"


def seconds_until_session_open_et(now: DatetimeLike = None, mode: str = SESSION_RTH) -> float:
    """Seconds until the next open bar for the given session mode."""
    et = _to_et(now)
    mode = coerce_session_mode(mode)
    if is_session_open_et(et, mode):
        return 0.0
    probe = et
    for _ in range(4 * 24 * 60):
        probe += timedelta(minutes=1)
        if is_session_open_et(probe, mode):
            return max(60.0, (probe - et).total_seconds())
    return 3600.0


def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.strip().split(":")
    return int(h), int(m)


def parse_session_windows(raw: str) -> List[SessionWindow]:
    """Parse `label=HH:MM-HH:MM;label2=HH:MM-HH:MM` (same shape as FVS1_SESSIONS)."""
    windows: List[SessionWindow] = []
    for part in (raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        label, times = part.split("=", 1) if "=" in part else ("session", part)
        if "-" not in times:
            continue
        start, end = times.split("-", 1)
        windows.append(SessionWindow(label.strip(), start.strip(), end.strip()))
    return windows


def load_scalp_sessions_from_env(
    env_key: str = "SCALP_SESSIONS",
    *,
    use_defaults: bool = True,
) -> List[SessionWindow]:
    """
    Load SCALP_SESSIONS windows.

    Empty / unset → DEFAULT_SCALP_SESSIONS when use_defaults=True.
    Set SCALP_SESSIONS=off|none|all|rth to disable intraday filtering (full outer session).
    """
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return list(DEFAULT_SCALP_SESSIONS) if use_defaults else []
    low = raw.lower()
    if low in ("off", "none", "all", "rth", "full", "false", "0"):
        return []
    parsed = parse_session_windows(raw)
    if parsed:
        return parsed
    return list(DEFAULT_SCALP_SESSIONS) if use_defaults else []


def is_in_session_windows_et(
    now: DatetimeLike = None,
    windows: Optional[List[SessionWindow]] = None,
) -> Tuple[bool, str]:
    """Return (in_window, label_or_reason). Empty windows → always True (no filter)."""
    if not windows:
        return True, "all"
    et = _to_et(now)
    if et.weekday() >= 5:
        return False, "weekend"
    mins = et.hour * 60 + et.minute
    for w in windows:
        sh, sm = _parse_hhmm(w.start)
        eh, em = _parse_hhmm(w.end)
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        if start_m <= mins < end_m:
            return True, w.label
    return False, "outside scalp windows"


def seconds_until_scalp_window_et(
    now: DatetimeLike = None,
    windows: Optional[List[SessionWindow]] = None,
    outer_mode: str = SESSION_RTH,
) -> float:
    """Seconds until next moment that is both outer-session open and inside a scalp window."""
    if not windows:
        return seconds_until_session_open_et(now, outer_mode)
    et = _to_et(now)
    if is_session_open_et(et, outer_mode):
        ok, _ = is_in_session_windows_et(et, windows)
        if ok:
            return 0.0
    probe = et
    for _ in range(4 * 24 * 60):
        probe += timedelta(minutes=1)
        if not is_session_open_et(probe, outer_mode):
            continue
        ok, _ = is_in_session_windows_et(probe, windows)
        if ok:
            return max(60.0, (probe - et).total_seconds())
    return 3600.0


def format_session_windows(windows: Optional[List[SessionWindow]]) -> str:
    if not windows:
        return "full outer session (no SCALP_SESSIONS filter)"
    return "; ".join(f"{w.label} {w.start}-{w.end}" for w in windows)
