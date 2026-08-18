"""FVS-1 (Fabio Valentini Scalper) parameters — env + optional YAML sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    return float(raw) if raw else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    return int(raw) if raw else default


@dataclass
class FVS1SessionWindow:
    label: str
    start: str  # HH:MM ET
    end: str


@dataclass
class FVS1Config:
    # Layer 0 — market state
    adx_min: int = 25
    balance_lookback: int = 20
    balance_atr_mult: float = 0.5
    impulse_min_bars: int = 3
    impulse_min_body_pct: float = 0.35
    impulse_max_bars: int = 16

    # Layer 1 — volume profile
    vp_bins: int = 40
    value_area_pct: float = 0.70
    lvn_percentile: float = 0.25

    # Layer 2 — absorption
    absorption_min_bars: int = 2
    absorption_vol_mult: float = 1.05
    absorption_range_mult: float = 0.75
    absorption_level_tol_pts: float = 2.0

    # Layer 3 — trigger / exits
    max_hold_seconds: int = 90
    aggression_body_pct: float = 0.55
    min_rr: float = 0.8
    sl_buffer_pts: float = 1.0

    # Fees / expectancy
    round_trip_fee: float = 1.50
    expectancy_window: int = 50
    max_consecutive_losses: int = 3
    pause_on_negative_e: bool = False

    # Live
    log_only: bool = True
    sessions: List[FVS1SessionWindow] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "FVS1Config":
        cfg = cls(
            adx_min=_env_int("FVS1_ADX_MIN", 25),
            balance_lookback=_env_int("FVS1_BALANCE_LOOKBACK", 20),
            balance_atr_mult=_env_float("FVS1_BALANCE_ATR_MULT", 0.5),
            impulse_min_bars=_env_int("FVS1_IMPULSE_MIN_BARS", 3),
            impulse_min_body_pct=_env_float("FVS1_IMPULSE_BODY_PCT", 0.35),
            impulse_max_bars=_env_int("FVS1_IMPULSE_MAX_BARS", 16),
            vp_bins=_env_int("FVS1_VP_BINS", 40),
            value_area_pct=_env_float("FVS1_VALUE_AREA_PCT", 0.70),
            lvn_percentile=_env_float("FVS1_LVN_PERCENTILE", 0.25),
            absorption_min_bars=_env_int("FVS1_ABSORPTION_MIN_BARS", 2),
            absorption_vol_mult=_env_float("FVS1_ABSORPTION_VOL_MULT", 1.05),
            absorption_range_mult=_env_float("FVS1_ABSORPTION_RANGE_MULT", 0.75),
            absorption_level_tol_pts=_env_float("FVS1_ABSORPTION_TOL_PTS", 2.0),
            max_hold_seconds=_env_int("FVS1_MAX_HOLD_SECONDS", 90),
            aggression_body_pct=_env_float("FVS1_AGGRESSION_BODY_PCT", 0.55),
            min_rr=_env_float("FVS1_MIN_RR", 0.8),
            sl_buffer_pts=_env_float("FVS1_SL_BUFFER_PTS", 1.0),
            round_trip_fee=_env_float("FVS1_ROUND_TRIP_FEE", 1.50),
            expectancy_window=_env_int("FVS1_EXPECTANCY_WINDOW", 50),
            max_consecutive_losses=_env_int("FVS1_MAX_CONSECUTIVE_LOSSES", 3),
            pause_on_negative_e=_env_bool("FVS1_PAUSE_ON_NEGATIVE_E", False),
            log_only=_env_bool("FVS1_LOG_ONLY", True),
        )
        cfg.sessions = load_fvs1_sessions()
        return cfg


def load_fvs1_sessions() -> List[FVS1SessionWindow]:
    """Parse FVS1_SESSIONS env or config/fvs1_sessions.yaml."""
    env_raw = os.getenv("FVS1_SESSIONS", "").strip()
    if env_raw:
        windows: List[FVS1SessionWindow] = []
        for part in env_raw.split(";"):
            part = part.strip()
            if not part:
                continue
            label, times = part.split("=", 1) if "=" in part else ("session", part)
            start, end = times.split("-", 1)
            windows.append(FVS1SessionWindow(label.strip(), start.strip(), end.strip()))
        if windows:
            return windows

    yaml_path = PROJECT_ROOT / "config" / "fvs1_sessions.yaml"
    if yaml_path.is_file() and yaml is not None:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out: List[FVS1SessionWindow] = []
        for w in data.get("sessions", []):
            out.append(FVS1SessionWindow(
                str(w.get("label", "session")),
                str(w["start"]),
                str(w["end"]),
            ))
        if out:
            return out

    return [
        FVS1SessionWindow("ny_open", "09:30", "10:00"),
        FVS1SessionWindow("pm", "14:00", "15:00"),
    ]


def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)


def is_fvs1_session_et(now_et, sessions: List[FVS1SessionWindow]) -> Tuple[bool, str]:
    """Return (in_session, label_or_reason)."""
    try:
        import pytz
        if now_et.tzinfo is None:
            now_et = pytz.timezone("US/Eastern").localize(now_et)
        else:
            now_et = now_et.astimezone(pytz.timezone("US/Eastern"))
    except ImportError:
        pass

    if now_et.weekday() >= 5:
        return False, "weekend"

    mins = now_et.hour * 60 + now_et.minute
    for w in sessions:
        sh, sm = parse_hhmm(w.start)
        eh, em = parse_hhmm(w.end)
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        if start_m <= mins < end_m:
            return True, w.label
    return False, "outside FVS1 session"
