"""Locked MNQ S16 mid — HybridS15_IB (ORB20 + IB45), long-only.

Ports HybridS15_IB signal math for Lucid paper. Lock fingerprint must stay
e8ad933573b8bbf3. Does not mutate the lock JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import pytz
    ET = pytz.timezone("US/Eastern")
except ImportError:  # pragma: no cover
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

SESSION_NAME = "s16_mid"
EXPECTED_FP = "e8ad933573b8bbf3"

OR_MINUTES = 20
ORB_WINDOW = 120
IB_MINUTES = 45
IB_WINDOW = 180
TARGET_R = 0.75
RTH_OPEN = 9 * 60 + 30
RTH_END = 16 * 60
FLAT_MINUTE = 15 * 60 + 55
IB_AFTER_MINUTE = 11 * 60
DAILY_LOSS_HALT_PCT = 2.0
MAX_CONTRACTS = 10


def lock_path() -> Path:
    env = os.getenv("S16_MID_LOCK", "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "MNQ_PASS_s16_mid.json"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_locked_params(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or lock_path()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    params = dict(cfg.get("strategy", {}).get("params") or {})
    engine = dict(cfg.get("engine") or {})
    return {
        "or_minutes": int(params.get("or_minutes", OR_MINUTES)),
        "orb_window": int(params.get("orb_window", ORB_WINDOW)),
        "ib_minutes": int(params.get("ib_minutes", IB_MINUTES)),
        "ib_window": int(params.get("ib_window", IB_WINDOW)),
        "target_r": float(params.get("target_r", TARGET_R)),
        "daily_loss_halt_pct": float(engine.get("daily_loss_halt_pct", DAILY_LOSS_HALT_PCT)),
        "max_contracts": int(engine.get("max_contracts", MAX_CONTRACTS)),
        "flatten_at_rth_close": bool(engine.get("flatten_at_rth_close", True)),
        "rth_entries_only": bool(engine.get("rth_entries_only", True)),
        "class": str(cfg.get("strategy", {}).get("class") or "HybridS15_IB"),
        "name": str(cfg.get("strategy", {}).get("name") or ""),
        "lock_path": str(p),
        "lock_fp": fingerprint(p),
        "status": cfg.get("status"),
        "variant": cfg.get("variant"),
    }


def verify_lock(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or lock_path()
    if not p.is_file():
        raise FileNotFoundError(f"S16 mid lock missing: {p}")
    fp = fingerprint(p)
    if fp != EXPECTED_FP:
        raise RuntimeError(f"S16 mid lock fingerprint mismatch: {fp} != {EXPECTED_FP}")
    params = load_locked_params(p)
    if params["class"] != "HybridS15_IB":
        raise RuntimeError(f"Unexpected strategy class: {params['class']}")
    return params


def paper_size() -> int:
    raw = os.getenv("S16_MID_SIZE") or os.getenv("PAPER_SIZE") or "1"
    try:
        n = int(float(str(raw).strip()))
    except ValueError:
        n = 1
    return max(1, min(n, MAX_CONTRACTS))


def account_equity_guess() -> float:
    raw = os.getenv("S16_MID_ACCOUNT") or os.getenv("PAPER_ACCOUNT") or "50000"
    try:
        return max(1000.0, float(str(raw).strip()))
    except ValueError:
        return 50000.0


def _to_et(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET)


def et_minutes(ts) -> int:
    t = _to_et(ts)
    return int(t.hour * 60 + t.minute)


def et_date(ts) -> date:
    return _to_et(ts).date()


def _as_et(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        try:
            return ET.localize(now)  # pytz
        except Exception:
            return now.replace(tzinfo=ET)
    return now.astimezone(ET)


def is_rth_et(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    if et.weekday() >= 5:
        return False
    m = et.hour * 60 + et.minute
    return RTH_OPEN <= m < RTH_END


def should_flatten_rth(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    if et.weekday() >= 5:
        return False
    m = et.hour * 60 + et.minute
    return m >= FLAT_MINUTE


def entries_allowed_et(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    if et.weekday() >= 5:
        return False
    m = et.hour * 60 + et.minute
    return RTH_OPEN <= m < FLAT_MINUTE


def recipe_banner_lines(params: Optional[Dict[str, Any]] = None) -> List[str]:
    p = params or load_locked_params()
    return [
        "============================================",
        "  BottTrader - MNQ S16 MID PAPER (Lucid TEST)",
        f"  HybridS15_IB  ORB{p['or_minutes']}+IB{p['ib_minutes']}  target_r={p['target_r']}",
        f"  long-only | RTH entries | flatten ~{FLAT_MINUTE // 60}:{FLAT_MINUTE % 60:02d} ET",
        f"  lock_fp={p.get('lock_fp', EXPECTED_FP)}  size={paper_size()}",
        "  PAPER / TEST only — not funded live",
        "  Close Lucid desktop / one login only",
        "============================================",
    ]


def ensure_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "datetime" not in out.columns:
                out = out.rename(columns={out.columns[0]: "datetime"})
        else:
            raise ValueError("1m frame needs datetime column or DatetimeIndex")
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"missing {col}")
        out[col] = out[col].astype(float)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    else:
        out["volume"] = out["volume"].astype(float).fillna(0.0)
    return out.reset_index(drop=True)


@dataclass
class DayState:
    day: Optional[date] = None
    orb_taken: bool = False
    ib_taken: bool = False
    last_signal_bar: Optional[pd.Timestamp] = None
    day_pnl: float = 0.0
    halted: bool = False


def _range_of(day_bars: pd.DataFrame, start_m: int, end_m: int) -> Optional[Tuple[float, float]]:
    if day_bars.empty:
        return None
    mins = day_bars["datetime"].map(et_minutes).to_numpy()
    mask = (mins >= start_m) & (mins < end_m)
    chunk = day_bars.loc[mask]
    if chunk.empty:
        return None
    hi = float(chunk["high"].max())
    lo = float(chunk["low"].min())
    if hi <= lo:
        return None
    return hi, lo


def _day_bars(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df.empty:
        return df
    dates = df["datetime"].map(et_date)
    return df.loc[dates == day].reset_index(drop=True)


def generate_hybrid_signals_for_day(
    day_bars: pd.DataFrame,
    *,
    or_minutes: int = OR_MINUTES,
    orb_window: int = ORB_WINDOW,
    ib_minutes: int = IB_MINUTES,
    ib_window: int = IB_WINDOW,
    target_r: float = TARGET_R,
) -> pd.DataFrame:
    """Causal HybridS15_IB for one RTH day (long-only)."""
    cols = ["datetime", "entry", "stop", "target", "leg"]
    if day_bars.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    mins = day_bars["datetime"].map(et_minutes).to_numpy()
    close = day_bars["close"].to_numpy()

    or_hi_lo = _range_of(day_bars, RTH_OPEN, RTH_OPEN + or_minutes)
    if or_hi_lo is not None:
        or_high, or_low = or_hi_lo
        risk = or_high - or_low
        entry_lo = RTH_OPEN + or_minutes
        entry_hi = entry_lo + orb_window
        for i in range(len(day_bars)):
            if not (entry_lo <= mins[i] < entry_hi):
                continue
            if close[i] > or_high:
                rows.append(
                    {
                        "datetime": day_bars["datetime"].iloc[i],
                        "entry": float(close[i]),
                        "stop": float(or_low),
                        "target": float(or_high + target_r * risk),
                        "leg": "orb",
                    }
                )
                break

    ib_hi_lo = _range_of(day_bars, RTH_OPEN, RTH_OPEN + ib_minutes)
    if ib_hi_lo is not None:
        ib_high, ib_low = ib_hi_lo
        risk = ib_high - ib_low
        entry_lo = RTH_OPEN + ib_minutes
        entry_hi = entry_lo + ib_window
        armed_long = True
        seen_above = False
        for i in range(len(day_bars)):
            m = int(mins[i])
            if m < IB_AFTER_MINUTE:
                continue
            if not (entry_lo <= m < entry_hi):
                continue
            if close[i] > ib_high:
                seen_above = True
            if seen_above and close[i] <= ib_high:
                armed_long = True
            if not seen_above:
                armed_long = True
            if close[i] > ib_high and armed_long:
                rows.append(
                    {
                        "datetime": day_bars["datetime"].iloc[i],
                        "entry": float(close[i]),
                        "stop": float(ib_low),
                        "target": float(ib_high + target_r * risk),
                        "leg": "ib",
                    }
                )
                break
            if close[i] > ib_high:
                armed_long = False
                seen_above = True

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def check_entry(
    df: pd.DataFrame,
    state: DayState,
    *,
    open_count: int = 0,
    now: Optional[datetime] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], DayState]:
    now = now or datetime.now(ET)
    p = params or load_locked_params()
    if state.halted or open_count > 0 or not entries_allowed_et(now):
        return None, state

    frame = ensure_datetime_frame(df)
    if frame.empty:
        return None, state

    today = _as_et(now).date()
    if state.day != today:
        state = DayState(day=today)

    day = _day_bars(frame, today)
    if day.empty:
        return None, state

    last_ts = _to_et(day["datetime"].iloc[-1])
    now_et = _as_et(now)
    if last_ts.floor("min") >= now_et.replace(second=0, microsecond=0):
        if len(day) < 2:
            return None, state
        day = day.iloc[:-1].reset_index(drop=True)
        if day.empty:
            return None, state

    sigs = generate_hybrid_signals_for_day(
        day,
        or_minutes=int(p["or_minutes"]),
        orb_window=int(p["orb_window"]),
        ib_minutes=int(p["ib_minutes"]),
        ib_window=int(p["ib_window"]),
        target_r=float(p["target_r"]),
    )
    if sigs.empty:
        return None, state

    last_bar = day["datetime"].iloc[-1]
    for _, row in sigs.iterrows():
        if pd.Timestamp(row["datetime"]) != pd.Timestamp(last_bar):
            continue
        leg = str(row["leg"])
        if leg == "orb" and state.orb_taken:
            continue
        if leg == "ib" and state.ib_taken:
            continue
        entry = float(row["entry"])
        sl = float(row["stop"])
        tp = float(row["target"])
        if tp <= entry or sl >= entry:
            continue
        sig = {
            "direction": "long",
            "leg": leg,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_pts": entry - sl,
            "tp_pts": tp - entry,
            "size": paper_size(),
            "session": SESSION_NAME,
            "signal_bar": str(last_bar),
            "strategy": "HybridS15_IB",
        }
        # Do not mark orb/ib taken here — runner marks after a successful Lucid fill.
        return sig, state

    return None, state


def check_bar_exit(
    pos: Dict[str, Any],
    bar_hi: float,
    bar_lo: float,
    bar_cl: float,
    *,
    flat_eod: bool = False,
) -> Optional[Tuple[str, float]]:
    if flat_eod:
        return "EOD", float(bar_cl)
    sl = float(pos["sl"])
    tp = float(pos["tp"])
    if bar_lo <= sl:
        return "SL", sl
    if bar_hi >= tp:
        return "TP", tp
    return None


def record_day_pnl(state: DayState, pnl: float, account: float, halt_pct: float) -> DayState:
    state.day_pnl += float(pnl)
    if account > 0 and state.day_pnl <= -abs(halt_pct) / 100.0 * account:
        state.halted = True
    return state


def explain_skip(
    state: DayState,
    *,
    open_count: int,
    now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now(ET)
    if state.halted:
        return f"daily loss halt (pnl={state.day_pnl:.2f})"
    if open_count:
        return "in trade (no scale-in)"
    if not entries_allowed_et(now):
        if should_flatten_rth(now):
            return "flatten / past entry window"
        if not is_rth_et(now):
            return "outside RTH"
        return "waiting RTH"
    if state.orb_taken and state.ib_taken:
        return "day legs done"
    parts = []
    if state.orb_taken:
        parts.append("ORB done")
    if state.ib_taken:
        parts.append("IB done")
    return "scanning " + ("+".join(parts) if parts else "ORB/IB")
