"""Clara-style day bias gate for revamp daytime entries (research, 2026-10-27).

Bias = side of the first 5m close after 09:50 ET (and before 11:00 ET) that is beyond the
09:30-09:50 opening range AND on the same side of session VWAP. Opening range wider than
CLARA_BIAS_WMAX points (default 250) -> no bias. While a bias exists (until 16:00 ET),
day-session entries opposite to it are blocked. Overnight entries are never touched.
Env: REVAMP_CLARA_BIAS=1 to enable (default OFF).
"""
import os
from datetime import time as dtime
from typing import Optional
import pandas as pd

ET = "America/New_York"


def enabled() -> bool:
    return str(os.getenv("REVAMP_CLARA_BIAS", "0")).strip().lower() in ("1", "true", "yes", "on")


def _et_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index if not "timestamp" in df.columns else pd.to_datetime(df["timestamp"]))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ET)


def day_bias(df: pd.DataFrame, now_et: Optional[pd.Timestamp] = None, bar_minutes: int = 5) -> int:
    """+1 long bias, -1 short bias, 0 none. Uses only bars that have CLOSED by now_et."""
    if df is None or len(df) == 0:
        return 0
    idx = _et_index(df)
    d = df.copy()
    d.index = idx
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ET)
    day = now_et.normalize()
    g = d[(d.index >= day + pd.Timedelta(hours=9, minutes=30)) & (d.index < day + pd.Timedelta(hours=16))]
    g = g[g.index + pd.Timedelta(minutes=bar_minutes) <= now_et]  # closed bars only
    o = g[g.index < day + pd.Timedelta(hours=9, minutes=50)]
    if len(o) == 0 or len(g) == len(o):
        return 0
    if now_et >= day + pd.Timedelta(hours=16):
        return 0
    hi, lo = float(o["high"].max()), float(o["low"].min())
    if hi - lo > float(os.getenv("CLARA_BIAS_WMAX", "250")):
        return 0
    vol = g["volume"].astype(float).clip(lower=1e-9) if "volume" in g else pd.Series(1.0, index=g.index)
    tp = (g["high"] + g["low"] + g["close"]) / 3
    vw = (tp * vol).cumsum() / vol.cumsum()
    rest = g[g.index >= day + pd.Timedelta(hours=9, minutes=50)]
    cut = day + pd.Timedelta(hours=float(os.getenv("CLARA_BIAS_CUT_H", "11")))
    for ts, row in rest.iterrows():
        if ts >= cut:
            break
        c = float(row["close"])
        if c > hi and c > vw.loc[ts]:
            return 1
        if c < lo and c < vw.loc[ts]:
            return -1
    return 0


def block_reason(df: pd.DataFrame, direction: str, now_et: Optional[pd.Timestamp] = None, bar_minutes: int = 5) -> Optional[str]:
    if not enabled():
        return None
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ET)
    t = now_et.time()
    if not (dtime(9, 30) <= t < dtime(16, 0)) or now_et.weekday() >= 5:
        return None
    b = day_bias(df, now_et, bar_minutes)
    s = 1 if str(direction).upper() in ("LONG", "BUY", "1") else -1
    if b != 0 and s != b:
        return f"clara bias {'LONG' if b > 0 else 'SHORT'} blocks {direction}"
    return None
