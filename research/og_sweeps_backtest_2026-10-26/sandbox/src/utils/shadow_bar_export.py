"""Append closed MNQ 5-minute bars for the NEAT shadow paper trader.

SHADOW_BAR_EXPORT defaults to on (set 0/off to disable). Writes data/live_bars_MNQ_5m.csv
(SHADOW_BAR_EXPORT_PATH overrides). Never places orders. Caller must swallow exceptions.

Newest row: Rithmic history frames are stamped by bar END and hold only finished bars
(rithmic_connector tags them df.attrs["bar_time"] == "end"). That newest bar is written as
soon as its end time has passed. Any other frame (Yahoo fallback, local-history merge,
untagged) may end in a still-forming bar, so its newest row is dropped as before.
Rows are never rewritten, so a forming bar must never get in.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

_last_written: dict[str, str] = {}


def export_path() -> Path:
    raw = os.getenv("SHADOW_BAR_EXPORT_PATH", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "data" / "live_bars_MNQ_5m.csv"


def _utc(val) -> pd.Timestamp:
    ts = pd.Timestamp(val)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def newest_is_finished(df, time_col: str, now=None) -> bool:
    """True only for an end-stamped Rithmic frame whose newest bar end is not in the future."""
    try:
        if getattr(df, "attrs", {}).get("bar_time") != "end":
            return False
        now_ts = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        return _utc(df[time_col].iloc[-1]) <= now_ts
    except Exception:
        return False


def export_closed_bars(df, symbol: str = "MNQ", path: Optional[Path] = None, now=None) -> int:
    if os.getenv("SHADOW_BAR_EXPORT", "1").strip().lower() in ("0", "false", "no", "off"):
        return 0
    if df is None or len(df) < 2:
        return 0
    if not str(symbol).upper().startswith("MNQ"):
        return 0
    path = Path(path) if path is not None else export_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    time_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        return 0
    closed = df if newest_is_finished(df, time_col, now) else df.iloc[:-1]
    last = _last_written.get(symbol)
    if last is None and path.is_file() and path.stat().st_size > 0:
        try:
            _t = pd.Timestamp(str(pd.read_csv(path, usecols=["ts_et"]).iloc[-1]["ts_et"]))
            if _t.tzinfo is None:
                _t = _t.tz_localize("America/New_York")
            last = _t.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
            _last_written[symbol] = last
        except Exception:
            last = None
    rows = []
    for _, row in closed.iterrows():
        ts = pd.Timestamp(row[time_col])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        et = ts.tz_convert("America/New_York")
        key = et.strftime("%Y-%m-%d %H:%M")
        if last is not None and key <= last:
            continue
        rows.append(dict(ts_et=et.isoformat(), open=float(row["open"]), high=float(row["high"]),
                         low=float(row["low"]), close=float(row["close"]),
                         volume=float(row["volume"]) if "volume" in closed.columns and pd.notna(row.get("volume", None)) else 0.0))
        last = key
    if not rows:
        return 0
    out = pd.DataFrame(rows)
    out.to_csv(path, mode="a", header=not path.is_file() or path.stat().st_size == 0, index=False)
    _last_written[symbol] = last
    return len(rows)
