"""Read-only MNQ 1m bar export for NEAT DEMO watch.

Enable with NEAT_BAR_EXPORT=1 (optional NEAT_BAR_EXPORT_PATH).
Never places orders; failures are swallowed by the caller.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_PATH = Path(r"C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv")

_last_fetch_mono: float = 0.0
_last_written_ts: dict[str, str] = {}
_header_ok: dict[str, bool] = {}


def enabled() -> bool:
    return os.getenv("NEAT_BAR_EXPORT", "").strip().lower() in ("1", "true", "yes", "on")


def export_path() -> Path:
    raw = os.getenv("NEAT_BAR_EXPORT_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_PATH


def _ts_key(val: Any) -> str:
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def maybe_export_1m(broker: Any, symbol: str, *, min_interval_s: float = 20.0) -> int:
    """Fetch recent 1m bars and append new closed rows. Returns rows written."""
    global _last_fetch_mono
    if not enabled() or broker is None:
        return 0
    sym = str(symbol or "").upper()
    if not sym.startswith("MNQ"):
        return 0
    now = time.monotonic()
    if now - _last_fetch_mono < min_interval_s:
        return 0
    _last_fetch_mono = now

    df = broker.get_candles(sym, timeframe_minutes=1, num_candles=30)
    if df is None or len(df) < 2:
        return 0

    # Drop the newest bar — may still be forming.
    closed = df.iloc[:-1].copy()
    path = export_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    last = _last_written_ts.get(sym)
    if last is None and path.is_file() and path.stat().st_size > 0:
        try:
            tip = pd.read_csv(path, usecols=["datetime"]).iloc[-1]["datetime"]
            last = _ts_key(tip)
            _last_written_ts[sym] = last
            _header_ok[str(path)] = True
        except Exception:
            last = None

    rows_out = []
    for _, row in closed.iterrows():
        key = _ts_key(row["datetime"] if "datetime" in closed.columns else row.get("time"))
        if last is not None and key <= last:
            continue
        rows_out.append(
            {
                "datetime": key,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0) or 0),
            }
        )
        last = key

    if not rows_out:
        return 0

    write_header = not path.is_file() or path.stat().st_size == 0 or not _header_ok.get(str(path), False)
    out = pd.DataFrame(rows_out)
    out.to_csv(path, mode="a", header=write_header, index=False)
    _header_ok[str(path)] = True
    _last_written_ts[sym] = last or _last_written_ts.get(sym, "")
    return len(rows_out)
