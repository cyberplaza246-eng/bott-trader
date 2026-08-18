"""Paper-mode market data: Databento 1m refresh + local CSV. No Rithmic, no Yahoo."""
from __future__ import annotations

import importlib.util
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.strategy.mnq_15m_ema_eod import load_1m_seed_csv
from src.utils.redact import redact_secrets


def paper_rithmic_brackets_enabled(environ=None) -> bool:
    """Paper overnight may send Lucid sim orders (ticker+order, live_mode=False)."""
    env = environ if environ is not None else os.environ
    return str(env.get("PAPER_RITHMIC_BRACKETS", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def paper_uses_rithmic(environ=None) -> bool:
    """Paper skips Rithmic unless the operator explicitly wants a paper plant."""
    env = environ if environ is not None else os.environ
    return (
        str(env.get("PAPER_USE_RITHMIC", "")).strip().lower() == "true"
        or paper_rithmic_brackets_enabled(env)
    )


def default_1m_csv_path() -> str:
    return os.getenv("EMA15_1M_SEED", os.path.join("data", "MNQ_1m.csv"))


def _load_refresh_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "download_databento_latest.py"
    spec = importlib.util.spec_from_file_location("download_databento_latest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Databento download script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refresh_mnq_1m(lookback_days: int = 3) -> Dict[str, Any]:
    """Pull latest MNQ 1m into data/MNQ_1m.csv. Never prints keys."""
    mod = _load_refresh_mod()
    return mod.refresh_latest_1m(lookback_days=lookback_days)


def resample_ohlcv(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df is None or df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    minutes = max(1, int(timeframe_minutes or 1))
    if minutes <= 1:
        return df.copy().reset_index(drop=True)
    from src.strategy.mnq_15m_ema_eod import parse_mnq_1m_datetime

    d = df.copy()
    d["datetime"] = parse_mnq_1m_datetime(d["datetime"])
    d = d.dropna(subset=["datetime"])
    d = d.set_index("datetime").sort_index()
    out = (
        d.resample(f"{minutes}min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index()
    )
    if "volume" in out.columns:
        out = out[out["volume"] > 0].reset_index(drop=True)
    return out


class PaperCsvFeed:
    """CSV/Databento candle source used as the paper `broker` stand-in."""

    connected = False

    def __init__(self, csv_path: Optional[str] = None, refresh_sec: int = 60):
        self.csv_path = csv_path or default_1m_csv_path()
        self.refresh_sec = max(15, int(refresh_sec))
        self._df_1m: Optional[pd.DataFrame] = None
        self._last_refresh = 0.0
        self._last_meta: Dict[str, Any] = {}

    def initialize(self) -> None:
        try:
            self.refresh(force=True)
        except Exception as exc:
            print(f"CSV/Databento seed skipped: {redact_secrets(exc)}")
        try:
            self._reload_csv()
        except Exception as exc:
            print(f"CSV seed load skipped: {redact_secrets(exc)}")
            self._df_1m = None

    def shutdown(self) -> None:
        return

    def refresh(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and now - self._last_refresh < self.refresh_sec:
            return self._last_meta
        self._last_refresh = now
        try:
            self._last_meta = refresh_mnq_1m(lookback_days=3) or {}
        except Exception as exc:
            self._last_meta = {
                "ok": False,
                "error": redact_secrets(exc),
                "source": "databento",
                "path": self.csv_path,
            }
            print(f"MNQ 1m refresh failed: {self._last_meta['error']}")
        try:
            self._reload_csv()
        except Exception as exc:
            print(f"CSV seed load skipped: {redact_secrets(exc)}")
            self._df_1m = None
        return self._last_meta

    def _reload_csv(self) -> None:
        try:
            self._df_1m = load_1m_seed_csv(self.csv_path)
        except Exception:
            self._df_1m = None

    def _ensure_1m(self) -> Optional[pd.DataFrame]:
        self.refresh(force=False)
        if self._df_1m is None or self._df_1m.empty:
            self._reload_csv()
        return self._df_1m

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        num_candles: int = 100,
        end_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        df_1m = self._ensure_1m()
        if df_1m is None or df_1m.empty:
            return None
        out = resample_ohlcv(df_1m, timeframe_minutes)
        if out is None or out.empty:
            return None
        n = max(1, int(num_candles or 100))
        return out.tail(n).reset_index(drop=True)

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        df_1m = self._ensure_1m()
        if df_1m is None or df_1m.empty:
            return None
        last = float(df_1m.iloc[-1]["close"])
        if last <= 0:
            return None
        return {"bid": last, "ask": last, "last": last}

    def get_tick_flow(self, symbol: str):
        return None

    def last_bar_summary(self) -> str:
        df = self._df_1m
        if df is None or df.empty:
            return f"no bars at {self.csv_path}"
        row = df.iloc[-1]
        ts = pd.to_datetime(row["datetime"], utc=True)
        et = ts.tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M ET")
        age = int((datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds())
        return f"{float(row['close']):.2f} as of {et} ({len(df):,} bars, age {max(0, age)}s)"
