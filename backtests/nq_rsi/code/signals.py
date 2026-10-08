"""5m REVAMP ensemble signals (repo code, unchanged) + sleeve tags, exactly like live_signals.classify_last_sleeve."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/bott-trader-revamp")
from src.research_ensemble.ensemble import EnsembleStrategy  # noqa: E402
from src.research_ensemble.indicators import adx  # noqa: E402

TICK = 0.25


def rtick(x):
    return np.round(np.asarray(x, dtype=float) / TICK) * TICK


def live_atr_sma14(df5: pd.DataFrame) -> pd.Series:
    """ATR used by the live trailing stop (start_live_rithmic.calculate_indicators: SMA14 of TR)."""
    h, l, c = df5["high"], df5["low"], df5["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def build_signals(df5: pd.DataFrame) -> pd.DataFrame:
    strat = EnsembleStrategy(rth_only=True, revamp_exits=True, breakout_target_r=2.0, mr_target="bb_far")
    sig = strat.generate_signals(df5)
    adx_ = adx(df5["high"], df5["low"], df5["close"], 14)
    trending = adx_ >= strat.adx_trend_min
    ranging = adx_ < strat.adx_range_max
    bo = strat._breakout.generate_signals(df5)
    bo_fires = (bo.entries != 0) & (strat.breakout_in_range | ~ranging)
    sleeve = np.where(bo_fires, "breakout", np.where(trending, "trend", np.where(ranging, "mean_reversion", "")))
    atr_live = live_atr_sma14(df5)
    out = pd.DataFrame({
        "dir": sig.entries.astype(int), "ref": df5["close"], "sl": sig.stop_price, "tp": sig.target_price,
        "sleeve": sleeve, "adx": adx_, "atr_live": atr_live,
    }, index=df5.index)
    out = out[out["dir"] != 0].copy()
    # live: trend -> trail only (no TP); breakout/MR -> fixed TP, no trail
    out.loc[out["sleeve"] == "trend", "tp"] = np.nan
    out["trailing"] = out["sleeve"] == "trend"
    out["sl"] = rtick(out["sl"])
    out["tp"] = np.where(out["tp"].notna(), rtick(out["tp"].fillna(0)), np.nan)
    out = out[np.isfinite(out["sl"])]
    out["signal_time"] = out.index + pd.Timedelta(minutes=5)  # known at 5m bar close
    return out
