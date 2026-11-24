"""Live helper: research 5m ADX ensemble signals + sleeve tags for Rithmic exits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_ensemble.ensemble import EnsembleStrategy
from src.research_ensemble.indicators import adx, bollinger_bands
from src.research_ensemble import revamp_exits as _rx
from src.research_ensemble.session import rth_mask

ADX_PERIOD = 14


def dataframe_for_research(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize live Rithmic/Yahoo OHLCV into a DatetimeIndex frame for the research ensemble."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        out = out.dropna(subset=["datetime"]).set_index("datetime")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("research ensemble needs datetime column or DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"missing column {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 1.0
    else:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(1.0)
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def classify_last_sleeve(df: pd.DataFrame, strategy: EnsembleStrategy | None = None) -> dict | None:
    """
    Return the latest bar entry from the research ensemble with sleeve + SL/TP.

    Keys: direction ('long'|'short'), entry, sl, tp (None if trail-only),
          sleeve ('trend'|'mean_reversion'|'breakout'), trailing (bool), atr
    """
    strat = strategy or EnsembleStrategy()
    # Force revamp flags from env when strategy was built with defaults
    if _rx.revamp_enabled():
        strat.revamp_exits = True
        strat.breakout_target_r = _rx.revamp_breakout_target_r()
        strat.mr_target = _rx.revamp_mr_target()

    research_df = dataframe_for_research(df)
    if len(research_df) < 220:
        return None

    sig = strat.generate_signals(research_df)
    i = -1
    entry = int(sig.entries.iloc[i])
    if entry == 0:
        return None

    # Determine which sleeve won on this bar (same ADX router as EnsembleStrategy)
    adx_ = adx(research_df["high"], research_df["low"], research_df["close"], ADX_PERIOD)
    adx_i = float(adx_.iloc[i])
    trending = adx_i >= strat.adx_trend_min
    ranging = adx_i < strat.adx_range_max

    # Breakout override (same as ensemble): fires unless ranging and breakout_in_range=False
    bo = strat._breakout.generate_signals(research_df)
    bo_entry = int(bo.entries.iloc[i])
    bo_fires = bo_entry != 0 and (strat.breakout_in_range or not ranging)

    if bo_fires:
        sleeve = "breakout"
    elif trending:
        sleeve = "trend"
    elif ranging:
        sleeve = "mean_reversion"
    else:
        sleeve = "trend"  # transition band shouldn't enter, but keep a label

    close = float(research_df["close"].iloc[i])
    sl = float(sig.stop_price.iloc[i])
    tgt_raw = sig.target_price.iloc[i]
    tp = None if (tgt_raw is None or (isinstance(tgt_raw, float) and np.isnan(tgt_raw))) else float(tgt_raw)

    # Revamp breakout: fixed R target, no trail. Revamp MR: bb_far fixed, no trail.
    # Trend (and original breakout): trail-only.
    if sleeve == "breakout" and strat.revamp_exits and strat.breakout_target_r:
        trailing = False
        if tp is None:
            risk = abs(close - sl)
            tp = close + entry * float(strat.breakout_target_r) * risk
    elif sleeve == "mean_reversion":
        trailing = False
        if strat.revamp_exits and strat.mr_target == "bb_far" and tp is None:
            up, _mid, lo = bollinger_bands(research_df["close"], 20, 2.0)
            tp = float(up.iloc[i] if entry > 0 else lo.iloc[i])
    else:
        trailing = True
        tp = None

    if not np.isfinite(sl):
        return None
    if tp is not None and not np.isfinite(tp):
        tp = None

    # RTH gate already applied inside EnsembleStrategy when rth_only=True
    if strat.rth_only and not bool(rth_mask(research_df.index).iloc[i]):
        return None

    atr_series = research_df["high"] - research_df["low"]  # rough; prefer indicator atr
    from src.research_ensemble.indicators import atr as _atr
    atr_v = float(_atr(research_df["high"], research_df["low"], research_df["close"], 14).iloc[i])
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = float(atr_series.iloc[i]) if np.isfinite(atr_series.iloc[i]) else 10.0

    return {
        "direction": "long" if entry > 0 else "short",
        "entry": close,
        "sl": sl,
        "tp": tp,
        "sleeve": sleeve,
        "trailing": trailing,
        "atr": atr_v,
        "adx": adx_i,
        "ensemble_signal": "BUY" if entry > 0 else "SELL",
        "regime": "trending" if trending else ("ranging" if ranging else "transition"),
    }
