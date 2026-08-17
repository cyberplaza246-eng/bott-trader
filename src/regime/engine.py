"""
Regime SCORE engine — deliberately not called "probabilities." The softmax
output sums to 1 and looks calibrated, but it isn't: each component is a
measurement of the current bar relative to its own recent history (a
z-score or a rolling fraction), not a forecast validated against actual
future outcomes. Treat trend_score=0.45 as "trend-like conditions are
somewhat more present than range/breakout-like conditions right now," not
as "45% chance of a trend." Calibration against realized outcomes is future
work, not something this module claims to have done.

It's also still built entirely from OHLCV+volume (no order flow or breadth
data), so it's a continuous, less-lagging alternative to a hard threshold
like `ADX >= 25` — not genuine early detection. It cannot see anything the
market itself hasn't already priced in.

Four heuristic component signals, each normalized to roughly [0, 1]:

  volatility_expansion  — current ATR vs its own recent history (z-score,
                           sigmoid-squashed). High = a real move is already
                           underway (supports TREND).
  compression            — the inverse: ATR currently low relative to its
                           recent history. High = the market has gone quiet,
                           a classic precursor to a range breaking
                           (supports BREAKOUT).
  vwap_acceptance         — fraction of the last N bars spent meaningfully
                           away from VWAP (> 1 ATR). High = price is
                           "accepting" a move away from value instead of
                           snapping back (supports TREND); low = price keeps
                           reverting to VWAP (supports RANGE).
  failed_reversal_rate    — fraction of recent local-extreme bars where a
                           push against the immediate move got rejected
                           (new N-bar high/low followed by a close back the
                           other way). High = counter-moves keep failing
                           (supports TREND continuing).
  volume_participation    — relative volume (current vs rolling average),
                           sigmoid-squashed around 1.0x.

These combine into three raw regime scores, then a per-bar softmax turns
them into a distribution that sums to 1: trend_score, range_score,
breakout_score. A fourth column, transition_score, measures how fast the
leading score has been changing over the last 6 bars (30 min on 5m data) —
high when the market's apparent regime is actively shifting rather than
settled, used downstream to size down rather than commit fully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.indicators import atr, session_vwap

ATR_ZSCORE_WINDOW = 500
VWAP_DISTANCE_ATR_MULT = 1.0
ACCEPTANCE_WINDOW = 60
SWING_LOOKBACK = 20
FAILED_REVERSAL_WINDOW = 60
VOLUME_LOOKBACK = 20
TRANSITION_LOOKBACK_BARS = 6


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x.clip(-20, 20)))


def compute_regime_scores(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    atr_ = atr(high, low, close, 14)
    atr_z = _zscore(atr_, ATR_ZSCORE_WINDOW)
    volatility_expansion = _sigmoid(atr_z).fillna(0.5)
    compression = _sigmoid(-atr_z).fillna(0.5)

    vwap = session_vwap(high, low, close, volume)
    vwap_distance = (close - vwap).abs() / atr_.replace(0, np.nan)
    away_from_vwap = (vwap_distance > VWAP_DISTANCE_ATR_MULT).astype(float)
    vwap_acceptance = away_from_vwap.rolling(ACCEPTANCE_WINDOW).mean().fillna(0.5)

    new_high = high == high.rolling(SWING_LOOKBACK).max()
    new_low = low == low.rolling(SWING_LOOKBACK).min()
    failed_breakout = new_high & (close < close.shift(1))
    failed_breakdown = new_low & (close > close.shift(1))
    failed_reversal_rate = (
        (failed_breakout | failed_breakdown).rolling(FAILED_REVERSAL_WINDOW).mean().fillna(0.0)
    )

    relative_volume = volume / volume.rolling(VOLUME_LOOKBACK).mean().replace(0, np.nan)
    volume_participation = _sigmoid(relative_volume.fillna(1.0) - 1.0)

    trend_raw = (
        0.25 * volatility_expansion
        + 0.25 * vwap_acceptance
        + 0.25 * failed_reversal_rate
        + 0.25 * volume_participation
    )
    range_raw = 0.5 * (1 - vwap_acceptance) + 0.5 * (1 - volatility_expansion)
    breakout_raw = 0.6 * compression + 0.4 * volume_participation

    raw = np.column_stack([trend_raw.to_numpy(), range_raw.to_numpy(), breakout_raw.to_numpy()])
    raw = raw - raw.max(axis=1, keepdims=True)  # numerical stability
    exp_raw = np.exp(raw)
    scores = exp_raw / exp_raw.sum(axis=1, keepdims=True)

    leading_score = scores.max(axis=1)
    leading_score_series = pd.Series(leading_score, index=df.index)
    transition_score = leading_score_series.diff(TRANSITION_LOOKBACK_BARS).abs().fillna(0.0)

    return pd.DataFrame(
        {
            "trend_score": scores[:, 0],
            "range_score": scores[:, 1],
            "breakout_score": scores[:, 2],
            "transition_score": transition_score,
        },
        index=df.index,
    )
