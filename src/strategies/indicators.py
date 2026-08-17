"""Minimal pandas/numpy technical indicators — no extra dependencies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    return upper, lower


def session_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, session_reset_hour_utc: int = 23
) -> pd.Series:
    """VWAP that resets each CME trading session (23:00 UTC, matching
    InstrumentSpec.session_start) rather than at UTC calendar midnight."""
    session_key = (close.index - pd.Timedelta(hours=session_reset_hour_utc)).date
    typical_price = (high + low + close) / 3
    cum_pv = (typical_price * volume).groupby(session_key).cumsum()
    cum_vol = volume.groupby(session_key).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)

    di_sum = plus_di + minus_di
    # Guard against dividing near-zero by near-zero during flat/no-trend
    # periods, which otherwise produces unstable DX spikes from float noise.
    dx = pd.Series(
        np.where(di_sum > 1e-6, 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan), 0.0),
        index=high.index,
    )
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)
