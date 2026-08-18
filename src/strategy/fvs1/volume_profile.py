"""Impulse leg detection and volume profile (POC / VAL / VAH / LVN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ImpulseLeg:
    start_idx: int
    end_idx: int
    direction: int  # 1 long, -1 short
    low: float
    high: float


@dataclass
class VolumeProfile:
    poc: float
    val: float
    vah: float
    lvn_prices: List[float]
    total_volume: float
    bin_centers: np.ndarray
    bin_volumes: np.ndarray


def _body_pct(row: pd.Series) -> float:
    rng = float(row["high"]) - float(row["low"])
    if rng <= 0:
        return 0.0
    return abs(float(row["close"]) - float(row["open"])) / rng


def detect_out_of_balance(
    df: pd.DataFrame,
    idx: int,
    *,
    lookback: int,
    atr_mult: float,
) -> Tuple[bool, str]:
    """Price displaced outside recent balance range."""
    if idx < lookback + 1:
        return False, "insufficient history"
    window = df.iloc[idx - lookback : idx]
    bal_high = float(window["high"].max())
    bal_low = float(window["low"].min())
    close = float(df.iloc[idx]["close"])
    atr = float(df.iloc[idx].get("atr", 0) or 0)
    if pd.isna(atr) or atr <= 0:
        atr = float(window["high"].max() - window["low"].min()) / lookback
    threshold = atr * atr_mult
    if close > bal_high + threshold:
        return True, f"above balance {bal_high:.2f} by {close - bal_high:.2f}pts"
    if close < bal_low - threshold:
        return True, f"below balance {bal_low:.2f} by {bal_low - close:.2f}pts"
    return False, f"inside balance [{bal_low:.2f}, {bal_high:.2f}]"


def detect_impulse_leg(
    df: pd.DataFrame,
    idx: int,
    direction: int,
    *,
    min_bars: int,
    min_body_pct: float,
    max_scan: int = 12,
) -> Optional[ImpulseLeg]:
    """Walk backward from idx to find contiguous impulse leg ending at idx."""
    if idx < min_bars:
        return None
    end = idx
    start = end
    for j in range(end, max(end - max_scan, -1), -1):
        row = df.iloc[j]
        body = _body_pct(row)
        o, c = float(row["open"]), float(row["close"])
        bar_dir = 1 if c >= o else -1
        if bar_dir != direction or body < min_body_pct:
            break
        start = j
    if end - start + 1 < min_bars:
        return None
    leg = df.iloc[start : end + 1]
    return ImpulseLeg(
        start_idx=start,
        end_idx=end,
        direction=direction,
        low=float(leg["low"].min()),
        high=float(leg["high"].max()),
    )


def build_volume_profile(
    df: pd.DataFrame,
    leg: ImpulseLeg,
    *,
    n_bins: int,
    value_area_pct: float,
    lvn_percentile: float,
) -> Optional[VolumeProfile]:
    """Tick-weighted VP: distribute each bar's volume uniformly across H-L into bins."""
    segment = df.iloc[leg.start_idx : leg.end_idx + 1]
    if segment.empty:
        return None
    lo, hi = leg.low, leg.high
    if hi <= lo:
        return None

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_vol = np.zeros(n_bins, dtype=float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for _, row in segment.iterrows():
        h, l = float(row["high"]), float(row["low"])
        vol = float(row.get("volume", 0) or 0)
        if vol <= 0:
            continue
        bar_lo = max(l, lo)
        bar_hi = min(h, hi)
        if bar_hi <= bar_lo:
            # pin bar — assign to nearest bin
            mid = (float(row["open"]) + float(row["close"])) / 2.0
            bi = int(np.clip(np.searchsorted(bin_edges, mid) - 1, 0, n_bins - 1))
            bin_vol[bi] += vol
            continue
        # overlap bins
        for b in range(n_bins):
            b_lo, b_hi = bin_edges[b], bin_edges[b + 1]
            overlap = max(0.0, min(bar_hi, b_hi) - max(bar_lo, b_lo))
            span = bar_hi - bar_lo
            if span > 0 and overlap > 0:
                bin_vol[b] += vol * (overlap / span)

    total = float(bin_vol.sum())
    if total <= 0:
        return None

    poc_idx = int(np.argmax(bin_vol))
    poc = float(bin_centers[poc_idx])

    # Expand from POC until value_area_pct of volume captured
    target = total * value_area_pct
    captured = bin_vol[poc_idx]
    lo_i = hi_i = poc_idx
    while captured < target and (lo_i > 0 or hi_i < n_bins - 1):
        vol_below = bin_vol[lo_i - 1] if lo_i > 0 else -1.0
        vol_above = bin_vol[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if vol_above >= vol_below:
            hi_i += 1
            captured += bin_vol[hi_i]
        else:
            lo_i -= 1
            captured += bin_vol[lo_i]

    val = float(bin_edges[lo_i])
    vah = float(bin_edges[hi_i + 1])

    # LVN: bins below lvn_percentile of max bin volume (excluding POC neighborhood)
    max_v = float(bin_vol.max())
    cutoff = max_v * lvn_percentile
    lvn_prices = [
        float(bin_centers[i])
        for i in range(n_bins)
        if i != poc_idx and bin_vol[i] <= cutoff and bin_vol[i] > 0
    ]

    return VolumeProfile(
        poc=poc,
        val=val,
        vah=vah,
        lvn_prices=lvn_prices,
        total_volume=total,
        bin_centers=bin_centers,
        bin_volumes=bin_vol,
    )


def find_recent_impulse_leg(
    df: pd.DataFrame,
    end_idx: int,
    direction: int,
    *,
    min_bars: int,
    min_body_pct: float,
    max_scan: int,
) -> Optional[ImpulseLeg]:
    """Impulse leg may have completed before absorption/pullback bars."""
    low = max(min_bars - 1, end_idx - max_scan)
    for end in range(end_idx, low - 1, -1):
        leg = detect_impulse_leg(
            df, end, direction,
            min_bars=min_bars, min_body_pct=min_body_pct, max_scan=max_scan,
        )
        if leg is not None:
            return leg
    return None


def price_at_value_edge(
    close: float,
    profile: VolumeProfile,
    direction: int,
    tol_pts: float,
) -> Tuple[bool, float, str]:
    """Pullback to VAL (long) or VAH (short) after impulse."""
    if direction == 1:
        level = profile.val
        if close <= profile.vah and close >= level - tol_pts:
            return True, level, "at/below VAL pullback"
        return False, level, f"close {close:.2f} not at VAL {level:.2f}"
    level = profile.vah
    if close >= profile.val and close <= level + tol_pts:
        return True, level, "at/above VAH pullback"
    return False, level, f"close {close:.2f} not at VAH {level:.2f}"


def nearest_lvn(level: float, lvn_prices: List[float], direction: int) -> Optional[float]:
    """LVN between entry level and POC for structural context."""
    if not lvn_prices:
        return None
    if direction == 1:
        candidates = [p for p in lvn_prices if p <= level]
        return max(candidates) if candidates else None
    candidates = [p for p in lvn_prices if p >= level]
    return min(candidates) if candidates else None
