"""Absorption bar detection — high effort, low result at structural level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class AbsorptionZone:
    start_idx: int
    end_idx: int
    level: float
    zone_low: float
    zone_high: float
    bar_count: int


def _bar_range(row: pd.Series) -> float:
    return float(row["high"]) - float(row["low"])


def _touches_level(row: pd.Series, level: float, tol: float) -> bool:
    lo, hi = float(row["low"]), float(row["high"])
    return lo - tol <= level <= hi + tol


def is_absorption_bar(
    row: pd.Series,
    level: float,
    avg_vol: float,
    avg_range: float,
    *,
    vol_mult: float,
    range_mult: float,
    tol_pts: float,
) -> Tuple[bool, str]:
    """Single bar absorption at level."""
    if not _touches_level(row, level, tol_pts):
        return False, "level not touched"
    vol = float(row.get("volume", 0) or 0)
    rng = _bar_range(row)
    if avg_vol <= 0 or avg_range <= 0:
        return False, "no baseline"
    if vol < avg_vol * vol_mult:
        return False, f"vol {vol:.0f} < {vol_mult}x avg {avg_vol:.0f}"
    if rng > avg_range * range_mult:
        return False, f"range {rng:.2f} > {range_mult}x avg {avg_range:.2f}"
    return True, f"absorption vol={vol:.0f} range={rng:.2f}"


def detect_absorption_sequence(
    df: pd.DataFrame,
    end_idx: int,
    level: float,
    direction: int,
    *,
    min_bars: int,
    vol_mult: float,
    range_mult: float,
    tol_pts: float,
    lookback: int = 8,
) -> Optional[AbsorptionZone]:
    """
    Find 2+ consecutive absorption bars ending at end_idx, at VAL (long) or VAH (short).
    Long: absorption at VAL with rejection wicks (close off lows).
    Short: absorption at VAH with rejection wicks (close off highs).
    """
    start_scan = max(0, end_idx - lookback + 1)
    avg_vol = float(df.iloc[start_scan:end_idx + 1]["volume"].mean() or 0)
    ranges = df.iloc[start_scan:end_idx + 1].apply(_bar_range, axis=1)
    avg_range = float(ranges.mean() or 0)

    seq: List[int] = []
    for j in range(end_idx, start_scan - 1, -1):
        row = df.iloc[j]
        ok, _ = is_absorption_bar(
            row, level, avg_vol, avg_range,
            vol_mult=vol_mult, range_mult=range_mult, tol_pts=tol_pts,
        )
        if not ok:
            break
        o, c, h, l = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
        rng = h - l if h > l else 1e-9
        if direction == 1:
            # close in upper half — sellers absorbed
            if (c - l) / rng < 0.45:
                break
        else:
            if (h - c) / rng < 0.45:
                break
        seq.append(j)

    if len(seq) < min_bars:
        return None

    start_idx = min(seq)
    end_i = max(seq)
    zone = df.iloc[start_idx:end_i + 1]
    return AbsorptionZone(
        start_idx=start_idx,
        end_idx=end_i,
        level=level,
        zone_low=float(zone["low"].min()),
        zone_high=float(zone["high"].max()),
        bar_count=len(seq),
    )
