"""ATR-aware stop/target sizing for scalp / hybrid entries."""

from __future__ import annotations

from typing import Tuple, Union

import pandas as pd

Number = Union[int, float]


def compute_scalp_bracket_pts(
    atr_1m: Number,
    *,
    sl_pts: float = 8.0,
    tp_pts: float = 14.0,
    sl_min: float = 6.0,
    sl_max: float = 12.0,
    tp_min: float = 10.0,
    tp_max: float = 20.0,
    min_rr: float = 1.4,
    use_atr_bounds: bool = True,
) -> Tuple[float, float, float, float]:
    """Return (sl_pts, tp_pts, atr_used, rr_ratio) for a scalp bracket."""
    atr = 0.0
    try:
        v = float(atr_1m)
        if not pd.isna(v) and v > 0:
            atr = v
    except (TypeError, ValueError):
        pass

    if use_atr_bounds and atr > 0:
        sl_floor = max(sl_min, 0.4 * atr)
        sl_cap = min(sl_max, 1.0 * atr)
        if sl_floor > sl_cap:
            sl = min(sl_floor, sl_max)
        else:
            sl = max(sl_floor, min(sl_pts, sl_cap))
    else:
        sl = max(sl_min, min(sl_pts, sl_max))

    tp_floor = max(tp_min, sl * min_rr)
    tp = max(tp_floor, min(tp_pts, tp_max))
    rr = tp / sl if sl > 0 else 0.0
    return sl, tp, atr, rr


def format_scalp_bracket_log(sl_pts: float, tp_pts: float, atr: float, rr: float) -> str:
    """One-line bracket summary for entry logs."""
    return f"Bracket: SL={sl_pts:.0f}pt TP={tp_pts:.0f}pt (ATR={atr:.1f}, R:R={rr:.1f})"
