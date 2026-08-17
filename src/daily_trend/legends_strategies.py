"""
Four daily-bar strategies, each modeled on a specific, real trading
philosophy, tested individually before any combination is considered
(matching the "one hypothesis at a time" discipline used throughout this
project -- combining signals has made things worse every other time it was
tried here, so nothing gets combined until each piece is tested alone).

All four share the PTJ 200-SMA regime gate as an OPTIONAL hard directional
filter (toggle via use_ptj_filter, so it can be isolated as its own
variable in a comparison matrix):
    Long entries only when Close[t-1] > SMA200[t-1]
    Short entries only when Close[t-1] < SMA200[t-1]
    Position liquidated immediately if the regime flips while held
    (only enforced when use_ptj_filter=True).

1. ptj_dennis_breakout   -- N-day Donchian breakout (entry lookback is a
                             parameter, tested across a plateau of values,
                             not a single tuned number), exit on a fixed
                             10-day opposite structural extreme. Breakout
                             is detected using the CURRENT bar's own
                             high/low against the PRIOR N-day channel (no
                             lookahead -- the channel itself only uses
                             shift(1) history); fill happens at the NEXT
                             bar's close, deliberately more conservative
                             than an intrabar stop-order fill, because
                             mixing an intrabar-fill assumption with a
                             same-bar filter is exactly what produced the
                             lookahead artifact found in enhanced_breakout
                             elsewhere in this project.
2. simons_zscore_fade    -- mean-reversion within the prevailing regime:
                             Z = (Close[t-1] - EMA30[t-1]) / std30[t-1];
                             |Z| >= z_threshold triggers a fade back toward
                             the mean, only in the direction the regime
                             filter allows (when the filter is on).
3. raschke_turtle_soup   -- fade a failed N-day-high/low breakout: price
                             makes a new N-day extreme, then closes back
                             inside the prior range within a few bars ->
                             trade the reversal, target the N-day midpoint.

Each function returns a list of trade dicts (entry_time, direction, entry,
exit, exit_type, pnl) using next-session execution and realistic costs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.daily_trend.instruments import REGISTRY


def _costs(symbol: str):
    spec = REGISTRY[symbol]
    return spec.tick_size, spec.tick_value_usd, spec.contract_multiplier, spec.commission_rt


def _add_common_indicators(df: pd.DataFrame, entry_lookback: int, exit_lookback: int) -> pd.DataFrame:
    df = df.copy()
    df["sma200"] = df["close"].rolling(200).mean()
    df["ema30"] = df["close"].ewm(span=30, adjust=False).mean()
    df["std30"] = df["close"].rolling(30).std()
    df["high_entry"] = df["high"].rolling(entry_lookback).max()
    df["low_entry"] = df["low"].rolling(entry_lookback).min()
    df["high_exit"] = df["high"].rolling(exit_lookback).max()
    df["low_exit"] = df["low"].rolling(exit_lookback).min()
    # Regime known as of t-1 (prior day close vs prior day SMA200) -- avoids
    # using today's own not-yet-closed bar to gate today's entry.
    df["regime"] = np.where(df["close"].shift(1) > df["sma200"].shift(1), 1,
                    np.where(df["close"].shift(1) < df["sma200"].shift(1), -1, 0))
    return df


def _pnl(direction: int, entry: float, exit_: float, contracts: int, tick_size: float,
         tick_value: float, commission_rt: float) -> float:
    ticks = (exit_ - entry) / tick_size * direction
    gross = ticks * tick_value * contracts
    return gross - commission_rt * contracts


def ptj_dennis_breakout(df: pd.DataFrame, symbol: str, contracts: int = 1,
                         entry_lookback: int = 20, exit_lookback: int = 10,
                         use_ptj_filter: bool = True) -> list[dict]:
    """N-day Donchian breakout, optionally PTJ-regime-gated, exit on a
    fixed 10-day opposite structural extreme."""
    tick_size, tick_value, _, commission_rt = _costs(symbol)
    df = _add_common_indicators(df, entry_lookback, exit_lookback)
    trades = []
    position: Optional[dict] = None
    n = len(df)

    for i in range(200, n):
        row = df.iloc[i]
        if position is not None:
            d = position["dir"]
            if use_ptj_filter and row["regime"] != 0 and row["regime"] != d:
                trades.append({"entry_time": position["entry_time"], "direction": "long" if d == 1 else "short",
                                "entry": position["entry"], "exit": row["close"], "exit_type": "REGIME_FLIP",
                                "pnl": _pnl(d, position["entry"], row["close"], contracts, tick_size, tick_value, commission_rt)})
                position = None
            else:
                exit_level = row["low_exit"] if d == 1 else row["high_exit"]
                hit = (d == 1 and row["low"] <= exit_level) or (d == -1 and row["high"] >= exit_level)
                if hit:
                    trades.append({"entry_time": position["entry_time"], "direction": "long" if d == 1 else "short",
                                    "entry": position["entry"], "exit": exit_level, "exit_type": "STRUCTURE_EXIT",
                                    "pnl": _pnl(d, position["entry"], exit_level, contracts, tick_size, tick_value, commission_rt)})
                    position = None

        if position is None and i < n - 1:
            regime = row["regime"]
            allow_long = (regime == 1) if use_ptj_filter else True
            allow_short = (regime == -1) if use_ptj_filter else True
            prior_high = df["high_entry"].iloc[i - 1]
            prior_low = df["low_entry"].iloc[i - 1]
            if allow_long and row["high"] > prior_high:
                nxt = df.iloc[i + 1]
                position = {"dir": 1, "entry": nxt["close"], "entry_time": nxt.name}
            elif allow_short and row["low"] < prior_low:
                nxt = df.iloc[i + 1]
                position = {"dir": -1, "entry": nxt["close"], "entry_time": nxt.name}

    return trades


def simons_zscore_fade(df: pd.DataFrame, symbol: str, contracts: int = 1, z_threshold: float = 3.0,
                        use_ptj_filter: bool = True) -> list[dict]:
    """Mean-reversion fade, optionally gated to only fade dips in a bull
    regime / rallies in a bear regime (never fades against the macro trend
    when the filter is on)."""
    tick_size, tick_value, _, commission_rt = _costs(symbol)
    df = _add_common_indicators(df, entry_lookback=20, exit_lookback=10)
    df["zscore"] = (df["close"] - df["ema30"]) / df["std30"].replace(0, np.nan)
    trades = []
    position: Optional[dict] = None
    n = len(df)
    max_hold = 10

    for i in range(200, n):
        row = df.iloc[i]
        if position is not None:
            bars_held = i - position["entry_idx"]
            target = df["ema30"].iloc[i]
            d = position["dir"]
            hit_target = (d == 1 and row["high"] >= target) or (d == -1 and row["low"] <= target)
            timed_out = bars_held >= max_hold
            regime_flip = use_ptj_filter and row["regime"] != 0 and row["regime"] != d
            if hit_target or timed_out or regime_flip:
                exit_price = target if hit_target else row["close"]
                exit_type = "MEAN_TARGET" if hit_target else ("TIMEOUT" if timed_out else "REGIME_FLIP")
                trades.append({"entry_time": position["entry_time"], "direction": "long" if d == 1 else "short",
                                "entry": position["entry"], "exit": exit_price, "exit_type": exit_type,
                                "pnl": _pnl(d, position["entry"], exit_price, contracts, tick_size, tick_value, commission_rt)})
                position = None

        if position is None and i < n - 1:
            regime = row["regime"]
            z = row["zscore"]
            if pd.isna(z):
                continue
            allow_long = (regime == 1) if use_ptj_filter else True
            allow_short = (regime == -1) if use_ptj_filter else True
            if allow_long and z <= -z_threshold:
                nxt = df.iloc[i + 1]
                position = {"dir": 1, "entry": nxt["close"], "entry_time": nxt.name, "entry_idx": i + 1}
            elif allow_short and z >= z_threshold:
                nxt = df.iloc[i + 1]
                position = {"dir": -1, "entry": nxt["close"], "entry_time": nxt.name, "entry_idx": i + 1}

    return trades


def raschke_turtle_soup(df: pd.DataFrame, symbol: str, contracts: int = 1, fail_bars: int = 3,
                         entry_lookback: int = 20, use_ptj_filter: bool = False) -> list[dict]:
    """Fade a failed N-day breakout: new N-day extreme that closes back
    inside the prior range within `fail_bars` -> trade the reversal,
    target the N-day midpoint. use_ptj_filter is off by default (the whole
    point is fading trapped breakout traders, which can happen against or
    with the regime) but can be toggled on for the comparison matrix."""
    tick_size, tick_value, _, commission_rt = _costs(symbol)
    df = _add_common_indicators(df, entry_lookback, exit_lookback=10)
    df["mid_entry"] = (df["high_entry"] + df["low_entry"]) / 2
    trades = []
    position: Optional[dict] = None
    n = len(df)
    max_hold = 15
    breakout_watch: Optional[dict] = None

    for i in range(200, n):
        row = df.iloc[i]

        if position is not None:
            bars_held = i - position["entry_idx"]
            target = row["mid_entry"]
            d = position["dir"]
            hit_target = (d == 1 and row["high"] >= target) or (d == -1 and row["low"] <= target)
            timed_out = bars_held >= max_hold
            regime_flip = use_ptj_filter and row["regime"] != 0 and row["regime"] != d
            if hit_target or timed_out or regime_flip:
                exit_price = target if hit_target else row["close"]
                exit_type = "MIDPOINT_TARGET" if hit_target else ("TIMEOUT" if timed_out else "REGIME_FLIP")
                trades.append({"entry_time": position["entry_time"], "direction": "long" if d == 1 else "short",
                                "entry": position["entry"], "exit": exit_price, "exit_type": exit_type,
                                "pnl": _pnl(d, position["entry"], exit_price, contracts, tick_size, tick_value, commission_rt)})
                position = None
            continue

        prior_high = df["high_entry"].iloc[i - 1]
        prior_low = df["low_entry"].iloc[i - 1]

        if breakout_watch is None:
            if row["high"] > prior_high:
                breakout_watch = {"dir": 1, "level": prior_high, "bars_since": 0, "idx": i}
            elif row["low"] < prior_low:
                breakout_watch = {"dir": -1, "level": prior_low, "bars_since": 0, "idx": i}
        else:
            breakout_watch["bars_since"] = i - breakout_watch["idx"]
            failed = (breakout_watch["dir"] == 1 and row["close"] < breakout_watch["level"]) or \
                     (breakout_watch["dir"] == -1 and row["close"] > breakout_watch["level"])
            fade_dir = -breakout_watch["dir"] if breakout_watch else 0
            allow = True
            if use_ptj_filter and fade_dir != 0:
                allow = (row["regime"] == 0) or (row["regime"] == fade_dir)
            if failed and allow and i < n - 1:
                nxt = df.iloc[i + 1]
                position = {"dir": fade_dir, "entry": nxt["close"], "entry_time": nxt.name, "entry_idx": i + 1}
                breakout_watch = None
            elif failed:
                breakout_watch = None  # failed but blocked by regime filter -- don't take it
            elif breakout_watch["bars_since"] > fail_bars:
                breakout_watch = None

    return trades
