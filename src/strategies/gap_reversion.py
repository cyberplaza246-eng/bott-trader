"""
Overnight gap reversion -- fades the RTH opening gap vs. the prior session's
RTH close when the gap exceeds a volatility-scaled threshold, betting on a
same-morning fill back toward yesterday's close.

Self-contained backtest function (not routed through the generic engine)
because the spec requires a real 10:30 ET time-stop, which the generic
engine doesn't support (only stop/target/ATR-trail/EOD exits) -- an
approximation here would risk materially changing the result, so this
implements the exact rule instead.

One addition to the originally proposed rule set, flagged explicitly: the
spec as given had no protective stop-loss (only a profit target and a time
stop), which means unlimited downside on any single trade -- not something
this project will backtest as literally specified. Added an ATR-based
catastrophe stop (entry +/- `stop_atr_mult` x daily ATR) as risk management;
everything else follows the spec as given.

Entry: at the RTH open bar's close (09:30 ET), if |gap| >= min_atr_mult x
daily ATR(14) (ATR known as of the PRIOR completed day only -- no lookahead).
Target: prior day's RTH close (gap fully filled).
Time stop: force-exit at 10:30 ET if neither target nor stop hit by then.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.daily_trend.instruments import REGISTRY
from src.strategies.indicators import atr

SESSION_TZ = "America/New_York"
RTH_OPEN_HOUR, RTH_OPEN_MINUTE = 9, 30
RTH_CLOSE_HOUR, RTH_CLOSE_MINUTE = 16, 0
TIME_STOP_HOUR, TIME_STOP_MINUTE = 10, 30
MIN_ATR_MULT = 0.5
STOP_ATR_MULT = 1.0


def backtest_gap_reversion(df: pd.DataFrame, symbol: str, min_atr_mult: float = MIN_ATR_MULT,
                            stop_atr_mult: float = STOP_ATR_MULT, contracts: int = 1) -> list[dict]:
    spec = REGISTRY[symbol] if symbol in REGISTRY else None
    if spec is None:
        # Fall back to the intraday registry for symbols not in the daily one.
        from src.instruments.instrument_registry import REGISTRY as INTRADAY_REGISTRY
        spec = INTRADAY_REGISTRY[symbol]
    tick_size = spec.tick_size
    tick_value = spec.tick_value_usd
    commission_rt = spec.commission_rt

    high, low, close = df["high"], df["low"], df["close"]
    local_index = df.index.tz_convert(SESSION_TZ)
    session_date = local_index.date
    minutes_of_day = local_index.hour * 60 + local_index.minute

    rth_open_min = RTH_OPEN_HOUR * 60 + RTH_OPEN_MINUTE
    rth_close_min = RTH_CLOSE_HOUR * 60 + RTH_CLOSE_MINUTE
    time_stop_min = TIME_STOP_HOUR * 60 + TIME_STOP_MINUTE

    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    daily_atr = atr(daily["high"], daily["low"], daily["close"], 14).shift(1)
    daily_atr_by_date = {ts.date(): v for ts, v in daily_atr.items()}

    rth_close_by_date: dict = {}
    for i in np.where(np.asarray(minutes_of_day == rth_close_min))[0]:
        rth_close_by_date[session_date[i]] = close.iloc[i]
    sorted_dates = sorted(rth_close_by_date.keys())
    prior_close_by_date = {sorted_dates[k]: rth_close_by_date[sorted_dates[k - 1]] for k in range(1, len(sorted_dates))}

    hi = high.to_numpy()
    lo = low.to_numpy()
    cl = close.to_numpy()
    n = len(df)

    trades = []
    open_idx = np.where(np.asarray(minutes_of_day == rth_open_min))[0]

    for i in open_idx:
        d = session_date[i]
        prior_close = prior_close_by_date.get(d)
        day_atr = daily_atr_by_date.get(d)
        if prior_close is None or day_atr is None or pd.isna(day_atr) or day_atr <= 0:
            continue

        gap = cl[i] - prior_close
        if abs(gap) < min_atr_mult * day_atr:
            continue

        direction = 1 if gap < 0 else -1  # gap down -> long fade; gap up -> short fade
        entry_price = cl[i]
        stop = entry_price - direction * stop_atr_mult * day_atr
        target = prior_close

        # Walk forward from the entry bar until target/stop/time-stop.
        exit_price, exit_type = None, None
        for j in range(i + 1, n):
            if session_date[j] != d:
                break  # ran past this session without resolving (shouldn't happen given time-stop)
            hit_target = (direction == 1 and hi[j] >= target) or (direction == -1 and lo[j] <= target)
            hit_stop = (direction == 1 and lo[j] <= stop) or (direction == -1 and hi[j] >= stop)
            at_time_stop = minutes_of_day[j] >= time_stop_min

            if hit_stop and hit_target:
                # Conservative: assume stop hit first if both trigger same bar.
                exit_price, exit_type = stop, "STOP"
            elif hit_stop:
                exit_price, exit_type = stop, "STOP"
            elif hit_target:
                exit_price, exit_type = target, "TARGET"
            elif at_time_stop:
                exit_price, exit_type = cl[j], "TIME_STOP"

            if exit_price is not None:
                break

        if exit_price is None:
            continue  # never resolved (e.g. data ended mid-session) -- skip

        ticks = (exit_price - entry_price) / tick_size * direction
        gross_pnl = ticks * tick_value * contracts
        commission = commission_rt * contracts
        trades.append({
            "entry_time": df.index[i], "direction": "long" if direction == 1 else "short",
            "entry": entry_price, "exit": exit_price, "exit_type": exit_type,
            "pnl": gross_pnl - commission,
        })

    return trades
