"""
VWAP Pullback + Higher-Timeframe Trend — implements the discretionary
playbook: only trade with the daily trend, enter on a pullback to VWAP that
holds (doesn't break the session low/high) and gets rejected back in the
trend direction with a strong candle, enter on the next bar.

Daily trend filter (computed from the *prior* completed session only, to
avoid lookahead into the still-forming current day):
  Bullish day: prior day close > daily EMA(10) > daily EMA(20)
  Bearish day: prior day close < daily EMA(10) < daily EMA(20)
  Otherwise: no trades that session.

Long setup: price above VWAP, pulls back to touch/dip through VWAP without
breaking the session low made so far, then a bullish candle (body >=
`min_body_atr_mult` x ATR) closes back above VWAP -> enter on the next bar.
Short setup is the mirror image.

Risk plan (see engine.py for the breakeven mechanic): stop below the
pullback low, fixed 2R target on the full position, stop moved to breakeven
at 1R. This approximates the intended "2R first target / trail remainder
under swing lows" without partial-position support in the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import atr, ema, session_vwap

DAILY_FAST_EMA, DAILY_SLOW_EMA = 10, 20
PULLBACK_LOOKBACK_BARS = 5
MIN_BODY_ATR_MULT = 0.3
R_MULTIPLE_TARGET = 2.0
BREAKEVEN_R_MULT = 1.0
SESSION_TZ = "America/New_York"
RTH_OPEN_MINUTES = 9 * 60 + 30   # 09:30 ET
RTH_CLOSE_MINUTES = 16 * 60      # 16:00 ET


class VwapPullbackTrendStrategy:
    name = "vwap_pullback_trend"

    def __init__(self, daily_fast_ema: int = DAILY_FAST_EMA, daily_slow_ema: int = DAILY_SLOW_EMA,
                 pullback_lookback_bars: int = PULLBACK_LOOKBACK_BARS,
                 min_body_atr_mult: float = MIN_BODY_ATR_MULT, r_multiple_target: float = R_MULTIPLE_TARGET,
                 breakeven_r_mult: float = BREAKEVEN_R_MULT):
        self.daily_fast_ema = daily_fast_ema
        self.daily_slow_ema = daily_slow_ema
        self.pullback_lookback_bars = pullback_lookback_bars
        self.min_body_atr_mult = min_body_atr_mult
        self.r_multiple_target = r_multiple_target
        self.breakeven_r_mult = breakeven_r_mult  # read by the engine directly

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
        open_ = df["open"]

        local_index = df.index.tz_convert(SESSION_TZ)
        session_date = local_index.date

        # Daily trend bias from the prior completed session only.
        daily_close = close.groupby(session_date).last()
        daily_fast = daily_close.ewm(span=self.daily_fast_ema, adjust=False).mean()
        daily_slow = daily_close.ewm(span=self.daily_slow_ema, adjust=False).mean()
        prior_close = daily_close.shift(1)
        prior_fast = daily_fast.shift(1)
        prior_slow = daily_slow.shift(1)
        daily_bias = pd.Series(
            np.select(
                [(prior_close > prior_fast) & (prior_fast > prior_slow),
                 (prior_close < prior_fast) & (prior_fast < prior_slow)],
                [1, -1], default=0,
            ),
            index=daily_close.index,
        )
        bias_by_date = daily_bias.to_dict()
        bar_bias = np.array([bias_by_date.get(d, 0) for d in session_date])

        # Session low/high made so far (excludes current bar to avoid using
        # the bar's own extreme as "the level it must hold above").
        session_low_so_far = low.groupby(session_date).cummin().shift(1)
        session_high_so_far = high.groupby(session_date).cummax().shift(1)

        vwap = session_vwap(high, low, close, volume)
        atr_ = atr(high, low, close, 14)
        body = (close - open_).to_numpy()

        minutes_of_day = local_index.hour * 60 + local_index.minute
        in_rth = np.asarray((minutes_of_day >= RTH_OPEN_MINUTES) & (minutes_of_day < RTH_CLOSE_MINUTES))

        cl = close.to_numpy()
        op = open_.to_numpy()
        lo = low.to_numpy()
        hi = high.to_numpy()
        vw = vwap.to_numpy()
        at = atr_.to_numpy()
        sess_lo = session_low_so_far.to_numpy()
        sess_hi = session_high_so_far.to_numpy()

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        n = len(df)
        pending_entry_dir = 0  # set on the rejection bar, fired on the *next* bar
        pending_stop = np.nan
        pending_target = np.nan

        for i in range(1, n):
            if pending_entry_dir != 0:
                entries.iloc[i] = pending_entry_dir
                stop_price.iloc[i] = pending_stop
                target_price.iloc[i] = pending_target
                pending_entry_dir = 0
                continue

            if bar_bias[i] == 0 or np.isnan(vw[i]) or np.isnan(at[i]) or not in_rth[i]:
                continue

            strong_body = abs(body[i]) >= self.min_body_atr_mult * at[i]
            window_lo = lo[max(0, i - self.pullback_lookback_bars) : i + 1]
            window_hi = hi[max(0, i - self.pullback_lookback_bars) : i + 1]

            if bar_bias[i] == 1:
                pulled_back = cl[i - 1] <= vw[i - 1]
                held_session_low = np.isnan(sess_lo[i]) or window_lo.min() >= sess_lo[i]
                rejects_up = cl[i] > vw[i] and cl[i] > op[i] and strong_body
                if pulled_back and held_session_low and rejects_up:
                    stop = window_lo.min()
                    risk = cl[i] - stop
                    if risk > 0:
                        pending_entry_dir = 1
                        pending_stop = stop
                        pending_target = cl[i] + self.r_multiple_target * risk
            elif bar_bias[i] == -1:
                pulled_back = cl[i - 1] >= vw[i - 1]
                held_session_high = np.isnan(sess_hi[i]) or window_hi.max() <= sess_hi[i]
                rejects_down = cl[i] < vw[i] and cl[i] < op[i] and strong_body
                if pulled_back and held_session_high and rejects_down:
                    stop = window_hi.max()
                    risk = stop - cl[i]
                    if risk > 0:
                        pending_entry_dir = -1
                        pending_stop = stop
                        pending_target = cl[i] - self.r_multiple_target * risk

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)
