"""
VWAP Pullback + Trend, v2 — a synthesis, not a fresh guess.

Base mechanics unchanged from VwapPullbackTrendStrategy (daily trend filter,
VWAP pullback holding the session extreme, rejection-candle confirmation,
next-bar entry, 2R target with breakeven-at-1R): see vwap_pullback_trend.py
for the full rationale.

Two evidence-grounded changes on top of v1, not a rebuild:

1. Defaults changed from the untested originals (pullback_lookback_bars=5,
   min_body_atr_mult=0.3) to the specific combination the walk-forward
   selection-bias audit showed was a REAL, consistently-selected preference
   across folds (pullback_lookback_bars=8, min_body_atr_mult=0.5) -- not
   picked because it looked good once, but because it won repeatedly across
   independent train windows (see scripts/audit_selection_bias.py results).

2. New: require ADX(14) >= min_adx at the rejection bar. Grounded in two
   separate observations across this project's whole test history: (a)
   mean_reversion (the intraday strategy that survived best) used an ADX
   regime gate; (b) every strategy that fired often, regardless of
   direction, lost -- selectivity via a real regime filter is the one
   property that correlated with less damage everywhere it was tried. This
   is the first time it's been combined specifically with vwap_pullback_trend.

Still subject to the same walk-forward + significance + cross-instrument
replication gate as everything else -- this is hypothesis #16, evidence-
informed, not validated by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import StrategySignals
from src.strategies.indicators import adx, atr, ema, session_vwap

DAILY_FAST_EMA, DAILY_SLOW_EMA = 10, 20
PULLBACK_LOOKBACK_BARS = 8    # audit-preferred, not the untested v1 default of 5
MIN_BODY_ATR_MULT = 0.5       # audit-preferred, not the untested v1 default of 0.3
MIN_ADX = 20.0                # new: require real trend strength at the entry bar
ADX_PERIOD = 14
R_MULTIPLE_TARGET = 2.0
BREAKEVEN_R_MULT = 1.0
SESSION_TZ = "America/New_York"
RTH_OPEN_MINUTES = 9 * 60 + 30
RTH_CLOSE_MINUTES = 16 * 60


class VwapPullbackTrendV2Strategy:
    name = "vwap_pullback_trend_v2"

    def __init__(self, daily_fast_ema: int = DAILY_FAST_EMA, daily_slow_ema: int = DAILY_SLOW_EMA,
                 pullback_lookback_bars: int = PULLBACK_LOOKBACK_BARS,
                 min_body_atr_mult: float = MIN_BODY_ATR_MULT, min_adx: float = MIN_ADX,
                 adx_period: int = ADX_PERIOD, r_multiple_target: float = R_MULTIPLE_TARGET,
                 breakeven_r_mult: float = BREAKEVEN_R_MULT):
        self.daily_fast_ema = daily_fast_ema
        self.daily_slow_ema = daily_slow_ema
        self.pullback_lookback_bars = pullback_lookback_bars
        self.min_body_atr_mult = min_body_atr_mult
        self.min_adx = min_adx
        self.adx_period = adx_period
        self.r_multiple_target = r_multiple_target
        self.breakeven_r_mult = breakeven_r_mult

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
        open_ = df["open"]

        local_index = df.index.tz_convert(SESSION_TZ)
        session_date = local_index.date

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

        session_low_so_far = low.groupby(session_date).cummin().shift(1)
        session_high_so_far = high.groupby(session_date).cummax().shift(1)

        vwap = session_vwap(high, low, close, volume)
        atr_ = atr(high, low, close, 14)
        adx_ = adx(high, low, close, self.adx_period)
        body = (close - open_).to_numpy()

        minutes_of_day = local_index.hour * 60 + local_index.minute
        in_rth = np.asarray((minutes_of_day >= RTH_OPEN_MINUTES) & (minutes_of_day < RTH_CLOSE_MINUTES))

        cl = close.to_numpy()
        op = open_.to_numpy()
        lo = low.to_numpy()
        hi = high.to_numpy()
        vw = vwap.to_numpy()
        at = atr_.to_numpy()
        ax = adx_.to_numpy()
        sess_lo = session_low_so_far.to_numpy()
        sess_hi = session_high_so_far.to_numpy()

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        n = len(df)
        pending_entry_dir = 0
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

            if ax[i] < self.min_adx:
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
