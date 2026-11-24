"""
Ensemble: ADX-based regime router rather than naive signal voting (trend and
mean-reversion signals routinely contradict each other, so voting would
cancel out most trades). Per bar:

  - ADX >= 25 (trending)   -> take trend-following signals
  - ADX < 20  (ranging)    -> take mean-reversion signals
  - ADX 20-25 (transition) -> neither trend nor mean-reversion trades
  - breakout signals are event-triggered Donchian breaks; by default they
    do *not* fire in a ranging regime (ADX < 20). The pre-sprint behaviour
    of "breakout always overrides, including inside a range" is available
    via `breakout_in_range=True` for reproduction.

Optional `rth_only=True` zeros entries outside the NYSE cash session
(09:30–16:00 America/New_York). Overnight Globex 5m bars are liquid enough
to exist in the tape but fills/slippage there are worse than the 1-tick
model; RTH-only is a fill-realism / risk overlay, not a claimed edge.

REVAMP exits (opt-in, default OFF -> identical to original): `revamp_exits=True`
or env REVAMP_EXITS=1 sets breakout sleeve to a fixed R target (default 2R, no
trail) and MR sleeve target to the far Bollinger band (long->upper, short->lower).
The engine-side 4.0 ATR trail is applied in run_backtest (same flag).
See src/strategies/revamp_exits.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_ensemble.session import rth_mask
from src.research_ensemble.base import StrategySignals
from src.research_ensemble.breakout import BreakoutStrategy
from src.research_ensemble.indicators import adx, bollinger_bands
from src.research_ensemble import revamp_exits as _rx
from src.research_ensemble.mean_reversion import MeanReversionStrategy
from src.research_ensemble.trend_following import TrendFollowingStrategy

ADX_TREND_MIN = 25
ADX_RANGE_MAX = 20
ADX_PERIOD = 14


class EnsembleStrategy:
    name = "ensemble"

    def __init__(
        self,
        adx_trend_min: float = ADX_TREND_MIN,
        adx_range_max: float = ADX_RANGE_MAX,
        rth_only: bool = True,
        breakout_in_range: bool = False,
        event_trigger: bool = True,
        revamp_exits: bool | None = None,
        breakout_target_r: float | None = None,
        mr_target: str | None = None,
    ) -> None:
        self.adx_trend_min = adx_trend_min
        self.adx_range_max = adx_range_max
        self.rth_only = rth_only
        self.breakout_in_range = breakout_in_range
        self.event_trigger = event_trigger
        self.revamp_exits = _rx.revamp_enabled() if revamp_exits is None else bool(revamp_exits)
        self.breakout_target_r = (breakout_target_r if breakout_target_r is not None
                                  else (_rx.revamp_breakout_target_r() if self.revamp_exits else None))
        self.mr_target = mr_target or (_rx.revamp_mr_target() if self.revamp_exits else "bb_mid")
        self._trend = TrendFollowingStrategy()
        self._mean_reversion = MeanReversionStrategy(event_trigger=event_trigger)
        self._breakout = BreakoutStrategy(event_trigger=event_trigger)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        trend_sig = self._trend.generate_signals(df)
        mr_sig = self._mean_reversion.generate_signals(df)
        bo_sig = self._breakout.generate_signals(df)
        if self.revamp_exits:
            mr_sig, bo_sig = self._apply_revamp_targets(df, mr_sig, bo_sig)

        adx_ = adx(df["high"], df["low"], df["close"], ADX_PERIOD)
        trending = adx_ >= self.adx_trend_min
        ranging = adx_ < self.adx_range_max

        entries = pd.Series(0, index=df.index)
        stop_price = pd.Series(np.nan, index=df.index)
        target_price = pd.Series(np.nan, index=df.index)

        entries[trending] = trend_sig.entries[trending]
        stop_price[trending] = trend_sig.stop_price[trending]
        target_price[trending] = trend_sig.target_price[trending]

        entries[ranging] = mr_sig.entries[ranging]
        stop_price[ranging] = mr_sig.stop_price[ranging]
        target_price[ranging] = mr_sig.target_price[ranging]

        bo_fires = bo_sig.entries != 0
        if not self.breakout_in_range:
            bo_fires = bo_fires & ~ranging
        entries[bo_fires] = bo_sig.entries[bo_fires]
        stop_price[bo_fires] = bo_sig.stop_price[bo_fires]
        target_price[bo_fires] = bo_sig.target_price[bo_fires]

        if self.rth_only:
            in_rth = rth_mask(df.index)
            entries = entries.where(in_rth, 0)
            stop_price = stop_price.where(in_rth, np.nan)
            target_price = target_price.where(in_rth, np.nan)

        if self.revamp_exits:
            # same sanity as research harness: a target on the wrong side -> NaN (engine trails)
            c = df["close"]
            badt = ((entries == 1) & (target_price <= c)) | ((entries == -1) & (target_price >= c))
            target_price = target_price.where(~badt, np.nan)
            target_price = target_price.where(entries != 0, np.nan)

        return StrategySignals(entries=entries, stop_price=stop_price, target_price=target_price)

    def _apply_revamp_targets(self, df, mr_sig, bo_sig):
        c = df["close"]
        if self.breakout_target_r:
            d = bo_sig.entries.astype(float)
            fired = bo_sig.entries != 0
            tgt = c + d * float(self.breakout_target_r) * (c - bo_sig.stop_price).abs()
            bo_sig = StrategySignals(entries=bo_sig.entries, stop_price=bo_sig.stop_price,
                                     target_price=bo_sig.target_price.where(~fired, tgt))
        if self.mr_target == "bb_far":
            up, _mid, lo = bollinger_bands(c, 20, 2.0)
            fired = mr_sig.entries != 0
            far = pd.Series(np.where(mr_sig.entries > 0, up, lo), index=df.index)
            mr_sig = StrategySignals(entries=mr_sig.entries, stop_price=mr_sig.stop_price,
                                     target_price=mr_sig.target_price.where(~fired, far))
        return mr_sig, bo_sig
