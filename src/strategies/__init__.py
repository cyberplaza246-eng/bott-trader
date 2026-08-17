from src.strategies.breakout import BreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.extreme_displacement_reversion import ExtremeDisplacementReversionStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.strategies.orb_failure import OrbFailureStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.vol_expansion_momentum import VolExpansionMomentumStrategy
from src.strategies.volume_shock_continuation import VolumeShockContinuationStrategy
from src.strategies.vwap_ema_cross import VwapEmaCrossStrategy
from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy
from src.strategies.vwap_pullback_trend_v2 import VwapPullbackTrendV2Strategy

STRATEGIES = {
    "trend": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "ensemble": EnsembleStrategy,
    "vwap_cross": VwapEmaCrossStrategy,
    "orb": OpeningRangeBreakoutStrategy,
    "orb_failure": OrbFailureStrategy,
    "vwap_pullback_trend": VwapPullbackTrendStrategy,
    "vwap_pullback_trend_v2": VwapPullbackTrendV2Strategy,
    "vol_expansion_momentum": VolExpansionMomentumStrategy,
    "extreme_displacement_reversion": ExtremeDisplacementReversionStrategy,
    "volume_shock_continuation": VolumeShockContinuationStrategy,
}


def get_strategy(name: str):
    if name == "regime_bot":
        from src.regime.allocator import RegimeAllocatorStrategy
        return RegimeAllocatorStrategy()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; expected one of {list(STRATEGIES) + ['regime_bot']}")
    return STRATEGIES[name]()
