"""
Backtester defaults — overridable via CLI flags in scripts/run_backtest.py.
"""
from __future__ import annotations

DEFAULT_ACCOUNT_SIZE = 50_000.0
DEFAULT_RISK_PCT = 0.5          # % of account risked per trade
DEFAULT_TIMEFRAME = "5m"        # 5-minute bars: less noise than 1m, still intraday
DEFAULT_SLIPPAGE_TICKS = 1      # assumed slippage per fill, in ticks

DATA_DIR = "data"
REPORTS_DIR = "reports"

SUPPORTED_SYMBOLS = ["MNQ", "NQ", "MES", "ES"]
SUPPORTED_TIMEFRAMES = ["1m", "5m"]
SUPPORTED_STRATEGIES = [
    "trend", "mean_reversion", "breakout", "ensemble", "vwap_cross", "orb",
    "orb_failure", "vwap_pullback_trend", "vwap_pullback_trend_v2", "regime_bot",
    "vol_expansion_momentum", "extreme_displacement_reversion", "volume_shock_continuation",
]

# NQ now has its own real Databento feed (data/NQ_1m.csv, NQ_5m.csv) — no
# longer derived from MNQ. Kept as an empty mapping in case a future symbol
# needs the same derive-from-a-related-instrument fallback.
DERIVED_SYMBOLS: dict[str, str] = {}
