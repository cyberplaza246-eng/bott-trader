"""
Unified Instrument Registry — Single source of truth for all tradeable instruments.

Replaces the 10+ duplicate PAIR_CONFIG / PIP_VALUES dictionaries scattered
across the codebase.  Every module should import from here instead.

Covers:
  - Futures: MES, MNQ, ES, NQ, CL, GC
  - Forex (backward-compat): EUR/USD, GBP/USD, USD/JPY
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone
from typing import Dict, List, Optional


# ── Core Data Model ─────────────────────────────────────────────────

@dataclass(frozen=True)
class InstrumentSpec:
    """Immutable specification for a single tradeable instrument."""

    symbol: str                        # e.g. "MES", "EUR/USD"
    asset_class: str                   # "futures" | "forex"
    exchange: str                      # "CME", "FOREX"

    # Price mechanics
    tick_size: float                   # Minimum price increment (e.g. 0.25 for ES)
    tick_value_usd: float              # Dollar value of one tick move per contract/lot
    decimal_places: int                # Display precision
    contract_multiplier: float         # Notional multiplier (ES=50, MES=5, forex=100_000)

    # Costs
    commission_rt: float               # Round-trip commission per contract in USD
    spread_default: float              # Default simulated spread in price units

    # Risk / volatility defaults
    atr_minimum: float                 # Minimum ATR to allow trades (price units)
    sl_max_ticks_1m: int               # Hard-cap SL in ticks for 1M trades
    sl_max_ticks_5m: int               # Hard-cap SL in ticks for 5M trades
    tp_max_ticks_1m: int               # Hard-cap TP in ticks for 1M trades
    tp_max_ticks_5m: int               # Hard-cap TP in ticks for 5M trades
    sweep_tolerance_ticks: int         # Liquidity sweep penetration tolerance

    # Session
    session_start: dt_time             # Normal trading start (UTC)
    session_end: dt_time               # Normal trading end (UTC)
    maintenance_start: dt_time         # CME maintenance start (UTC)
    maintenance_end: dt_time           # CME maintenance end (UTC)

    # Margin (intraday / overnight)
    margin_intraday: float             # Margin per contract (intraday)
    margin_overnight: float            # Margin per contract (overnight hold)

    # Metadata
    correlation_group: str             # e.g. "equity_index", "energy", "metal", "eur_bloc"
    description: str                   # Human-friendly label

    # ── Convenience helpers ──────────────────────────────────────

    @property
    def pip_size(self) -> float:
        """Backward-compat alias — returns tick_size for futures, pip for forex."""
        return self.tick_size

    @property
    def pip_value_per_lot(self) -> float:
        """Backward-compat alias for forex lot-based code."""
        return self.tick_value_usd

    @property
    def spread_sim(self) -> float:
        """Backward-compat alias used by sweep/scalping analyzers."""
        return self.spread_default

    @property
    def session_atr_min(self) -> float:
        """Backward-compat alias used by scalping/sweep analyzers."""
        return self.atr_minimum


# ── Futures Instruments ─────────────────────────────────────────────

_MES = InstrumentSpec(
    symbol="MES",
    asset_class="futures",
    exchange="CME",
    tick_size=0.25,
    tick_value_usd=1.25,         # $5 per point × 0.25 = $1.25/tick
    decimal_places=2,
    contract_multiplier=5.0,
    commission_rt=0.62,          # Typical micro commission RT
    spread_default=0.25,         # 1 tick typical
    atr_minimum=2.0,             # ~2 points min ATR (filters low-vol noise)
    sl_max_ticks_1m=10,          # 2.5 points = $3.12 on MES
    sl_max_ticks_5m=14,          # 3.5 points = $4.38
    tp_max_ticks_1m=18,          # 4.5 points = $5.62
    tp_max_ticks_5m=24,          # 6 points = $7.50
    sweep_tolerance_ticks=2,     # 0.5 point sweep past level
    session_start=dt_time(23, 0),   # Sun 6pm ET = 23 UTC
    session_end=dt_time(22, 0),     # Fri 5pm ET = 22 UTC
    maintenance_start=dt_time(21, 0),  # 4pm-5pm CT = 21-22 UTC
    maintenance_end=dt_time(22, 0),
    margin_intraday=40.0,
    margin_overnight=1_596.0,
    correlation_group="equity_index",
    description="Micro E-mini S&P 500",
)

_MNQ = InstrumentSpec(
    symbol="MNQ",
    asset_class="futures",
    exchange="CME",
    tick_size=0.25,
    tick_value_usd=0.50,         # $2 per point × 0.25 = $0.50/tick
    decimal_places=2,
    contract_multiplier=2.0,
    commission_rt=0.62,
    spread_default=0.50,         # 2 ticks typical
    atr_minimum=5.0,            # NQ more volatile, 5m ATR typically 8-12
    sl_max_ticks_1m=20,          # 5 points = $2.50 on MNQ
    sl_max_ticks_5m=28,          # 7 points = $3.50
    tp_max_ticks_1m=35,          # 8.75 points = $4.38
    tp_max_ticks_5m=48,          # 12 points = $6.00
    sweep_tolerance_ticks=5,     # 1.25 points
    session_start=dt_time(23, 0),
    session_end=dt_time(22, 0),
    maintenance_start=dt_time(21, 0),
    maintenance_end=dt_time(22, 0),
    margin_intraday=50.0,
    margin_overnight=1_886.0,
    correlation_group="equity_index",
    description="Micro E-mini Nasdaq-100",
)

_ES = InstrumentSpec(
    symbol="ES",
    asset_class="futures",
    exchange="CME",
    tick_size=0.25,
    tick_value_usd=12.50,        # $50 per point × 0.25 = $12.50/tick
    decimal_places=2,
    contract_multiplier=50.0,
    commission_rt=2.04,
    spread_default=0.25,
    atr_minimum=3.0,
    sl_max_ticks_1m=16,          # 4 points = $50
    sl_max_ticks_5m=32,          # 8 points = $100
    tp_max_ticks_1m=24,          # 6 points
    tp_max_ticks_5m=48,          # 12 points
    sweep_tolerance_ticks=4,
    session_start=dt_time(23, 0),
    session_end=dt_time(22, 0),
    maintenance_start=dt_time(21, 0),
    maintenance_end=dt_time(22, 0),
    margin_intraday=500.0,
    margin_overnight=15_960.0,
    correlation_group="equity_index",
    description="E-mini S&P 500",
)

_NQ = InstrumentSpec(
    symbol="NQ",
    asset_class="futures",
    exchange="CME",
    tick_size=0.25,
    tick_value_usd=5.00,         # $20 per point × 0.25
    decimal_places=2,
    contract_multiplier=20.0,
    commission_rt=2.04,
    spread_default=0.50,
    atr_minimum=15.0,
    sl_max_ticks_1m=40,
    sl_max_ticks_5m=80,
    tp_max_ticks_1m=60,
    tp_max_ticks_5m=120,
    sweep_tolerance_ticks=8,
    session_start=dt_time(23, 0),
    session_end=dt_time(22, 0),
    maintenance_start=dt_time(21, 0),
    maintenance_end=dt_time(22, 0),
    margin_intraday=1_000.0,
    margin_overnight=18_860.0,
    correlation_group="equity_index",
    description="E-mini Nasdaq-100",
)

_CL = InstrumentSpec(
    symbol="CL",
    asset_class="futures",
    exchange="NYMEX",
    tick_size=0.01,
    tick_value_usd=10.00,        # $1000 per point × 0.01
    decimal_places=2,
    contract_multiplier=1_000.0,
    commission_rt=2.04,
    spread_default=0.02,         # 2 ticks
    atr_minimum=0.30,            # $0.30 min ATR
    sl_max_ticks_1m=30,          # $0.30 = $300
    sl_max_ticks_5m=50,          # $0.50 = $500
    tp_max_ticks_1m=45,
    tp_max_ticks_5m=75,
    sweep_tolerance_ticks=3,
    session_start=dt_time(23, 0),
    session_end=dt_time(22, 0),
    maintenance_start=dt_time(21, 0),
    maintenance_end=dt_time(22, 0),
    margin_intraday=2_500.0,
    margin_overnight=6_600.0,
    correlation_group="energy",
    description="Crude Oil WTI",
)

_GC = InstrumentSpec(
    symbol="GC",
    asset_class="futures",
    exchange="COMEX",
    tick_size=0.10,
    tick_value_usd=10.00,        # $100 per point × 0.10
    decimal_places=1,
    contract_multiplier=100.0,
    commission_rt=2.04,
    spread_default=0.20,
    atr_minimum=3.0,
    sl_max_ticks_1m=50,
    sl_max_ticks_5m=100,
    tp_max_ticks_1m=75,
    tp_max_ticks_5m=150,
    sweep_tolerance_ticks=5,
    session_start=dt_time(23, 0),
    session_end=dt_time(22, 0),
    maintenance_start=dt_time(21, 0),
    maintenance_end=dt_time(22, 0),
    margin_intraday=5_000.0,
    margin_overnight=11_000.0,
    correlation_group="metal",
    description="Gold (COMEX)",
)


# ── Forex Instruments (backward-compat) ─────────────────────────────

_EURUSD = InstrumentSpec(
    symbol="EUR/USD",
    asset_class="forex",
    exchange="FOREX",
    tick_size=0.0001,
    tick_value_usd=10.0,         # $10 per pip per standard lot
    decimal_places=5,
    contract_multiplier=100_000.0,
    commission_rt=7.0,
    spread_default=0.00006,      # 0.6 pips ECN
    atr_minimum=0.00010,         # 1.0 pip
    sl_max_ticks_1m=150,         # 15 pips
    sl_max_ticks_5m=250,         # 25 pips
    tp_max_ticks_1m=200,         # 20 pips
    tp_max_ticks_5m=350,         # 35 pips
    sweep_tolerance_ticks=5,     # 0.5 pips
    session_start=dt_time(0, 0),
    session_end=dt_time(0, 0),   # 24/5
    maintenance_start=dt_time(21, 50),
    maintenance_end=dt_time(22, 5),
    margin_intraday=0.0,
    margin_overnight=0.0,
    correlation_group="eur_bloc",
    description="Euro vs US Dollar",
)

_GBPUSD = InstrumentSpec(
    symbol="GBP/USD",
    asset_class="forex",
    exchange="FOREX",
    tick_size=0.0001,
    tick_value_usd=10.0,
    decimal_places=5,
    contract_multiplier=100_000.0,
    commission_rt=7.0,
    spread_default=0.00010,      # 1.0 pip ECN
    atr_minimum=0.00015,         # 1.5 pips
    sl_max_ticks_1m=150,
    sl_max_ticks_5m=250,
    tp_max_ticks_1m=200,
    tp_max_ticks_5m=350,
    sweep_tolerance_ticks=5,
    session_start=dt_time(0, 0),
    session_end=dt_time(0, 0),
    maintenance_start=dt_time(21, 50),
    maintenance_end=dt_time(22, 5),
    margin_intraday=0.0,
    margin_overnight=0.0,
    correlation_group="eur_bloc",
    description="British Pound vs US Dollar",
)

_USDJPY = InstrumentSpec(
    symbol="USD/JPY",
    asset_class="forex",
    exchange="FOREX",
    tick_size=0.01,
    tick_value_usd=6.5,          # ~$6.50 per pip at ~153 JPY
    decimal_places=3,
    contract_multiplier=100_000.0,
    commission_rt=7.0,
    spread_default=0.008,        # 0.8 pips ECN
    atr_minimum=0.015,           # 1.5 pips
    sl_max_ticks_1m=150,         # 15 pips
    sl_max_ticks_5m=250,         # 25 pips
    tp_max_ticks_1m=200,
    tp_max_ticks_5m=350,
    sweep_tolerance_ticks=5,
    session_start=dt_time(0, 0),
    session_end=dt_time(0, 0),
    maintenance_start=dt_time(21, 50),
    maintenance_end=dt_time(22, 5),
    margin_intraday=0.0,
    margin_overnight=0.0,
    correlation_group="jpy_bloc",
    description="US Dollar vs Japanese Yen",
)


# ── Registry ────────────────────────────────────────────────────────

REGISTRY: Dict[str, InstrumentSpec] = {
    # Futures
    "MES": _MES,
    "MNQ": _MNQ,
    "ES":  _ES,
    "NQ":  _NQ,
    "CL":  _CL,
    "GC":  _GC,
    # Forex
    "EUR/USD": _EURUSD,
    "GBP/USD": _GBPUSD,
    "USD/JPY": _USDJPY,
}


# ── Public API ──────────────────────────────────────────────────────

def get_instrument(symbol: str) -> InstrumentSpec:
    """Retrieve an instrument spec by symbol.  Raises KeyError if unknown."""
    return REGISTRY[symbol]


def get_all_instruments(asset_class: Optional[str] = None) -> List[InstrumentSpec]:
    """Return all instruments, optionally filtered by asset class."""
    specs = list(REGISTRY.values())
    if asset_class:
        specs = [s for s in specs if s.asset_class == asset_class]
    return specs


def tick_distance(symbol: str, price_a: float, price_b: float) -> int:
    """Number of ticks between two prices (always positive)."""
    spec = REGISTRY[symbol]
    return round(abs(price_a - price_b) / spec.tick_size)


def dollar_risk(symbol: str, entry: float, stop: float, contracts: int = 1) -> float:
    """Dollar amount at risk for a given entry/stop and contract count."""
    spec = REGISTRY[symbol]
    ticks = tick_distance(symbol, entry, stop)
    return ticks * spec.tick_value_usd * contracts


def is_futures(symbol: str) -> bool:
    return REGISTRY[symbol].asset_class == "futures"


def is_forex(symbol: str) -> bool:
    return REGISTRY[symbol].asset_class == "forex"


def is_maintenance_window(symbol: str, utc_now: Optional[datetime] = None) -> bool:
    """Check if the instrument is currently in its maintenance window."""
    spec = REGISTRY[symbol]
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    t = utc_now.time()
    if spec.maintenance_start <= spec.maintenance_end:
        return spec.maintenance_start <= t < spec.maintenance_end
    # Wraps midnight
    return t >= spec.maintenance_start or t < spec.maintenance_end
