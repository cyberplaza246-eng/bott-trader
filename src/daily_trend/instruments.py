"""
Instrument specs for the daily multi-market trend-following universe.

Deliberately separate from src/instruments/instrument_registry.py, which is
shaped for intraday scalping (session windows, 1m/5m SL/TP tick caps, sweep
tolerance) — none of which means anything for a daily-bar system. This is
the minimal spec a daily trend system actually needs: tick mechanics,
costs, and margin.

Margin figures are approximate typical CME/CBOT/COMEX/NYMEX initial margins
and will drift over time — treat as order-of-magnitude for position sizing,
not a live-trading reference (same caveat as the intraday registry).
Commission is a typical retail futures round-turn estimate per contract.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyInstrumentSpec:
    symbol: str
    databento_continuous_symbol: str  # e.g. "ES.c.0"
    exchange: str
    description: str
    tick_size: float
    tick_value_usd: float
    contract_multiplier: float
    commission_rt: float
    margin_approx: float


REGISTRY: dict[str, DailyInstrumentSpec] = {
    "ES": DailyInstrumentSpec(
        symbol="ES", databento_continuous_symbol="ES.c.0", exchange="CME",
        description="E-mini S&P 500",
        tick_size=0.25, tick_value_usd=12.50, contract_multiplier=50.0,
        commission_rt=2.04, margin_approx=13_200.0,
    ),
    "NQ": DailyInstrumentSpec(
        symbol="NQ", databento_continuous_symbol="NQ.c.0", exchange="CME",
        description="E-mini Nasdaq-100",
        tick_size=0.25, tick_value_usd=5.00, contract_multiplier=20.0,
        commission_rt=2.04, margin_approx=18_860.0,
    ),
    "MES": DailyInstrumentSpec(
        symbol="MES", databento_continuous_symbol="MES.c.0", exchange="CME",
        description="Micro E-mini S&P 500",
        tick_size=0.25, tick_value_usd=1.25, contract_multiplier=5.0,
        commission_rt=0.62, margin_approx=1_596.0,
    ),
    "MNQ": DailyInstrumentSpec(
        symbol="MNQ", databento_continuous_symbol="MNQ.c.0", exchange="CME",
        description="Micro E-mini Nasdaq-100",
        tick_size=0.25, tick_value_usd=0.50, contract_multiplier=2.0,
        commission_rt=0.62, margin_approx=1_886.0,
    ),
    "GC": DailyInstrumentSpec(
        symbol="GC", databento_continuous_symbol="GC.c.0", exchange="COMEX",
        description="Gold (100 troy oz)",
        tick_size=0.10, tick_value_usd=10.00, contract_multiplier=100.0,
        commission_rt=2.04, margin_approx=11_000.0,
    ),
    "CL": DailyInstrumentSpec(
        symbol="CL", databento_continuous_symbol="CL.c.0", exchange="NYMEX",
        description="Crude Oil WTI (1,000 bbl)",
        tick_size=0.01, tick_value_usd=10.00, contract_multiplier=1_000.0,
        commission_rt=2.04, margin_approx=6_600.0,
    ),
    "HG": DailyInstrumentSpec(
        symbol="HG", databento_continuous_symbol="HG.c.0", exchange="COMEX",
        description="Copper (25,000 lbs)",
        tick_size=0.0005, tick_value_usd=12.50, contract_multiplier=25_000.0,
        commission_rt=2.04, margin_approx=6_600.0,
    ),
    "ZS": DailyInstrumentSpec(
        symbol="ZS", databento_continuous_symbol="ZS.c.0", exchange="CBOT",
        description="Soybeans (5,000 bu)",
        tick_size=0.0025, tick_value_usd=12.50, contract_multiplier=5_000.0,
        commission_rt=2.04, margin_approx=3_300.0,
    ),
    "ZN": DailyInstrumentSpec(
        symbol="ZN", databento_continuous_symbol="ZN.c.0", exchange="CBOT",
        description="10-Year T-Note ($100,000 face)",
        tick_size=0.015625, tick_value_usd=15.625, contract_multiplier=1_000.0,
        commission_rt=2.04, margin_approx=2_200.0,
    ),
    "6E": DailyInstrumentSpec(
        symbol="6E", databento_continuous_symbol="6E.c.0", exchange="CME",
        description="Euro FX (EUR 125,000)",
        tick_size=0.00005, tick_value_usd=6.25, contract_multiplier=125_000.0,
        commission_rt=2.04, margin_approx=2_750.0,
    ),
    "6J": DailyInstrumentSpec(
        symbol="6J", databento_continuous_symbol="6J.c.0", exchange="CME",
        description="Japanese Yen (JPY 12,500,000)",
        tick_size=0.0000005, tick_value_usd=6.25, contract_multiplier=12_500_000.0,
        commission_rt=2.04, margin_approx=3_850.0,
    ),
}


def get_spec(symbol: str) -> DailyInstrumentSpec:
    if symbol not in REGISTRY:
        raise ValueError(f"Unsupported daily-trend symbol {symbol!r}; expected one of {list(REGISTRY)}")
    return REGISTRY[symbol]
