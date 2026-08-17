"""
Thin instrument-spec accessor for the backtester.

Reuses the existing contract specs in src/instruments/instrument_registry.py
(tick size, tick value, contract multiplier, commissions, margin) instead of
duplicating them. Only exposes the symbols this bot trades.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.instruments.instrument_registry import REGISTRY, InstrumentSpec  # noqa: E402

from config.settings import SUPPORTED_SYMBOLS  # noqa: E402


def get_spec(symbol: str) -> InstrumentSpec:
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol!r}; expected one of {SUPPORTED_SYMBOLS}")
    return REGISTRY[symbol]
