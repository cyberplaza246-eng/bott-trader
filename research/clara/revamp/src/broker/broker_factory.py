"""
Broker Factory — Instantiate the correct broker connector based on config.

Usage:
    from src.broker.broker_factory import create_broker
    broker = create_broker()   # reads BROKER_TYPE env / strategy_config
"""
from __future__ import annotations

import os

from src.broker.base_broker import BaseBroker


def create_broker(broker_type: str | None = None) -> BaseBroker:
    """Create and return an initialized broker connector.

    Args:
        broker_type: "mt5", "traderspost", "rithmic", or None (auto-detect from env).

    Returns:
        An initialized BaseBroker instance.
    """
    if broker_type is None:
        broker_type = os.getenv("BROKER_TYPE", "mt5").lower().strip()

    if broker_type == "rithmic":
        from src.broker.rithmic_connector import RithmicConnector
        broker = RithmicConnector()
    elif broker_type == "traderspost":
        from src.broker.traderspost_connector import TradersPostConnector
        broker = TradersPostConnector()
    elif broker_type == "mt5":
        from src.broker.mt5_connector import MT5Connector
        broker = MT5Connector()
    else:
        raise ValueError(f"Unknown BROKER_TYPE: {broker_type!r}. Use 'mt5', 'traderspost', or 'rithmic'.")

    return broker
