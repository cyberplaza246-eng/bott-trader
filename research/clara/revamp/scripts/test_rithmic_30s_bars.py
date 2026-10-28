#!/usr/bin/env python
"""Spike test: connect to Rithmic and print the last 5 sub-minute bars."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    user = os.getenv("RITHMIC_USER_ID", "").strip()
    password = os.getenv("RITHMIC_PASSWORD", "").strip()
    if not user or not password:
        print("RITHMIC_USER_ID / RITHMIC_PASSWORD not set — skipping live 30s bar test")
        return 0

    symbol = os.getenv("TEST_SYMBOL", "MNQ").strip().upper()
    period = max(1, int(os.getenv("TRIGGER_BAR_SECONDS", "30")))

    from src.broker.rithmic_connector import RithmicConnector

    print(f"Connecting to Rithmic for {symbol} {period}s bars...")
    broker = RithmicConnector(live_mode=False)
    broker._symbols_to_watch = [symbol]
    broker.initialize()

    if not broker.connected:
        print(f"Could not connect: {broker._last_connect_error or 'unknown error'}")
        return 1

    try:
        df = broker.get_candles_seconds(symbol, period_seconds=period, num_candles=5)
        src = broker.second_bar_source
        if df is None or df.empty:
            print(f"No {period}s bars returned (source={src})")
            if src == "ticks":
                print(
                    "Lucid may not expose SECOND_BAR history — run during market hours "
                    "and wait for LAST_TRADE ticks to build bars."
                )
            return 0

        print(f"Got {len(df)} bars (source={src}):")
        for _, row in df.iterrows():
            print(
                f"  {row['datetime']}  "
                f"O={row['open']:.2f} H={row['high']:.2f} "
                f"L={row['low']:.2f} C={row['close']:.2f} V={int(row['volume'])}"
            )
        return 0
    finally:
        broker.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
