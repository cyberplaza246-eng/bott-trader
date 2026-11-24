#!/usr/bin/env python
"""Inspect TEST011 MNQ and flatten if the Lucid test fill is still open."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

os.environ["RITHMIC_QUOTES_ONLY"] = "false"
os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
os.environ["RITHMIC_ALLOW_SIMULATOR"] = "true"
os.environ["RITHMIC_SKIP_HISTORY_PNL"] = "true"
os.environ["RITHMIC_ORDER_PLANT_ONLY"] = "false"

from src.broker.rithmic_connector import (
    RithmicConnector,
    _response_to_dict_safe,
    apply_lucid_ticker_order_plants,
)
from src.utils.redact import redact_secrets

apply_lucid_ticker_order_plants()

WATCH_TAGS = ("bot_39fa453b6a", "bot_3335864db6", "bot_")


def _safe(data: dict) -> dict:
    keep = (
        "symbol",
        "status",
        "qty",
        "quantity",
        "filled_qty",
        "fill_qty",
        "user_tag",
        "basket_id",
        "transaction_type",
        "order_type",
        "avg_fill_price",
        "avg_price",
        "price",
        "account_id",
        "exchange",
        "text",
        "notify_type",
    )
    out = {k: data.get(k) for k in keep if data.get(k) not in (None, "", [])}
    return out


def main() -> int:
    broker = RithmicConnector(live_mode=False, quotes_only=False)
    broker._symbols_to_watch = ["MNQ"]
    broker._disable_yahoo_fallback = True
    broker.initialize()
    if not broker.connected:
        print("NOT CONNECTED")
        print(redact_secrets(broker._last_connect_error or "unknown"))
        return 1

    acct = broker._get_account_id() or "?"
    print(f"account={acct} live_mode=False ticker+order")

    net = broker.get_symbol_list_positions_net("MNQ")
    print(f"list_positions net MNQ: {net}")
    try:
        report = broker.reconcile_symbol_exposure("MNQ")
        print(
            f"reconcile broker_net={report.get('broker_net')} "
            f"source={report.get('exposure_source')} "
            f"working={report.get('working_entries')}"
        )
    except Exception as exc:
        print(f"reconcile failed: {redact_secrets(exc)}")

    tickets = []
    working = []
    filled_bot = []
    try:
        orders = broker._run_sync(
            broker._client.list_orders(account_id=acct),
            timeout=20,
        )
        print(f"list_orders count: {len(orders or [])}")
        for order in orders or []:
            data = _response_to_dict_safe(order)
            our = broker._reverse_resolve(str(data.get("symbol") or ""))
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            status = str(data.get("status") or "")
            notify = str(data.get("notify_type") or "").upper()
            if our != "MNQ" and not tag.startswith("bot_"):
                continue
            print(f"  order: {_safe(data)} resolved={our}")
            tickets.append(tag)
            if tag.startswith("bot_") and RithmicConnector._is_working_order_status(status):
                working.append(tag)
            if tag.startswith("bot_") and (notify == "FILL" or "fill" in status.lower()):
                filled_bot.append(tag)
    except Exception as exc:
        print(f"list_orders failed: {redact_secrets(exc)}")
        orders = []

    should_flatten = bool(working)
    if working:
        print(f"working bot orders: {working}")
    if not should_flatten and net in (None, 0):
        print("No working MNQ protective orders.")
        broker.shutdown()
        return 0

    print("Flattening MNQ test exposure (simulator)...")
    result = broker.flatten_symbol("MNQ", max_attempts=1, verify_delay=1.5)
    print(f"flatten result: {result}")
    net2 = broker.get_symbol_list_positions_net("MNQ")
    print(f"list_positions net after flatten: {net2}")
    if result.get("flat"):
        print("STATUS: flatten market filled (PnL plant unavailable).")
    elif net2 == 0:
        print("STATUS: flat")
    else:
        print(f"STATUS: still open net={net2}")
    broker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
