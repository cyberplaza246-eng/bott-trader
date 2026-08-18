#!/usr/bin/env python
"""
Flatten net position for one symbol on Rithmic and cancel working bot orders.

Usage:
    python scripts/rithmic_flatten_symbol.py MNQ
    python scripts/rithmic_flatten_symbol.py MNQ --dry-run
    python scripts/rithmic_flatten_symbol.py MNQ --force
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from src.broker.rithmic_connector import RithmicConnector, _response_to_dict_safe


def _flatten_side_label(net: int) -> str:
    if net < 0:
        return f"SHORT {abs(net)} → flatten with BUY {abs(net)}"
    if net > 0:
        return f"LONG {abs(net)} → flatten with SELL {abs(net)}"
    return "FLAT"


def _print_attempts(attempts: list) -> None:
    for info in attempts:
        print(
            f"  attempt {info.get('attempt')}: "
            f"net_before={info.get('net_before')} "
            f"contract={info.get('broker_symbol')} "
            f"close={info.get('close_side')} qty={info.get('qty')}"
        )
        exit_ok = info.get("exit_position_ok")
        if exit_ok is not None:
            print(
                f"    exit_position: {'OK' if exit_ok else 'FAIL'} "
                f"net_after_exit={info.get('net_after_exit')}"
            )
        market_ok = info.get("market_order_ok")
        if market_ok is not None:
            print(
                f"    market_order: {'OK' if market_ok else 'FAIL'} "
                f"net_after={info.get('net_after')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten Rithmic symbol position")
    parser.add_argument("symbol", help="Symbol to flatten (e.g. MNQ)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show position and orders only; do not flatten or cancel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Extra flatten retries and longer verify delay",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max flatten attempts (default 5, or 10 with --force)",
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()
    max_attempts = args.max_attempts or (10 if args.force else 5)
    verify_delay = 2.0 if args.force else 1.5

    connector = RithmicConnector(live_mode=True)
    print(f"Connecting to flatten {symbol}...")
    connector.initialize()
    if not connector.connected:
        err = connector._last_connect_error or "unknown error"
        print(f"Connection failed: {err}")
        return 1

    acct = connector._get_account_id() or "?"
    route = connector.get_trade_route("CME") or "?"
    rith_sym, exchange = connector._resolve_symbol(symbol)
    print(f"account={acct} route={route}")
    print(f"resolved contract (config): {rith_sym} @ {exchange}")

    net = connector.get_symbol_list_positions_net(symbol)
    report = connector.reconcile_symbol_exposure(symbol)
    print(f"list_positions net: {net}")
    print(
        f"reconcile: broker_net={report.get('broker_net')} "
        f"source={report.get('exposure_source')} "
        f"tag_inferred={report.get('tag_inferred_entries')} "
        f"working={report.get('working_entries')}"
    )
    if net not in (None, 0):
        print(f"exposure: {_flatten_side_label(int(net))}")

    try:
        orders = connector._run_sync(
            connector._client.list_orders(account_id=connector._get_account_id()),
            timeout=15,
        )
        working_bot = 0
        total_sym_orders = 0
        for order in orders or []:
            data = _response_to_dict_safe(order)
            our_sym = connector._reverse_resolve(data.get("symbol", ""))
            if our_sym != symbol:
                continue
            total_sym_orders += 1
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            if not connector._is_working_order_status(str(data.get("status") or "")):
                continue
            if tag.startswith("bot_"):
                working_bot += 1
        print(f"orders for {symbol}: {total_sym_orders} total, {working_bot} working bot_*")
    except Exception as e:
        print(f"list_orders failed: {e}")

    if args.dry_run:
        print("Dry run — no orders sent.")
        return 0

    should_flatten = net not in (None, 0) or args.force
    if not should_flatten:
        print("No list_positions exposure — skipping flatten.")
    else:
        if net in (None, 0) and args.force:
            print("Force mode — flatten retry despite list_positions net=0")
        print(
            f"Flattening {symbol} (max_attempts={max_attempts}, "
            f"verify_delay={verify_delay}s)..."
        )
        result = connector.flatten_symbol(
            symbol,
            max_attempts=max_attempts,
            verify_delay=verify_delay,
            cancel_working_first=True,
        )
        cancelled = result.get("cancelled_bot", 0)
        if cancelled:
            print(f"cancelled working bot_* orders: {cancelled}")
        broker_sym = result.get("broker_symbol") or rith_sym
        if broker_sym != rith_sym:
            print(f"broker contract from list_positions: {broker_sym} @ {result.get('exchange', exchange)}")
        attempts = result.get("attempts") or []
        if attempts:
            print("flatten attempts:")
            _print_attempts(attempts)
        if result.get("flat"):
            print("flatten: FLAT")
        else:
            err = result.get("error")
            final_net = result.get("final_net", net)
            print(f"flatten: FAILED (final_net={final_net})")
            if err:
                print(f"error: {err}")

    purge = connector.purge_orphan_protective_orders([symbol], threshold=1)
    purge_info = purge.get(symbol) or {}
    cancelled_stale = purge_info.get("cancelled", 0)
    order_count = purge_info.get("order_count", 0)
    if cancelled_stale:
        print(f"purged {cancelled_stale} stale working bot/protective order(s)")
    elif order_count > 1:
        net_after_purge = connector.get_symbol_list_positions_net(symbol)
        if net_after_purge not in (None, 0):
            print(
                f"orphan purge skipped: orders={order_count} "
                f"list_positions_net={net_after_purge} (must be flat to purge)"
            )
        else:
            print(f"orphan purge: nothing cancelled (orders={order_count})")

    net_after = connector.get_symbol_list_positions_net(symbol)
    print(f"list_positions net after: {net_after}")
    return 0 if not net_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
