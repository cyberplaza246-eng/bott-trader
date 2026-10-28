#!/usr/bin/env python
"""
One-time helper: list Rithmic accounts and trade routes from your .env credentials.
Connects like the live bot but does NOT place orders.

Usage:
    python scripts/list_rithmic_accounts.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows consoles often default to cp1252; connector may print unicode warnings.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from src.broker.rithmic_connector import (  # noqa: E402
    RITHMIC_GATEWAYS,
    RithmicConnector,
    _response_to_dict_safe,
)


def _route_label(route_name: str) -> str:
    if RithmicConnector._is_simulator_route_name(route_name):
        return "SIMULATOR"
    return "LIVE"


def _account_balance(
    connector: RithmicConnector, account_id: str
) -> Tuple[Optional[float], Optional[float]]:
    """Return (balance, equity) for one account, or (None, None) if unavailable."""
    if not connector.connected or not connector._client:
        return None, None
    try:
        summaries = connector._run_sync(
            connector._client.plants["pnl"].list_account_summary(account_id=account_id),
            timeout=15,
        )
    except Exception:
        return None, None
    if not summaries:
        return None, None
    data = _response_to_dict_safe(summaries[0])
    balance = data.get("account_balance")
    equity = data.get("cash_on_hand") or data.get("margin_balance")
    try:
        bal_f = float(balance) if balance is not None else None
    except (TypeError, ValueError):
        bal_f = None
    try:
        eq_f = float(equity) if equity is not None else None
    except (TypeError, ValueError):
        eq_f = None
    return bal_f, eq_f


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _group_routes_by_exchange(routes: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for route in routes:
        exchange = str(getattr(route, "exchange", "") or "?")
        grouped.setdefault(exchange, []).append(route)
    return dict(sorted(grouped.items()))


def _recommend_live_cme_route(routes: List[Any]) -> str:
    """First non-simulator CME route (preferred for .env recommendation)."""
    cme_routes = [
        r for r in routes if str(getattr(r, "exchange", "") or "").upper() == "CME"
    ]
    if not cme_routes:
        return ""
    live_routes = [
        r
        for r in cme_routes
        if not RithmicConnector._is_simulator_route_name(
            str(getattr(r, "trade_route", "") or "")
        )
    ]
    if live_routes:
        preferred = os.getenv("RITHMIC_TRADE_ROUTE", "").strip()
        sorted_routes = sorted(
            live_routes,
            key=lambda r: RithmicConnector._route_sort_key(
                r, preferred=preferred, reject_simulator=True
            ),
        )
        return str(getattr(sorted_routes[0], "trade_route", "") or "")
    return ""


def main() -> int:
    user = os.getenv("RITHMIC_USER_ID", "").strip()
    password = os.getenv("RITHMIC_PASSWORD", "").strip()
    system = os.getenv("RITHMIC_SYSTEM", "Rithmic Paper Trading")
    gateway = os.getenv("RITHMIC_GATEWAY", "").strip()
    gateway = gateway or RITHMIC_GATEWAYS.get(system, "")

    print("=" * 72)
    print("  Rithmic Account & Trade Route Lister")
    print("  (read-only — no orders placed)")
    print("=" * 72)

    if not user or not password:
        print("\nMissing credentials. Set in .env:")
        print("  RITHMIC_USER_ID=...")
        print("  RITHMIC_PASSWORD=...")
        return 1

    print(f"\nSystem:  {system}")
    print(f"Gateway: {gateway or '(unknown — check RITHMIC_SYSTEM / RITHMIC_GATEWAY)'}")

    connector = RithmicConnector(live_mode=True)
    try:
        print("\nConnecting (same path as live bot)...")
        connector.initialize()
        if not connector.connected:
            err = connector._last_connect_error or "unknown error"
            print(f"\nConnection failed: {err}")
            print("Close other R|Trader / NinjaTrader sessions, wait ~60s, retry.")
            return 1

        order_plant = connector._client.plants["order"]
        accounts = list(order_plant.accounts or [])
        routes = list(order_plant.trade_routes or [])

        print(f"\n{'─' * 72}")
        print(f"ACCOUNTS ({len(accounts)})")
        print(f"{'─' * 72}")
        if not accounts:
            print("  (none returned)")
        for acct in accounts:
            acct_id = str(getattr(acct, "account_id", "") or "")
            acct_name = str(getattr(acct, "account_name", "") or "unnamed")
            balance, equity = _account_balance(connector, acct_id)
            bal_str = _format_money(balance)
            eq_str = _format_money(equity)
            selected = "  <-- connector default" if acct_id == connector._get_account_id() else ""
            print(f"  {acct_id}")
            print(f"    name:    {acct_name}")
            print(f"    balance: {bal_str}   equity: {eq_str}{selected}")

        print(f"\n{'─' * 72}")
        print(f"TRADE ROUTES BY EXCHANGE ({len(routes)} total)")
        print(f"{'─' * 72}")
        if not routes:
            print("  (none returned)")
        for exchange, exchange_routes in _group_routes_by_exchange(routes).items():
            print(f"\n  [{exchange}]")
            for route in exchange_routes:
                route_name = str(getattr(route, "trade_route", "") or "")
                label = _route_label(route_name)
                is_default = bool(getattr(route, "is_default", False))
                status = str(getattr(route, "status", "") or "")
                chosen = (
                    "  <-- selected for this exchange"
                    if route_name == connector.get_trade_route(exchange)
                    else ""
                )
                marker = " *** SIMULATOR ***" if label == "SIMULATOR" else ""
                print(
                    f"    {route_name}  [{label}]"
                    f"  default={is_default}  status={status or 'n/a'}"
                    f"{marker}{chosen}"
                )

        recommended_account = connector._get_account_id() or ""
        recommended_route = _recommend_live_cme_route(routes)

        print(f"\n{'─' * 72}")
        print("RECOMMENDED .env LINES")
        print(f"{'─' * 72}")
        if recommended_account:
            print(f"RITHMIC_ACCOUNT_ID={recommended_account}")
        else:
            print("# No account_id resolved")

        if recommended_route:
            print(f"RITHMIC_TRADE_ROUTE={recommended_route}")
        else:
            print("# No live CME trade route found — only simulator routes available")
            sim_cme = [
                str(getattr(r, "trade_route", "") or "")
                for r in routes
                if str(getattr(r, "exchange", "") or "").upper() == "CME"
            ]
            if sim_cme:
                print(f"# CME simulator route(s): {', '.join(sim_cme)}")

        if connector.using_simulator_route:
            print(
                "\nWARNING: Connector would route LIVE orders to SIMULATOR with current settings."
            )
            print("Set RITHMIC_TRADE_ROUTE to your funded live route (not 'simulator').")

        print("\nDone.")
        return 0
    finally:
        connector.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
