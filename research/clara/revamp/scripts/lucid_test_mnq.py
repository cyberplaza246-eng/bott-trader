#!/usr/bin/env python
"""One MNQ market order to Lucid/Rithmic TEST simulator. Not the locked live loop.

Connects TICKER + ORDER plants only (no history/pnl). live_mode=False.

Usage (stop overnight paper first — one Rithmic session):
  python scripts/lucid_test_mnq.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from src.broker.rithmic_connector import (
    apply_lucid_ticker_order_plants,
    is_forced_logout_text,
)

apply_lucid_ticker_order_plants()

# June MNQ is expired in Aug 2026; keep ticker/orders on the live front month.
_ov = os.getenv("RITHMIC_CONTRACT_OVERRIDES", "")
if "MNQ=MNQM6" in _ov.upper():
    os.environ["RITHMIC_CONTRACT_OVERRIDES"] = "MNQ=MNQU6"


def _csv_last() -> float:
    from src.strategy.mnq_15m_ema_eod import load_1m_seed_csv

    path = os.path.join(ROOT, "data", "MNQ_1m.csv")
    df = load_1m_seed_csv(path)
    if df is None or df.empty:
        return 0.0
    col = "close" if "close" in df.columns else df.columns[-1]
    return float(df[col].iloc[-1])


def _order_tags(orders) -> set[str]:
    from src.broker.rithmic_connector import _response_to_dict_safe

    tags: set[str] = set()
    for order in orders or []:
        data = _response_to_dict_safe(order)
        tag = str(data.get("user_tag") or data.get("basket_id") or "").strip()
        if tag:
            tags.add(tag)
        oid = str(data.get("order_id") or data.get("id") or "").strip()
        if oid:
            tags.add(oid)
    return tags


def _snapshot_orders(broker, acct):
    from src.utils.redact import redact_secrets

    try:
        orders = broker._run_sync(
            broker._client.list_orders(account_id=acct),
            timeout=20,
        )
        return list(orders or []), None
    except Exception as exc:
        return [], redact_secrets(exc)


def main() -> int:
    from src.broker.rithmic_connector import RithmicConnector, _response_to_dict_safe
    from src.utils.redact import redact_secrets

    print("=" * 60)
    print("  LUCID TEST FILL — 1 MNQ to TEST simulator")
    print("  ticker+order plants  live_mode=False  (not funded)")
    print("=" * 60)

    broker = RithmicConnector(live_mode=False, quotes_only=False)
    broker._symbols_to_watch = ["MNQ"]
    broker._disable_yahoo_fallback = True
    broker.initialize()
    if not broker.connected:
        err = redact_secrets(broker._last_connect_error or "not connected")
        forced = is_forced_logout_text(err)
        print(f"Rithmic not connected. ForcedLogout={'yes' if forced else 'no'}")
        print(f"  {err}")
        return 1

    acct = broker._get_account_id() or "?"
    print(f"Connected account={acct} live_mode=False ticker+order")
    print("Waiting 15s for ForcedLogout / session steal...")
    time.sleep(15)

    if not broker.connected:
        print("FAILED — disconnected during settle wait (ForcedLogout yes)")
        try:
            broker.shutdown()
        except Exception:
            pass
        return 1

    before_orders, list_err = _snapshot_orders(broker, acct)
    if list_err and is_forced_logout_text(list_err):
        print("ForcedLogout yes — something else is still logged in. Stop.")
        print(f"  {list_err}")
        broker.shutdown()
        return 1
    if list_err:
        print(f"list_orders during settle: {list_err}")
        if is_forced_logout_text(list_err):
            broker.shutdown()
            return 1

    before_tags = _order_tags(before_orders)
    print(f"list_orders before: {len(before_orders)} (tags={len(before_tags)})")

    ticker_last = 0.0
    try:
        broker._ensure_market_data()
        for _ in range(20):
            q = broker.get_latest_price("MNQ") or {}
            ticker_last = float(q.get("last") or q.get("bid") or 0)
            if ticker_last > 0:
                break
            time.sleep(0.5)
    except Exception as exc:
        err = redact_secrets(exc)
        print(f"Ticker subscribe failed: {err}")
        if is_forced_logout_text(err):
            print("ForcedLogout yes — aborting before order.")
            broker.shutdown()
            return 1

    csv_last = _csv_last()
    print(f"MNQ last ticker={ticker_last:.2f} csv={csv_last:.2f}")
    if ticker_last <= 0:
        print("FAILED — no MNQ last print from ticker (not using CSV for the order).")
        broker.shutdown()
        return 1

    price = ticker_last
    sl_pts = 20.0
    tp_pts = 20.0
    sl = round(price - sl_pts, 2)
    tp = round(price + tp_pts, 2)
    print(f"BUY 1 MNQ market  SL {sl:.2f}  TP {tp:.2f}")
    print("Submitting to Lucid/Rithmic TEST simulator...")

    result = None
    try:
        result = broker.place_order(
            symbol="MNQ",
            order_type="buy",
            size=1,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
        )
    except Exception as exc:
        err = redact_secrets(exc)
        print(f"place_order raised: {err}")
        if is_forced_logout_text(err):
            print("ForcedLogout yes")
        broker.shutdown()
        return 1

    ticket = ""
    entry_ticket = ""
    if result:
        entry_ticket = str(result.get("ticket") or result.get("order_id") or "")
        ticket = entry_ticket

    found = None
    after_orders = []
    for _ in range(10):
        after_orders, list_err = _snapshot_orders(broker, acct)
        if list_err and is_forced_logout_text(list_err):
            print("ForcedLogout yes during list_orders. Stop.")
            print(f"  {list_err}")
            broker.shutdown()
            return 1
        after_tags = _order_tags(after_orders)
        for order in after_orders:
            data = _response_to_dict_safe(order)
            tag = str(data.get("user_tag") or data.get("basket_id") or "")
            notify = str(data.get("notify_type") or "").upper()
            status = str(data.get("status") or "").lower()
            side = str(data.get("transaction_type") or "").upper()
            if ticket and (tag == ticket or tag.startswith(ticket)):
                found = data
                if tag == ticket:
                    break
                continue
            if tag.startswith("bot_") and tag not in before_tags:
                if found is None or (ticket and tag == ticket):
                    found = data
                    if not ticket:
                        ticket = tag
                if ticket and tag == ticket:
                    break
                continue
            if (
                side == "BUY"
                and notify in ("FILL", "COMPLETE")
                and "fill" in status
                and tag not in before_tags
            ):
                found = data
                ticket = tag or ticket
                break
        if found:
            break
        new_tags = after_tags - before_tags
        if ticket and (ticket in after_tags or any(ticket in t for t in after_tags)):
            for order in after_orders:
                data = _response_to_dict_safe(order)
                tag = str(data.get("user_tag") or data.get("basket_id") or "")
                if tag == ticket:
                    found = data
                    break
            if found:
                break
        if new_tags and ticket and ticket in new_tags:
            break
        time.sleep(1.0)

    after_tags = _order_tags(after_orders)
    booked = bool(found) or (
        bool(ticket) and any(t == ticket or t.startswith(ticket) for t in after_tags)
    )
    if ticket and ticket not in after_tags and not found:
        print("FAILED — local submit only; ticket not in Lucid list_orders.")
        print(f"  local ticket={ticket} (not on broker)")
        print("  list_orders unchanged vs pre-order snapshot")
        broker.shutdown()
        return 1
    if not booked:
        print("FAILED — no new bot_* ticket or fill on Lucid list_orders.")
        if ticket:
            print(f"  local ticket={ticket} (not on broker)")
        broker.shutdown()
        return 1

    notify = str((found or {}).get("notify_type") or (found or {}).get("status") or "")
    fill_px = (found or {}).get("avg_fill_price") or (found or {}).get("price")
    print("ORDER SENT TO LUCID SIMULATOR")
    print(f"  ticket={ticket}")
    print(f"  status={notify} fill={fill_px}")
    filled = str((found or {}).get("notify_type") or "").upper() == "FILL" or (
        "fill" in str((found or {}).get("status") or "").lower()
    )
    if booked and not filled:
        # Working order on Lucid counts as booked; treat as fill if position appears.
        net_chk = broker.get_symbol_list_positions_net("MNQ")
        if net_chk not in (None, 0):
            filled = True
    sl_ok = bool(result and result.get("native_stop_attached"))
    net = broker.get_symbol_list_positions_net("MNQ")
    print(f"  list_positions net={net} native_stop={sl_ok}")

    if booked:
        if not sl_ok:
            print("Booked with no broker stop — flattening test position.")
        else:
            print("Fill/ticket confirmed — flattening so the session can be released.")
        flat = broker.flatten_symbol("MNQ", max_attempts=8, verify_delay=1.5)
        print(f"  flatten={flat.get('flat')} final_net={flat.get('final_net')}")
        net = broker.get_symbol_list_positions_net("MNQ")
        print(f"  net after flatten={net}")

    report_ticket = entry_ticket or ticket
    open_flat = "unknown" if net is None else ("flat" if net == 0 else f"open net={net}")
    print(
        f"RESULT connected=yes ForcedLogout=no ticker_last={ticker_last:.2f} "
        f"fill={'yes' if filled or booked else 'no'} ticket={report_ticket} {open_flat}"
    )
    broker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
