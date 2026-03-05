"""
MT5 Relay Server — Windows-side Flask API that bridges MetaTrader5
to the trading bot via HTTP.

Endpoints:
    GET  /ping               — health check
    GET  /account             — full account info (balance, leverage, margin)
    GET  /balance             — account balance
    GET  /equity              — account equity
    GET  /candles             — OHLCV candle data
    GET  /price               — latest bid/ask
    GET  /positions           — open positions
    GET  /history             — recently closed deals
    POST /order               — place a market order
    POST /close               — close a position by ticket (or first matching pair)
    POST /close_all           — close ALL open positions

Usage:
    1. pip install flask MetaTrader5 waitress
    2. python relay_server.py
    3. Bot connects to http://127.0.0.1:5555
"""

import sys
import os
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import MetaTrader5 as mt5

app = Flask(__name__)

# Optional bearer token for basic auth
RELAY_TOKEN = os.getenv("MT5_RELAY_TOKEN", "change-me-to-a-secret")

# Lock to serialise MT5 API calls (MT5 library is not thread-safe)
# Use RLock (reentrant) so _ensure_mt5() can be called inside a locked block
_mt5_lock = threading.RLock()


# ── Auth Middleware ───────────────────────────────────────────────────

@app.before_request
def check_auth():
    if request.path == "/ping":
        return None
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if RELAY_TOKEN and token != RELAY_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401


# ── Health ────────────────────────────────────────────────────────────

def _ensure_mt5():
    """Re-initialize MT5 if the connection was lost (e.g. terminal restart)."""
    with _mt5_lock:
        if mt5.terminal_info() is not None:
            return True
        print("⚠️  MT5 connection lost — attempting re-init...")
        if mt5.initialize():
            info = mt5.account_info()
            if info:
                print(f"✅ MT5 re-connected | Account: {info.login} | Balance: {info.balance}")
                return True
        print(f"❌ MT5 re-init failed: {mt5.last_error()}")
        return False

@app.route("/ping")
def ping():
    connected = _ensure_mt5()
    return jsonify({"status": "ok", "mt5_connected": connected})


# ── Account ───────────────────────────────────────────────────────────

@app.route("/account")
def account_info():
    with _mt5_lock:
        if not _ensure_mt5():
            return jsonify({"error": "MT5 not connected"}), 503
        info = mt5.account_info()
    if not info:
        return jsonify({"error": "Cannot read account info"}), 500
    return jsonify({
        "login": info.login,
        "server": info.server,
        "currency": info.currency,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "margin_free": info.margin_free,
        "profit": info.profit,
        "leverage": info.leverage,
    })


@app.route("/balance")
def balance():
    with _mt5_lock:
        info = mt5.account_info()
    return jsonify({"balance": info.balance if info else 0})


@app.route("/equity")
def equity():
    with _mt5_lock:
        info = mt5.account_info()
    return jsonify({"equity": info.equity if info else 0})


# ── Market Data ───────────────────────────────────────────────────────

@app.route("/candles")
def candles():
    pair = request.args.get("pair", "EURUSD")
    tf = int(request.args.get("timeframe", 60))
    count = int(request.args.get("count", 100))

    tf_map = {
        1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15, 60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4, 1440: mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(tf, mt5.TIMEFRAME_H1)

    with _mt5_lock:
        if not _ensure_mt5():
            return jsonify({"error": "MT5 not connected"}), 503

        symbol = pair.replace("/", "")
        if not mt5.symbol_select(symbol, True):
            symbol = pair
            if not mt5.symbol_select(symbol, True):
                return jsonify({"error": f"Symbol {pair} not found"}), 404

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        return jsonify({"error": "No candle data"}), 500

    candle_list = []
    for r in rates:
        candle_list.append({
            "datetime": datetime.utcfromtimestamp(r['time']).isoformat(),
            "open": float(r['open']),
            "high": float(r['high']),
            "low": float(r['low']),
            "close": float(r['close']),
            "volume": int(r['tick_volume']),
        })
    return jsonify({"candles": candle_list})


@app.route("/price")
def price():
    pair = request.args.get("pair", "EURUSD")
    symbol = pair.replace("/", "")
    with _mt5_lock:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            tick = mt5.symbol_info_tick(pair)
    if not tick:
        return jsonify({"error": f"No tick for {pair}"}), 404
    return jsonify({"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid})


# ── Positions ─────────────────────────────────────────────────────────

@app.route("/positions")
def positions():
    pair = request.args.get("pair")
    with _mt5_lock:
        if pair:
            symbol = pair.replace("/", "")
            pos = mt5.positions_get(symbol=symbol)
            if not pos:
                pos = mt5.positions_get(symbol=pair)
        else:
            pos = mt5.positions_get()

    if pos is None:
        pos = []

    result = []
    for p in pos:
        result.append({
            "ticket": p.ticket,
            "pair": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "open_price": p.price_open,
            "current_price": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "magic": p.magic,
            "comment": p.comment,
            "open_time": datetime.utcfromtimestamp(p.time).isoformat(),
        })
    return jsonify({"positions": result})


# ── Shared: Multi-filling-type Order Sender ───────────────────────────

def _try_order(req_base):
    """Try placing an order with multiple filling types for broker compat.
    
    If order with SL/TP fails, retries without SL/TP (modify later).
    Caller must hold _mt5_lock.
    Returns: (result, sl_tp_included) tuple or error string
    """
    filling_types = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN,
    ]

    # First pass: try with SL/TP included
    last_error = None
    for filling in filling_types:
        req = dict(req_base)
        req["type_filling"] = filling
        print(f"  Trying filling={filling} (with SL/TP)...")
        result = mt5.order_send(req)

        if result is None:
            last_error = f"order_send returned None: {mt5.last_error()}"
            print(f"  x {last_error}")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  OK  ticket={result.order} (SL/TP included in order)")
            return result, True

        last_error = f"{result.comment} (retcode={result.retcode})"
        print(f"  x {last_error}")
        if result.retcode not in [10030, 10033]:
            break

    # Second pass: retry WITHOUT SL/TP (some brokers reject SL/TP on market orders)
    has_sl_tp = req_base.get('sl', 0) > 0 or req_base.get('tp', 0) > 0
    if has_sl_tp:
        print(f"  Retrying WITHOUT SL/TP...")
        req_no_sltp = dict(req_base)
        req_no_sltp.pop('sl', None)
        req_no_sltp.pop('tp', None)
        for filling in filling_types:
            req = dict(req_no_sltp)
            req["type_filling"] = filling
            print(f"  Trying filling={filling} (no SL/TP)...")
            result = mt5.order_send(req)

            if result is None:
                last_error = f"order_send returned None: {mt5.last_error()}"
                print(f"  x {last_error}")
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  OK  ticket={result.order} (SL/TP will be added via modify)")
                return result, False

            last_error = f"{result.comment} (retcode={result.retcode})"
            print(f"  x {last_error}")
            if result.retcode not in [10030, 10033]:
                break

    return last_error, False


# ── Order Placement ───────────────────────────────────────────────────

@app.route("/order", methods=["POST"])
def place_order():
    data = request.json
    try:
        pair = data.get("pair", "EURUSD")
        order_type = data.get("type", "BUY")
        lot_size = float(data.get("lot_size", 0.01))
        sl = float(data.get("stop_loss", 0))
        tp = float(data.get("take_profit", 0))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field format: {str(e)}"}), 400

    with _mt5_lock:
        if not _ensure_mt5():
            return jsonify({"error": "MT5 not connected"}), 503

        symbol = pair.replace("/", "")
        if not mt5.symbol_select(symbol, True):
            symbol = pair
            if not mt5.symbol_select(symbol, True):
                return jsonify({"error": f"Symbol {pair} not found"}), 404

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return jsonify({"error": "Cannot get current price"}), 500

        action_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        entry_price = tick.ask if order_type == "BUY" else tick.bid

        # Try placing order WITH SL/TP first (fastest, avoids modify race)
        req_base = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": action_type,
            "price": entry_price,
            "sl": sl if sl > 0 else 0.0,
            "tp": tp if tp > 0 else 0.0,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        print(f"\n{'='*50}")
        print(f"ORDER: {symbol} {order_type} {lot_size} lots | SL={sl} TP={tp}")

        # Snapshot positions BEFORE placing order
        pre_positions = set()
        existing = mt5.positions_get(symbol=symbol)
        if existing:
            pre_positions = {p.ticket for p in existing}

        result, sl_tp_included = _try_order(req_base)

        if isinstance(result, str):
            return jsonify({"error": f"Order failed: {result}"}), 400

        # Find the actual position ticket by diffing before/after
        import time
        position_ticket = None
        sl_tp_set = sl_tp_included  # Already set if broker accepted SL/TP in order

        if (sl > 0 or tp > 0) and not sl_tp_included:
            # Find the NEW position by comparing before/after snapshots
            for attempt in range(5):
                time.sleep(0.3 * (attempt + 1))
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    new_positions = [p for p in positions if p.ticket not in pre_positions]
                    if new_positions:
                        position_ticket = new_positions[0].ticket
                        print(f"  Found NEW position ticket: {position_ticket} (attempt {attempt+1})")
                        break
                    # Fallback: try matching by order ID
                    for p in positions:
                        if p.ticket == result.order:
                            position_ticket = p.ticket
                            print(f"  Found position by order ticket: {position_ticket}")
                            break
                    if position_ticket:
                        break

            if not position_ticket:
                # Last resort: newest position with our magic number
                if positions:
                    bot_positions = [p for p in positions if p.magic == 234000]
                    if bot_positions:
                        position_ticket = max(p.ticket for p in bot_positions)
                        print(f"  ⚠️ Using newest magic position: {position_ticket}")
                    else:
                        position_ticket = result.order
                        print(f"  ⚠️ No magic positions found, using order ticket: {position_ticket}")
                else:
                    position_ticket = result.order
                    print(f"  ⚠️ No positions found, using order ticket: {position_ticket}")

            # Retry SL/TP modify up to 3 times
            for attempt in range(3):
                modify_req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": position_ticket,
                    "sl": sl,
                    "tp": tp,
                }
                print(f"  Adding SL/TP to ticket {position_ticket} (attempt {attempt+1})...")
                modify_result = mt5.order_send(modify_req)
                if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  ✅ SL/TP set: SL={sl} TP={tp}")
                    sl_tp_set = True
                    break
                else:
                    err = modify_result.comment if modify_result else mt5.last_error()
                    retcode = modify_result.retcode if modify_result else "N/A"
                    print(f"  ⚠️ SL/TP modify attempt {attempt+1} failed: {err} (retcode={retcode})")
                    time.sleep(0.5)

            if not sl_tp_set:
                print(f"  ❌ CRITICAL: SL/TP NOT SET after 3 attempts for {position_ticket}")
        elif sl_tp_included:
            # SL/TP was in the order — find the position ticket
            time.sleep(0.5)
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                new_positions = [p for p in positions if p.ticket not in pre_positions]
                if new_positions:
                    position_ticket = new_positions[0].ticket
                else:
                    position_ticket = result.order
            else:
                position_ticket = result.order
        else:
            position_ticket = result.order

        # Final verification: confirm SL/TP are actually on the position
        if position_ticket and (sl > 0 or tp > 0):
            time.sleep(0.3)
            verify = mt5.positions_get(ticket=position_ticket)
            if not verify:
                # Try without ticket filter
                verify_all = mt5.positions_get(symbol=symbol)
                if verify_all:
                    verify = [p for p in verify_all if p.ticket not in pre_positions]
            if verify:
                v = verify[0]
                if v.sl == 0 and v.tp == 0:
                    print(f"  ❌ VERIFY FAILED: SL/TP still 0! Attempting emergency modify...")
                    sl_tp_set = False
                    # Emergency modify attempt
                    for attempt in range(3):
                        mod_req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": v.ticket,
                            "sl": sl,
                            "tp": tp,
                        }
                        mod_result = mt5.order_send(mod_req)
                        if mod_result and mod_result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"  ✅ Emergency SL/TP set on ticket {v.ticket}")
                            sl_tp_set = True
                            position_ticket = v.ticket
                            break
                        time.sleep(0.5)
                else:
                    print(f"  ✅ VERIFIED: SL={v.sl} TP={v.tp}")
                    sl_tp_set = True

        return jsonify({
            "ticket": position_ticket or result.order,
            "pair": symbol,
            "type": order_type,
            "volume": lot_size,
            "price": entry_price,
            "sl_tp_set": sl_tp_set,
        })


# ── Close Position ────────────────────────────────────────────────────

@app.route("/close", methods=["POST"])
def close_position():
    """
    Close a position.
    Body:  {"ticket": 12345}                — close specific ticket
           {"pair": "EURUSD", "volume": 0.01}  — close first matching pair
    """
    data = request.json
    ticket = data.get("ticket")
    pair = data.get("pair", "EURUSD")
    volume = float(data.get("volume", 0))

    with _mt5_lock:
        # Locate the position -------------------------------------------------
        if ticket:
            all_pos = mt5.positions_get()
            pos = None
            if all_pos:
                for p in all_pos:
                    if p.ticket == int(ticket):
                        pos = p
                        break
            if not pos:
                return jsonify({"error": f"No position with ticket {ticket}"}), 404
        else:
            symbol = pair.replace("/", "")
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                positions = mt5.positions_get(symbol=pair)
            if not positions:
                return jsonify({"error": f"No open position for {pair}"}), 404
            pos = positions[0]

        # Build the close request ----------------------------------------------
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return jsonify({"error": f"Cannot get tick for {pos.symbol}"}), 500
        close_price = tick.bid if pos.type == 0 else tick.ask
        close_volume = volume if volume > 0 else pos.volume

        req_base = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": pos.ticket,          # ← critical: links to existing position
            "price": close_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        print(f"\n{'='*50}")
        print(f"CLOSE: ticket={pos.ticket} {pos.symbol} {close_volume} lots")
        result, _ = _try_order(req_base)

    if isinstance(result, str):
        return jsonify({"error": f"Close failed: {result}"}), 400

    return jsonify({"closed": pos.ticket, "order": result.order})


# ── Close ALL Positions ───────────────────────────────────────────────

@app.route("/close_all", methods=["POST"])
def close_all():
    """Close every open position on the account."""
    with _mt5_lock:
        all_positions = mt5.positions_get()
        if not all_positions:
            return jsonify({"message": "No positions to close", "results": []})

        results = []
        for pos in all_positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                results.append({"ticket": pos.ticket, "error": "No tick data"})
                continue
            close_price = tick.bid if pos.type == 0 else tick.ask

            req_base = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Bot Close All",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result, _ = _try_order(req_base)
            if isinstance(result, str):
                results.append({"ticket": pos.ticket, "error": result})
            else:
                results.append({"ticket": pos.ticket, "closed": True, "order": result.order})

    return jsonify({"results": results})


# ── Modify SL/TP ──────────────────────────────────────────────────────

@app.route("/modify", methods=["POST"])
def modify_position():
    """
    Modify SL and/or TP on an existing position.
    Body: {"ticket": 12345, "sl": 1.08500, "tp": 1.09200}
    """
    data = request.json
    ticket = data.get("ticket")
    new_sl = data.get("sl")
    new_tp = data.get("tp")

    if not ticket:
        return jsonify({"error": "ticket is required"}), 400

    with _mt5_lock:
        # Find the position
        all_pos = mt5.positions_get()
        pos = None
        if all_pos:
            for p in all_pos:
                if p.ticket == int(ticket):
                    pos = p
                    break
        if not pos:
            return jsonify({"error": f"No position with ticket {ticket}"}), 404

        # Use existing SL/TP if not provided
        sl_val = float(new_sl) if new_sl is not None else pos.sl
        tp_val = float(new_tp) if new_tp is not None else pos.tp

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": sl_val,
            "tp": tp_val,
        }

        print(f"\n{'='*50}")
        print(f"MODIFY: ticket={pos.ticket} {pos.symbol} | SL={sl_val} TP={tp_val}")
        result = mt5.order_send(req)

    if result is None:
        err = f"order_send returned None: {mt5.last_error()}"
        print(f"  x {err}")
        return jsonify({"error": err}), 400

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"{result.comment} (retcode={result.retcode})"
        print(f"  x {err}")
        return jsonify({"error": err}), 400

    print(f"  OK  modified ticket={pos.ticket}")
    return jsonify({
        "modified": pos.ticket,
        "sl": sl_val,
        "tp": tp_val,
    })


# ── Trade History ─────────────────────────────────────────────────────

@app.route("/history")
def trade_history():
    """
    Return recently closed deals (last N hours).
    Query params:
        hours  — look-back window (default 24)
    """
    hours = int(request.args.get("hours", 24))
    now = datetime.now()
    from_date = now - timedelta(hours=hours)

    # Get completed deals
    with _mt5_lock:
        deals = mt5.history_deals_get(from_date, now)
    if deals is None:
        return jsonify({"deals": [], "error": str(mt5.last_error())})

    closed = []
    for d in deals:
        # Type 0=BUY, 1=SELL — entry(0) vs exit(1) deals
        # We only want exit deals (entry=1) which represent closed trades
        if d.entry != 1:
            continue

        closed.append({
            "ticket": d.ticket,
            "order": d.order,
            "position_id": d.position_id,
            "pair": d.symbol,
            "type": "BUY" if d.type == 0 else "SELL",  # closing direction
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "commission": d.commission,
            "swap": d.swap,
            "time": datetime.fromtimestamp(d.time).isoformat(),
            "comment": d.comment,
        })

    return jsonify({"deals": closed})


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing MT5...")
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.account_info()
    print(f"Connected to MT5 | Account: {info.login} | "
          f"Balance: {info.balance} | Leverage: {info.leverage}:1")
    print(f"Relay server starting on http://0.0.0.0:5555")

    # Use waitress (production WSGI server) if available, otherwise
    # fall back to Flask's threaded dev server.  Either way the server
    # handles concurrent requests so one slow MT5 call can't block /ping.
    try:
        from waitress import serve
        print("Using waitress (multi-threaded production server)")
        serve(app, host="0.0.0.0", port=5555, threads=4)
    except ImportError:
        print("waitress not installed — using Flask threaded mode")
        print("  (pip install waitress for better performance)")
        app.run(host="0.0.0.0", port=5555, debug=False, threaded=True)
