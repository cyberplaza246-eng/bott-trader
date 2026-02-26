"""
MT5 Relay Server — Windows-side Flask API that bridges MetaTrader5
to a Linux/Mac bot via HTTP (exposed with ngrok).

Endpoints:
    GET  /ping               — health check
    GET  /account             — full account info (balance, leverage, margin)
    GET  /balance             — account balance
    GET  /equity              — account equity
    GET  /candles             — OHLCV candle data
    GET  /price               — latest bid/ask
    GET  /positions           — open positions
    POST /order               — place a market order
    POST /close               — close a position by ticket (or first matching pair)
    POST /close_all           — close ALL open positions

Usage:
    1. pip install flask MetaTrader5
    2. python relay_server.py
    3. In another terminal: ngrok http 5555
    4. Set MT5_RELAY_URL in your .env to the ngrok URL
"""

import sys
import os
from datetime import datetime
from flask import Flask, request, jsonify
import MetaTrader5 as mt5

app = Flask(__name__)

# Optional bearer token for basic auth
RELAY_TOKEN = os.getenv("MT5_RELAY_TOKEN", "change-me-to-a-secret")


# ── Auth Middleware ───────────────────────────────────────────────────

@app.before_request
def check_auth():
    if request.path == "/ping":
        return None
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if RELAY_TOKEN and token != RELAY_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401


# ── Health ────────────────────────────────────────────────────────────

@app.route("/ping")
def ping():
    connected = mt5.terminal_info() is not None
    return jsonify({"status": "ok", "mt5_connected": connected})


# ── Account ───────────────────────────────────────────────────────────

@app.route("/account")
def account_info():
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
    info = mt5.account_info()
    return jsonify({"balance": info.balance if info else 0})


@app.route("/equity")
def equity():
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
            "open_time": datetime.utcfromtimestamp(p.time).isoformat(),
        })
    return jsonify({"positions": result})


# ── Shared: Multi-filling-type Order Sender ───────────────────────────

def _try_order(req_base):
    """Try placing an order with multiple filling types for broker compat."""
    filling_types = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN,
    ]
    last_error = None
    for filling in filling_types:
        req = dict(req_base)
        req["type_filling"] = filling
        print(f"  Trying filling={filling} ...")
        result = mt5.order_send(req)

        if result is None:
            last_error = f"order_send returned None: {mt5.last_error()}"
            print(f"  x {last_error}")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  OK  ticket={result.order}")
            return result

        last_error = f"{result.comment} (retcode={result.retcode})"
        print(f"  x {last_error}")
        # Only retry on filling-type errors
        if result.retcode not in [10030, 10033]:
            break

    return last_error


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

    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": action_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI Trading Bot",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    print(f"\n{'='*50}")
    print(f"ORDER: {symbol} {order_type} {lot_size} lots | SL={sl} TP={tp}")
    result = _try_order(req_base)

    if isinstance(result, str):
        return jsonify({"error": f"Order failed: {result}"}), 400

    return jsonify({
        "ticket": result.order,
        "pair": symbol,
        "type": order_type,
        "volume": lot_size,
        "price": entry_price,
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
    result = _try_order(req_base)

    if isinstance(result, str):
        return jsonify({"error": f"Close failed: {result}"}), 400

    return jsonify({"closed": pos.ticket, "order": result.order})


# ── Close ALL Positions ───────────────────────────────────────────────

@app.route("/close_all", methods=["POST"])
def close_all():
    """Close every open position on the account."""
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

        result = _try_order(req_base)
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
    print(f"  Expose with: ngrok http 5555")
    print(f"  Then set MT5_RELAY_URL in your .env")

    app.run(host="0.0.0.0", port=5555, debug=False)
