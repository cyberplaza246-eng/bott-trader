#!/usr/bin/env python
"""
Auto place one trade based on breakout strategy signal.
Usage: python place_trade_now.py [long|short|auto]
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def main():
    direction = sys.argv[1] if len(sys.argv) > 1 else 'auto'
    
    print("=" * 60)
    print(f"  PLACING TRADE - Direction: {direction.upper()}")
    print("=" * 60)
    
    from src.broker.rithmic_connector import RithmicConnector
    
    print("\nConnecting to Rithmic...")
    broker = RithmicConnector()
    broker.initialize()
    
    if not broker.connected:
        print("❌ Not connected to Rithmic!")
        return 1
    
    acct = broker.get_account_info()
    print(f"✅ Connected | Balance: ${acct.get('balance', 0):,.2f}")
    
    # Get candles for signal
    print("\nGetting market data...")
    df = broker.get_candles('MES', 5, 100)
    
    if df is None or len(df) < 20:
        print("⚠️  Insufficient Rithmic data, using Yahoo Finance fallback...")
        import yfinance as yf
        data = yf.download('ES=F', period='5d', interval='5m', progress=False)
        if not data.empty:
            df = data.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            if 'datetime' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'})
            df = df.tail(100).reset_index(drop=True)
            print(f"✅ Yahoo Finance fallback: {len(df)} candles")
        else:
            print("❌ Could not get candles from Yahoo either")
            broker.shutdown()
            return 1
    
    # Calculate indicators
    df['high_10'] = df['high'].rolling(10).max().shift(1)
    df['low_10'] = df['low'].rolling(10).min().shift(1)
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    row = df.iloc[-1]
    price = row['close']
    atr = row['atr']
    ema = row['ema50']
    high_10 = row['high_10']
    low_10 = row['low_10']
    
    print(f"\nMarket State:")
    print(f"  Price: {price:.2f}")
    print(f"  ATR(14): {atr:.2f}")
    print(f"  EMA50: {ema:.2f}")
    print(f"  10-bar High: {high_10:.2f}")
    print(f"  10-bar Low: {low_10:.2f}")
    print(f"  Trend: {'UP' if price > ema else 'DOWN'}")
    
    # Determine direction
    if direction == 'auto':
        if price > ema and price > high_10:
            direction = 'long'
            print(f"\n📈 AUTO: Bullish setup - going LONG")
        elif price < ema and price < low_10:
            direction = 'short'
            print(f"\n📉 AUTO: Bearish setup - going SHORT")
        else:
            print(f"\n⏸️  AUTO: No clear signal - price between levels")
            direction = 'long' if price > ema else 'short'
            print(f"    Defaulting to trend direction: {direction.upper()}")
    
    # --- Use structure-based TP (resistance) via ScalpingAnalyzer ---
    from src.ai.scalping_analyzer import ScalpingAnalyzer
    analyzer = ScalpingAnalyzer()
    # Use last 70 bars for structure, as in original candle fetch
    df_struct = df.copy()
    # Analyzer expects 'BUY'/'SELL'
    analyzer_direction = 'BUY' if direction == 'long' else 'SELL'
    # Use 8pt SL as base, but let analyzer adjust if needed
    sl_dist = 8.0
    entry = price + 0.25 if direction == 'long' else price - 0.25
    # Calculate structure-based TP/SL
    rr = analyzer.calculate_risk_reward(
        df_struct,
        analyzer_direction,
        pair='MES',
        spread=None,
        tp_ratio_override=None,
        recent_sl_values=None
    )
    if rr is not None:
        sl = rr['stop_loss']
        tp = rr['take_profit']
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        reason = rr.get('tp_ratio_used', '')
        print(f"\n[Structure TP/SL] Reason: {rr.get('tp_ratio_used','')}, R:R={rr.get('rr_ratio',0):.2f}")
    else:
        # Fallback to fixed if analyzer fails
        print("\n[WARNING] Structure TP/SL unavailable, using fixed 8/16pt.")
        if direction == 'long':
            sl = entry - sl_dist
            tp = entry + 16.0
        else:
            sl = entry + sl_dist
            tp = entry - 16.0
        tp_dist = 16.0
        reason = 'fixed fallback'
    order_type = 'buy' if direction == 'long' else 'sell'
    
    print(f"\n{'='*50}")
    print(f"  ORDER:")
    print(f"  Direction: {direction.upper()}")
    print(f"  Entry: {entry:.2f}")
    print(f"  Stop Loss: {sl:.2f} ({sl_dist:.2f} pts away)")
    print(f"  Take Profit: {tp:.2f} ({tp_dist:.2f} pts away)")
    print(f"  Risk: ${sl_dist * 5:.2f}")
    print(f"  Reward: ${tp_dist * 5:.2f}")
    print(f"  TP/SL logic: {reason}")
    print(f"{'='*50}")
    
    print(f"\n⏳ Placing order...")
    
    result = broker.place_order(
        symbol='MES',
        order_type=order_type,
        size=1,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp
    )
    
    if result:
        print(f"\n✅ ORDER PLACED!")
        print(f"   Ticket: {result.get('ticket')}")
        print(f"\n🔔 Check your Tradesea account!")
    else:
        print(f"\n❌ Order failed!")
    
    broker.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
