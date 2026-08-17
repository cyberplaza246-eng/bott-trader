#!/usr/bin/env python
"""
Simple single-trade test via Rithmic.
Places one long or short trade based on current breakout conditions.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def main():
    print("=" * 60)
    print("  SINGLE TRADE TEST - Rithmic/Tradesea")
    print("=" * 60)
    
    from src.broker.rithmic_connector import RithmicConnector
    
    print("\nConnecting to Rithmic...")
    broker = RithmicConnector()
    broker.initialize()
    
    if not broker.connected:
        print("❌ Not connected!")
        return
    
    # Get account info
    acct = broker.get_account_info()
    print(f"✅ Connected | Balance: ${acct.get('balance', 0):,.2f}")
    
    # Get current price
    print("\nGetting MES price...")
    price = broker.get_latest_price('MES')
    if price:
        print(f"MES: Bid={price.get('bid')} Ask={price.get('ask')} Last={price.get('last')}")
        current = price.get('last', price.get('bid', 0))
    else:
        print("Using Yahoo fallback for price...")
        import yfinance as yf
        es = yf.Ticker("ES=F")
        current = es.info.get('regularMarketPrice', 6700)
        print(f"ES futures price: {current}")
    
    # Simple trade: Long 1 MES with 10 point stop, 20 point target
    direction = input("\nEnter direction (long/short/skip): ").strip().lower()
    
    if direction == 'skip':
        print("Skipping trade.")
        broker.shutdown()
        return
    
    if direction not in ['long', 'short']:
        print("Invalid direction. Use 'long' or 'short'")
        broker.shutdown()
        return
    
    # Calculate SL/TP
    if direction == 'long':
        entry = current + 0.25  # 1 tick above
        sl = entry - 10  # 10 point stop
        tp = entry + 20  # 20 point target
        order_type = 'buy'
    else:
        entry = current - 0.25  # 1 tick below
        sl = entry + 10  # 10 point stop
        tp = entry - 20  # 20 point target
        order_type = 'sell'
    
    print(f"\n{'='*40}")
    print(f"ORDER PREVIEW:")
    print(f"  Direction: {direction.upper()}")
    print(f"  Entry: ~{entry:.2f}")
    print(f"  Stop Loss: {sl:.2f}")
    print(f"  Take Profit: {tp:.2f}")
    print(f"  Risk: ${10 * 5:.2f} (10 pts × $5)")
    print(f"  Reward: ${20 * 5:.2f} (20 pts × $5)")
    print(f"{'='*40}")
    
    confirm = input("\nPlace this trade? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("Trade cancelled.")
        broker.shutdown()
        return
    
    print(f"\n⏳ Placing {direction} order...")
    
    result = broker.place_order(
        symbol='MES',
        order_type=order_type,
        size=1,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp
    )
    
    if result:
        print(f"✅ ORDER PLACED!")
        print(f"   Ticket: {result.get('ticket')}")
        print(f"   Check your Tradesea account!")
    else:
        print("❌ Order failed!")
    
    print("\nShutting down...")
    broker.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
