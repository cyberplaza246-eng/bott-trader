#!/usr/bin/env python
"""
Test MT5 connection
Usage: python scripts/test_mt5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.broker.mt5_connector import MT5Connector
from src.utils.logger import bot_logger


def test_mt5():
    """Test MT5 connection"""
    load_dotenv()
    
    bot_logger.info("Testing MT5 connection...")
    
    try:
        connector = MT5Connector()
        
        # Get account info
        account_info = connector.get_account_info()
        if account_info:
            print("\n✅ MT5 Connection Successful!")
            print(f"  Account: {account_info['login']}")
            print(f"  Balance: ${account_info['balance']:,.2f}")
            print(f"  Equity: ${account_info['equity']:,.2f}")
            print(f"  Free Margin: ${account_info['margin_free']:,.2f}")
        
        # Get latest price
        price = connector.get_latest_price('EUR/USD')
        if price:
            print(f"\nLatest EUR/USD Price:")
            print(f"  Bid: {price['bid']:.5f}")
            print(f"  Ask: {price['ask']:.5f}")
        
        # Cleanup
        connector.shutdown()
        
    except Exception as e:
        bot_logger.error(f"MT5 Connection Failed: {str(e)}")
        print("\n❌ MT5 Connection Failed!")
        print(f"Error: {str(e)}")
        print("\nChecklist:")
        print("[ ] MT5 terminal is running")
        print("[ ] Account number (.env MT5_ACCOUNT) is correct")
        print("[ ] Password (.env MT5_PASSWORD) is correct")
        print("[ ] Server name (.env MT5_SERVER) is correct")
        print("[ ] Internet connection is stable")


if __name__ == '__main__':
    test_mt5()
