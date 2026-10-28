#!/usr/bin/env python
"""
Main entry point for the trading bot
Usage: python -m scripts.run_bot [options]
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot import TradingBot
from config.strategy_config import TRADING_MODE
from src.utils.logger import bot_logger


def main():
    """Main entry point"""
    load_dotenv()
    
    bot_logger.info("=" * 70)
    bot_logger.info("🚀 AI-Powered Forex Trading Bot Starting...")
    bot_logger.info("=" * 70)
    
    # Get API keys
    newsapi_key = os.getenv('NEWSAPI_KEY')
    
    # Initialize bot
    bot = TradingBot(newsapi_key=newsapi_key, enable_dashboard=True)
    
    # Run bot
    try:
        bot.start()
    except KeyboardInterrupt:
        bot_logger.info("\nShutting down...")
        bot.stop()
    except Exception as e:
        bot_logger.error(f"Fatal error: {str(e)}", exc_info=True)
        bot.stop()


if __name__ == '__main__':
    main()
