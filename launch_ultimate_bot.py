#!/usr/bin/env python3
"""
🚀 THE GREATEST AI AUTO TRADING BOT EVER 🚀

Combines:
- Deep RL (TD3) from Deep-RL-Stocks
- Multi-source data from HKUDS/AI-Trader
- Advanced ensemble methods
- Ultimate risk management
- Real-time execution

Usage:
python launch_ultimate_bot.py --mode live
python launch_ultimate_bot.py --mode paper
python launch_ultimate_bot.py --mode train
"""

import argparse
import os
import sys
import logging
import signal
import time
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.ultimate_trading_bot import UltimateTradingBot
from ai.rl_trainer import train_ultimate_model
from ai.data_integrator import UltimateDataIntegrator
from ai.risk_manager import UltimateRiskManager

class UltimateBotLauncher:
    """Launcher for the greatest AI trading bot ever"""

    def __init__(self):
        self.bot: Optional[UltimateTradingBot] = None
        self.setup_logging()

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/ultimate_bot.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('UltimateBot')

    def launch_live_trading(self):
        """Launch live trading mode"""
        self.logger.info("🚀 Launching ULTIMATE AI TRADING BOT - LIVE MODE 🚀")
        self.logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")

        # Confirm live trading
        if not self.confirm_live_trading():
            self.logger.info("Live trading cancelled by user")
            return

        try:
            self.bot = UltimateTradingBot()
            self.bot.start_trading()
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            if self.bot:
                self.bot.stop_trading()
        except Exception as e:
            self.logger.error(f"Fatal error in live trading: {e}")
            if self.bot:
                self.bot.stop_trading()
            raise

    def launch_paper_trading(self):
        """Launch paper trading mode"""
        self.logger.info("📊 Launching ULTIMATE AI TRADING BOT - PAPER TRADING MODE 📊")

        try:
            # Set paper trading environment
            os.environ['TRADING_MODE'] = 'paper'

            self.bot = UltimateTradingBot()
            self.bot.start_trading()
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            if self.bot:
                self.bot.stop_trading()
        except Exception as e:
            self.logger.error(f"Fatal error in paper trading: {e}")
            if self.bot:
                self.bot.stop_trading()
            raise

    def launch_training_mode(self):
        """Launch training mode for RL models"""
        self.logger.info("🎓 Launching ULTIMATE AI TRADING BOT - TRAINING MODE 🎓")

        try:
            # Train RL models
            symbols = ['MES', 'MNQ', 'SPY', 'QQQ']
            self.logger.info(f"Training RL models for symbols: {symbols}")

            agent, rewards = train_ultimate_model(symbols, episodes=1000)

            # Save trained models
            self.logger.info("Training completed! Models saved.")

            # Print final performance
            final_avg_reward = sum(rewards[-100:]) / 100
            self.logger.info(f"Final 100-episode average reward: {final_avg_reward:.4f}")

        except Exception as e:
            self.logger.error(f"Error in training mode: {e}")
            raise

    def launch_backtest_mode(self):
        """Launch backtesting mode"""
        self.logger.info("📈 Launching ULTIMATE AI TRADING BOT - BACKTEST MODE 📈")

        try:
            # Import backtesting module (would need to be created)
            from backtest.ultimate_backtester import UltimateBacktester

            backtester = UltimateBacktester()
            results = backtester.run_backtest()

            self.logger.info("Backtest completed!")
            self.logger.info(f"Results: {results}")

        except ImportError:
            self.logger.error("Backtesting module not implemented yet")
        except Exception as e:
            self.logger.error(f"Error in backtest mode: {e}")
            raise

    def launch_data_collection(self):
        """Launch data collection mode"""
        self.logger.info("📊 Launching ULTIMATE AI TRADING BOT - DATA COLLECTION MODE 📊")

        try:
            integrator = UltimateDataIntegrator()
            symbols = ['MES', 'MNQ', 'SPY', 'QQQ', 'IWM']

            self.logger.info(f"Collecting comprehensive data for {len(symbols)} symbols")

            multi_data = integrator.get_multi_asset_data(symbols)

            # Save data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/comprehensive_data_{timestamp}.json'

            with open(filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for symbol, data in multi_data.items():
                    serializable_data[symbol] = {}
                    for key, value in data.items():
                        if isinstance(value, pd.DataFrame):
                            serializable_data[symbol][key] = value.to_dict()
                        elif isinstance(value, np.ndarray):
                            serializable_data[symbol][key] = value.tolist()
                        else:
                            serializable_data[symbol][key] = value

                json.dump(serializable_data, f, indent=2, default=str)

            self.logger.info(f"Data collection completed! Saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            raise

    def confirm_live_trading(self) -> bool:
        """Get user confirmation for live trading"""
        print("\n" + "="*60)
        print("⚠️  LIVE TRADING CONFIRMATION ⚠️")
        print("="*60)
        print("You are about to start LIVE TRADING with real money!")
        print("This bot will:")
        print("• Execute real trades in live markets")
        print("• Risk actual capital")
        print("• Generate real profits or losses")
        print()
        print("Risk Management Features Active:")
        print("• Maximum 2% risk per trade")
        print("• Maximum 5% daily loss limit")
        print("• Maximum 10% drawdown protection")
        print("• Automatic emergency stops")
        print()
        response = input("Are you sure you want to proceed? (type 'YES' to confirm): ")

        return response.upper() == 'YES'

    def show_status(self):
        """Show current bot status"""
        print("\n" + "="*60)
        print("🤖 ULTIMATE AI TRADING BOT STATUS 🤖")
        print("="*60)

        if self.bot and hasattr(self.bot, 'is_running') and self.bot.is_running:
            print("✅ Bot Status: RUNNING")
            print(f"📊 Portfolio Value: ${self.bot.portfolio.cash:.2f}")
            print(f"📈 Open Positions: {len(self.bot.positions)}")
            print(f"🎯 Active Signals: {len(self.bot.last_signals)}")
        else:
            print("❌ Bot Status: STOPPED")

        print("\nAvailable Commands:")
        print("• python launch_ultimate_bot.py --mode live    # Live trading")
        print("• python launch_ultimate_bot.py --mode paper   # Paper trading")
        print("• python launch_ultimate_bot.py --mode train   # Train RL models")
        print("• python launch_ultimate_bot.py --mode backtest # Run backtests")
        print("• python launch_ultimate_bot.py --mode data    # Collect data")

def main():
    parser = argparse.ArgumentParser(description='The Greatest AI Auto Trading Bot Ever')
    parser.add_argument('--mode', required=True,
                       choices=['live', 'paper', 'train', 'backtest', 'data', 'status'],
                       help='Bot operating mode')
    parser.add_argument('--symbols', nargs='+', default=['MES', 'MNQ'],
                       help='Symbols to trade (default: MES MNQ)')
    parser.add_argument('--balance', type=float, default=50000,
                       help='Starting balance (default: 50000)')

    args = parser.parse_args()

    # Set environment variables
    os.environ['INITIAL_BALANCE'] = str(args.balance)
    os.environ['TRADING_SYMBOLS'] = ','.join(args.symbols)

    launcher = UltimateBotLauncher()

    try:
        if args.mode == 'live':
            launcher.launch_live_trading()
        elif args.mode == 'paper':
            launcher.launch_paper_trading()
        elif args.mode == 'train':
            launcher.launch_training_mode()
        elif args.mode == 'backtest':
            launcher.launch_backtest_mode()
        elif args.mode == 'data':
            launcher.launch_data_collection()
        elif args.mode == 'status':
            launcher.show_status()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    print("""
    🚀 THE GREATEST AI AUTO TRADING BOT EVER 🚀

    Combining the best from all repositories:
    • Deep RL (TD3) from Deep-RL-Stocks
    • Multi-source data from HKUDS/AI-Trader
    • Advanced ensemble methods
    • Ultimate risk management
    • Real-time execution

    """)

    main()