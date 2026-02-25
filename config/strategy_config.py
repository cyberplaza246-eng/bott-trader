"""
Strategy configuration for the AI trading bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MT5 Connection
MT5_ACCOUNT = os.getenv('MT5_ACCOUNT')
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5')

# Trading Parameters
RISK_PER_TRADE_PERCENT = float(os.getenv('RISK_PER_TRADE_PERCENT', 1.0))
MAX_CONCURRENT_TRADES = int(os.getenv('MAX_CONCURRENT_TRADES', 3))
DAILY_LOSS_LIMIT_PERCENT = float(os.getenv('DAILY_LOSS_LIMIT_PERCENT', 10))
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 50))

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'live', 'paper', 'backtest'
AUTOTRADING_ENABLED = TRADING_MODE == 'live'

# Currency Pairs to Trade
PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY']

# Timeframes (in minutes)
TIMEFRAMES = {
    'fast': 5,      # 5-minute candles
    'medium': 60,   # 1-hour candles
    'slow': 240,    # 4-hour candles
}

# AI Model Thresholds
ENSEMBLE_CONFIDENCE_THRESHOLD = float(os.getenv('ENSEMBLE_CONFIDENCE_THRESHOLD', 0.45))
MIN_MODELS_AGREEMENT = int(os.getenv('MIN_MODELS_AGREEMENT', 2))

# Technical Analysis Parameters
INDICATORS = {
    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER_BANDS': {'period': 20, 'std_dev': 2},
    'ATR': {'period': 14},
    'VOLUME_PERIOD': 20,
}

# Risk Management
STOP_LOSS_MULTIPLIER = 1.5  # ATR multiplier for stop loss
TAKE_PROFIT_RATIO = 2.0     # Risk:Reward ratio (1:2)
MAX_DAILY_LOSS_AMOUNT = INITIAL_BALANCE * (DAILY_LOSS_LIMIT_PERCENT / 100)

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = 'logs'
LOG_FILE = f'{LOG_DIR}/trading_bot.log'

# Backtesting
BACKTEST_START_DATE = '2022-01-01'
BACKTEST_END_DATE = '2024-12-31'
BACKTEST_LEVERAGE = 100

# Model Paths
LSTM_MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'

print(f"✅ Strategy Config Loaded - Mode: {TRADING_MODE}, Risk: {RISK_PER_TRADE_PERCENT}% per trade")
