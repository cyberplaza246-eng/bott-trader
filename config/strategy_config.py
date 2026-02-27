"""
Strategy configuration for the AI scalping bot
Optimized for 1M + 5M dual-timeframe scalping on EUR/USD & GBP/USD
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

# Currency Pairs to Trade (scalping focus)
PAIRS = ['EUR/USD', 'GBP/USD']

# Timeframes (in minutes) — dual-timeframe scalping
TIMEFRAMES = {
    'scalp_fast': 1,    # 1-minute candles (primary scalp)
    'scalp_slow': 5,    # 5-minute candles (confluence)
    'fast': 5,          # Kept for backward compat with ensemble models
    'medium': 60,       # 1-hour candles (trend context)
    'slow': 240,        # 4-hour candles (macro bias)
}

# ── Scalping-Specific Configuration ────────────────────────────────

# Per-pair per-timeframe scalping parameters
SCALPING_PAIRS = {
    'EUR/USD': {
        '1m': {
            'sl_pips_min': 4, 'sl_pips_max': 6,
            'tp_ratio': [1.0, 1.2],       # 1:1 R:R — needs >50% WR
            'max_hold_seconds': 600,       # 10 minutes
            'cooldown_seconds': 60,
        },
        '5m': {
            'sl_pips_min': 5, 'sl_pips_max': 8,
            'tp_ratio': [1.0, 1.2],       # 1:1 R:R — needs >50% WR
            'max_hold_seconds': 2400,      # 40 minutes (8 bars)
            'cooldown_seconds': 180,
        },
    },
    'GBP/USD': {
        '1m': {
            'sl_pips_min': 5, 'sl_pips_max': 8,
            'tp_ratio': [1.0, 1.2],
            'max_hold_seconds': 600,
            'cooldown_seconds': 60,
        },
        '5m': {
            'sl_pips_min': 6, 'sl_pips_max': 10,
            'tp_ratio': [1.0, 1.2],
            'max_hold_seconds': 3000,      # 50 minutes (10 bars)
            'cooldown_seconds': 180,
        },
    },
}

# Session windows for scalping (tighter than swing)
SCALPING_SESSION_WINDOWS = {
    'EUR/USD': {'start': 7, 'end': 22},   # Extended: London through NY close
    'GBP/USD': {'start': 7, 'end': 22},   # Extended: London through NY close
}

# Spread limits for scalping (pips — tighter than swing)
SCALPING_SPREAD_LIMITS = {
    'EUR/USD': 2.0,   # Max 2.0 pips
    'GBP/USD': 2.5,   # Max 2.5 pips
}

# Confluence bonus: both timeframes agree → boost confidence
CONFLUENCE_BONUS = float(os.getenv('CONFLUENCE_BONUS', 0.15))   # +15%
DIVERGENCE_PENALTY = float(os.getenv('DIVERGENCE_PENALTY', 0.05))  # -5% (reduced to allow more trades)

# Optimal trading hours (bonus confidence during peak liquidity)
OPTIMAL_HOURS_UTC = list(range(8, 12))  # 08:00-11:59 UTC
OPTIMAL_HOUR_BONUS = float(os.getenv('OPTIMAL_HOUR_BONUS', 0.05))  # +5%

# AI Model Thresholds (tuned for scalping speed)
ENSEMBLE_CONFIDENCE_THRESHOLD = float(os.getenv('ENSEMBLE_CONFIDENCE_THRESHOLD', 0.35))
MIN_MODELS_AGREEMENT = int(os.getenv('MIN_MODELS_AGREEMENT', 2))

# Technical Analysis Parameters (tuned for scalping)
INDICATORS = {
    'RSI': {'period': 9, 'overbought': 70, 'oversold': 30},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER_BANDS': {'period': 20, 'std_dev': 2},
    'ATR': {'period': 14},
    'VOLUME_PERIOD': 20,
}

# Risk Management
STOP_LOSS_MULTIPLIER = 1.5  # ATR multiplier for stop loss
TAKE_PROFIT_RATIO = 2.0     # Risk:Reward ratio (1:2)
MICRO_TAKE_PROFIT_RATIO = float(os.getenv('MICRO_TAKE_PROFIT_RATIO', 2.0))
HIGH_CERTAINTY_THRESHOLD = float(os.getenv('HIGH_CERTAINTY_THRESHOLD', 0.70))
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

print(f"✅ Strategy Config Loaded - Mode: {TRADING_MODE}, Scalping 1M+5M, Pairs: {PAIRS}")
