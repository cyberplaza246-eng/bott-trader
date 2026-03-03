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
MAX_CONCURRENT_TRADES = int(os.getenv('MAX_CONCURRENT_TRADES', 1))
DAILY_LOSS_LIMIT_PERCENT = float(os.getenv('DAILY_LOSS_LIMIT_PERCENT', 5))
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 50))

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'live', 'paper', 'backtest'
AUTOTRADING_ENABLED = TRADING_MODE == 'live'

# Currency Pairs to Trade (scalping focus)
PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY']

# Timeframes (in minutes) — dual-timeframe scalping
TIMEFRAMES = {
    'scalp_fast': 1,    # 1-minute candles (primary scalp)
    'scalp_slow': 5,    # 5-minute candles (confluence)
    'fast': 5,          # Kept for backward compat with ensemble models
    'medium': 60,       # 1-hour candles (trend context)
    'slow': 240,        # 4-hour candles (macro bias)
}

# ── Scalping-Specific Configuration ────────────────────────────────

# Per-pair ATR-based scalping parameters (no fixed pips)
# SL/TP are fully derived from ATR at trade time
SCALPING_PAIRS = {
    'EUR/USD': {
        'session_atr_min': 0.00040,    # Min ATR to trade (~4 pips)
        'spread_sim': 0.00015,         # Simulated spread (1.5 pips)
        'pip_size': 0.0001,
        'cooldown_seconds': 30,
        'max_hold_candles': 8,         # Time-exit: 8 candles max
    },
    'GBP/USD': {
        'session_atr_min': 0.00055,    # Min ATR to trade (~5.5 pips)
        'spread_sim': 0.00020,         # Simulated spread (2 pips)
        'pip_size': 0.0001,
        'cooldown_seconds': 30,
        'max_hold_candles': 8,
    },
    'USD/JPY': {
        'session_atr_min': 0.060,      # Min ATR to trade (~6 pips in JPY terms)
        'spread_sim': 0.020,           # Simulated spread (2 pips)
        'pip_size': 0.01,
        'cooldown_seconds': 30,
        'max_hold_candles': 8,
    },
}

# Session windows for scalping (UTC hours)
# start <= hour < end.  Use start=0, end=24 for 24/7.
SCALPING_SESSION_WINDOWS = {
    'EUR/USD': {'start': 7, 'end': 17},   # London open → NY overlap
    'GBP/USD': {'start': 7, 'end': 17},   # London open → NY overlap
    'USD/JPY': {'start': 0, 'end': 17},   # Asian + London + NY
}

# Spread limits for scalping (pips — tighter than swing)
SCALPING_SPREAD_LIMITS = {
    'EUR/USD': 2.0,   # Max 2.0 pips
    'GBP/USD': 2.5,   # Max 2.5 pips
    'USD/JPY': 2.5,   # Max 2.5 pips
}

# Confluence bonus: both timeframes agree → boost confidence
CONFLUENCE_BONUS = float(os.getenv('CONFLUENCE_BONUS', 0.15))   # +15%
DIVERGENCE_PENALTY = float(os.getenv('DIVERGENCE_PENALTY', 0.05))  # -5% (reduced to allow more trades)

# Optimal trading hours (bonus confidence during peak liquidity)
OPTIMAL_HOURS_UTC = list(range(8, 12))  # 08:00-11:59 UTC
OPTIMAL_HOUR_BONUS = float(os.getenv('OPTIMAL_HOUR_BONUS', 0.05))  # +5%

# AI Model Thresholds
# Ensemble conviction scores typicallsy range 0.10–0.40
# 0.15 = minimum confidence to enter a trade
_raw_threshold = float(os.getenv('ENSEMBLE_CONFIDENCE_THRESHOLD', 0.15))
ENSEMBLE_CONFIDENCE_THRESHOLD = max(_raw_threshold, 0.15)  # Hard floor: 15%
MIN_MODELS_AGREEMENT = int(os.getenv('MIN_MODELS_AGREEMENT', 3))

# Technical Analysis Parameters (ATR-centric scalping)
INDICATORS = {
    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},   # RSI 14 per strategy
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER_BANDS': {'period': 20, 'std_dev': 2},
    'ATR': {'period': 14},
    'ADX': {'period': 14, 'trend_threshold': 18},
    'EMA': {'short': 20, 'medium': 50, 'long': 200},
    'VOLUME_PERIOD': 20,
    'VOLUME_SPIKE_THRESHOLD': 1.2,   # Volume > 1.2x average
}

# Risk Management (ATR-based — no fixed pips)
STOP_LOSS_MULTIPLIER = 1.2   # SL = 1.2 x ATR(14) — wider for breathing room
TAKE_PROFIT_RATIO = 1.8      # TP = 1.8 x SL — better R:R
TP_EXPANDING = 2.0            # Wider TP in expanding volatility
TP_CONTRACTING = 1.5          # Tighter TP in contracting volatility
MIN_RISK_REWARD_RATIO = 1.5   # Minimum R:R to enter
MICRO_TAKE_PROFIT_RATIO = float(os.getenv('MICRO_TAKE_PROFIT_RATIO', 1.8))
HIGH_CERTAINTY_THRESHOLD = float(os.getenv('HIGH_CERTAINTY_THRESHOLD', 0.40))
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
