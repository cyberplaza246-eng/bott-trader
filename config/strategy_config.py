"""
Strategy configuration for the AI scalping bot
Optimized for 1M + 5M dual-timeframe scalping on EUR/USD & GBP/USD
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Broker selection: "mt5" (forex), "traderspost" (futures via TradersPost), "rithmic" (direct Rithmic)
BROKER_TYPE = os.getenv('BROKER_TYPE', 'mt5')

# Asset class: "forex" or "futures"
ASSET_CLASS = os.getenv('ASSET_CLASS', 'forex')

# MT5 Connection (forex mode)
MT5_ACCOUNT = os.getenv('MT5_ACCOUNT')
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5')

# Rithmic Connection (direct futures execution, preferred)
RITHMIC_USER_ID = os.getenv('RITHMIC_USER_ID', '')
RITHMIC_PASSWORD = os.getenv('RITHMIC_PASSWORD', '')
RITHMIC_SYSTEM = os.getenv('RITHMIC_SYSTEM', 'Rithmic Paper Trading')
RITHMIC_GATEWAY = os.getenv('RITHMIC_GATEWAY', '')

# TradersPost Connection (alternative futures mode — webhook-based)
TRADERSPOST_WEBHOOK_URL = os.getenv('TRADERSPOST_WEBHOOK_URL', '')
TRADERSPOST_API_KEY = os.getenv('TRADERSPOST_API_KEY', '')
TRADERSPOST_WEBHOOK_SECRET = os.getenv('TRADERSPOST_WEBHOOK_SECRET', '')
TRADERSPOST_ACCOUNT_ID = os.getenv('TRADERSPOST_ACCOUNT_ID', '')

# Trading Parameters
RISK_PER_TRADE_PERCENT = float(os.getenv('RISK_PER_TRADE_PERCENT', 1.0))
MAX_CONCURRENT_TRADES = int(os.getenv('MAX_CONCURRENT_TRADES', 1))
DAILY_LOSS_LIMIT_PERCENT = float(os.getenv('DAILY_LOSS_LIMIT_PERCENT', 3))
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 50000 if ASSET_CLASS == 'futures' else 50))

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'live')  # 'live', 'paper', 'backtest'
AUTOTRADING_ENABLED = TRADING_MODE == 'live'

# Symbols to Trade — auto-selected by asset class
FOREX_PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY']
FUTURES_SYMBOLS = ['MES', 'MNQ']  # Start with micros for eval
PAIRS = FUTURES_SYMBOLS if ASSET_CLASS == 'futures' else FOREX_PAIRS

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
        'session_atr_min': 0.00010,    # Min ATR 1.0 pip (lowered for frequency)
        'spread_sim': 0.00015,         # Simulated spread (1.5 pips)
        'pip_size': 0.0001,
        'cooldown_seconds': 30,
        'max_hold_candles': 30,        # Time-exit: 30 candles max (2.5h on 5M)
    },
    'GBP/USD': {
        'session_atr_min': 0.00015,    # Min ATR 1.5 pips
        'spread_sim': 0.00020,         # Simulated spread (2 pips)
        'pip_size': 0.0001,
        'cooldown_seconds': 30,
        'max_hold_candles': 30,
    },
    'USD/JPY': {
        'session_atr_min': 0.015,      # Min ATR 1.5 pips
        'spread_sim': 0.020,           # Simulated spread (2 pips)
        'pip_size': 0.01,
        'cooldown_seconds': 30,
        'max_hold_candles': 30,
    },
    # ── Futures scalping params ─────────────────────────────────
    'MES': {
        'session_atr_min': 1.0,        # Lowered ATR for quiet/Sunday trading
        'spread_sim': 0.25,            # 1 tick spread
        'pip_size': 0.25,              # tick_size
        'cooldown_seconds': 300,
        'max_hold_candles': 20,        # Tighter hold for futures
    },
    'MNQ': {
        'session_atr_min': 2.0,        # Lowered ATR for quiet/Sunday trading
        'spread_sim': 0.50,            # 2 ticks spread
        'pip_size': 0.25,
        'cooldown_seconds': 300,
        'max_hold_candles': 20,
    },
}

# Session windows for scalping (UTC hours)
# start <= hour < end.  start == end means 24/7.
SCALPING_SESSION_WINDOWS = {
    'EUR/USD': {'start': 0, 'end': 0},    # 24/7 trading (start==end means always)
    'GBP/USD': {'start': 0, 'end': 0},    # 24/7 trading
    'USD/JPY': {'start': 0, 'end': 0},    # 24/7 trading
    'MES': {'start': 13, 'end': 21},        # US extended session: 13-21 UTC (8am-4pm ET)
    'MNQ': {'start': 13, 'end': 21},
}

# Spread limits for scalping (in tick/pip units)
SCALPING_SPREAD_LIMITS = {
    'EUR/USD': 2.0,   # Max 2.0 pips
    'GBP/USD': 2.5,   # Max 2.5 pips
    'USD/JPY': 2.5,   # Max 2.5 pips
    'MES': 2.0,       # Max 2 ticks (0.50 points)
    'MNQ': 4.0,       # Max 4 ticks (1.0 point)
}

# Confluence bonus: both timeframes agree → boost confidence
CONFLUENCE_BONUS = float(os.getenv('CONFLUENCE_BONUS', 0.15))   # +15%
DIVERGENCE_PENALTY = float(os.getenv('DIVERGENCE_PENALTY', 0.12))  # -12% (was 5% — too lenient)

# Optimal trading hours (bonus confidence during peak liquidity)
# Forex: London Open 08-12 UTC | Futures: US RTH 13:30-20:00 UTC (9:30am-4pm ET)
if ASSET_CLASS == 'futures':
    OPTIMAL_HOURS_UTC = list(range(13, 20))   # US regular trading hours
else:
    OPTIMAL_HOURS_UTC = list(range(8, 12))    # London Open
OPTIMAL_HOUR_BONUS = float(os.getenv('OPTIMAL_HOUR_BONUS', 0.05))  # +5%

# AI Model Thresholds
# With sweep-gated architecture, sweep must fire (4-layer validation)
# then confidence is boosted/reduced by EMA + Technical confirmation.
# Futures tuning: 0.60 threshold + 3 model agreement was optimal in backtest.
ENSEMBLE_CONFIDENCE_THRESHOLD = float(os.getenv('ENSEMBLE_CONFIDENCE_THRESHOLD', 0.45))
MIN_MODELS_AGREEMENT = int(os.getenv('MIN_MODELS_AGREEMENT', 3))  # Sweep + at least 2 context models must agree

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
STOP_LOSS_MULTIPLIER = 0.8    # SL = 0.8 x ATR(14) — matches ScalpingAnalyzer.SL_ATR_MULT
TAKE_PROFIT_RATIO = 1.8      # TP = 1.8 x SL — better R:R
TP_EXPANDING = 2.0            # Wider TP in expanding volatility
TP_CONTRACTING = 1.5          # Tighter TP in contracting volatility
MIN_RISK_REWARD_RATIO = 1.5   # Minimum R:R to enter
MICRO_TAKE_PROFIT_RATIO = float(os.getenv('MICRO_TAKE_PROFIT_RATIO', 1.8))
HIGH_CERTAINTY_THRESHOLD = float(os.getenv('HIGH_CERTAINTY_THRESHOLD', 0.40))
MAX_DAILY_LOSS_AMOUNT = INITIAL_BALANCE * (DAILY_LOSS_LIMIT_PERCENT / 100)

# Logging
LOG_LEVEL = 'DEBUG'
LOG_DIR = 'logs'
LOG_FILE = f'{LOG_DIR}/trading_bot.log'

# Backtesting
BACKTEST_START_DATE = '2022-01-01'
BACKTEST_END_DATE = '2024-12-31'
BACKTEST_LEVERAGE = 100

# Walk-Forward Analysis
WALKFORWARD_SPLITS = int(os.getenv('WALKFORWARD_SPLITS', 5))
WALKFORWARD_TRAIN_PCT = float(os.getenv('WALKFORWARD_TRAIN_PCT', 0.70))
WALKFORWARD_MODE = os.getenv('WALKFORWARD_MODE', 'rolling')  # 'rolling' or 'anchored'

# Slippage Modeling (pips per trade — applied to both entry and exit)
BACKTEST_SLIPPAGE_PIPS = float(os.getenv('BACKTEST_SLIPPAGE_PIPS', 0.5))

# Reinforcement Learning
RL_ENABLED = os.getenv('RL_ENABLED', 'true').lower() == 'true'
RL_EXPLORATION_TRADES = int(os.getenv('RL_EXPLORATION_TRADES', 0))  # 0 = exploit-only (agent is pre-trained)

# Model Paths
LSTM_MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'

print(f"✅ Strategy Config Loaded - Mode: {TRADING_MODE}, Scalping 1M+5M, Pairs: {PAIRS}")
