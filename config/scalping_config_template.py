"""
Scalping Configuration Template

Copy this file and customize parameters for your specific needs.
Includes advanced tuning options for different market conditions.
"""

# ═══════════════════════════════════════════════════════════════════════════
# BASIC SCALPING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Pairs to trade
SCALPING_PAIRS = ['GBP/USD', 'EUR/USD']

# Timeframe (must be 5 minutes for this strategy)
SCALPING_TIMEFRAME = 'M5'  # 5-minute candles

# Maximum number of concurrent scalp trades
MAX_CONCURRENT_SCALPS = 2

# Signal cooldown (minimum time between signals per pair, in minutes)
SIGNAL_COOLDOWN_MINUTES = 2

# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

INDICATORS = {
    'RSI': {
        'period': 9,  # Short period for 5-min scalping
        'overbought': 70,
        'oversold': 30,
        'buy_range': [50, 55],      # Buy: RSI between these levels
        'sell_range': [45, 50],     # Sell: RSI between these levels
        'buy_minimum': 45,          # Don't buy if RSI below this
        'sell_maximum': 55,         # Don't sell if RSI above this
    },
    'EMA': {
        'entry_zone': 20,           # EMA 20 defines pullback zone
        'trend': 50,                # EMA 50 defines trend direction
        'bias': 200,                # EMA 200 defines macro bias
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# PAIR-SPECIFIC RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

PAIR_RISK_CONFIG = {
    'GBP/USD': {
        # Stop loss range (pips)
        'min_stop_loss_pips': 8,
        'max_stop_loss_pips': 12,
        'preferred_stop_loss_pips': 10,
        
        # Take profit ratios (Risk:Reward)
        'tp_ratio_aggressive': 2.0,      # 2R - aim for this
        'tp_ratio_conservative': 1.5,    # 1.5R - fallback
        
        # Position sizing
        'risk_percent_per_trade': 1.0,   # % of account to risk
        'max_lot_size': 0.05,
        
        # Time limits
        'max_hold_minutes': 15,          # Force-close after this time
        'optimal_hold_minutes': 5,       # Ideal hold time
    },
    'EUR/USD': {
        # EUR moves slower than GBP - adjust accordingly
        'min_stop_loss_pips': 6,
        'max_stop_loss_pips': 10,
        'preferred_stop_loss_pips': 8,
        
        'tp_ratio_aggressive': 1.5,      # Less aggressive for slower pair
        'tp_ratio_conservative': 1.0,
        
        'risk_percent_per_trade': 0.8,   # Slightly more conservative
        'max_lot_size': 0.03,
        
        'max_hold_minutes': 20,
        'optimal_hold_minutes': 8,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# TRADE ENTRY REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════

ENTRY_REQUIREMENTS = {
    # Minimum trend strength (EMA spread normalized)
    'min_trend_strength': 0.1,  # 10+ pip spread between EMAs
    
    # Pullback requirements
    'pullback_threshold': 15,   # Max distance from EMA 20 to enter
    'pullback_min_distance': 3, # Min pullback distance to confirm
    
    # Confluence scoring
    'min_confidence_score': 0.70,  # Minimum signal confidence
    
    # Session preferences
    'prefer_london_session': True,
    'allow_outside_london': True,  # Can trade outside session, but with warnings
    
    # News event buffer
    'hours_before_news': 0.5,  # Avoid entries 30 min before events
    'hours_after_news': 0.5,   # Avoid entries 30 min after events
}

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED FILTERING
# ═══════════════════════════════════════════════════════════════════════════

FILTERS = {
    # Spread limits (in pips)
    'max_spread_pips': {
        'GBP/USD': 3.5,
        'EUR/USD': 3.0,
    },
    
    # RSI divergence detection (optional extra confirmation)
    'require_rsi_divergence': False,  # Set to True for stricter entries
    
    # Candle confirmation strength
    'require_strong_candle_close': True,  # Require strong bullish/bearish close
    
    # Filter counter-trend noise
    'avoid_counter_trend_noise': True,  # Don't trade if strong movement against signal
}

# ═══════════════════════════════════════════════════════════════════════════
# EXIT STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

EXIT_RULES = {
    # Take profit levels
    'tp_management': 'two_level',  # 'single' or 'two_level' scale-out
    
    # Two-level exit (if using two_level mode)
    'tp1_ratio': 1.0,          # Close 50% at 1R
    'tp1_quantity_percent': 0.5,
    'tp2_ratio': 2.0,          # Close remaining 50% at 2R
    
    # Time-based forced exit
    'max_minutes_without_profit': 10,  # Exit if 10+ min in red
    'max_minutes_open': 15,             # Hard stop at 15 min
    
    # Trailing stop
    'use_trailing_stop': False,  # Set to True for dynamic trailing
    'trailing_stop_atr_multiple': 1.5,
}

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION MODES
# ═══════════════════════════════════════════════════════════════════════════

# Choose a mode based on market conditions
OPTIMIZATION_MODES = {
    'conservative': {
        'description': 'High win rate, smaller profits',
        'min_confidence': 0.80,         # Higher threshold
        'tp_ratios': [1.0, 1.5],       # Take profit early
        'max_sl_pips': 8,              # Tighter stops
        'require_divergence': True,    # Extra confirmation
    },
    'balanced': {
        'description': 'Default - Good balance of wins and profits',
        'min_confidence': 0.70,
        'tp_ratios': [1.5, 2.0],
        'max_sl_pips': 10,
        'require_divergence': False,
    },
    'aggressive': {
        'description': 'More trades, but more losses',
        'min_confidence': 0.60,         # Lower threshold
        'tp_ratios': [2.0, 3.0],       # Aim for bigger wins
        'max_sl_pips': 12,             # Wider stops
        'require_divergence': False,
    },
    'trend_following': {
        'description': 'Focus on strong trends only',
        'min_confidence': 0.75,
        'min_trend_strength': 0.15,    # Higher EMA spread requirement
        'tp_ratios': [1.5, 2.5],
        'max_sl_pips': 10,
    },
}

# Current mode (change to customize)
CURRENT_MODE = 'balanced'

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING & MONITORING
# ═══════════════════════════════════════════════════════════════════════════

LOGGING = {
    'log_every_candle': False,      # Verbose: log each candle analysis
    'log_signals_only': True,       # Log only when signals generated
    'log_trades': True,             # Log all trade executions
    'save_trade_csv': True,         # Save trades to CSV for analysis
    'debug_mode': False,            # Extra debug output
}

# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

BACKTEST_CONFIG = {
    'starting_balance': 1000,
    'slippage_pips': 0.5,           # Assume 0.5 pip slippage
    'commission_per_trade': 0.0,    # Some brokers charge commission
    'report_frequency': 'daily',    # Show daily P&L
    'plot_equity_curve': True,      # Generate equity curve chart
}

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLES OF CONFIGURATION PRESETS
# ═══════════════════════════════════════════════════════════════════════════

"""
EXAMPLE 1: Ultra-Conservative (Perfect for beginners)
────────────────────────────────────────────────────
- Min confidence: 0.80
- TP targets: 1R only (exit fast)
- SL: 8 pips (tight)
- Result: High win rate (60%+), small profits

EXAMPLE 2: London Session Only (Best liquidity)
────────────────────────────────────────────────
- Only trade 08:00-22:00 UTC
- Min confidence: 0.70
- TP targets: 1.5-2R
- SL: 10 pips

EXAMPLE 3: High-Frequency Scalp (Many trades)
────────────────────────────────────────────────
- Min confidence: 0.65
- Signal cooldown: 1 minute (aggressive)
- Max hold: 10 minutes
- TP: 0.5-1R (very quick)
- SL: 5-7 pips

EXAMPLE 4: Trend Rider (Follow strong trends)
────────────────────────────────────────────────
- Min trend strength: 0.20 (very strong trends)
- Min confidence: 0.75
- Only trade if EMA 50/200 spread > 15 pips
- TP: 2-3R (ride the trend)
"""

# ═══════════════════════════════════════════════════════════════════════════
# QUICK START: Just change the mode above!
# ═══════════════════════════════════════════════════════════════════════════

# Load appropriate settings based on mode
if CURRENT_MODE == 'balanced':
    # Default settings (already configured above)
    pass
elif CURRENT_MODE == 'conservative':
    ENTRY_REQUIREMENTS['min_confidence_score'] = 0.80
    FILTERS['require_rsi_divergence'] = True
elif CURRENT_MODE == 'aggressive':
    ENTRY_REQUIREMENTS['min_confidence_score'] = 0.60
elif CURRENT_MODE == 'trend_following':
    ENTRY_REQUIREMENTS['min_trend_strength'] = 0.15
