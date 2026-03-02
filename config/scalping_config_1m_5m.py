# Multi-Timeframe Scalping Configuration (1M & 5M)
# Use this config for the enhanced multi-timeframe bot

class MultiTimeframeScalpingConfig:
    """Configuration for 1-minute AND 5-minute scalping strategy"""
    
    # =========================================================================
    # MODE SELECTION
    # =========================================================================
    
    MODE = "balanced"  # Options: "conservative", "balanced", "aggressive"
    PROFIT_MODE = "quick_wins"  # Options: "normal", "quick_wins" (take small wins fast)
    
    # =========================================================================
    # CONSERVATIVE MODE
    # Only trade confluence (both 1M & 5M agree) — safest but fewer trades
    # =========================================================================
    
    CONSERVATIVE = {
        'name': 'Conservative Multi-TF',
        'description': 'Confluence only — highest probability trades',
        
        # Trading Rules
        'confluence_required': True,  # Only trade when both TF agree
        'min_1m_confidence': 0.80,    # Stricter than default
        'min_5m_confidence': 0.75,
        
        # Position Management
        'max_concurrent_scalps': 1,   # One position at a time
        'max_total_loss_daily': 100,  # Stop trading if down this much
        
        # 1-Minute Settings
        '1m_enabled': True,
        '1m_candles_history': 100,
        '1m_max_hold_seconds': 480,   # 8 minutes max
        '1m_gbpusd_sl_min': 5,
        '1m_gbpusd_sl_max': 8,
        '1m_eurusd_sl_min': 4,
        '1m_eurusd_sl_max': 6,
        
        # 5-Minute Settings
        '5m_enabled': True,
        '5m_candles_history': 200,
        '5m_max_hold_seconds': 1200,  # 20 minutes max
        '5m_gbpusd_sl_min': 8,
        '5m_gbpusd_sl_max': 12,
        '5m_eurusd_sl_min': 6,
        '5m_eurusd_sl_max': 10,
        
        # Account Risk
        'account_risk_percent': 1.0,  # Risk 1% per trade
        'account_tier': 'mini',       # micro/mini/standard/professional
    }
    
    # =========================================================================
    # BALANCED MODE
    # Trade single TF signals + confluence for higher frequency
    # =========================================================================
    
    BALANCED = {
        'name': 'Balanced Multi-TF',
        'description': 'Trade any confident setup on either TF',
        
        # Trading Rules
        'confluence_required': False,  # Trade single TF if confident enough
        'min_1m_confidence': 0.75,     # Standard for 1M
        'min_5m_confidence': 0.70,     # Standard for 5M
        
        # Position Management
        'max_concurrent_scalps': 2,   # Two positions allowed
        'max_total_loss_daily': 200,
        
        # 1-Minute Settings
        '1m_enabled': True,
        '1m_candles_history': 100,
        '1m_max_hold_seconds': 480,
        '1m_gbpusd_sl_min': 5,
        '1m_gbpusd_sl_max': 8,
        '1m_eurusd_sl_min': 4,
        '1m_eurusd_sl_max': 6,
        
        # 5-Minute Settings
        '5m_enabled': True,
        '5m_candles_history': 200,
        '5m_max_hold_seconds': 1200,
        '5m_gbpusd_sl_min': 8,
        '5m_gbpusd_sl_max': 12,
        '5m_eurusd_sl_min': 6,
        '5m_eurusd_sl_max': 10,
        
        # Account Risk
        'account_risk_percent': 1.5,
        'account_tier': 'mini',
    }
    
    # =========================================================================
    # QUICK WINS PROFIT MODE
    # Take small profits quickly — higher win rate, smaller average wins
    # =========================================================================
    
    QUICK_WINS = {
        'tp_ratios': [0.5, 0.8],       # 0.5R-0.8R (quick ~5-8 pip targets)
        'breakeven_r': 0.3,             # Move SL to breakeven at 0.3R profit
        'partial_close_pct': 0.40,      # Partial close at 40% of TP distance
        'trail_atr_mult': 0.5,          # Trail very tight at 0.5× ATR
        'max_hold_multiplier': 0.6,     # 60% of normal hold time
        'description': 'Take small wins fast — higher win rate, smaller targets',
    }
    
    NORMAL_PROFITS = {
        'tp_ratios': [1.0, 1.5],        # 1R-1.5R (standard targets)
        'breakeven_r': 0.5,             # Move SL to breakeven at 0.5R profit
        'partial_close_pct': 0.60,      # Partial close at 60% of TP distance
        'trail_atr_mult': 0.8,          # Trail at 0.8× ATR
        'max_hold_multiplier': 1.0,     # Normal hold time
        'description': 'Standard profit targets',
    }
    
    # =========================================================================
    # AGGRESSIVE MODE
    # Trade any signal including divergences — highest frequency
    # =========================================================================
    
    AGGRESSIVE = {
        'name': 'Aggressive Multi-TF',
        'description': 'High frequency: trade any TF, even divergences',
        
        # Trading Rules
        'confluence_required': False,
        'min_1m_confidence': 0.70,     # Lower threshold
        'min_5m_confidence': 0.65,
        'allow_divergent_trades': True,  # Trade even if 1M & 5M disagree
        
        # Position Management
        'max_concurrent_scalps': 3,   # Three positions allowed
        'max_total_loss_daily': 300,
        
        # 1-Minute Settings
        '1m_enabled': True,
        '1m_candles_history': 100,
        '1m_max_hold_seconds': 480,
        '1m_gbpusd_sl_min': 5,
        '1m_gbpusd_sl_max': 8,
        '1m_eurusd_sl_min': 4,
        '1m_eurusd_sl_max': 6,
        
        # 5-Minute Settings
        '5m_enabled': True,
        '5m_candles_history': 200,
        '5m_max_hold_seconds': 1200,
        '5m_gbpusd_sl_min': 8,
        '5m_gbpusd_sl_max': 12,
        '5m_eurusd_sl_min': 6,
        '5m_eurusd_sl_max': 10,
        
        # Account Risk
        'account_risk_percent': 2.0,
        'account_tier': 'standard',
    }
    
    # =========================================================================
    # PAIRS TO TRADE & RISK PARAMETERS
    # =========================================================================
    
    PAIRS = {
        'GBP/USD': {
            'enabled': True,
            'base_lot_size': 0.1,
            
            # 1-Minute specific
            '1m_sl_range': [5, 8],           # pips
            '1m_tp_ratio': [1.0, 1.5],       # 1R, 1.5R
            '1m_max_hold': 480,              # seconds (8 min)
            '1m_cooldown': 120,              # seconds (2 min)
            
            # 5-Minute specific
            '5m_sl_range': [8, 12],
            '5m_tp_ratio': [1.5, 2.0],
            '5m_max_hold': 1200,             # seconds (20 min)
            '5m_cooldown': 300,              # seconds (5 min)
            
            # Session filters
            'trading_sessions': ['london', 'newyork'],
            'avoid_times': ['05:00-08:00'],  # Avoid low-volume Asian
        },
        
        'EUR/USD': {
            'enabled': True,
            'base_lot_size': 0.15,  # Slightly larger (slower moves)
            
            # 1-Minute specific (tighter for slow EUR)
            '1m_sl_range': [4, 6],
            '1m_tp_ratio': [0.8, 1.2],
            '1m_max_hold': 480,
            '1m_cooldown': 120,
            '1m_extra_confirmation': True,  # Wait for RSI divergence
            
            # 5-Minute specific
            '5m_sl_range': [6, 10],
            '5m_tp_ratio': [1.0, 1.5],
            '5m_max_hold': 1200,
            '5m_cooldown': 300,
            
            # Session filters
            'trading_sessions': ['london', 'overlap'],
            'avoid_times': ['03:00-08:00'],  # Avoid quiet Asian
        },
    }
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    INDICATORS = {
        # RSI (Relative Strength Index)
        'rsi': {
            'period': 9,
            'overbought': 70,
            'oversold': 30,
            'midpoint': 50,
        },
        
        # EMA (Exponential Moving Averages)
        'ema': {
            'fast': 20,      # Pullback target
            'medium': 50,    # Trend filter
            'slow': 200,     # Major trend
        },
        
        # Confluence filtering
        'confluence': {
            'score_weight_1m': 0.4,   # 40% weight on 1M signal
            'score_weight_5m': 0.6,   # 60% weight on 5M signal
            'require_alignment': True, # Both TF must align for buy/sell
        },
    }
    
    # =========================================================================
    # ACCOUNT & RISK MANAGEMENT
    # =========================================================================
    
    ACCOUNT = {
        'account_type': 'demo',  # 'demo' for paper testing, 'real' for live
        'tier': 'mini',          # Account tier affects max position size
        
        # Risk per trade
        'risk_percent': 1.0,     # Risk 1% of account per trade
        'max_loss_daily': 100,   # Stop trading if down 100 USD
        'max_loss_weekly': 500,  # Weekly loss limit
        
        # Position limits
        'max_positions': 2,      # Max 2 open scalps simultaneously
        'max_positions_same_pair': 1,  # Only one GBP/USD at a time
        'min_distance_buy_sell': 10,   # Min 10 pips between buy & sell
        
        # Other
        'spread_tolerance_pips': 3.5,  # Warn if spread > 3.5 pips
    }
    
    # =========================================================================
    # EXECUTION & LOGGING
    # =========================================================================
    
    EXECUTION = {
        'live_trading_enabled': True,    # LIVE TRADING ENABLED
        'send_alerts': True,             # Alert on confluent signals
        'alert_destinations': ['log', 'console'],  # Where to send alerts
        'log_all_signals': True,         # Log every signal (even skipped)
        'log_confluence': True,          # Highlight confluence trades
    }
    
    # =========================================================================
    # CONFIGURATION PRESETS
    # =========================================================================
    
    @classmethod
    def get_active_config(cls):
        """Return active config based on MODE"""
        if cls.MODE == "conservative":
            return cls.CONSERVATIVE
        elif cls.MODE == "aggressive":
            return cls.AGGRESSIVE
        else:
            return cls.BALANCED
    
    @classmethod
    def get_pair_config(cls, pair):
        """Get config for specific pair"""
        return cls.PAIRS.get(pair)
    
    @classmethod
    def get_1m_sl_range(cls, pair):
        """Get 1-minute stop loss range for pair"""
        cfg = cls.PAIRS.get(pair)
        return cfg['1m_sl_range'] if cfg else [5, 8]
    
    @classmethod
    def get_5m_sl_range(cls, pair):
        """Get 5-minute stop loss range for pair"""
        cfg = cls.PAIRS.get(pair)
        return cfg['5m_sl_range'] if cfg else [8, 12]
    
    # =========================================================================
    # RECOMMENDED PRESETS
    # =========================================================================
    
    PRESETS = {
        'confluence_trader': {
            'name': 'Confluence-Only Trader',
            'description': 'Wait for both 1M & 5M agreement only',
            'config_overrides': {
                'confluence_required': True,
                'min_1m_confidence': 0.80,
                'min_5m_confidence': 0.75,
                'max_concurrent_scalps': 1,
                'account_risk_percent': 0.8,
            },
        },
        
        'multi_timeframe_trader': {
            'name': 'Multi-Timeframe Balanced',
            'description': 'Trade both 1M & 5M with single TF OK',
            'config_overrides': {
                'confluence_required': False,
                'min_1m_confidence': 0.75,
                'min_5m_confidence': 0.70,
                'max_concurrent_scalps': 2,
                'account_risk_percent': 1.5,
            },
        },
        
        '1m_only': {
            'name': '1-Minute Only Scalper',
            'description': 'High-frequency 1M scalping only',
            'config_overrides': {
                '1m_enabled': True,
                '5m_enabled': False,
                'min_1m_confidence': 0.70,
                'max_concurrent_scalps': 2,
                'account_risk_percent': 1.0,
            },
        },
        
        '5m_only': {
            'name': '5-Minute Only Scalper',
            'description': 'Traditional 5-minute scalping',
            'config_overrides': {
                '1m_enabled': False,
                '5m_enabled': True,
                'min_5m_confidence': 0.70,
                'max_concurrent_scalps': 2,
                'account_risk_percent': 1.5,
            },
        },
    }


# ============================================================================
# QUICK START
# ============================================================================
# 
# 1. Change MODE to your preference:
#    MODE = "conservative"  # Best for beginners
#    MODE = "balanced"      # Most traders use this
#    MODE = "aggressive"    # Experienced traders only
#
# 2. Set live trading when ready:
#    EXECUTION['live_trading_enabled'] = True
#
# 3. Run the bot:
#    python scripts/run_multi_timeframe_scalper.py paper
#    python scripts/run_multi_timeframe_scalper.py live
#
# ============================================================================


if __name__ == '__main__':
    # Print active configuration
    print(f"Active Mode: {MultiTimeframeScalpingConfig.MODE}")
    print(f"Config: {MultiTimeframeScalpingConfig.get_active_config()}")
    print(f"\nPairs Enabled:")
    for pair in MultiTimeframeScalpingConfig.PAIRS.values():
        if pair['enabled']:
            print(f"  - {pair}")
