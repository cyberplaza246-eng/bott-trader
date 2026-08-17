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
        'confluence_required': True,
        'min_1m_confidence': 0.80,
        'min_5m_confidence': 0.75,
        'entry_threshold': 0.75,        # Tighter than default 0.70
        
        # Position Management
        'max_concurrent_scalps': 1,
        'max_total_loss_daily': 100,
        
        # ATR-based settings (no fixed pip SL)
        '1m_enabled': True,
        '1m_candles_history': 200,
        '1m_max_hold_candles': 6,       # Tighter time exit
        '5m_enabled': True,
        '5m_candles_history': 250,
        
        # Account Risk
        'account_risk_percent': 1.0,
        'account_tier': 'mini',
    }
    
    # =========================================================================
    # BALANCED MODE
    # Trade single TF signals + confluence for higher frequency
    # =========================================================================
    
    BALANCED = {
        'name': 'Balanced Multi-TF',
        'description': 'Trade any confident setup on either TF',
        
        # Trading Rules
        'confluence_required': False,
        'min_1m_confidence': 0.75,
        'min_5m_confidence': 0.70,
        'entry_threshold': 0.70,        # Default ATR entry threshold
        
        # Position Management
        'max_concurrent_scalps': 2,
        'max_total_loss_daily': 200,
        
        # ATR-based settings
        '1m_enabled': True,
        '1m_candles_history': 200,
        '1m_max_hold_candles': 8,       # Standard time exit
        '5m_enabled': True,
        '5m_candles_history': 250,
        
        # Account Risk
        'account_risk_percent': 1.5,
        'account_tier': 'mini',
    }
    
    # =========================================================================
    # QUICK WINS PROFIT MODE
    # Take small profits quickly — higher win rate, smaller average wins
    # =========================================================================
    
    QUICK_WINS = {
        'tp_ratios': [1.3, 1.3],       # Flat 1.3R for all regimes
        'breakeven_r': 0.8,             # Move SL to breakeven at 0.8R profit
        'partial_close_pct': 0.50,      # Partial close 50% at 1.0R
        'trail_atr_mult': 0.8,          # Trail at 0.8× ATR
        'max_hold_multiplier': 1.0,     # Full ATR-calculated hold time
        'description': 'ATR-adaptive quick wins — breakeven fast, trail remainder',
    }
    
    NORMAL_PROFITS = {
        'tp_ratios': [1.3, 1.3],        # Flat 1.3R for all regimes
        'breakeven_r': 0.8,             # Move SL to breakeven at 0.8R profit
        'partial_close_pct': 0.50,      # Partial close at 50% of TP distance
        'trail_atr_mult': 0.8,          # Trail at 0.8× ATR
        'max_hold_multiplier': 1.0,     # Normal hold time
        'description': 'ATR-adaptive standard targets',
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
        'min_1m_confidence': 0.70,
        'min_5m_confidence': 0.65,
        'entry_threshold': 0.65,        # Lower entry threshold
        'allow_divergent_trades': True,
        
        # Position Management
        'max_concurrent_scalps': 3,
        'max_total_loss_daily': 300,
        
        # ATR-based settings
        '1m_enabled': True,
        '1m_candles_history': 200,
        '1m_max_hold_candles': 10,      # Wider time exit
        '5m_enabled': True,
        '5m_candles_history': 250,
        
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
            'session_atr_min': 0.00055,     # Min ATR to trade (~5.5 pips)
            'spread_sim': 0.00020,          # Simulated spread (2 pips)
            'pip_size': 0.0001,
            
            # Hold limits (candle-based, not second-based)
            'max_hold_candles': 8,          # Time-exit after 8 candles
            'cooldown_seconds': 30,
            
            # Session filters
            'trading_sessions': ['london', 'newyork'],
        },
        
        'EUR/USD': {
            'enabled': True,
            'base_lot_size': 0.15,
            'session_atr_min': 0.00040,     # Min ATR to trade (~4 pips)
            'spread_sim': 0.00015,          # Simulated spread (1.5 pips)
            'pip_size': 0.0001,
            
            # Hold limits (candle-based)
            'max_hold_candles': 8,
            'cooldown_seconds': 30,
            
            # Session filters
            'trading_sessions': ['london', 'overlap'],
        },
    }
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    INDICATORS = {
        # RSI (Relative Strength Index)
        'rsi': {
            'period': 14,         # RSI(14) per ATR strategy
            'overbought': 70,
            'oversold': 30,
            'entry_low': 40,      # Pre-expansion zone lower
            'entry_high': 60,     # Pre-expansion zone upper
        },
        
        # EMA (Exponential Moving Averages)
        'ema': {
            'fast': 20,      # Pullback target
            'medium': 50,    # 5M bias filter
            'slow': 200,     # Major trend
        },
        
        # ATR (Average True Range) — drives everything
        'atr': {
            'period': 14,
            'sma_period': 5,           # Rolling ATR average
            'sl_multiplier': 0.8,      # SL = 0.8 x ATR (grid-search optimal)
            'tp_base_ratio': 1.3,      # TP = 1.3 x SL (flat)
            'tp_expanding': 1.3,       # Flat — regime variance removed
            'tp_contracting': 1.3,     # Flat — contracting regime skipped
            'exhaustion_mult': 2.0,    # Reject if candle > 2x ATR
            'spread_max_pct': 0.20,    # Spread must be < 20% of ATR
        },
        
        # ADX (Trend strength)
        'adx': {
            'period': 14,
            'strong_threshold': 25,
            'preferred_threshold': 18,
        },
        
        # Volume
        'volume': {
            'period': 20,
            'spike_threshold': 1.2,    # Volume > 1.2x average
        },
        
        # Micro-structure
        'structure': {
            'lookback': 5,             # 5-candle structure break
        },
        
        # Confluence filtering
        'confluence': {
            'score_weight_1m': 0.4,
            'score_weight_5m': 0.6,
            'require_alignment': True,
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
        """Get ATR-based SL config for pair (no fixed pip range)."""
        cfg = cls.PAIRS.get(pair)
        if cfg:
            return {
                'atr_multiplier': 0.8,
                'session_atr_min': cfg.get('session_atr_min', 0.00040),
            }
        return {'atr_multiplier': 0.8, 'session_atr_min': 0.00040}
    
    @classmethod
    def get_5m_sl_range(cls, pair):
        """Get ATR-based SL config for pair (no fixed pip range)."""
        cfg = cls.PAIRS.get(pair)
        if cfg:
            return {
                'atr_multiplier': 0.8,
                'session_atr_min': cfg.get('session_atr_min', 0.00040),
            }
        return {'atr_multiplier': 0.8, 'session_atr_min': 0.00040}
    
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
