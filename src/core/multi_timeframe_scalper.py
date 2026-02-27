"""
Multi-Timeframe Scalping Bot — 1M & 5M High-Frequency Strategy

Extends the scalping analyzer to support multiple timeframes:
  - 1M (1-minute): Tight scalps, 4-8 pips SL, 0.8-1.5R TP
  - 5M (5-minute): Standard scalps, 5-12 pips SL, 1-2R TP

Same core strategy (RSI9 + EMA pullbacks) with timeframe-specific parameters.
"""
import pandas as pd
import numpy as np
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.utils.logger import bot_logger


class MultiTimeframeScalpingAnalyzer:
    """Multi-timeframe scalping analyzer (1M & 5M support)"""
    
    # Timeframe-specific configuration
    TIMEFRAME_CONFIG = {
        'M1': {
            'name': '1-Minute',
            'candles_history': 100,  # Less history for 1M
            'PAIR_CONFIG': {
                'GBP/USD': {
                    'min_sl_pips': 5,
                    'max_sl_pips': 8,
                    'tp_ratio': [1.0, 1.5],      # 1R to 1.5R (tighter)
                    'min_rsi_for_buy': 50,
                    'max_rsi_for_buy': 55,
                    'min_rsi_for_sell': 45,
                    'max_rsi_for_sell': 50,
                    'max_hold_minutes': 5,       # Very short holds
                    'min_rsi_confirmation': 45,  # Stricter RSI filter
                },
                'EUR/USD': {
                    'min_sl_pips': 4,
                    'max_sl_pips': 6,
                    'tp_ratio': [0.8, 1.2],      # 0.8R to 1.2R (tight)
                    'min_rsi_for_buy': 50,
                    'max_rsi_for_buy': 55,
                    'min_rsi_for_sell': 45,
                    'max_rsi_for_sell': 50,
                    'max_hold_minutes': 8,
                    'min_rsi_confirmation': 45,
                    'extra_confirmation_required': True,  # EUR slower, need extra
                }
            }
        },
        'M5': {
            'name': '5-Minute',
            'candles_history': 200,  # Standard history for 5M
            'PAIR_CONFIG': {
                'GBP/USD': {
                    'min_sl_pips': 8,
                    'max_sl_pips': 12,
                    'tp_ratio': [1.5, 2.0],
                    'min_rsi_for_buy': 50,
                    'max_rsi_for_buy': 55,
                    'min_rsi_for_sell': 45,
                    'max_rsi_for_sell': 50,
                    'max_hold_minutes': 15,
                    'min_rsi_confirmation': 40,
                },
                'EUR/USD': {
                    'min_sl_pips': 6,
                    'max_sl_pips': 10,
                    'tp_ratio': [1.0, 1.5],
                    'min_rsi_for_buy': 50,
                    'max_rsi_for_buy': 55,
                    'min_rsi_for_sell': 45,
                    'max_rsi_for_sell': 50,
                    'max_hold_minutes': 20,
                    'min_rsi_confirmation': 40,
                    'extra_confirmation_required': True,
                }
            }
        }
    }
    
    def __init__(self):
        """Initialize multi-timeframe analyzer"""
        self.analyzer_5m = ScalpingAnalyzer()
        self.timeframe_config = self.TIMEFRAME_CONFIG
        
        bot_logger.info("🔪 Multi-Timeframe Scalping Analyzer initialized (1M & 5M)")
    
    def get_config_for_pair(self, pair, timeframe='M5'):
        """Get configuration for specific pair and timeframe.
        
        Args:
            pair: Currency pair (EUR/USD or GBP/USD)
            timeframe: M1 or M5
            
        Returns:
            dict: Configuration for this pair/timeframe
        """
        tf_config = self.TIMEFRAME_CONFIG.get(timeframe)
        if not tf_config:
            return self.TIMEFRAME_CONFIG['M5']['PAIR_CONFIG'].get(pair)
        
        return tf_config['PAIR_CONFIG'].get(pair)
    
    def get_signal_1m(self, df, pair='EUR/USD'):
        """Generate 1-minute scalping signal.
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair
            
        Returns:
            dict: Trading signal (modified for 1M timeframe)
        """
        signal = self.analyzer_5m.get_signal(df, pair)
        
        if signal['signal'] == 'SKIP':
            return signal
        
        # Get 1M-specific config
        config = self.get_config_for_pair(pair, 'M1')
        
        # For 1M: Stricter requirements
        # 1. Higher confidence threshold
        if signal['confidence'] < 0.75:
            signal['signal'] = 'SKIP'
            signal['reasons'].append(
                f"1M timeframe: Confidence {signal['confidence']:.2f} below 0.75 threshold"
            )
            return signal
        
        # 2. Recalculate R:R for 1M
        current_price = df['close'].iloc[-1]
        sl_pips = (config['min_sl_pips'] + config['max_sl_pips']) / 2
        
        if pair in ['EUR/USD', 'GBP/USD']:
            sl_price_distance = sl_pips / 10000
        else:
            sl_price_distance = sl_pips / 100
        
        if signal['signal'] == 'BUY':
            sl = current_price - sl_price_distance
            tp1 = current_price + (sl_price_distance * config['tp_ratio'][0])
            tp2 = current_price + (sl_price_distance * config['tp_ratio'][1])
        else:  # SELL
            sl = current_price + sl_price_distance
            tp1 = current_price - (sl_price_distance * config['tp_ratio'][0])
            tp2 = current_price - (sl_price_distance * config['tp_ratio'][1])
        
        signal['stop_loss'] = sl
        signal['take_profit'] = tp1
        signal['risk_reward'] = {
            'risk_pips': sl_pips,
            'reward_pips_1': abs(tp1 - current_price) * 10000,
            'reward_pips_2': abs(tp2 - current_price) * 10000,
        }
        
        signal['timeframe'] = '1M'
        signal['reasons'].append(f"1M scalp: SL {sl_pips:.0f}p, TP1 {signal['risk_reward']['reward_pips_1']:.0f}p")
        
        return signal
    
    def get_signal_5m(self, df, pair='EUR/USD'):
        """Generate 5-minute scalping signal.
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair
            
        Returns:
            dict: Trading signal (standard 5M)
        """
        signal = self.analyzer_5m.get_signal(df, pair)
        signal['timeframe'] = '5M'
        return signal
    
    def get_signal(self, df, pair='EUR/USD', timeframe='M5'):
        """Generate trading signal for specified timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair
            timeframe: 'M1' or 'M5'
            
        Returns:
            dict: Trading signal with timeframe-specific parameters
        """
        if timeframe == 'M1':
            return self.get_signal_1m(df, pair)
        else:
            return self.get_signal_5m(df, pair)


class MultiTimeframeScalpingTrader:
    """Trade execution for multiple timeframes"""
    
    def __init__(self, broker=None, risk_manager=None):
        """Initialize multi-timeframe trader.
        
        Args:
            broker: MT5Connector instance
            risk_manager: RiskManager instance
        """
        from src.core.scalping_trader import ScalpingTrader
        
        self.broker = broker
        self.risk_manager = risk_manager
        self.analyzer = MultiTimeframeScalpingAnalyzer()
        self.trader_1m = ScalpingTrader(broker=broker, risk_manager=risk_manager)
        self.trader_5m = ScalpingTrader(broker=broker, risk_manager=risk_manager)
        
        # Track active trades by timeframe
        self.active_trades_1m = {}
        self.active_trades_5m = {}
        
        bot_logger.info("🔪 Multi-Timeframe Scalping Trader initialized (1M & 5M)")
    
    def analyze_pair_multi_tf(self, df_1m, df_5m, pair):
        """Analyze pair on both timeframes.
        
        Args:
            df_1m: 1-minute OHLCV DataFrame
            df_5m: 5-minute OHLCV DataFrame
            pair: Currency pair
            
        Returns:
            dict: Signals from both timeframes
        """
        signal_1m = self.analyzer.get_signal(df_1m, pair, 'M1')
        signal_5m = self.analyzer.get_signal(df_5m, pair, 'M5')
        
        return {
            'signal_1m': signal_1m,
            'signal_5m': signal_5m,
            'pair': pair,
            'confluence': self._check_confluence(signal_1m, signal_5m),
        }
    
    def _check_confluence(self, signal_1m, signal_5m):
        """Check if both timeframes agree on direction.
        
        Args:
            signal_1m: 1M signal
            signal_5m: 5M signal
            
        Returns:
            dict: Confluence analysis
        """
        confluence = {
            'both_buy': signal_1m['signal'] == 'BUY' and signal_5m['signal'] == 'BUY',
            'both_sell': signal_1m['signal'] == 'SELL' and signal_5m['signal'] == 'SELL',
            'divergent': (signal_1m['signal'] in ['BUY', 'SELL'] and 
                         signal_5m['signal'] in ['BUY', 'SELL'] and 
                         signal_1m['signal'] != signal_5m['signal']),
            'score': 0.0,
        }
        
        if confluence['both_buy'] or confluence['both_sell']:
            confluence['score'] = 1.0
        elif confluence['divergent']:
            confluence['score'] = 0.0
        
        return confluence
    
    def process_candles_multi_tf(self, df_gbp_1m, df_eur_1m, df_gbp_5m, df_eur_5m):
        """Process new candles across all timeframes.
        
        Args:
            df_gbp_1m: GBP/USD 1M
            df_eur_1m: EUR/USD 1M
            df_gbp_5m: GBP/USD 5M
            df_eur_5m: EUR/USD 5M
            
        Returns:
            list: New trades opened
        """
        new_trades = []
        
        # GBP/USD analysis
        gbp_analysis = self.analyze_pair_multi_tf(df_gbp_1m, df_gbp_5m, 'GBP/USD')
        
        # EUR/USD analysis
        eur_analysis = self.analyze_pair_multi_tf(df_eur_1m, df_eur_5m, 'EUR/USD')
        
        # Priority: Trade on confluence > 5M only > 1M only
        # (But be cautious of divergence)
        
        # Execute based on confluence
        results = {
            'gbp_analysis': gbp_analysis,
            'eur_analysis': eur_analysis,
            'trades_opened': new_trades,
        }
        
        return results
    
    def get_summary(self):
        """Get current trading status across timeframes.
        
        Returns:
            dict: Summary of active trades by timeframe
        """
        return {
            'active_1m': len(self.active_trades_1m),
            'active_5m': len(self.active_trades_5m),
            'active_total': len(self.active_trades_1m) + len(self.active_trades_5m),
            'trades_1m': list(self.active_trades_1m.keys()),
            'trades_5m': list(self.active_trades_5m.keys()),
        }
