"""
Multi-Timeframe Scalping Bot — ATR-Centric 1M & 5M Strategy

Uses separate ScalpingAnalyzer instances for each timeframe.
5M provides directional bias (EMA 20/50), 1M provides entry signals.
All SL/TP are ATR-derived — no fixed pips anywhere.
"""
import pandas as pd
import numpy as np
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.utils.logger import bot_logger

# Import profit mode config (with fallback)
try:
    from config.scalping_config_1m_5m import MultiTimeframeScalpingConfig
    PROFIT_MODE = getattr(MultiTimeframeScalpingConfig, 'PROFIT_MODE', 'quick_wins')
except ImportError:
    PROFIT_MODE = 'quick_wins'


class MultiTimeframeScalpingAnalyzer:
    """Multi-timeframe scalping analyzer with separate 1M/5M instances.

    Flow:
      1. 5M analyzer detects directional bias (EMA 20/50 + ADX)
      2. 1M analyzer detects pullback entry within bias direction
      3. All SL/TP come from ScalpingAnalyzer.calculate_risk_reward()
    """

    def __init__(self, profit_mode=None):
        """Initialize with separate analyzers for each timeframe."""
        mode = profit_mode if profit_mode else PROFIT_MODE

        # Separate instances — each manages its own ATR/indicator state
        self.analyzer_1m = ScalpingAnalyzer(profit_mode=mode, timeframe='1m')
        self.analyzer_5m = ScalpingAnalyzer(profit_mode=mode, timeframe='5m')
        self.profit_mode = mode

        mode_label = "QUICK_WINS" if mode == 'quick_wins' else "NORMAL"
        bot_logger.info(
            f"🔪 Multi-TF Scalping Analyzer initialized (ATR-centric, 1M & 5M) [{mode_label} mode]"
        )

    def get_signal_1m(self, df_1m, pair='EUR/USD', df_5m=None, spread=None,
                      recent_sl_values=None):
        """Generate 1-minute scalping signal with 5M bias.

        This is the primary entry path: 5M provides bias, 1M provides timing.

        Args:
            df_1m: 1M candle DataFrame (200+ rows)
            pair: Currency pair
            df_5m: 5M candle DataFrame for bias detection
            spread: Actual broker spread
            recent_sl_values: Recent SL values for median check

        Returns:
            dict: Full signal with ATR-based SL/TP
        """
        signal = self.analyzer_1m.get_signal(
            df_1m, pair, timeframe='1m',
            df_5m=df_5m, spread=spread,
            recent_sl_values=recent_sl_values,
        )
        signal['timeframe'] = '1M'
        return signal

    def get_signal_5m(self, df_5m, pair='EUR/USD', spread=None,
                      recent_sl_values=None):
        """Generate 5-minute scalping signal (standalone).

        Used when no 1M data is available or for 5M-only mode.

        Args:
            df_5m: 5M candle DataFrame (200+ rows)
            pair: Currency pair
            spread: Actual broker spread
            recent_sl_values: Recent SL values for median check

        Returns:
            dict: Full signal with ATR-based SL/TP
        """
        signal = self.analyzer_5m.get_signal(
            df_5m, pair, timeframe='5m',
            spread=spread,
            recent_sl_values=recent_sl_values,
        )
        signal['timeframe'] = '5M'
        return signal

    def get_signal(self, df, pair='EUR/USD', timeframe='M5', df_5m=None,
                   spread=None, recent_sl_values=None):
        """Generate trading signal for specified timeframe.

        Args:
            df: OHLCV DataFrame for the primary timeframe
            pair: Currency pair
            timeframe: 'M1' or 'M5'
            df_5m: Optional 5M data for M1 bias detection
            spread: Actual broker spread
            recent_sl_values: Recent SL values for median check

        Returns:
            dict: Trading signal with ATR-based SL/TP
        """
        if timeframe == 'M1':
            return self.get_signal_1m(df, pair, df_5m=df_5m, spread=spread,
                                      recent_sl_values=recent_sl_values)
        else:
            return self.get_signal_5m(df, pair, spread=spread,
                                      recent_sl_values=recent_sl_values)


class MultiTimeframeScalpingTrader:
    """Trade execution for multiple timeframes — ATR-centric."""

    def __init__(self, broker=None, risk_manager=None, profit_mode=None):
        from src.core.scalping_trader import ScalpingTrader

        mode = profit_mode if profit_mode else PROFIT_MODE

        self.broker = broker
        self.risk_manager = risk_manager
        self.profit_mode = mode
        self.analyzer = MultiTimeframeScalpingAnalyzer(profit_mode=mode)
        self.trader_1m = ScalpingTrader(broker=broker, risk_manager=risk_manager, profit_mode=mode)
        self.trader_5m = ScalpingTrader(broker=broker, risk_manager=risk_manager, profit_mode=mode)

        # Track active trades by timeframe
        self.active_trades_1m = {}
        self.active_trades_5m = {}

        mode_label = "QUICK_WINS" if mode == 'quick_wins' else "NORMAL"
        bot_logger.info(
            f"🔪 Multi-TF Scalping Trader initialized (ATR-centric, 1M & 5M) [{mode_label} mode]"
        )

    def set_profit_mode(self, mode):
        """Switch profit mode at runtime."""
        self.profit_mode = mode
        self.analyzer.profit_mode = mode
        self.analyzer.analyzer_1m.profit_mode = mode
        self.analyzer.analyzer_5m.profit_mode = mode
        self.trader_1m.set_profit_mode(mode)
        self.trader_5m.set_profit_mode(mode)

        mode_label = "QUICK_WINS" if mode == 'quick_wins' else "NORMAL"
        bot_logger.info(f"🔄 Multi-TF profit mode changed to [{mode_label}]")

    def analyze_pair_multi_tf(self, df_1m, df_5m, pair, spread=None,
                               recent_sl_values=None):
        """Analyze pair on both timeframes.

        Primary flow: 5M bias → 1M entry (ATR-based SL/TP)
        """
        # 1M signal with 5M bias
        signal_1m = self.analyzer.get_signal_1m(
            df_1m, pair, df_5m=df_5m, spread=spread,
            recent_sl_values=recent_sl_values,
        )
        # 5M standalone (backup)
        signal_5m = self.analyzer.get_signal_5m(
            df_5m, pair, spread=spread,
            recent_sl_values=recent_sl_values,
        )

        return {
            'signal_1m': signal_1m,
            'signal_5m': signal_5m,
            'pair': pair,
            'confluence': self._check_confluence(signal_1m, signal_5m),
        }

    def _check_confluence(self, signal_1m, signal_5m):
        """Check if both timeframes agree on direction."""
        s1 = signal_1m.get('signal', 'SKIP')
        s5 = signal_5m.get('signal', 'SKIP')

        both_buy = s1 == 'BUY' and s5 == 'BUY'
        both_sell = s1 == 'SELL' and s5 == 'SELL'
        divergent = (
            s1 in ('BUY', 'SELL') and s5 in ('BUY', 'SELL') and s1 != s5
        )

        score = 1.0 if (both_buy or both_sell) else (0.0 if divergent else 0.5)

        return {
            'both_buy': both_buy,
            'both_sell': both_sell,
            'divergent': divergent,
            'score': score,
        }

    def process_candles_multi_tf(self, data_by_pair=None, spreads=None,
                                  # Legacy positional args for backward compat
                                  df_gbp_1m=None, df_eur_1m=None,
                                  df_gbp_5m=None, df_eur_5m=None,
                                  spread_gbp=None, spread_eur=None):
        """Process new candles across all timeframes.
        
        Args:
            data_by_pair: dict mapping pair -> {'1m': df, '5m': df}
            spreads: dict mapping pair -> spread value
        """
        new_trades = []
        results = {}

        # Support legacy call signature
        if data_by_pair is None:
            data_by_pair = {}
            if df_gbp_1m is not None or df_gbp_5m is not None:
                data_by_pair['GBP/USD'] = {'1m': df_gbp_1m, '5m': df_gbp_5m}
            if df_eur_1m is not None or df_eur_5m is not None:
                data_by_pair['EUR/USD'] = {'1m': df_eur_1m, '5m': df_eur_5m}
            spreads = spreads or {}
            if spread_gbp is not None:
                spreads['GBP/USD'] = spread_gbp
            if spread_eur is not None:
                spreads['EUR/USD'] = spread_eur

        spreads = spreads or {}
        for pair, dfs in data_by_pair.items():
            df_1m = dfs.get('1m')
            df_5m = dfs.get('5m')
            analysis = self.analyze_pair_multi_tf(
                df_1m, df_5m, pair, spread=spreads.get(pair),
            )
            results[f'{pair}_analysis'] = analysis

        # Backward-compat keys
        if 'GBP/USD_analysis' in results:
            results['gbp_analysis'] = results['GBP/USD_analysis']
        if 'EUR/USD_analysis' in results:
            results['eur_analysis'] = results['EUR/USD_analysis']

        results['trades_opened'] = new_trades
        return results

    def get_summary(self):
        """Get current trading status across timeframes."""
        return {
            'active_1m': len(self.active_trades_1m),
            'active_5m': len(self.active_trades_5m),
            'active_total': len(self.active_trades_1m) + len(self.active_trades_5m),
            'trades_1m': list(self.active_trades_1m.keys()),
            'trades_5m': list(self.active_trades_5m.keys()),
        }
