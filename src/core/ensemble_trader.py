"""
8-Model Ensemble Trading Decision System
Combines: LSTM, Sentiment, Technical, Volume, Multi-Timeframe, Support/Resistance, Candlestick Patterns, EMA Crossover
Adaptive weights via learning system
"""
from src.ai.lstm_predictor import LSTMPredictor
from src.ai.sentiment_analyzer import SentimentAnalyzer
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.multi_timeframe import MultiTimeframeAnalyzer
from src.ai.support_resistance import SupportResistanceDetector
from src.ai.candlestick_patterns import CandlestickPatternDetector
from src.ai.ema_crossover import EMACrossoverAnalyzer
from src.ai.adaptive_learner import AdaptiveLearner
from src.utils.logger import TradeLogger
from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD, MIN_MODELS_AGREEMENT


class EnsembleTrader:
    """8-model ensemble for high-confidence trading signals with adaptive learning"""
    
    def __init__(self, newsapi_key=None, broker=None):
        self.lstm = LSTMPredictor(lookback_window=60)
        self.sentiment = SentimentAnalyzer(newsapi_key=newsapi_key)
        self.technical = TechnicalAnalyzer()
        self.volume = VolumeAnalyzer(volume_period=20)
        self.multi_tf = MultiTimeframeAnalyzer()
        self.sr_detector = SupportResistanceDetector()
        self.candle_detector = CandlestickPatternDetector()
        self.ema_crossover = EMACrossoverAnalyzer()
        self.learner = AdaptiveLearner()
        self.broker = broker  # Needed for multi-timeframe data
        
        # Base weights (will be overridden by adaptive learner)
        self.model_weights = {
            'lstm': 0.18,
            'sentiment': 0.12,
            'technical': 0.14,
            'volume': 0.08,
            'multi_tf': 0.14,
            'support_resistance': 0.10,
            'candlestick': 0.12,
            'ema_crossover': 0.12,
        }
    
    def get_trading_signal(self, df, pair):
        """
        Generate trading signal from 7-model ensemble
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair (e.g., 'EUR/USD')
        
        Returns:
            {
                'signal': 'BUY', 'SELL', or 'SKIP',
                'confidence': 0.0-1.0,
                'models_agreement': number of models agreeing,
                'details': {...details from each model...}
            }
        """
        
        # Get adaptive weights
        adaptive_weights = self.learner.get_adjusted_weights()
        # Merge: use adaptive weights for known models, base for new ones
        weights = dict(self.model_weights)
        for k, v in adaptive_weights.items():
            if k in weights:
                weights[k] = v
        # Normalise
        w_sum = sum(weights.values())
        weights = {k: v / w_sum for k, v in weights.items()}
        
        # Calculate indicators first so all models use enriched data
        df_enriched = self.technical.calculate_indicators(df)
        
        # === Run all 7 models ===
        # 1. LSTM
        lstm_signal = self.lstm.predict_direction(df_enriched)
        
        # 2. Sentiment
        sentiment_signal = self.sentiment.get_pair_sentiment(pair)
        sentiment_signal_type = 'BUY' if sentiment_signal['sentiment_score'] > 0.2 else (
            'SELL' if sentiment_signal['sentiment_score'] < -0.2 else 'HOLD'
        )
        sentiment_confidence = abs(sentiment_signal['sentiment_score'])
        
        # 3. Technical
        technical_signal = self.technical.get_signal(df_enriched)
        
        # 4. Volume
        volume_signal = self.volume.get_volume_signal(df_enriched)
        
        # 5. Multi-Timeframe (needs broker for data)
        if self.broker:
            mtf_signal = self.multi_tf.analyze(self.broker, pair)
            mtf_signal_type = mtf_signal['signal']
            mtf_confidence = mtf_signal['confluence_score']
        else:
            mtf_signal_type = 'HOLD'
            mtf_confidence = 0.0
            mtf_signal = {'signal': 'HOLD', 'confluence_score': 0.0, 'reason': 'No broker'}
        
        # 6. Support/Resistance
        sr_signal = self.sr_detector.get_sr_signal(df_enriched)
        
        # 7. Candlestick Patterns
        candle_signal = self.candle_detector.get_pattern_signal(df_enriched)
        
        # 8. EMA Crossover (fast — pure math)
        ema_signal = self.ema_crossover.get_signal(df_enriched)
        
        # === Vote counting ===
        all_signals = {
            'lstm': {'signal': lstm_signal['signal'], 'confidence': lstm_signal['confidence']},
            'sentiment': {'signal': sentiment_signal_type, 'confidence': sentiment_confidence},
            'technical': {'signal': technical_signal['signal'], 'confidence': technical_signal['confidence']},
            'volume': {'signal': volume_signal['signal'], 'confidence': volume_signal['confidence']},
            'multi_tf': {'signal': mtf_signal_type, 'confidence': mtf_confidence},
            'support_resistance': {'signal': sr_signal['signal'], 'confidence': sr_signal['confidence']},
            'candlestick': {'signal': candle_signal['signal'], 'confidence': candle_signal['confidence']},
            'ema_crossover': {'signal': ema_signal['signal'], 'confidence': ema_signal['confidence']},
        }
        
        buy_votes = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in all_signals.values() if s['signal'] == 'SELL')
        
        # Determine final signal
        models_agreement = max(buy_votes, sell_votes)
        total_models = len(all_signals)
        
        # Use adaptive threshold
        min_agreement = MIN_MODELS_AGREEMENT
        
        if buy_votes > sell_votes and models_agreement >= min_agreement:
            final_signal = 'BUY'
        elif sell_votes > buy_votes and models_agreement >= min_agreement:
            final_signal = 'SELL'
        else:
            final_signal = 'SKIP'
        
        # Check S/R context: don't buy into resistance, don't sell into support
        if final_signal == 'BUY' and sr_signal.get('levels', {}).get('price_zone') == 'AT_RESISTANCE':
            final_signal = 'SKIP'  # Safety: don't buy at the ceiling
        if final_signal == 'SELL' and sr_signal.get('levels', {}).get('price_zone') == 'AT_SUPPORT':
            final_signal = 'SKIP'  # Safety: don't sell at the floor
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            all_signals[model]['confidence'] * weights.get(model, 0)
            for model in all_signals
        )
        
        # Generate reasoning
        reason_parts = [
            f"LSTM: {lstm_signal['signal']} ({lstm_signal['confidence']:.0%})",
            f"Sent: {sentiment_signal_type} ({sentiment_confidence:.0%})",
            f"Tech: {technical_signal['signal']} ({technical_signal['confidence']:.0%})",
            f"Vol: {volume_signal['signal']} ({volume_signal['confidence']:.0%})",
            f"MTF: {mtf_signal_type} ({mtf_confidence:.0%})",
            f"S/R: {sr_signal['signal']} ({sr_signal['confidence']:.0%})",
            f"Candle: {candle_signal['signal']} ({candle_signal['confidence']:.0%})",
            f"EMA: {ema_signal['signal']} ({ema_signal['confidence']:.0%})",
        ]
        
        detailed_reason = " | ".join(reason_parts)
        
        result = {
            'signal': final_signal,
            'confidence': weighted_confidence,
            'models_agreement': models_agreement,
            'total_models': total_models,
            'min_agreement_required': min_agreement,
            'detailed_reason': detailed_reason,
            'enriched_df': df_enriched,
            'models': {
                'lstm': lstm_signal,
                'sentiment': {
                    'signal': sentiment_signal_type,
                    'confidence': sentiment_confidence,
                    'score': sentiment_signal['sentiment_score'],
                    'news_count': sentiment_signal['news_count']
                },
                'technical': technical_signal,
                'volume': volume_signal,
                'multi_tf': mtf_signal,
                'support_resistance': sr_signal,
                'candlestick': candle_signal,
                'ema_crossover': ema_signal,
            },
            'sr_levels': sr_signal.get('levels', {}),
            'patterns': candle_signal.get('patterns', []),
        }
        
        # Log the signal
        if final_signal != 'SKIP':
            TradeLogger.log_signal(
                pair=pair,
                signal_type=final_signal,
                confidence=weighted_confidence,
                reason=detailed_reason,
                models_agreement=models_agreement
            )
        
        return result
    
    def should_trade(self, signal_result):
        """
        Determine if signal is strong enough to trade.
        Uses adaptive confidence threshold.
        """
        threshold = self.learner.get_adjusted_threshold()
        
        return (
            signal_result['signal'] != 'SKIP' and
            signal_result['confidence'] >= threshold and
            signal_result['models_agreement'] >= MIN_MODELS_AGREEMENT
        )
    
    def record_trade_result(self, trade_data: dict):
        """Pass trade result to adaptive learner."""
        self.learner.record_trade(trade_data)
