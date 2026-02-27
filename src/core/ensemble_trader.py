"""
8-Model Ensemble Trading Decision System — God Tier v2

Combines: LSTM, Sentiment, Technical, Volume, Multi-Timeframe, S/R, Candlestick, EMA Crossover
With: Regime-aware voting, conviction scaling, confluence bonuses, pair-specific weights

LSTM is optional — if TensorFlow is not installed, the ensemble
automatically runs with 7 models and redistributes its weight.
"""
import pandas as pd
from src.ai.lstm_predictor import LSTMPredictor, TF_AVAILABLE
from src.ai.sentiment_analyzer import SentimentAnalyzer
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.multi_timeframe import MultiTimeframeAnalyzer
from src.ai.support_resistance import SupportResistanceDetector
from src.ai.candlestick_patterns import CandlestickPatternDetector
from src.ai.ema_crossover import EMACrossoverAnalyzer
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.ai.adaptive_learner import AdaptiveLearner
from src.ai.cross_pair_analyzer import CrossPairAnalyzer
from src.utils.logger import TradeLogger, bot_logger
from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD, MIN_MODELS_AGREEMENT


class EnsembleTrader:
    """8-model ensemble with regime awareness, conviction scaling, and adaptive learning"""

    # Regime-based model weight boosts (scalping-tuned)
    REGIME_BOOSTS = {
        'trending': {
            'scalping': 1.5,
            'ema_crossover': 1.4,
            'multi_tf': 1.3,
            'technical': 1.2,
            'lstm': 1.1,
            'support_resistance': 0.7,  # S/R less useful in trends
            'volume': 0.9,
        },
        'ranging': {
            'scalping': 0.6,
            'support_resistance': 1.5,
            'candlestick': 1.3,
            'technical': 1.2,  # RSI/BB work well in ranges
            'ema_crossover': 0.6,  # Crossovers whipsaw in ranges
            'multi_tf': 0.8,
        },
        'volatile': {
            'scalping': 0.8,
            'volume': 1.4,
            'support_resistance': 1.2,
            'candlestick': 1.1,
            'ema_crossover': 0.5,  # Crossovers unreliable
            'technical': 0.8,
        },
    }

    def __init__(self, newsapi_key=None, broker=None):
        self.lstm = LSTMPredictor(lookback_window=60)
        self.lstm_available = TF_AVAILABLE and self.lstm.available
        self.sentiment = SentimentAnalyzer(newsapi_key=newsapi_key)
        self.technical = TechnicalAnalyzer()
        self.volume = VolumeAnalyzer(volume_period=20)
        self.multi_tf = MultiTimeframeAnalyzer()
        self.sr_detector = SupportResistanceDetector()
        self.candle_detector = CandlestickPatternDetector()
        self.ema_crossover = EMACrossoverAnalyzer()
        self.scalping = ScalpingAnalyzer()
        self.learner = AdaptiveLearner()
        self.cross_pair = CrossPairAnalyzer()
        self.broker = broker

        # Scalping-tuned weights: scalping model is the heaviest
        self.model_weights = {
            'scalping': 0.22,
            'ema_crossover': 0.14,
            'candlestick': 0.13,
            'technical': 0.12,
            'volume': 0.11,
            'multi_tf': 0.10,
            'support_resistance': 0.06,
            'lstm': 0.06,
            'sentiment': 0.06,
        }

        model_count = 9
        if not self.lstm_available:
            bot_logger.info("🧠 Scalping ensemble running with 8 models (LSTM disabled)")
            model_count = 8
        else:
            bot_logger.info("🧠 Scalping ensemble running with all 9 models")
        bot_logger.info(f"🔪 ScalpingAnalyzer active as primary signal (weight 0.22)")

    def get_trading_signal(self, df, pair):
        """
        Generate trading signal from ensemble with regime awareness.

        Returns:
            {
                'signal': 'BUY', 'SELL', or 'SKIP',
                'confidence': 0.0-1.0,
                'models_agreement': int,
                'regime': str,
                'details': {...}
            }
        """
        # Calculate indicators first
        df_enriched = self.technical.calculate_indicators(df)

        # Detect market regime
        regime = self.learner.detect_regime(df_enriched)

        # Get pair-specific adaptive weights
        adaptive_weights = self.learner.get_adjusted_weights(pair=pair)
        weights = dict(self.model_weights)
        for k, v in adaptive_weights.items():
            if k in weights:
                weights[k] = v

        # If LSTM is disabled, redistribute
        if not self.lstm_available:
            lstm_w = weights.pop('lstm', 0)
            if weights:
                bonus = lstm_w / len(weights)
                weights = {k: v + bonus for k, v in weights.items()}

        # Apply regime-based boosts
        regime_boosts = self.REGIME_BOOSTS.get(regime, {})
        for model, boost in regime_boosts.items():
            if model in weights:
                weights[model] *= boost

        # Normalize weights
        w_sum = sum(weights.values())
        weights = {k: v / w_sum for k, v in weights.items()}

        # === Run all models ===
        if self.lstm_available:
            lstm_signal = self.lstm.predict_direction(df_enriched)
        else:
            lstm_signal = {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'LSTM disabled'}

        sentiment_signal = self.sentiment.get_pair_sentiment(pair)
        sentiment_signal_type = 'BUY' if sentiment_signal['sentiment_score'] > 0.1 else (
            'SELL' if sentiment_signal['sentiment_score'] < -0.1 else 'HOLD'
        )
        sentiment_confidence = abs(sentiment_signal['sentiment_score'])

        technical_signal = self.technical.get_signal(df_enriched)
        volume_signal = self.volume.get_volume_signal(df_enriched)

        if self.broker:
            mtf_signal = self.multi_tf.analyze(self.broker, pair)
            mtf_signal_type = mtf_signal['signal']
            mtf_confidence = mtf_signal['confluence_score']
        else:
            mtf_signal_type = 'HOLD'
            mtf_confidence = 0.0
            mtf_signal = {'signal': 'HOLD', 'confluence_score': 0.0, 'reason': 'No broker'}

        sr_signal = self.sr_detector.get_sr_signal(df_enriched)
        candle_signal = self.candle_detector.get_pattern_signal(df_enriched)
        ema_signal = self.ema_crossover.get_signal(df_enriched)

        # Scalping signal (9th model — highest weight)
        scalping_signal = self.scalping.get_signal(df_enriched, pair)
        scalping_signal_type = scalping_signal.get('signal', 'HOLD')
        if scalping_signal_type == 'SKIP':
            scalping_signal_type = 'HOLD'
        scalping_confidence = scalping_signal.get('confidence', 0.0)

        # === Build signal map ===
        all_signals = {}
        if self.lstm_available:
            all_signals['lstm'] = {'signal': lstm_signal['signal'], 'confidence': lstm_signal['confidence']}

        all_signals.update({
            'scalping': {'signal': scalping_signal_type, 'confidence': scalping_confidence},
            'sentiment': {'signal': sentiment_signal_type, 'confidence': sentiment_confidence},
            'technical': {'signal': technical_signal['signal'], 'confidence': technical_signal['confidence']},
            'volume': {'signal': volume_signal['signal'], 'confidence': volume_signal['confidence']},
            'multi_tf': {'signal': mtf_signal_type, 'confidence': mtf_confidence},
            'support_resistance': {'signal': sr_signal['signal'], 'confidence': sr_signal['confidence']},
            'candlestick': {'signal': candle_signal['signal'], 'confidence': candle_signal['confidence']},
            'ema_crossover': {'signal': ema_signal['signal'], 'confidence': ema_signal['confidence']},
        })

        # Count votes (only from active models)
        active_signals = {k: v for k, v in all_signals.items() if v['signal'] != 'HOLD' or v['confidence'] > 0.0}
        buy_votes = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in all_signals.values() if s['signal'] == 'SELL')

        models_agreement = max(buy_votes, sell_votes)
        total_models = len(all_signals)
        active_model_count = len(active_signals)

        # Scale min_agreement by active model count
        min_agreement = max(2, int(MIN_MODELS_AGREEMENT * active_model_count / total_models + 0.5))

        if buy_votes > sell_votes and models_agreement >= min_agreement:
            final_signal = 'BUY'
        elif sell_votes > buy_votes and models_agreement >= min_agreement:
            final_signal = 'SELL'
        else:
            final_signal = 'SKIP'

        # === Conviction Scoring ===
        # Instead of just counting votes, calculate conviction based on:
        # 1. Number of agreeing models (more = stronger)
        # 2. Average confidence of agreeing models (higher = stronger)
        # 3. Whether high-weight models agree (weighted conviction)
        if final_signal != 'SKIP' and active_signals:
            direction = final_signal
            agreeing = {k: v for k, v in active_signals.items() if v['signal'] == direction}
            opposing = {k: v for k, v in active_signals.items() if v['signal'] != direction and v['signal'] != 'HOLD'}

            # Weighted conviction: sum of (weight * confidence) for agreeing models
            agreeing_conviction = sum(
                weights.get(m, 0) * v['confidence'] for m, v in agreeing.items()
            )
            opposing_conviction = sum(
                weights.get(m, 0) * v['confidence'] for m, v in opposing.items()
            )

            # Net conviction (positive = signal is strong)
            net_conviction = agreeing_conviction - opposing_conviction * 0.5

            # Confluence bonus: stronger signal when more models agree
            agreement_ratio = len(agreeing) / max(active_model_count, 1)
            confluence_bonus = 0.0
            if agreement_ratio >= 0.75:  # 75%+ models agree
                confluence_bonus = 0.10
                bot_logger.info(f"🎯 Strong confluence: {len(agreeing)}/{active_model_count} models agree ({agreement_ratio:.0%})")
            elif agreement_ratio >= 0.60:
                confluence_bonus = 0.05

            # Average confidence of agreeing models
            avg_agreeing_conf = sum(v['confidence'] for v in agreeing.values()) / max(len(agreeing), 1)

            # Final weighted confidence
            weighted_confidence = net_conviction + confluence_bonus

            # Regime confidence modifier from learner
            regime_modifier = self.learner.get_regime_confidence_modifier(regime)
            weighted_confidence *= regime_modifier

        else:
            weighted_confidence = 0.0
            net_conviction = 0.0

        # === EMA 200 Trend Filter ===
        EMA_COUNTER_TREND_PENALTY = 0.08  # Increased from 0.05
        ema_200 = df_enriched['ema_200'].iloc[-1] if 'ema_200' in df_enriched.columns else None
        cur_price = df_enriched['close'].iloc[-1]
        ema_counter_trend = False
        if ema_200 is not None and not pd.isna(ema_200) and final_signal != 'SKIP':
            if (final_signal == 'BUY' and cur_price < ema_200) or \
               (final_signal == 'SELL' and cur_price > ema_200):
                trend_dir = 'bearish' if cur_price < ema_200 else 'bullish'
                bot_logger.info(
                    f"⚠️  EMA200 penalty: {final_signal} counter-trend "
                    f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f})"
                )
                ema_counter_trend = True
            else:
                # Trend-aligned bonus
                weighted_confidence *= 1.05
                bot_logger.info(
                    f"✅ EMA200 aligned: {final_signal} with trend "
                    f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f}) +5% confidence"
                )

        if ema_counter_trend and weighted_confidence > 0:
            weighted_confidence = max(0.0, weighted_confidence - EMA_COUNTER_TREND_PENALTY)

        # === S/R Context Advisory ===
        price_zone = sr_signal.get('levels', {}).get('price_zone', '')
        if final_signal == 'BUY' and price_zone == 'AT_RESISTANCE':
            bot_logger.info("⚠️  S/R advisory: BUY near resistance")
        if final_signal == 'SELL' and price_zone == 'AT_SUPPORT':
            bot_logger.info("⚠️  S/R advisory: SELL near support")

        # === Momentum Divergence Check ===
        # If RSI diverges from price direction, reduce confidence
        if final_signal != 'SKIP' and 'rsi' in df_enriched.columns:
            rsi = df_enriched['rsi'].iloc[-1]
            if final_signal == 'BUY' and rsi > 70:
                weighted_confidence *= 0.85
                bot_logger.info(f"⚠️  RSI overbought ({rsi:.0f}) on BUY signal — reducing confidence")
            elif final_signal == 'SELL' and rsi < 30:
                weighted_confidence *= 0.85
                bot_logger.info(f"⚠️  RSI oversold ({rsi:.0f}) on SELL signal — reducing confidence")

        # === Cross-Pair Correlation Modifier ===
        if final_signal != 'SKIP' and pair:
            cross_modifier = self.cross_pair.get_confidence_modifier(pair, final_signal)
            if cross_modifier != 1.0:
                weighted_confidence *= cross_modifier
                direction = 'confirms ✅' if cross_modifier > 1.0 else 'diverges ⚠️'
                bot_logger.info(
                    f"🔗 Cross-pair {direction}: {pair} confidence x{cross_modifier:.3f}"
                )

        # Cap confidence at 1.0
        weighted_confidence = min(1.0, max(0.0, weighted_confidence))

        # Generate reasoning
        reason_parts = []
        reason_parts.append(f"Scalp: {scalping_signal_type} ({scalping_confidence:.0%})")
        if self.lstm_available:
            reason_parts.append(f"LSTM: {lstm_signal['signal']} ({lstm_signal['confidence']:.0%})")
        reason_parts.extend([
            f"Sent: {sentiment_signal_type} ({sentiment_confidence:.0%})",
            f"Tech: {technical_signal['signal']} ({technical_signal['confidence']:.0%})",
            f"Vol: {volume_signal['signal']} ({volume_signal['confidence']:.0%})",
            f"MTF: {mtf_signal_type} ({mtf_confidence:.0%})",
            f"S/R: {sr_signal['signal']} ({sr_signal['confidence']:.0%})",
            f"Candle: {candle_signal['signal']} ({candle_signal['confidence']:.0%})",
            f"EMA: {ema_signal['signal']} ({ema_signal['confidence']:.0%})",
            f"Regime: {regime}",
        ])

        detailed_reason = " | ".join(reason_parts)

        result = {
            'signal': final_signal,
            'confidence': weighted_confidence,
            'models_agreement': models_agreement,
            'total_models': total_models,
            'min_agreement_required': min_agreement,
            'regime': regime,
            'detailed_reason': detailed_reason,
            'enriched_df': df_enriched,
            'models': {
                'scalping': {
                    'signal': scalping_signal_type,
                    'confidence': scalping_confidence,
                    'setup': scalping_signal.get('setup', 'none'),
                },
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
        Uses adaptive confidence threshold + regime awareness.
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
