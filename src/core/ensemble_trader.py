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
from src.ai.ml_trade_scorer import MLTradeScorer
from src.utils.logger import TradeLogger, bot_logger
from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD, MIN_MODELS_AGREEMENT


class EnsembleTrader:
    """8-model ensemble with regime awareness, conviction scaling, and adaptive learning"""

    # Regime-based model weight boosts (ATR-centric scalping)
    REGIME_BOOSTS = {
        'trending': {
            'scalping': 1.5,       # ATR-pullback thrives in trends
            'ema_crossover': 1.3,  # EMA alignment confirms
            'technical': 1.2,
            'volume': 1.1,
            'multi_tf': 1.2,
            'lstm': 1.0,
            'support_resistance': 0.6,  # S/R less useful in trends
            'candlestick': 0.9,
        },
        'ranging': {
            'scalping': 0.5,       # ATR too low in ranges
            'support_resistance': 1.5,
            'candlestick': 1.3,
            'technical': 1.2,
            'volume': 1.0,
            'ema_crossover': 0.5,  # Crossovers whipsaw in ranges
            'multi_tf': 0.7,
        },
        'volatile': {
            'scalping': 1.3,       # ATR expanding = scalping loves it
            'volume': 1.4,
            'technical': 0.9,
            'ema_crossover': 0.6,
            'support_resistance': 1.1,
            'candlestick': 1.1,
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
        self.ml_scorer = MLTradeScorer()
        self.broker = broker

        # ATR-centric weights: scalping analyzer is the dominant signal
        self.model_weights = {
            'scalping': 0.28,         # Primary ATR-pullback signal
            'technical': 0.18,        # Multi-indicator confirmation
            'volume': 0.14,           # Volume spike confirmation
            'ema_crossover': 0.12,    # EMA alignment + ATR momentum
            'candlestick': 0.10,      # Candle pattern confirmation
            'multi_tf': 0.08,         # 5M timeframe alignment
            'support_resistance': 0.04,
            'lstm': 0.03,
            'sentiment': 0.03,
        }

        model_count = 9
        if not self.lstm_available:
            bot_logger.info("🧠 ATR-centric ensemble running with 8 models (LSTM disabled)")
            model_count = 8
        else:
            bot_logger.info("🧠 ATR-centric ensemble running with all 9 models")
        bot_logger.info(f"🔪 ScalpingAnalyzer active as primary ATR signal (weight 0.28)")

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

        # Scalping signal (9th model — heaviest weight, ATR-centric)
        # Pass 5M data and spread if available
        df_5m = None
        broker_spread = None
        if self.broker:
            try:
                df_5m = self.broker.get_candles(pair, '5m', count=250)
            except Exception:
                pass
            try:
                broker_spread = self.broker.get_spread(pair)
            except Exception:
                pass

        scalping_signal = self.scalping.get_signal(
            df_enriched, pair,
            df_5m=df_5m,
            spread=broker_spread,
        )
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

        # Require at least 3 models to agree for a trade signal
        min_agreement = max(3, int(MIN_MODELS_AGREEMENT * active_model_count / total_models + 0.5))

        if buy_votes > sell_votes and models_agreement >= min_agreement:
            final_signal = 'BUY'
        elif sell_votes > buy_votes and models_agreement >= min_agreement:
            final_signal = 'SELL'
        else:
            final_signal = 'SKIP'

        # === Conviction Scoring ===
        # Average conviction of agreeing models (weighted), scaled by agreement breadth.
        # HOLD models are neutral — they don't drag down the score.
        if final_signal != 'SKIP' and active_signals:
            direction = final_signal
            agreeing = {k: v for k, v in all_signals.items() if v['signal'] == direction}
            opposing = {k: v for k, v in all_signals.items() if v['signal'] != direction and v['signal'] != 'HOLD'}

            # Weighted confidence of agreeing models
            weighted_agree = sum(weights.get(k, 0) * v['confidence'] for k, v in agreeing.items())
            weight_agree_sum = sum(weights.get(k, 0) for k in agreeing)
            weighted_oppose = sum(weights.get(k, 0) * v['confidence'] for k, v in opposing.items())

            # Average conviction: how confident are the agreeing models? (0-1)
            avg_conviction = weighted_agree / weight_agree_sum if weight_agree_sum > 0 else 0.0

            # Agreement breadth: what fraction of ALL models agree? (0-1)
            agreement_ratio = len(agreeing) / max(total_models, 1)

            # Confluence bonus for broad agreement
            confluence_bonus = 0.0
            if agreement_ratio >= 0.60:
                confluence_bonus = 0.10
                bot_logger.info(f"🎯 Strong confluence: {len(agreeing)}/{total_models} models agree ({agreement_ratio:.0%})")
            elif agreement_ratio >= 0.40:
                confluence_bonus = 0.05

            # Opposing penalty (from models that actively disagree)
            opposing_penalty = weighted_oppose * 0.5

            # Final confidence = avg conviction × (base + agreement scaling) + bonus - penalty
            # Base 0.5 ensures 3 models at decent conviction can still clear threshold
            # Scaling 0.5 rewards broader agreement
            weighted_confidence = avg_conviction * (0.5 + 0.5 * agreement_ratio) + confluence_bonus - opposing_penalty

            # Regime confidence modifier from learner
            regime_modifier = self.learner.get_regime_confidence_modifier(regime)
            weighted_confidence *= regime_modifier

            net_conviction = weighted_confidence  # For logging compatibility

        else:
            weighted_confidence = 0.0
            net_conviction = 0.0

        # === EMA 200 Trend Filter — block counter-trend trades ===
        ema_200 = df_enriched['ema_200'].iloc[-1] if 'ema_200' in df_enriched.columns else None
        cur_price = df_enriched['close'].iloc[-1]
        if ema_200 is not None and not pd.isna(ema_200) and final_signal != 'SKIP':
            if (final_signal == 'BUY' and cur_price < ema_200) or \
               (final_signal == 'SELL' and cur_price > ema_200):
                bot_logger.info(
                    f"🚫 EMA200 BLOCK: {final_signal} counter-trend "
                    f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f}) — forcing SKIP"
                )
                final_signal = 'SKIP'
                weighted_confidence = 0.0
            else:
                weighted_confidence *= 1.10
                bot_logger.info(
                    f"✅ EMA200 aligned: {final_signal} with trend "
                    f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f}) +10% confidence"
                )

        # === High-Weight Model Disagreement Filter ===
        # If a heavy model (scalping, LSTM, technical) STRONGLY opposes the signal, penalize
        heavy_models = ['scalping', 'lstm', 'technical', 'ema_crossover']
        strong_opposition_count = 0
        for model_name in heavy_models:
            m = all_signals.get(model_name)
            if m and m['signal'] != 'HOLD' and m['signal'] != final_signal and m['confidence'] >= 0.50:
                strong_opposition_count += 1
                bot_logger.info(
                    f"⚠️  Heavy model {model_name} opposes {final_signal} "
                    f"with {m['signal']} ({m['confidence']:.0%})"
                )
        if strong_opposition_count >= 2:
            weighted_confidence *= 0.85  # -15% if 2+ heavy models oppose
            bot_logger.info(
                f"⚠️ {strong_opposition_count} heavy models oppose {final_signal} — slight reduction"
            )
        elif strong_opposition_count == 1:
            weighted_confidence *= 0.95  # -5% if 1 heavy model opposes

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
            # ATR-centric fields
            'atr_regime': scalping_signal.get('atr_regime', 'neutral'),
            'atr_tp_ratio': scalping_signal.get('atr_tp_ratio', 1.4),
            'scalping_risk_reward': scalping_signal.get('risk_reward', {}),
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
        Requires at least one core model (scalping, technical, or EMA) to agree.
        """
        threshold = self.learner.get_adjusted_threshold()

        # Minimum 3 models must agree
        if signal_result['models_agreement'] < MIN_MODELS_AGREEMENT:
            bot_logger.info(
                f"📊 Only {signal_result['models_agreement']} models agree "
                f"(need {MIN_MODELS_AGREEMENT}) — skipping"
            )
            return False

        # At least one core model must agree with the direction
        core_models = ['scalping', 'technical', 'ema_crossover']
        models = signal_result.get('models', {})
        direction = signal_result['signal']
        core_agrees = any(
            models.get(m, {}).get('signal') == direction
            for m in core_models
        )
        if not core_agrees and direction != 'SKIP':
            bot_logger.info(
                f"📊 No core model (scalping/technical/EMA) agrees with {direction} — skipping"
            )
            return False

        return (
            signal_result['signal'] != 'SKIP' and
            signal_result['confidence'] >= threshold and
            signal_result['models_agreement'] >= MIN_MODELS_AGREEMENT
        )

    def get_ml_win_probability(self, signal_result: dict, pair: str) -> float:
        """Get ML model's predicted win probability for this trade setup."""
        return self.ml_scorer.predict_win_probability(signal_result, pair, self.learner)

    def capture_ml_features(self, signal_result: dict, pair: str) -> list:
        """Capture feature snapshot at trade entry time for later training."""
        return MLTradeScorer.extract_features(signal_result, pair, self.learner)

    def record_ml_trade(self, features: list, is_win: bool):
        """Record a completed trade's features + outcome for ML training."""
        self.ml_scorer.record_trade(features, is_win)

    def record_trade_result(self, trade_data: dict):
        """Pass trade result to adaptive learner."""
        self.learner.record_trade(trade_data)
