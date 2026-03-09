"""
Sweep-Gated Entry System — RayAlgo v3

Architecture:
  Gate:    LiquiditySweepAnalyzer (4-layer: Bias → Sweep → MSS → Entry)
           If sweep does NOT fire → signal = SKIP.  No vote can override.
  Confirm: EMA Crossover (trend alignment) + Technical (MACD/BB momentum)
           These BOOST or REDUCE sweep confidence — they cannot create a signal.
  Context: All other models (scalping, volume, sentiment, LSTM, S/R, candle, MTF)
           still run for logging / adaptive learning — but do NOT affect entry.

LSTM is optional — if TensorFlow is not installed the system runs without it.
"""
import pandas as pd
from src.ai.lstm_predictor import LSTMPredictor, TF_AVAILABLE
from src.ai.sentiment_analyzer import SentimentAnalyzer
from src.ai.nlp_sentiment import NLPSentimentAnalyzer, FINBERT_AVAILABLE
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.multi_timeframe import MultiTimeframeAnalyzer
from src.ai.support_resistance import SupportResistanceDetector
from src.ai.candlestick_patterns import CandlestickPatternDetector
from src.ai.ema_crossover import EMACrossoverAnalyzer
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
from src.ai.adaptive_learner import AdaptiveLearner
from src.ai.cross_pair_analyzer import CrossPairAnalyzer
from src.ai.ml_trade_scorer import MLTradeScorer
from src.ai.rl_agent import RLTradingAgent
from src.utils.logger import TradeLogger, bot_logger
from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD, MIN_MODELS_AGREEMENT


class EnsembleTrader:
    """Sweep-gated entry system with confirmation boosters and adaptive learning."""

    # Confirmation boost/penalty amounts
    EMA_CONFIRM_BOOST = 0.05     # EMA aligned with sweep direction
    EMA_OPPOSE_PENALTY = 0.10    # EMA opposes sweep direction
    TECH_CONFIRM_BOOST = 0.03    # Technical momentum matches sweep
    TECH_OPPOSE_PENALTY = 0.05   # Technical momentum opposes sweep
    LSTM_CONFIRM_BOOST = 0.08    # LSTM direction agrees with sweep
    LSTM_OPPOSE_PENALTY = 0.20   # LSTM direction opposes sweep (strong filter)
    RL_SKIP_PENALTY = 0.08       # RL agent recommends skipping

    def __init__(self, newsapi_key=None, broker=None):
        # ── Primary: sweep gate ──────────────────────────────────────
        self.sweep = LiquiditySweepAnalyzer()

        # ── Confirmation models (only these affect confidence) ───────
        self.ema_crossover = EMACrossoverAnalyzer()
        self.technical = TechnicalAnalyzer()

        # ── Context models (logging/learning only, no entry influence)
        self.lstm = LSTMPredictor(lookback_window=60)
        self.lstm_available = TF_AVAILABLE and self.lstm.available
        if FINBERT_AVAILABLE:
            self.sentiment = NLPSentimentAnalyzer(newsapi_key=newsapi_key)
            bot_logger.info("🧠 NLP sentiment: FinBERT active (context only)")
        else:
            self.sentiment = SentimentAnalyzer(newsapi_key=newsapi_key)
            bot_logger.info("⚠️ NLP sentiment: keyword fallback (context only)")
        self.volume = VolumeAnalyzer(volume_period=20)
        self.multi_tf = MultiTimeframeAnalyzer()
        self.sr_detector = SupportResistanceDetector()
        self.candle_detector = CandlestickPatternDetector()
        self.scalping = ScalpingAnalyzer()

        # ── Adaptive learning & cross-pair ───────────────────────────
        self.learner = AdaptiveLearner()
        self.cross_pair = CrossPairAnalyzer()
        self.ml_scorer = MLTradeScorer()
        self.rl_agent = RLTradingAgent()
        self.rl_available = hasattr(self.rl_agent, 'q_network') or hasattr(self.rl_agent, 'q_table')
        self.broker = broker

        # Legacy model_weights kept for adaptive learner compatibility
        self.model_weights = {
            'sweep': 1.00,
            'ema_crossover': 0.0,
            'technical': 0.0,
            'scalping': 0.0,
            'candlestick': 0.0,
            'multi_tf': 0.0,
            'support_resistance': 0.0,
            'volume': 0.0,
            'lstm': 0.0,
            'sentiment': 0.0,
        }

        bot_logger.info("🎯 Sweep-Gated Entry System (RayAlgo v3)")
        bot_logger.info("   Gate:    LiquiditySweep (4-layer: Bias → Sweep → MSS → Entry)")
        bot_logger.info("   Confirm: EMA Crossover + Technical (boost/reduce only)")
        bot_logger.info("   Context: 8 models for learning (no entry influence)")

    def get_trading_signal(self, df, pair):
        """
        Generate trading signal using sweep-gated architecture.

        Flow:
          1. Calculate indicators on 1M data
          2. Run LiquiditySweep 4-layer pipeline → produces signal + confidence
          3. If sweep = SKIP → final = SKIP (hard gate)
          4. If sweep fires → run EMA + Technical as confirmation boosters
          5. Apply EMA200 trend filter, cross-pair modifier, learner adjustments
          6. Run context models for logging / adaptive learning
          7. Return final signal

        Returns:
            dict with signal, confidence, models_agreement, regime, details, etc.
        """
        # ── Step 1: Calculate indicators ─────────────────────────────
        df_enriched = self.technical.calculate_indicators(df)

        # Detect market regime (for adaptive learner)
        regime = self.learner.detect_regime(df_enriched)

        # ── Step 2: Fetch 5M data + spread (shared across models) ────
        df_5m = None
        broker_spread = None
        if self.broker:
            try:
                df_5m = self.broker.get_candles(pair, 5, num_candles=250)
                if df_5m is not None:
                    bot_logger.info(f"📊 {pair} 5M fetch OK: {len(df_5m)} rows")
                else:
                    bot_logger.warning(f"⚠️ {pair} 5M fetch returned None")
            except Exception as e:
                bot_logger.warning(f"⚠️ {pair} 5M fetch EXCEPTION: {type(e).__name__}: {e}")
            get_spread = getattr(self.broker, 'get_spread', None)
            if callable(get_spread):
                try:
                    broker_spread = get_spread(pair)
                except Exception as e:
                    bot_logger.warning(f"⚠️ {pair} spread fetch failed: {e}")

        # ── Step 3: Run sweep gate (PRIMARY — decides entry) ─────────
        sweep_signal = self.sweep.get_signal(
            df_enriched, pair,
            df_5m=df_5m,
            spread=broker_spread,
        )
        sweep_direction = sweep_signal.get('signal', 'SKIP')
        sweep_confidence = sweep_signal.get('confidence', 0.0)

        # ── Step 4: Run confirmation models ──────────────────────────
        ema_signal = self.ema_crossover.get_signal(df_enriched)
        technical_signal = self.technical.get_signal(df_enriched)

        # ── Step 5: Run context models (for logging + learning) ──────
        if self.lstm_available:
            lstm_signal = self.lstm.predict_direction(df_enriched)
        else:
            lstm_signal = {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'LSTM disabled'}

        sentiment_signal = self.sentiment.get_pair_sentiment(pair)
        sentiment_signal_type = 'BUY' if sentiment_signal['sentiment_score'] > 0.1 else (
            'SELL' if sentiment_signal['sentiment_score'] < -0.1 else 'HOLD'
        )
        sentiment_confidence = abs(sentiment_signal['sentiment_score'])

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

        scalping_signal = self.scalping.get_signal(
            df_enriched, pair,
            df_5m=df_5m,
            spread=broker_spread,
        )

        # ── Step 6: Sweep gate decision ──────────────────────────────
        context_signals = {}  # defined early so result dict always has it

        if sweep_direction not in ('BUY', 'SELL'):
            # Sweep didn't fire — check for EMA+Technical consensus fallback
            # This lets the bot take high-conviction trend trades when no
            # liquidity sweep exists but EMA and Technical both agree.
            ema_dir = ema_signal['signal']
            tech_dir = technical_signal['signal']
            if ema_dir in ('BUY', 'SELL') and ema_dir == tech_dir:
                final_signal = ema_dir
                final_confidence = 0.40  # just at threshold — intentionally low
                models_agreement = 2
                bot_logger.info(
                    f"🔄 No sweep → EMA+Tech fallback: {ema_dir} "
                    f"(EMA={ema_signal['confidence']:.0%}, "
                    f"Tech={technical_signal['confidence']:.0%})"
                )
            else:
                final_signal = 'SKIP'
                final_confidence = 0.0
                models_agreement = 0
        else:
            final_signal = sweep_direction
            final_confidence = sweep_confidence

            bot_logger.info(f"🔍 Confidence breakdown: Initial sweep={sweep_confidence:.1%}")

            # ── Confirmation adjustments (boost/reduce only) ─────────
            # EMA Crossover
            if ema_signal['signal'] == sweep_direction:
                final_confidence += self.EMA_CONFIRM_BOOST
                bot_logger.info(
                    f"✅ EMA confirms {sweep_direction} (+{self.EMA_CONFIRM_BOOST:.0%}) → {final_confidence:.1%}"
                )
            elif ema_signal['signal'] != 'HOLD' and ema_signal['signal'] != sweep_direction:
                final_confidence -= self.EMA_OPPOSE_PENALTY
                bot_logger.info(
                    f"⚠️ EMA opposes {sweep_direction} (−{self.EMA_OPPOSE_PENALTY:.0%}) → {final_confidence:.1%}"
                )

            # Technical momentum
            if technical_signal['signal'] == sweep_direction:
                final_confidence += self.TECH_CONFIRM_BOOST
                bot_logger.info(
                    f"✅ Technical confirms {sweep_direction} (+{self.TECH_CONFIRM_BOOST:.0%}) → {final_confidence:.1%}"
                )
            elif technical_signal['signal'] != 'HOLD' and technical_signal['signal'] != sweep_direction:
                final_confidence -= self.TECH_OPPOSE_PENALTY
                bot_logger.info(
                    f"⚠️ Technical opposes {sweep_direction} (−{self.TECH_OPPOSE_PENALTY:.0%}) → {final_confidence:.1%}"
                )

            # ── EMA 200 Trend Filter — SOFT PENALTY counter-trend ─────
            ema_200 = df_enriched['ema_200'].iloc[-1] if 'ema_200' in df_enriched.columns else None
            cur_price = df_enriched['close'].iloc[-1]
            if ema_200 is not None and not pd.isna(ema_200):
                if (final_signal == 'BUY' and cur_price < ema_200) or \
                   (final_signal == 'SELL' and cur_price > ema_200):
                    final_confidence -= 0.10
                    bot_logger.info(
                        f"⚠️ EMA200 counter-trend penalty: {final_signal} "
                        f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f}) -0.10"
                    )
                else:
                    bot_logger.info(
                        f"✅ EMA200 aligned: {final_signal} with trend "
                        f"(price {cur_price:.5f} vs EMA200 {ema_200:.5f})"
                    )

            # ── Cross-pair correlation modifier ──────────────────────
            if final_signal != 'SKIP' and pair:
                cross_modifier = self.cross_pair.get_confidence_modifier(pair, final_signal)
                if cross_modifier != 1.0:
                    old_confidence = final_confidence
                    final_confidence *= cross_modifier
                    direction = 'confirms ✅' if cross_modifier > 1.0 else 'diverges ⚠️'
                    bot_logger.info(
                        f"🔗 Cross-pair {direction}: {pair} confidence x{cross_modifier:.3f} ({old_confidence:.1%} → {final_confidence:.1%})"
                    )

            # ── Regime confidence modifier from learner ──────────────
            old_confidence = final_confidence
            regime_modifier = self.learner.get_regime_confidence_modifier(regime)
            final_confidence *= regime_modifier
            if regime_modifier != 1.0:
                bot_logger.info(
                    f"🎭 Regime modifier ({regime}): x{regime_modifier:.3f} ({old_confidence:.1%} → {final_confidence:.1%})"
                )

            # Count how many context models agree (for logging & compatibility)
            context_signals = {
                'ema_crossover': ema_signal['signal'],
                'technical': technical_signal['signal'],
                'scalping': scalping_signal.get('signal', 'HOLD'),
                'volume': volume_signal['signal'],
                'multi_tf': mtf_signal_type,
                'candlestick': candle_signal['signal'],
                'support_resistance': sr_signal['signal'],
            }
            if self.lstm_available:
                context_signals['lstm'] = lstm_signal['signal']
            context_signals['sentiment'] = sentiment_signal_type

            agreeing = sum(1 for s in context_signals.values() if s == sweep_direction)
            models_agreement = agreeing + 1  # +1 for sweep itself

            # ── Adaptive weight bonus ────────────────────────────────
            # Use learned model weights to give more influence to models
            # that have historically predicted correctly.
            try:
                learned_weights = self.learner.get_adjusted_weights(pair=pair)
                weighted_agreement = 0.0
                weighted_total = 0.0
                for model_name, model_signal in context_signals.items():
                    w = learned_weights.get(model_name, 0.1)
                    weighted_total += w
                    if model_signal == sweep_direction:
                        weighted_agreement += w
                if weighted_total > 0:
                    weighted_ratio = weighted_agreement / weighted_total
                    # Scale: 0.0 (all disagree) → 0.10 (all agree) bonus
                    weight_bonus = (weighted_ratio - 0.5) * 0.20
                    if abs(weight_bonus) > 0.01:
                        old_conf = final_confidence
                        final_confidence += weight_bonus
                        bot_logger.info(
                            f"📊 Adaptive weight {'bonus' if weight_bonus > 0 else 'penalty'}: "
                            f"{weight_bonus:+.2%} ({old_conf:.1%} → {final_confidence:.1%})"
                        )
            except Exception:
                pass

            # ── LSTM direction filter (raw prediction, 0.02% threshold) ──
            if self.lstm_available and final_signal != 'SKIP':
                try:
                    pct_change = lstm_signal.get('predicted_change_percent', 0)
                    # Skip LSTM filter if prediction is pegged at clamp limit (unreliable)
                    if abs(pct_change) >= 4.99:
                        bot_logger.info(f"⚠️ LSTM at clamp limit ({pct_change:+.3f}%) — ignoring (model needs retraining)")
                    elif abs(pct_change) > 0.02:
                        if (pct_change > 0 and sweep_direction == 'BUY') or \
                           (pct_change < 0 and sweep_direction == 'SELL'):
                            final_confidence += self.LSTM_CONFIRM_BOOST
                            bot_logger.info(f"✅ LSTM confirms {sweep_direction} ({pct_change:+.3f}%) +{self.LSTM_CONFIRM_BOOST}")
                        else:
                            final_confidence -= self.LSTM_OPPOSE_PENALTY
                            bot_logger.info(f"⚠️ LSTM opposes {sweep_direction} ({pct_change:+.3f}%) -{self.LSTM_OPPOSE_PENALTY}")
                except Exception:
                    pass

            # ── RL quality filter ────────────────────────────────────
            if self.rl_available and final_signal != 'SKIP':
                try:
                    import numpy as np
                    rsi_val = float(df_enriched['rsi'].iloc[-1]) if 'rsi' in df_enriched.columns and not pd.isna(df_enriched['rsi'].iloc[-1]) else 50.0
                    adx_val = float(df_enriched['adx'].iloc[-1]) if 'adx' in df_enriched.columns and not pd.isna(df_enriched['adx'].iloc[-1]) else 25.0
                    atr_val = float(df_enriched['atr'].iloc[-1]) if 'atr' in df_enriched.columns and not pd.isna(df_enriched['atr'].iloc[-1]) else 0.001
                    atr_med = float(df_enriched['atr'].median()) if 'atr' in df_enriched.columns else 0.001
                    ema200_val = float(df_enriched['ema_200'].iloc[-1]) if 'ema_200' in df_enriched.columns and not pd.isna(df_enriched['ema_200'].iloc[-1]) else cur_price
                    ema200_dist = (cur_price - ema200_val) / (atr_val + 1e-8)
                    from datetime import datetime
                    hour = datetime.utcnow().hour
                    vol_ratio = float(df_enriched['volume'].iloc[-1] / (df_enriched['volume'].rolling(20).mean().iloc[-1] + 1)) if 'volume' in df_enriched.columns else 1.0

                    rl_state = self.rl_agent.build_state(
                        ensemble_confidence=final_confidence,
                        model_agreement=models_agreement,
                        total_models=4,
                        regime='trending' if regime in ('trend_up', 'trend_down') else 'ranging',
                        rsi=rsi_val, adx=adx_val,
                        atr=atr_val, atr_median=atr_med,
                        ema200_dist=ema200_dist,
                        hour=hour, spread=broker_spread or 0.00015,
                        volume_ratio=vol_ratio,
                        daily_trades=0, max_daily_trades=30,
                        current_drawdown=0
                    )
                    rl_action = self.rl_agent.select_action(rl_state, training=False)
                    if rl_action == 0:  # SKIP
                        final_confidence -= self.RL_SKIP_PENALTY
                        bot_logger.info(f"⚠️ RL recommends SKIP (-{self.RL_SKIP_PENALTY})")
                    else:
                        bot_logger.info(f"✅ RL action: {self.rl_agent.get_action_name(rl_action)}")
                except Exception as e:
                    bot_logger.debug(f"RL filter skipped: {e}")

        # Cap confidence
        final_confidence = min(1.0, max(0.0, final_confidence))

        # ── S/R Context Advisory (logging only) ──────────────────────
        price_zone = sr_signal.get('levels', {}).get('price_zone', '')
        if final_signal == 'BUY' and price_zone == 'AT_RESISTANCE':
            bot_logger.info("⚠️ S/R advisory: BUY near resistance")
        if final_signal == 'SELL' and price_zone == 'AT_SUPPORT':
            bot_logger.info("⚠️ S/R advisory: SELL near support")

        # ── Build detailed reason string ─────────────────────────────
        reason_parts = []
        reason_parts.append(
            f"SWEEP: {sweep_direction} ({sweep_confidence:.0%}) "
            f"[{sweep_signal.get('regime', '?')}, MSS={'✓' if sweep_signal.get('mss', {}).get('confirmed') else '✗'}]"
        )
        reason_parts.append(f"EMA: {ema_signal['signal']} ({ema_signal['confidence']:.0%})")
        reason_parts.append(f"Tech: {technical_signal['signal']} ({technical_signal['confidence']:.0%})")
        reason_parts.append(f"Regime: {regime}")
        if models_agreement > 0:
            reason_parts.append(f"Context: {models_agreement} models aligned")
        detailed_reason = " | ".join(reason_parts)

        scalping_signal_type = scalping_signal.get('signal', 'HOLD')
        scalping_confidence = scalping_signal.get('confidence', 0.0)

        result = {
            'signal': final_signal,
            'confidence': final_confidence,
            'models_agreement': models_agreement,
            'total_models': len(context_signals) + 1,  # context models + sweep
            'min_agreement_required': MIN_MODELS_AGREEMENT,
            'regime': regime,
            'detailed_reason': detailed_reason,
            'enriched_df': df_enriched,
            'models': {
                'sweep': {
                    'signal': sweep_direction,
                    'confidence': sweep_confidence,
                    'regime': sweep_signal.get('regime', 'unknown'),
                    'bias': sweep_signal.get('bias'),
                    'mss_confirmed': bool(sweep_signal.get('mss', {}).get('confirmed', False)),
                },
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
            'rsi': float(df_enriched['rsi'].iloc[-1]) if 'rsi' in df_enriched.columns and not pd.isna(df_enriched['rsi'].iloc[-1]) else 50.0,
            'sr_levels': sr_signal.get('levels', {}),
            'patterns': candle_signal.get('patterns', []),
            # ATR-centric fields
            'atr_regime': scalping_signal.get('atr_regime', 'neutral'),
            'atr_tp_ratio': scalping_signal.get('atr_tp_ratio', 1.4),
            'scalping_risk_reward': scalping_signal.get('risk_reward', {}),
            # Sweep data for unified SL/TP
            'sweep_sl_tp': sweep_signal.get('sweep_sl_tp'),
            'sweep_wick': (sweep_signal.get('sweep_sl_tp') or {}).get('sweep_wick')
                          or (sweep_signal.get('sweep') or {}).get('sweep_wick'),
        }

        if final_signal != 'SKIP':
            TradeLogger.log_signal(
                pair=pair,
                signal_type=final_signal,
                confidence=final_confidence,
                reason=detailed_reason,
                models_agreement=models_agreement
            )

        return result

    def should_trade(self, signal_result):
        """
        Determine if signal is strong enough to trade.
        With sweep-gated architecture, the sweep already validated structure.
        We only check confidence threshold.
        """
        threshold = self.learner.get_adjusted_threshold()

        if signal_result['signal'] == 'SKIP':
            return False

        effective_confidence = signal_result['confidence']

        # Sweep must have fired (it's the gate)
        sweep_model = signal_result.get('models', {}).get('sweep', {})
        if sweep_model.get('signal') not in ('BUY', 'SELL'):
            bot_logger.info("🚫 Sweep gate did not fire — no trade")
            return False

        # EMA crossover must not oppose sweep direction (HOLD = neutral, allowed)
        ema_model = signal_result.get('models', {}).get('ema_crossover', {})
        ema_dir = ema_model.get('signal', 'HOLD')
        if ema_dir in ('BUY', 'SELL') and ema_dir != signal_result['signal']:
            bot_logger.info(
                f"🚫 EMA crossover ({ema_dir}) opposes {signal_result['signal']} — no trade"
            )
            return False

        # RSI must not contradict trade direction
        rsi_val = signal_result.get('rsi', 50.0)
        if signal_result['signal'] == 'BUY' and rsi_val > 70:
            bot_logger.info(
                f"🚫 RSI overbought ({rsi_val:.1f} > 70) — blocking BUY"
            )
            return False
        if signal_result['signal'] == 'SELL' and rsi_val < 30:
            bot_logger.info(
                f"🚫 RSI oversold ({rsi_val:.1f} < 30) — blocking SELL"
            )
            return False

        # Require minimum model agreement
        agreement = signal_result.get('models_agreement', 0)
        if agreement < MIN_MODELS_AGREEMENT:
            bot_logger.info(
                f"🚫 Model agreement {agreement} < required {MIN_MODELS_AGREEMENT} — no trade"
            )
            return False

        if effective_confidence < threshold:
            bot_logger.info(
                f"📊 Confidence {effective_confidence:.2%} < threshold {threshold:.2%}"
            )
            return False

        return True

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
