"""
Sweep-Gated Entry System — RayAlgo v3 + Intelligent ML

Architecture:
  Gate:    LiquiditySweepAnalyzer (4-layer: Bias → Sweep → MSS → Entry)
           If sweep does NOT fire → signal = SKIP.  No vote can override.
  Confirm: EMA Crossover (trend alignment) + Technical (MACD/BB momentum)
           These BOOST or REDUCE sweep confidence — they cannot create a signal.
  Context: All other models (scalping, volume, sentiment, LSTM, S/R, candle, MTF)
           still run for logging / adaptive learning — but do NOT affect entry.
  ML:      IntelligentTrader provides ML-augmented signal analysis via:
           - Feature generators (TA-Lib, statistical, rolling)
           - Multiple ML classifiers (Neural Net, Gradient Boost, SVC)
           - Advanced signal generation with threshold/crossover rules

LSTM is optional — if TensorFlow is not installed the system runs without it.
IntelligentTrader is optional — provides ML boost when models are trained.
"""
import pandas as pd
import os
from src.risk.sl_tp import calculate_sl_tp as calculate_structure_sl_tp
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
from src.instruments import REGISTRY
from config.strategy_config import ASSET_CLASS, ENSEMBLE_CONFIDENCE_THRESHOLD, MIN_MODELS_AGREEMENT

# Import IntelligentTrader with availability check
try:
    from src.ai.intelligent_trading import IntelligentTrader
    INTELLIGENT_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_AVAILABLE = False
    bot_logger.warning(f"IntelligentTrader not available: {e}")

# Import AdvancedStrategies (Binance bot strategies)
try:
    from src.ai.advanced_strategies import AdvancedStrategies
    ADVANCED_STRATS_AVAILABLE = True
except ImportError as e:
    ADVANCED_STRATS_AVAILABLE = False
    bot_logger.warning(f"AdvancedStrategies not available: {e}")

# Import DynamicSLTP manager
try:
    from src.ai.dynamic_sltp import DynamicSLTPManager
    DYNAMIC_SLTP_AVAILABLE = True
except ImportError as e:
    DYNAMIC_SLTP_AVAILABLE = False
    bot_logger.warning(f"DynamicSLTP not available: {e}")


class EnsembleTrader:
    """Sweep-gated entry system with confirmation boosters, intelligent ML, and adaptive learning."""

    # Confirmation boost/penalty amounts
    EMA_CONFIRM_BOOST = 0.05     # EMA aligned with sweep direction
    EMA_OPPOSE_PENALTY = 0.10    # EMA opposes sweep direction
    TECH_CONFIRM_BOOST = 0.03    # Technical momentum matches sweep
    TECH_OPPOSE_PENALTY = 0.05   # Technical momentum opposes sweep
    LSTM_CONFIRM_BOOST = 0.08    # LSTM direction agrees with sweep
    LSTM_OPPOSE_PENALTY = 0.20   # LSTM direction opposes sweep (strong filter)
    RL_SKIP_PENALTY = 0.08       # RL agent recommends skipping
    INTEL_CONFIRM_BOOST = 0.10   # IntelligentTrader confirms sweep direction
    INTEL_OPPOSE_PENALTY = 0.15  # IntelligentTrader opposes sweep direction
    INTEL_HIGH_CONF_BOOST = 0.05 # Extra boost for high-confidence intelligent signal
    ADV_CONFIRM_BOOST = 0.08     # Advanced strategies confirm sweep direction
    ADV_OPPOSE_PENALTY = 0.12    # Advanced strategies oppose sweep direction
    ADV_MULTI_AGREE_BOOST = 0.05 # Extra boost when multiple advanced strategies agree

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        v = os.getenv(name)
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _looks_like_five_minute_data(df: pd.DataFrame) -> bool:
        """Best-effort cadence check used for safe 5M fallback."""
        if df is None or len(df) < 30 or 'datetime' not in df.columns:
            return False
        try:
            dt = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dropna()
            if len(dt) < 20:
                return False
            diffs = dt.sort_values().diff().dropna()
            if diffs.empty:
                return False
            median_seconds = float(diffs.dt.total_seconds().median())
            return 240 <= median_seconds <= 360
        except Exception:
            return False

    def __init__(self, newsapi_key=None, broker=None):
        # RSI filter thresholds (configurable at runtime via env vars)
        self.rsi_buy_block = self._env_float('RSI_BUY_BLOCK', 70.0)
        self.rsi_sell_block = self._env_float('RSI_SELL_BLOCK', 30.0)
        self.rsi_buy_block_high_vol = self._env_float('RSI_BUY_BLOCK_HIGH_VOL', self.rsi_buy_block)
        self.rsi_sell_block_high_vol = self._env_float('RSI_SELL_BLOCK_HIGH_VOL', self.rsi_sell_block)

        # ── Primary: sweep gate ──────────────────────────────────────
        self.sweep = LiquiditySweepAnalyzer()

        # ── Confirmation models (only these affect confidence) ───────
        self.ema_crossover = EMACrossoverAnalyzer()
        self.technical = TechnicalAnalyzer()

        # ── Intelligent ML Trading System ────────────────────────────
        if INTELLIGENT_AVAILABLE:
            try:
                self.intelligent = IntelligentTrader(broker=broker)
                self.intelligent_available = True
                bot_logger.info("🧠 IntelligentTrader: ML classifiers active (GB, SVC, NN)")
            except Exception as e:
                self.intelligent = None
                self.intelligent_available = False
                bot_logger.warning(f"⚠️ IntelligentTrader init failed: {e}")
        else:
            self.intelligent = None
            self.intelligent_available = False

        # ── Advanced Strategies (Binance bot strategies) ─────────────
        if ADVANCED_STRATS_AVAILABLE:
            try:
                self.advanced_strategies = AdvancedStrategies()
                self.advanced_strats_available = True
                bot_logger.info("📈 AdvancedStrategies: 8 strategies active (FibMACD, StochRSI, HA, etc)")
            except Exception as e:
                self.advanced_strategies = None
                self.advanced_strats_available = False
                bot_logger.warning(f"⚠️ AdvancedStrategies init failed: {e}")
        else:
            self.advanced_strategies = None
            self.advanced_strats_available = False

        # ── Dynamic SL/TP Manager ────────────────────────────────────
        if DYNAMIC_SLTP_AVAILABLE:
            try:
                self.sltp_manager = DynamicSLTPManager(
                    use_trailing_stop=True,
                    trailing_callback=0.002
                )
                self.sltp_available = True
                bot_logger.info("🎯 DynamicSLTP: Trailing stops + swing-based SL/TP active")
            except Exception as e:
                self.sltp_manager = None
                self.sltp_available = False
                bot_logger.warning(f"⚠️ DynamicSLTP init failed: {e}")
        else:
            self.sltp_manager = None
            self.sltp_available = False

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
            'intelligent': 0.0,  # Will be enabled when trained
            'advanced_strategies': 0.0,  # Binance bot strategies
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

        bot_logger.info("🎯 Sweep-Gated Entry System (RayAlgo v3 + Intelligent ML)")
        bot_logger.info("   Gate:    LiquiditySweep (4-layer: Bias → Sweep → MSS → Entry)")
        bot_logger.info("   Confirm: EMA Crossover + Technical (boost/reduce only)")
        bot_logger.info(f"   ML:      IntelligentTrader ({'active' if self.intelligent_available else 'disabled'})")
        bot_logger.info(f"   Strats:  AdvancedStrategies ({'active' if self.advanced_strats_available else 'disabled'})")
        bot_logger.info(f"   SL/TP:   DynamicSLTP ({'active' if self.sltp_available else 'disabled'})")
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

        # If broker 5M is missing, reuse the caller frame only when cadence looks 5M.
        if (df_5m is None or len(df_5m) < 30) and self._looks_like_five_minute_data(df_enriched):
            df_5m = df_enriched.tail(250).copy()
            bot_logger.info(
                f"📊 {pair} using caller 5M frame fallback: {len(df_5m)} rows"
            )

        has_valid_5m_bias = df_5m is not None and len(df_5m) >= 30

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

        # ── Step 4b: Run IntelligentTrader ML analysis ───────────────
        intelligent_signal = {'signal': 'HOLD', 'confidence': 0.0}
        if self.intelligent_available:
            try:
                intelligent_signal = self.intelligent.get_trading_signal(df_enriched, pair)
            except Exception as e:
                bot_logger.debug(f"IntelligentTrader analysis failed: {e}")

        # ── Step 4c: Run Advanced Strategies (Binance bot strategies) ─
        advanced_signal = {'signal': 'HOLD', 'confidence': 0.0, 'buy_votes': 0, 'sell_votes': 0}
        if self.advanced_strats_available:
            try:
                # Use combined signal from multiple strategies for robustness
                advanced_signal = self.advanced_strategies.get_combined_signal(
                    df_enriched,
                    strategies=['stoch_rsi_macd', 'golden_cross', 'triple_ema_stoch', 'heikin_ashi_ema'],
                    min_agreement=2
                )
            except Exception as e:
                bot_logger.debug(f"AdvancedStrategies analysis failed: {e}")

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
        sweep_bias = sweep_signal.get('bias', 'HOLD') or 'HOLD'  # always defined

        if sweep_direction not in ('BUY', 'SELL'):
            # Sweep didn't fire — check for fallback entries
            ema_dir = ema_signal['signal']
            tech_dir = technical_signal['signal']
            
            # Get directional bias from sweep signal (bias comes from 5M regime detection)
            sweep_bias = sweep_signal.get('bias', 'HOLD')
            if sweep_bias not in ('BUY', 'SELL'):
                sweep_bias = 'HOLD'
            elif not has_valid_5m_bias:
                # Do not trust inferred/partial bias when 5M context is unavailable.
                sweep_bias = 'HOLD'
                bot_logger.info(f"🚫 5M bias fallback disabled for {pair}: missing/insufficient 5M candles")
            
            bot_logger.info(f"🔍 No sweep - checking fallback: EMA={ema_dir}, Tech={tech_dir}, SweepBias={sweep_bias} (regime={regime})")
            
            # Volatile regimes are too unpredictable for fallback entries
            _volatile_regimes = ('volatile', 'high_volatility')

            # MSS (Market Structure Shift) from the sweep detector
            # A confirmed MSS means price has broken the last internal HH/HL (bullish) or LH/LL (bearish)
            _mss_confirmed = bool(sweep_signal.get('mss', {}).get('confirmed', False))

            # Fallback 1: EMA + Technical agree — only trade trending regimes, not ranging chaos
            # Require Tech confidence ≥ 40% — low-confidence Tech signals (e.g. 23%) shouldn't gate
            if (ema_dir in ('BUY', 'SELL') and ema_dir == tech_dir
                    and technical_signal.get('confidence', 0.0) >= 0.40
                    and regime not in _volatile_regimes
                    and regime in ('trend_up', 'trend_down', 'trending')):
                final_signal = ema_dir
                final_confidence = 0.65
                models_agreement = 2
                bot_logger.info(
                    f"🔄 No sweep → EMA+Tech fallback: {ema_dir} "
                    f"(EMA={ema_signal['confidence']:.0%}, "
                    f"Tech={technical_signal['confidence']:.0%})"
                )
            # Fallback 2: Requires MSS + Tech + SweepBias all aligned (no sweep fired but structure shifted)
            elif (_mss_confirmed
                    and sweep_bias in ('BUY', 'SELL')
                    and tech_dir == sweep_bias                    # Tech must actively agree
                    and technical_signal.get('confidence', 0.0) >= 0.40  # Tech must be confident
                    and regime not in _volatile_regimes           # No volatile markets
                    and ema_dir != ('SELL' if sweep_bias == 'BUY' else 'BUY')):  # EMA must not oppose
                final_signal = sweep_bias
                final_confidence = 0.65
                models_agreement = 2
                bot_logger.info(
                    f"🔄 No sweep → MSS+Bias+Tech fallback: {sweep_bias} "
                    f"(MSS=✓, Tech={tech_dir} {technical_signal['confidence']:.0%}, EMA={ema_dir})"
                )
            else:
                final_signal = 'SKIP'
                final_confidence = 0.0
                models_agreement = 0
                _skip_reason = []
                if regime in _volatile_regimes:
                    _skip_reason.append(f"volatile regime ({regime})")
                if not _mss_confirmed:
                    _skip_reason.append("MSS not confirmed")
                if sweep_bias not in ('BUY', 'SELL'):
                    _skip_reason.append("no directional bias")
                if tech_dir != sweep_bias:
                    _skip_reason.append(f"Tech={tech_dir} doesn't confirm bias={sweep_bias}")
                if technical_signal.get('confidence', 0.0) < 0.40:
                    _skip_reason.append(f"Tech confidence too low ({technical_signal.get('confidence', 0.0):.0%})")
                bot_logger.info(f"🚫 No sweep and no fallback conditions met: {', '.join(_skip_reason) or 'conditions not met'}")
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

            # ── IntelligentTrader ML confirmation ─────────────────────
            if self.intelligent_available and intelligent_signal.get('signal') != 'HOLD':
                intel_dir = intelligent_signal.get('signal')
                intel_conf = intelligent_signal.get('confidence', 0.0)
                if intel_dir == sweep_direction:
                    final_confidence += self.INTEL_CONFIRM_BOOST
                    bot_logger.info(
                        f"🧠 IntelligentTrader confirms {sweep_direction} (+{self.INTEL_CONFIRM_BOOST:.0%}) → {final_confidence:.1%}"
                    )
                    # Extra boost for high-confidence intelligent signal
                    if intel_conf > 0.7:
                        final_confidence += self.INTEL_HIGH_CONF_BOOST
                        bot_logger.info(
                            f"🧠 IntelligentTrader HIGH confidence {intel_conf:.0%} (+{self.INTEL_HIGH_CONF_BOOST:.0%}) → {final_confidence:.1%}"
                        )
                elif intel_dir in ('BUY', 'SELL') and intel_dir != sweep_direction:
                    final_confidence -= self.INTEL_OPPOSE_PENALTY
                    bot_logger.info(
                        f"⚠️ IntelligentTrader opposes {sweep_direction} (−{self.INTEL_OPPOSE_PENALTY:.0%}) → {final_confidence:.1%}"
                    )

            # ── Advanced Strategies confirmation (Binance bot strategies)
            if self.advanced_strats_available and advanced_signal.get('signal') != 'HOLD':
                adv_dir = advanced_signal.get('signal')
                adv_conf = advanced_signal.get('confidence', 0.0)
                buy_votes = advanced_signal.get('buy_votes', 0)
                sell_votes = advanced_signal.get('sell_votes', 0)
                
                if adv_dir == sweep_direction:
                    final_confidence += self.ADV_CONFIRM_BOOST
                    bot_logger.info(
                        f"📈 AdvancedStrategies confirms {sweep_direction} (+{self.ADV_CONFIRM_BOOST:.0%}) → {final_confidence:.1%}"
                    )
                    # Extra boost for strong multi-strategy agreement
                    agreeing_count = buy_votes if adv_dir == 'BUY' else sell_votes
                    if agreeing_count >= 3:
                        final_confidence += self.ADV_MULTI_AGREE_BOOST
                        bot_logger.info(
                            f"📈 AdvancedStrategies strong consensus ({agreeing_count} strategies) (+{self.ADV_MULTI_AGREE_BOOST:.0%}) → {final_confidence:.1%}"
                        )
                elif adv_dir in ('BUY', 'SELL') and adv_dir != sweep_direction:
                    final_confidence -= self.ADV_OPPOSE_PENALTY
                    bot_logger.info(
                        f"⚠️ AdvancedStrategies opposes {sweep_direction} (−{self.ADV_OPPOSE_PENALTY:.0%}) → {final_confidence:.1%}"
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
            if self.intelligent_available:
                context_signals['intelligent'] = intelligent_signal.get('signal', 'HOLD')
            if self.advanced_strats_available:
                context_signals['advanced_strategies'] = advanced_signal.get('signal', 'HOLD')

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
                        hour=hour, spread=broker_spread or (REGISTRY[pair].spread_default if pair in REGISTRY else 0.00015),
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

        # ── S/R Gate: block counter-S/R entries ──────────────────────
        # Sweep-confirmed entries intentionally pass through S/R (liquidity flip).
        # Fallback/bias entries must NOT buy resistance or sell support.
        _sweep_confirmed = sweep_direction in ('BUY', 'SELL')
        price_zone = sr_signal.get('levels', {}).get('price_zone', '')
        if final_signal == 'BUY' and price_zone == 'AT_RESISTANCE':
            if _sweep_confirmed:
                bot_logger.info("⚠️ S/R advisory: BUY near resistance (sweep-confirmed, allowing)")
            else:
                bot_logger.info("🚫 S/R block: BUY at resistance — no sweep to confirm breakout, skipping")
                final_signal = 'SKIP'
                final_confidence = 0.0
        if final_signal == 'SELL' and price_zone == 'AT_SUPPORT':
            if _sweep_confirmed:
                bot_logger.info("⚠️ S/R advisory: SELL near support (sweep-confirmed, allowing)")
            else:
                bot_logger.info("🚫 S/R block: SELL at support — no sweep to confirm breakdown, skipping")
                final_signal = 'SKIP'
                final_confidence = 0.0

        # ── Build detailed reason string ─────────────────────────────
        reason_parts = []
        reason_parts.append(
            f"SWEEP: {sweep_direction} ({sweep_confidence:.0%}) "
            f"[{sweep_signal.get('regime', '?')}, MSS={'✓' if sweep_signal.get('mss', {}).get('confirmed') else '✗'}]"
        )
        reason_parts.append(f"EMA: {ema_signal['signal']} ({ema_signal['confidence']:.0%})")
        reason_parts.append(f"Tech: {technical_signal['signal']} ({technical_signal['confidence']:.0%})")
        if self.intelligent_available:
            reason_parts.append(f"ML: {intelligent_signal.get('signal', 'HOLD')} ({intelligent_signal.get('confidence', 0):.0%})")
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
            'sweep_bias': sweep_bias,
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
                'intelligent': {
                    'signal': intelligent_signal.get('signal', 'HOLD'),
                    'confidence': intelligent_signal.get('confidence', 0.0),
                    'trade_score': intelligent_signal.get('trade_score', 0.0),
                    'signal_strength': intelligent_signal.get('signal_strength', 'weak'),
                    'available': self.intelligent_available,
                },
                'advanced_strategies': {
                    'signal': advanced_signal.get('signal', 'HOLD'),
                    'confidence': advanced_signal.get('confidence', 0.0),
                    'buy_votes': advanced_signal.get('buy_votes', 0),
                    'sell_votes': advanced_signal.get('sell_votes', 0),
                    'strategies': advanced_signal.get('strategies', {}),
                    'available': self.advanced_strats_available,
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
            'pair': pair,  # Include pair for symbol-based detection (e.g., futures vs forex)
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
        Fallback: If sweep didn't fire but EMA+Tech agree OR regime bias is strong, allow trade.
        """
        threshold = self.learner.get_adjusted_threshold()

        if signal_result['signal'] == 'SKIP':
            return False

        effective_confidence = signal_result['confidence']

        # Check if sweep fired or if this is a fallback entry
        sweep_model = signal_result.get('models', {}).get('sweep', {})
        sweep_fired = sweep_model.get('signal') in ('BUY', 'SELL')

        # Futures mode: enforce strict sweep-only entries to avoid wrong-direction
        # fallback trades when structure confirmation is absent.
        # Detect futures by symbol: MES, MNQ, ES, NQ, MESO, MNQO, YM, CL, etc.
        pair = signal_result.get('pair', '')
        is_futures = pair.upper() in ('MES', 'MNQ', 'ES', 'NQ', 'MESO', 'MNQO', 'YM', 'CL') or \
                     ASSET_CLASS == 'futures'
        
        if is_futures and not sweep_fired:
            bot_logger.info(f"🚫 Futures mode ({pair}): sweep did not fire — fallback entry disabled")
            return False
        
        # Check EMA and Technical directions
        ema_model = signal_result.get('models', {}).get('ema_crossover', {})
        tech_model = signal_result.get('models', {}).get('technical', {})
        ema_dir = ema_model.get('signal', 'HOLD')
        tech_dir = tech_model.get('signal', 'HOLD')
        
        # Fallback 1: EMA + Technical agree on direction
        is_ema_tech_fallback = (
            not sweep_fired and
            ema_dir == signal_result['signal'] and
            tech_dir == signal_result['signal'] and
            ema_dir in ('BUY', 'SELL')
        )
        
        # Fallback 2: Sweep bias matches signal and indicators don't oppose
        sweep_bias = signal_result.get('sweep_bias', 'HOLD')
        if sweep_bias not in ('BUY', 'SELL'):
            sweep_bias = 'HOLD'
            
        opposing_dir = 'SELL' if signal_result['signal'] == 'BUY' else 'BUY'
        is_bias_fallback = (
            not sweep_fired and
            sweep_bias == signal_result['signal'] and
            ema_dir != opposing_dir and
            tech_dir != opposing_dir
        )
        
        is_fallback_entry = is_ema_tech_fallback or is_bias_fallback
        
        if not sweep_fired and not is_fallback_entry:
            bot_logger.info("🚫 Sweep gate did not fire and no fallback conditions — no trade")
            return False
        
        if is_ema_tech_fallback:
            bot_logger.info(f"🔄 FALLBACK ENTRY: EMA+Tech agree on {signal_result['signal']} (no sweep required)")
        elif is_bias_fallback:
            bot_logger.info(f"🔄 FALLBACK ENTRY: 5M bias {sweep_bias} matches signal (no sweep required)")

        # EMA crossover must not oppose signal direction (HOLD = neutral, allowed)
        if ema_dir in ('BUY', 'SELL') and ema_dir != signal_result['signal']:
            bot_logger.info(
                f"🚫 EMA crossover ({ema_dir}) opposes {signal_result['signal']} — no trade"
            )
            return False

        # RSI must not contradict trade direction (with optional high-volatility thresholds)
        rsi_val = signal_result.get('rsi', 50.0)
        regime_name = str(signal_result.get('regime', '') or '').lower()
        sweep_regime = str(
            (signal_result.get('models', {}).get('sweep', {}) or {}).get('regime', '') or ''
        ).lower()
        is_high_vol = (
            regime_name in ('high_volatility', 'volatile', 'volatility')
            or sweep_regime in ('high_volatility', 'volatile', 'volatility')
        )
        buy_block = self.rsi_buy_block_high_vol if is_high_vol else self.rsi_buy_block
        sell_block = self.rsi_sell_block_high_vol if is_high_vol else self.rsi_sell_block

        if signal_result['signal'] == 'BUY' and rsi_val > buy_block:
            bot_logger.info(
                f"🚫 RSI overbought ({rsi_val:.1f} > {buy_block:.1f}) — blocking BUY"
            )
            return False
        if signal_result['signal'] == 'SELL' and rsi_val < sell_block:
            bot_logger.info(
                f"🚫 RSI oversold ({rsi_val:.1f} < {sell_block:.1f}) — blocking SELL"
            )
            return False

        # Require minimum model agreement (skip for fallback entries)
        agreement = signal_result.get('models_agreement', 0)
        if agreement < MIN_MODELS_AGREEMENT and not is_fallback_entry:
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

    def get_dynamic_sl_tp(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        symbol: str = None,
        sr_levels: dict = None,
        sweep_wick: float = None,
        timeframe: str = '5m',
    ) -> dict:
        """
        Calculate dynamic SL/TP using swing points and ATR.
        
        Args:
            df: OHLCV DataFrame with indicators
            direction: 'BUY' or 'SELL'
            entry_price: Current entry price
            symbol: Trading symbol (optional)
            
        Returns:
            Dict with sl_price, tp_price, risk_reward, etc.
        """
        # Prefer structure-aware SL/TP when symbol context is available.
        if symbol:
            try:
                structure_result = calculate_structure_sl_tp(
                    df=df,
                    direction=direction,
                    pair=symbol,
                    timeframe=timeframe,
                    sr_levels=sr_levels,
                    sweep_wick=sweep_wick,
                )
                if structure_result:
                    return {
                        'sl_price': structure_result['stop_loss'],
                        'tp_price': structure_result['take_profit'],
                        'sl_distance': structure_result.get('sl_distance'),
                        'tp_distance': structure_result.get('tp_distance'),
                        'risk_reward': structure_result.get('rr_ratio', 2.0),
                        'method': 'structure_aware',
                    }
            except Exception as e:
                bot_logger.warning(f"Structure-aware SL/TP failed: {e} — falling back")

        if not self.sltp_available:
            # Fallback to simple percentage-based SL/TP
            sl_pct = 0.005  # 0.5%
            tp_pct = 0.010  # 1.0%
            if direction == 'BUY':
                return {
                    'sl_price': entry_price * (1 - sl_pct),
                    'tp_price': entry_price * (1 + tp_pct),
                    'risk_reward': 2.0,
                    'method': 'fallback'
                }
            else:
                return {
                    'sl_price': entry_price * (1 + sl_pct),
                    'tp_price': entry_price * (1 - tp_pct),
                    'risk_reward': 2.0,
                    'method': 'fallback'
                }
        
        dynamic_result = self.sltp_manager.calculate_sl_tp(df, direction, entry_price, symbol)
        if isinstance(dynamic_result, dict):
            dynamic_result.setdefault('method', 'dynamic')
        return dynamic_result

    def start_trailing_stop(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        quantity: float = 1.0
    ):
        """Start tracking a position for trailing stop updates."""
        if self.sltp_available:
            self.sltp_manager.start_tracking(
                symbol, direction, entry_price, sl_price, tp_price, quantity
            )

    def update_trailing_stop(self, symbol: str, current_price: float) -> dict:
        """Update trailing stop for an active position."""
        if not self.sltp_available:
            return None
        return self.sltp_manager.update_trailing_stop(symbol, current_price)

    def check_sltp_exit(self, symbol: str, current_price: float) -> dict:
        """Check if position should exit due to SL or TP hit."""
        if not self.sltp_available:
            return None
        return self.sltp_manager.check_exit(symbol, current_price)

    def stop_trailing(self, symbol: str):
        """Stop tracking a position (position closed)."""
        if self.sltp_available:
            self.sltp_manager.stop_tracking(symbol)

    def get_advanced_strategy_signal(
        self,
        df: pd.DataFrame,
        strategy: str = 'stoch_rsi_macd'
    ) -> dict:
        """
        Get signal from a specific advanced strategy.
        
        Available strategies: fib_macd, stoch_rsi_macd, golden_cross,
        triple_ema_stoch, heikin_ashi_ema, breakout, stoch_bb, wick_reversal
        """
        if not self.advanced_strats_available:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Not available'}
        return self.advanced_strategies.get_signal(df, strategy)
        
        # Also pass to IntelligentTrader for ML learning
        if self.intelligent_available:
            try:
                prediction = trade_data.get('prediction', {})
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', 0)
                direction = 'long' if trade_data.get('signal') == 'BUY' else 'short'
                self.intelligent.record_trade_outcome(prediction, entry_price, exit_price, direction)
            except Exception as e:
                bot_logger.debug(f"IntelligentTrader outcome recording failed: {e}")

    def train_intelligent_models(self, df: pd.DataFrame, pair: str = None):
        """Train IntelligentTrader ML models on historical data."""
        if not self.intelligent_available:
            return {'error': 'IntelligentTrader not available'}
        return self.intelligent.train(df, pair, force=True)

    def get_intelligent_analysis(self, df: pd.DataFrame, pair: str = None) -> dict:
        """Get detailed analysis from IntelligentTrader."""
        if not self.intelligent_available:
            return {'error': 'IntelligentTrader not available'}
        return self.intelligent.get_trading_signal(df, pair)

    def backtest_intelligent(self, df: pd.DataFrame, pair: str = None) -> dict:
        """Run backtest using IntelligentTrader strategy."""
        if not self.intelligent_available:
            return {'error': 'IntelligentTrader not available'}
        return self.intelligent.backtest_strategy(df, pair)

    def get_feature_importance(self) -> dict:
        """Get feature importance from IntelligentTrader models."""
        if not self.intelligent_available:
            return {'error': 'IntelligentTrader not available'}
        return self.intelligent.get_feature_analysis()

    def save_intelligent_models(self):
        """Save IntelligentTrader models to disk."""
        if not self.intelligent_available:
            return {'error': 'IntelligentTrader not available'}
        self.intelligent.save_models()
        return {'status': 'saved'}

    def get_intelligent_summary(self) -> str:
        """Get summary of IntelligentTrader performance."""
        if not self.intelligent_available:
            return "IntelligentTrader not available"
        return self.intelligent.get_trading_summary()
