"""
ML Trade Scorer — Real AI-Based Trade Decision Learning

Uses a Gradient Boosted Decision Tree (scikit-learn) to learn which
trade setups WIN and which LOSE based on 25+ features captured at
entry time.  Retrains automatically as new trades are recorded.

Features extracted per trade:
  - Per-model signal direction & confidence (9 models × 2 = 18 features)
  - Market regime (one-hot: trending/ranging/volatile)
  - Session (one-hot: asian/london/ny_overlap/new_york/off_hours)
  - Hour of day (cyclical sin/cos encoding)
  - Technical indicators at entry: RSI, ADX, ATR%, BB width, MACD histogram
  - Ensemble confidence & models agreement count
  - Recent performance context (win streak, loss streak, recent WR)
  - Pair encoding

Target: binary (1 = win, 0 = loss).  Breakeven trades excluded from training.

The scorer outputs a win probability (0.0–1.0) that the bot uses as
an additional gate before entering any trade.
"""

import os
import json
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque
from src.utils.logger import bot_logger

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    bot_logger.warning("scikit-learn not available — ML Trade Scorer disabled")

MODEL_PATH = 'models/ml_trade_scorer.pkl'
TRAINING_DATA_PATH = 'data/ml_training_data.json'

# ── Feature schema ────────────────────────────────────────────────
MODEL_NAMES = [
    'scalping', 'ema_crossover', 'candlestick', 'technical',
    'volume', 'multi_tf', 'support_resistance', 'lstm', 'sentiment',
]
SIGNAL_MAP = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0, 'SKIP': 0.0}
REGIME_MAP = {'trending': 0, 'ranging': 1, 'volatile': 2}
SESSION_MAP = {'asian': 0, 'london': 1, 'ny_overlap': 2, 'new_york': 3, 'off_hours': 4}
PAIR_MAP = {'EUR/USD': 0, 'GBP/USD': 1, 'USD/JPY': 2, 'MES': 3, 'NQ': 4}

# Minimum trades before the ML model activates (need enough data to learn from)
# Lowered from 200 → 50: faster activation with GBM regularization (max_depth=3, min_samples_leaf=5)
MIN_TRADES_TO_TRAIN = 50
# Retrain every N new trades
RETRAIN_INTERVAL = 15


class MLTradeScorer:
    """
    Gradient-boosted classifier that learns from past trade outcomes
    to predict win probability for new trade setups.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_data = []       # list of {features: [...], label: 0/1}
        self.trades_since_retrain = 0
        self.model_version = 0
        self.last_train_accuracy = 0.0
        self.last_train_cv_score = 0.0
        self.feature_importances = {}

        if not SKLEARN_AVAILABLE:
            bot_logger.warning("🧠 ML Trade Scorer: DISABLED (scikit-learn not installed)")
            return

        self._load_training_data()
        self._load_model()

        n = len(self.training_data)
        if n >= MIN_TRADES_TO_TRAIN and not self.is_trained:
            bot_logger.info(f"🧠 ML Scorer: {n} training samples found — training initial model")
            self._train()
        else:
            status = f"trained (v{self.model_version})" if self.is_trained else f"waiting ({n}/{MIN_TRADES_TO_TRAIN} trades)"
            bot_logger.info(f"🧠 ML Trade Scorer: {status}")

    # ==================================================================
    # Feature Extraction
    # ==================================================================
    @staticmethod
    def extract_features(signal_result: dict, pair: str, learner=None) -> list:
        """
        Extract a fixed-length feature vector from a trade setup.

        Args:
            signal_result: output from ensemble.get_trading_signal()
            pair: e.g. 'EUR/USD'
            learner: AdaptiveLearner instance (for context features)

        Returns:
            list of floats — one feature vector
        """
        features = []
        models = signal_result.get('models', {})

        # ── 1. Per-model signal direction & confidence (18 features) ──
        for model_name in MODEL_NAMES:
            ms = models.get(model_name, {})
            sig = ms.get('signal', 'HOLD')
            conf = float(ms.get('confidence', 0.0))
            # Direction relative to ensemble signal (+1 agree, -1 disagree, 0 hold)
            ensemble_sig = signal_result.get('signal', 'SKIP')
            if sig == ensemble_sig:
                direction = 1.0
            elif sig == 'HOLD' or sig == 'SKIP':
                direction = 0.0
            else:
                direction = -1.0
            features.append(direction)
            features.append(conf)

        # ── 2. Ensemble-level features (3 features) ──────────────────
        features.append(float(signal_result.get('confidence', 0.0)))
        features.append(float(signal_result.get('models_agreement', 0)))
        features.append(float(signal_result.get('total_models', 9)))

        # ── 3. Market regime one-hot (3 features) ────────────────────
        regime = signal_result.get('regime', 'ranging')
        for r in ['trending', 'ranging', 'volatile']:
            features.append(1.0 if regime == r else 0.0)

        # ── 4. Session one-hot (5 features) ──────────────────────────
        hour = datetime.utcnow().hour
        if 0 <= hour < 8:
            session = 'asian'
        elif 8 <= hour < 13:
            session = 'london'
        elif 13 <= hour < 17:
            session = 'ny_overlap'
        elif 17 <= hour < 22:
            session = 'new_york'
        else:
            session = 'off_hours'
        for s in ['asian', 'london', 'ny_overlap', 'new_york', 'off_hours']:
            features.append(1.0 if session == s else 0.0)

        # ── 5. Hour cyclical encoding (2 features) ───────────────────
        features.append(math.sin(2 * math.pi * hour / 24))
        features.append(math.cos(2 * math.pi * hour / 24))

        # ── 6. Technical indicators at entry (6 features) ────────────
        enriched_df = signal_result.get('enriched_df', None)
        if enriched_df is not None and len(enriched_df) > 0:
            latest = enriched_df.iloc[-1]
            close = float(latest.get('close', 0))
            rsi = float(latest.get('rsi', 50) or 50)
            adx = float(latest.get('adx', 0) or 0)
            atr = float(latest.get('atr', 0) or 0)
            atr_pct = (atr / close * 100) if close > 0 else 0
            bb_upper = float(latest.get('bb_upper', close) or close)
            bb_lower = float(latest.get('bb_lower', close) or close)
            bb_width = (bb_upper - bb_lower) / close * 100 if close > 0 else 0
            macd_hist = float(latest.get('macd_histogram', 0) or 0)
            # Normalize MACD histogram relative to price
            macd_norm = (macd_hist / close * 10000) if close > 0 else 0
        else:
            rsi, adx, atr_pct, bb_width, macd_norm = 50, 0, 0, 0, 0

        features.append(rsi / 100.0)          # Normalize to 0-1
        features.append(adx / 100.0)          # Normalize to 0-1
        features.append(atr_pct)              # Already in %
        features.append(bb_width)             # Already in %
        features.append(macd_norm)            # Normalized
        # Price vs EMA200 (trend alignment)
        if enriched_df is not None and len(enriched_df) > 0:
            ema200 = float(latest.get('ema_200', close) or close)
            ema_deviation = ((close - ema200) / close * 100) if close > 0 else 0
        else:
            ema_deviation = 0
        features.append(ema_deviation)

        # ── 7. Pair encoding (2 features) ────────────────────────────
        pair_norm = pair.upper().replace(' ', '') if pair else 'EUR/USD'
        features.append(1.0 if ('EUR' in pair_norm or 'MES' in pair_norm) else 0.0)
        features.append(1.0 if ('GBP' in pair_norm or 'NQ' in pair_norm) else 0.0)

        # ── 8. Signal direction (1 feature) ──────────────────────────
        ensemble_sig = signal_result.get('signal', 'SKIP')
        features.append(SIGNAL_MAP.get(ensemble_sig, 0.0))

        # ── 9. Adaptive learner context (5 features) ─────────────────
        if learner is not None:
            features.append(float(learner.consecutive_wins))
            features.append(float(learner.consecutive_losses))
            features.append(learner.get_recent_win_rate(5))
            features.append(learner.get_recent_win_rate(10))
            features.append(1.0 if learner.in_drawdown_protection else 0.0)
        else:
            features.extend([0.0, 0.0, 0.5, 0.5, 0.0])

        return features

    @staticmethod
    def feature_names() -> list:
        """Return human-readable names for each feature index."""
        names = []
        for m in MODEL_NAMES:
            names.append(f'{m}_direction')
            names.append(f'{m}_confidence')
        names.extend([
            'ensemble_confidence', 'models_agreement', 'total_models',
            'regime_trending', 'regime_ranging', 'regime_volatile',
            'session_asian', 'session_london', 'session_ny_overlap',
            'session_new_york', 'session_off_hours',
            'hour_sin', 'hour_cos',
            'rsi', 'adx', 'atr_pct', 'bb_width', 'macd_norm', 'ema_deviation',
            'pair_eur', 'pair_gbp',
            'signal_direction',
            'consec_wins', 'consec_losses', 'recent_wr_5', 'recent_wr_10',
            'drawdown_protection',
        ])
        return names

    # ==================================================================
    # Prediction
    # ==================================================================
    def predict_win_probability(self, signal_result: dict, pair: str, learner=None) -> float:
        """
        Predict probability that this trade setup will win.

        Returns:
            float 0.0–1.0 (win probability), or 0.5 if model not trained yet
        """
        if not SKLEARN_AVAILABLE or not self.is_trained or self.model is None:
            return 0.5  # Neutral — don't block anything when not trained

        try:
            features = self.extract_features(signal_result, pair, learner)
            X = np.array(features).reshape(1, -1)
            # predict_proba returns [[p_loss, p_win]]
            proba = self.model.predict_proba(X)[0]
            win_prob = float(proba[1]) if len(proba) > 1 else 0.5
            return win_prob
        except Exception as e:
            bot_logger.warning(f"🧠 ML predict failed: {e}")
            return 0.5

    # ==================================================================
    # Recording (called after trade closes)
    # ==================================================================
    def record_trade(self, features: list, is_win: bool):
        """
        Record a completed trade's features + outcome for training.

        Args:
            features: feature vector captured at entry time
            is_win: True if trade was profitable
        """
        if not SKLEARN_AVAILABLE:
            return

        self.training_data.append({
            'features': features,
            'label': 1 if is_win else 0,
            'timestamp': datetime.now().isoformat(),
        })

        # Keep last 2000 trades (rolling window)
        if len(self.training_data) > 2000:
            self.training_data = self.training_data[-2000:]

        self.trades_since_retrain += 1
        self._save_training_data()

        n = len(self.training_data)
        # Auto-retrain after enough new data
        if n >= MIN_TRADES_TO_TRAIN and self.trades_since_retrain >= RETRAIN_INTERVAL:
            bot_logger.info(
                f"🧠 ML Scorer: {self.trades_since_retrain} new trades → triggering retrain "
                f"(total samples: {n})"
            )
            self._train()
        elif n < MIN_TRADES_TO_TRAIN:
            bot_logger.info(f"🧠 ML Scorer: {n}/{MIN_TRADES_TO_TRAIN} trades collected — not enough to train yet")

    # ==================================================================
    # Training
    # ==================================================================
    def _train(self):
        """Train or retrain the gradient boosted classifier."""
        if not SKLEARN_AVAILABLE:
            return

        n = len(self.training_data)
        if n < MIN_TRADES_TO_TRAIN:
            return

        try:
            X = np.array([d['features'] for d in self.training_data])
            y = np.array([d['label'] for d in self.training_data])

            # Check class balance
            n_wins = int(y.sum())
            n_losses = n - n_wins
            if n_wins < 3 or n_losses < 3:
                bot_logger.warning(
                    f"🧠 ML Scorer: skipping train — need at least 3 wins and 3 losses "
                    f"(have {n_wins}W / {n_losses}L)"
                )
                return

            # Scale sample weights: give more weight to recent trades
            # Exponential decay: most recent trade has weight 1.0, oldest has ~0.3
            decay = 0.997
            sample_weights = np.array([decay ** (n - 1 - i) for i in range(n)])

            # Adaptive hyperparameters based on dataset size
            n_estimators = min(200, max(50, n * 2))
            max_depth = 3 if n < 100 else 4
            learning_rate = 0.1 if n < 100 else 0.05
            min_samples_leaf = max(2, n // 20)

            base_model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
            )

            # Cross-validate if enough data
            if n >= 30:
                cv_folds = min(5, n // 6)
                cv_scores = cross_val_score(base_model, X, y, cv=cv_folds, scoring='accuracy')
                self.last_train_cv_score = float(np.mean(cv_scores))
                bot_logger.info(
                    f"🧠 ML Scorer CV: {self.last_train_cv_score:.1%} "
                    f"(±{np.std(cv_scores):.1%}) over {cv_folds} folds"
                )

            # Train final model on all data with sample weights
            base_model.fit(X, y, sample_weight=sample_weights)

            # Calibrate probabilities for better reliability
            # (only if enough data for calibration)
            if n >= 50:
                self.model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                self.model = base_model

            self.is_trained = True
            self.model_version += 1
            self.trades_since_retrain = 0
            self.last_train_accuracy = float(base_model.score(X, y))

            # Feature importance analysis
            importances = base_model.feature_importances_
            fnames = self.feature_names()
            ranked = sorted(
                zip(fnames, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            self.feature_importances = {name: float(imp) for name, imp in ranked[:10]}

            bot_logger.info(
                f"🧠 ML Scorer TRAINED v{self.model_version}: "
                f"accuracy={self.last_train_accuracy:.1%} | "
                f"samples={n} ({n_wins}W/{n_losses}L) | "
                f"estimators={n_estimators} | depth={max_depth}"
            )
            bot_logger.info(
                f"🧠 Top features: " +
                " | ".join(f"{name}: {imp:.1%}" for name, imp in ranked[:5])
            )

            self._save_model()

        except Exception as e:
            bot_logger.error(f"🧠 ML Scorer training failed: {e}", exc_info=True)

    def force_retrain(self):
        """Manually trigger a retrain (e.g., from dashboard)."""
        self.trades_since_retrain = RETRAIN_INTERVAL  # Force threshold
        self._train()

    # ==================================================================
    # Persistence
    # ==================================================================
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'version': self.model_version,
                    'accuracy': self.last_train_accuracy,
                    'cv_score': self.last_train_cv_score,
                    'feature_importances': self.feature_importances,
                    'timestamp': datetime.now().isoformat(),
                }, f)
            bot_logger.info(f"🧠 ML model saved: v{self.model_version}")
        except Exception as e:
            bot_logger.warning(f"🧠 ML model save failed: {e}")

    def _load_model(self):
        """Load a previously trained model."""
        if not os.path.exists(MODEL_PATH):
            return
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.model_version = data.get('version', 1)
            self.last_train_accuracy = data.get('accuracy', 0.0)
            self.last_train_cv_score = data.get('cv_score', 0.0)
            self.feature_importances = data.get('feature_importances', {})
            self.is_trained = True
            bot_logger.info(
                f"🧠 ML model loaded: v{self.model_version} | "
                f"accuracy={self.last_train_accuracy:.1%} | "
                f"cv={self.last_train_cv_score:.1%}"
            )
        except Exception as e:
            bot_logger.warning(f"🧠 ML model load failed: {e}")
            self.model = None
            self.is_trained = False

    def _save_training_data(self):
        """Persist training data to disk."""
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            # Only save features + label to keep file small
            compact = [
                {'f': d['features'], 'l': d['label'], 't': d.get('timestamp', '')}
                for d in self.training_data[-2000:]
            ]
            with open(TRAINING_DATA_PATH, 'w') as f:
                json.dump(compact, f)
        except Exception as e:
            bot_logger.warning(f"🧠 ML training data save failed: {e}")

    def _load_training_data(self):
        """Load previously saved training data."""
        if not os.path.exists(TRAINING_DATA_PATH):
            return
        try:
            with open(TRAINING_DATA_PATH, 'r') as f:
                compact = json.load(f)
            self.training_data = [
                {'features': d['f'], 'label': d['l'], 'timestamp': d.get('t', '')}
                for d in compact
            ]
            bot_logger.info(f"🧠 ML training data loaded: {len(self.training_data)} samples")
        except Exception as e:
            bot_logger.warning(f"🧠 ML training data load failed: {e}")

    # ==================================================================
    # Status / Diagnostics
    # ==================================================================
    def get_status(self) -> dict:
        """Return status dict for logging / dashboard."""
        return {
            'available': SKLEARN_AVAILABLE,
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'training_samples': len(self.training_data),
            'min_required': MIN_TRADES_TO_TRAIN,
            'trades_since_retrain': self.trades_since_retrain,
            'retrain_interval': RETRAIN_INTERVAL,
            'last_accuracy': self.last_train_accuracy,
            'last_cv_score': self.last_train_cv_score,
            'top_features': self.feature_importances,
        }
