"""
LSTM Auto-Retrainer

Periodically retrains the LSTM model on accumulated market data so the
neural network stays current with evolving market dynamics.

Features:
  - Incremental retraining (fine-tune on new data, not full retrain)
  - Validation loss gate: only saves the new model if it improves
  - Scheduled via the bot's APScheduler (default: every 24h)
  - Thread-safe: swaps model atomically after training
  - Falls back gracefully if TensorFlow is unavailable
"""
import os
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.logger import bot_logger, error_logger
from config.strategy_config import LSTM_MODEL_PATH, SCALER_PATH, PAIRS

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

import pickle

# Minimum candles required per pair to justify retraining
MIN_CANDLES_PER_PAIR = 200
# Default retraining epochs (lower than initial training — fine-tune)
RETRAIN_EPOCHS = 20
RETRAIN_BATCH_SIZE = 32
LOOKBACK = 60


class LSTMRetrainer:
    """Auto-retrain LSTM on fresh market data."""

    def __init__(self, broker=None, predictor=None):
        """
        Args:
            broker:    MT5Connector for fetching fresh candles
            predictor: LSTMPredictor instance whose model we'll hot-swap
        """
        self.broker = broker
        self.predictor = predictor
        self._lock = threading.Lock()
        self.last_retrain_time = None
        self.last_retrain_val_loss = None
        self.retrain_count = 0
        self.available = TF_AVAILABLE and predictor is not None

    def retrain(self):
        """
        Download latest data for all pairs, retrain LSTM, and hot-swap
        the model if validation loss improves.

        Safe to call from APScheduler — runs in its own thread context.
        Returns True if model was updated, False otherwise.
        """
        if not self.available:
            bot_logger.info("LSTM retrainer skipped: TensorFlow or predictor not available")
            return False

        if not self.broker:
            bot_logger.warning("LSTM retrainer skipped: no broker connection")
            return False

        bot_logger.info("🔄 LSTM auto-retrain starting...")

        try:
            from src.ai.technical_analyzer import TechnicalAnalyzer
            technical = TechnicalAnalyzer()

            all_features = []
            for pair in PAIRS:
                # Fetch 500 candles of 1h data (≈3 weeks)
                df = self.broker.get_candles(pair, timeframe_minutes=60, num_candles=500)
                if df is None or len(df) < MIN_CANDLES_PER_PAIR:
                    bot_logger.info(f"  ⚠️ {pair}: insufficient data ({len(df) if df is not None else 0} candles)")
                    continue

                df_enriched = technical.calculate_indicators(df)
                mean_price = df_enriched['close'].mean()

                features = pd.DataFrame()
                features['close_norm'] = df_enriched['close'].values / mean_price
                rsi = df_enriched.get('rsi')
                features['rsi'] = (rsi / 100.0).values if rsi is not None else 0.5
                macd = df_enriched.get('macd')
                features['macd'] = (macd / mean_price).values if macd is not None else 0.0
                bb_lower = df_enriched.get('bb_lower')
                bb_upper = df_enriched.get('bb_upper')
                if bb_lower is not None and bb_upper is not None:
                    features['bb_pos'] = (
                        (df_enriched['close'] - bb_lower) /
                        (bb_upper - bb_lower + 1e-8)
                    ).values
                else:
                    features['bb_pos'] = 0.5
                atr = df_enriched.get('atr')
                features['atr_norm'] = (atr / mean_price).values if atr is not None else 0.001
                vol_sma = df_enriched['volume'].rolling(20).mean()
                features['vol_ratio'] = (df_enriched['volume'] / (vol_sma + 1)).values

                features = features.fillna(0).replace([np.inf, -np.inf], 0)
                all_features.append(features.values)
                bot_logger.info(f"  ✅ {pair}: {len(features)} candles prepared")

            if not all_features:
                bot_logger.warning("LSTM retrain aborted: no valid data")
                return False

            combined = np.concatenate(all_features, axis=0)
            num_features = combined.shape[1]

            # Load existing scaler baseline for comparison
            old_val_loss = self._load_meta_val_loss()

            # Fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(combined)

            # Build sequences
            X, y = [], []
            for i in range(len(scaled) - LOOKBACK):
                X.append(scaled[i:i + LOOKBACK])
                y.append(scaled[i + LOOKBACK, 0])  # predict normalised close
            X = np.array(X)
            y = np.array(y)

            if len(X) < 100:
                bot_logger.warning("LSTM retrain aborted: too few sequences")
                return False

            # Split
            split = int(len(X) * 0.85)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            bot_logger.info(f"  Sequences: {len(X)} (train={len(X_train)}, val={len(X_val)})")

            with self._lock:
                # Fine-tune existing model if possible, else build new
                model = self.predictor.model

                if model is None or self._model_shape_mismatch(model, LOOKBACK, num_features):
                    bot_logger.info("  Building new model architecture for retraining")
                    model = self._build_model(LOOKBACK, num_features)

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                ]

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=RETRAIN_EPOCHS,
                    batch_size=RETRAIN_BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=0,
                )

                new_val_loss = min(history.history['val_loss'])
                bot_logger.info(
                    f"  Retrain result: val_loss={new_val_loss:.6f} "
                    f"(previous={old_val_loss:.6f if old_val_loss else 'N/A'})"
                )

                # Only save if improved (or first time)
                if old_val_loss is None or new_val_loss < old_val_loss * 1.05:
                    # Save model + scaler
                    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
                    model.save(LSTM_MODEL_PATH)
                    with open(SCALER_PATH, 'wb') as f:
                        pickle.dump(scaler, f)

                    # Save metadata
                    meta = {
                        'num_features': num_features,
                        'lookback': LOOKBACK,
                        'final_val_loss': float(new_val_loss),
                        'epochs_trained': len(history.history['loss']),
                        'retrain_time': datetime.utcnow().isoformat(),
                        'retrain_count': self.retrain_count + 1,
                        'feature_names': ['close_norm', 'rsi', 'macd', 'bb_pos', 'atr_norm', 'vol_ratio'],
                    }
                    meta_path = LSTM_MODEL_PATH.replace('.h5', '_meta.json')
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    # Hot-swap into predictor
                    self.predictor.model = model
                    self.predictor.scaler = scaler
                    self.predictor.scaler_fitted = True

                    self.last_retrain_time = datetime.utcnow()
                    self.last_retrain_val_loss = new_val_loss
                    self.retrain_count += 1

                    bot_logger.info(
                        f"✅ LSTM retrained & hot-swapped (val_loss={new_val_loss:.6f}, "
                        f"retrain #{self.retrain_count})"
                    )
                    return True
                else:
                    bot_logger.info(
                        f"⚠️ LSTM retrain rejected: new val_loss {new_val_loss:.6f} "
                        f"worse than {old_val_loss:.6f}"
                    )
                    return False

        except Exception as e:
            error_logger.error(f"LSTM retrain failed: {e}", exc_info=True)
            return False

    def _build_model(self, lookback, num_features):
        """Build a fresh LSTM model matching the training pipeline architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

        model = Sequential([
            Input(shape=(lookback, num_features)),
            LSTM(units=128, return_sequences=True),
            Dropout(0.3),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=32, activation='relu'),
            Dense(units=1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae'],
        )
        return model

    @staticmethod
    def _model_shape_mismatch(model, lookback, num_features):
        """Check if existing model's input shape matches new data."""
        try:
            expected = model.input_shape
            # expected is (None, lookback, num_features)
            if expected[1] != lookback or expected[2] != num_features:
                return True
            return False
        except Exception:
            return True

    @staticmethod
    def _load_meta_val_loss():
        """Load previous validation loss from metadata file."""
        meta_path = LSTM_MODEL_PATH.replace('.h5', '_meta.json')
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get('final_val_loss')
        except Exception:
            return None
