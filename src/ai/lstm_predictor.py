"""
LSTM Neural Network for Price Prediction
Pre-trained or trained from historical data

TensorFlow is OPTIONAL — if not installed, a lightweight stub is used
so the rest of the 7-model ensemble still works.
"""
import numpy as np
import pandas as pd
import os
from src.utils.logger import bot_logger, error_logger
from config.strategy_config import LSTM_MODEL_PATH, SCALER_PATH

# --- Graceful TensorFlow import ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    bot_logger.warning("⚠️  TensorFlow not installed — LSTM model disabled. "
                       "Install with: pip install tensorflow>=2.13.0 (requires Python ≤3.12)")

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

import pickle


class LSTMPredictor:
    """LSTM model for predicting next price direction"""
    
    def __init__(self, lookback_window=60):
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if MinMaxScaler else None
        self.scaler_fitted = False
        self.available = TF_AVAILABLE  # Expose availability flag
        if TF_AVAILABLE:
            self.load_or_create_model()
        else:
            bot_logger.info("LSTM predictor running in stub mode (no TensorFlow)")
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if not TF_AVAILABLE:
            return
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = tf.keras.models.load_model(LSTM_MODEL_PATH)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.scaler_fitted = True
                bot_logger.info(f"✅ LSTM model loaded from {LSTM_MODEL_PATH}")
                return
            except Exception as e:
                error_logger.error(f"Error loading model: {str(e)}, creating new model")
        
        # Create new model
        self.model = Sequential([
            Input(shape=(self.lookback_window, 1)),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)  # Output: next price
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        bot_logger.info("✅ New LSTM model created")
    
    def prepare_data(self, prices):
        """
        Prepare data for LSTM
        
        Args:
            prices: Array of closing prices
        
        Returns:
            Normalized data ready for model
        """
        if len(prices) < self.lookback_window:
            return None, None
        
        # Normalize prices
        data = prices.reshape(-1, 1).astype('float32')
        normalized_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(normalized_data) - self.lookback_window):
            X.append(normalized_data[i:i + self.lookback_window])
            y.append(normalized_data[i + self.lookback_window])
        
        return np.array(X), np.array(y)
    
    def train(self, prices, epochs=50, batch_size=32):
        """
        Train LSTM model on historical prices
        
        Args:
            prices: Historical closing prices
            epochs: Training epochs
            batch_size: Batch size
        """
        if not TF_AVAILABLE:
            error_logger.error("Cannot train LSTM: TensorFlow not installed")
            return False
        
        try:
            X, y = self.prepare_data(prices)
            
            if X is None:
                error_logger.error("Insufficient data for training")
                return False
            
            bot_logger.info(f"Training LSTM on {len(X)} sequences...")
            
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_split=0.1
            )
            
            # Save model
            os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
            self.model.save(LSTM_MODEL_PATH)
            
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            bot_logger.info(f"✅ LSTM model saved to {LSTM_MODEL_PATH}")
            return True
        
        except Exception as e:
            error_logger.error(f"Error training LSTM: {str(e)}")
            return False
    
    def predict_direction(self, df):
        """
        Predict next price direction (UP/DOWN)
        
        Supports both:
          - Single-feature model (close price only)
          - Multi-feature model (close + rsi + macd + bb_pos + atr + vol_ratio)
        
        Returns HOLD with 0 confidence when TensorFlow is not installed.
        
        Args:
            df: DataFrame with closing prices (and optionally indicators)
        
        Returns:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'confidence': 0.0-1.0,
                'reason': 'explanation'
            }
        """
        if not TF_AVAILABLE or self.model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'LSTM disabled (TensorFlow not installed)'
            }
        
        try:
            if len(df) < self.lookback_window:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Need {self.lookback_window} candles, have {len(df)}'
                }
            
            # Detect if scaler expects multiple features
            num_features = self._get_scaler_features()
            
            if num_features > 1:
                return self._predict_multi_feature(df, num_features)
            else:
                return self._predict_single_feature(df)
        
        except Exception as e:
            error_logger.error(f"Error in LSTM prediction: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Prediction error: {str(e)}'
            }
    
    def _get_scaler_features(self):
        """Detect how many features the scaler was fitted with."""
        if not self.scaler_fitted:
            return 1
        try:
            return self.scaler.n_features_in_
        except AttributeError:
            return 1
    
    def _predict_single_feature(self, df):
        """Predict using close-price-only model."""
        recent_prices = df['close'].tail(self.lookback_window).values.reshape(-1, 1).astype('float32')
        
        if not self.scaler_fitted:
            all_prices = df['close'].values.reshape(-1, 1).astype('float32')
            self.scaler.fit(all_prices)
            self.scaler_fitted = True
        
        normalized = self.scaler.transform(recent_prices)
        X = normalized.reshape(1, self.lookback_window, 1)
        
        next_price_normalized = self.model.predict(X, verbose=0)
        next_price = self.scaler.inverse_transform(next_price_normalized)[0][0]
        
        return self._format_prediction(df, next_price)
    
    def _predict_multi_feature(self, df, num_features):
        """Predict using multi-feature model (close + indicators)."""
        import pandas_ta as pta
        
        tail = df.tail(self.lookback_window + 30).copy()  # Extra for indicator warmup
        
        # Build features matching training: close_norm, rsi, macd, bb_pos, atr_norm, vol_ratio
        mean_price = tail['close'].mean()
        
        features = pd.DataFrame()
        features['close_norm'] = tail['close'].values / mean_price
        
        # RSI
        rsi = tail.get('rsi')
        if rsi is None or rsi.isna().all():
            rsi = pta.rsi(tail['close'], length=14)
        features['rsi'] = (rsi / 100.0).values if rsi is not None else 0.5
        
        # MACD
        macd = tail.get('macd')
        if macd is None or (hasattr(macd, 'isna') and macd.isna().all()):
            macd_df = pta.macd(tail['close'])
            macd = macd_df.iloc[:, 0] if macd_df is not None else pd.Series(0, index=tail.index)
        features['macd'] = (macd / mean_price).values
        
        # Bollinger position
        bb_lower = tail.get('bb_lower')
        bb_upper = tail.get('bb_upper')
        if bb_lower is not None and bb_upper is not None:
            features['bb_pos'] = ((tail['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)).values
        else:
            bb_df = pta.bbands(tail['close'], length=20)
            if bb_df is not None:
                features['bb_pos'] = ((tail['close'].values - bb_df.iloc[:, 0].values) /
                                       (bb_df.iloc[:, 2].values - bb_df.iloc[:, 0].values + 1e-8))
            else:
                features['bb_pos'] = 0.5
        
        # ATR normalised
        atr = tail.get('atr')
        if atr is None or (hasattr(atr, 'isna') and atr.isna().all()):
            atr = pta.atr(tail['high'], tail['low'], tail['close'], length=14)
        features['atr_norm'] = (atr / mean_price).values if atr is not None else 0.001
        
        # Volume ratio
        vol_sma = tail['volume'].rolling(20).mean()
        features['vol_ratio'] = (tail['volume'] / (vol_sma + 1)).values
        
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Take only last lookback_window rows
        if len(features) > self.lookback_window:
            features = features.iloc[-self.lookback_window:]
        
        if len(features) < self.lookback_window:
            return {
                'signal': 'HOLD', 'confidence': 0.0,
                'reason': f'Insufficient feature data ({len(features)})'
            }
        
        feature_array = features.values.astype('float32')
        
        # Pad/trim to expected feature count
        if feature_array.shape[1] < num_features:
            pad = np.zeros((feature_array.shape[0], num_features - feature_array.shape[1]))
            feature_array = np.hstack([feature_array, pad])
        elif feature_array.shape[1] > num_features:
            feature_array = feature_array[:, :num_features]
        
        normalized = self.scaler.transform(feature_array)
        X = normalized.reshape(1, self.lookback_window, num_features)
        
        pred = self.model.predict(X, verbose=0)
        
        # Inverse-transform: we need a full-width array for the scaler
        pred_full = np.zeros((1, num_features))
        pred_full[0, 0] = pred[0][0]
        inverse = self.scaler.inverse_transform(pred_full)
        predicted_close_norm = inverse[0][0]
        
        # Convert back to price
        next_price = predicted_close_norm * mean_price
        
        return self._format_prediction(df, next_price)
    
    def _format_prediction(self, df, next_price):
        """Format prediction result consistently."""
        current_price = df['close'].iloc[-1]
        price_change_percent = ((next_price - current_price) / current_price) * 100
        
        # Determine signal
        if price_change_percent > 0.1:
            signal = 'BUY'
            confidence = min(abs(price_change_percent) / 2.0, 1.0)
        elif price_change_percent < -0.1:
            signal = 'SELL'
            confidence = min(abs(price_change_percent) / 2.0, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.3
        
        reason = f"LSTM predicts {price_change_percent:+.2f}% move"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'predicted_price': next_price,
            'current_price': current_price,
            'predicted_change_percent': price_change_percent
        }
