"""
LSTM Training Pipeline
Downloads data → preprocesses → trains model → saves weights

Can be run standalone:
    python -m scripts.train_lstm

Requires TensorFlow (Python 3.10-3.12). If TensorFlow is not installed,
the bot still works with 7 AI models — just skip this step.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check TensorFlow availability early
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.data.historical_downloader import HistoricalDownloader
from src.ai.lstm_predictor import LSTMPredictor
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger
from config.strategy_config import PAIRS, LSTM_MODEL_PATH, SCALER_PATH


def train_lstm(
    pairs: list = None,
    days: int = 365 * 2,
    interval: str = '1h',
    epochs: int = 50,
    batch_size: int = 32,
    lookback: int = 60,
    use_features: bool = True,
):
    """
    Full LSTM training pipeline.

    Args:
        pairs:         list of pairs to train on (uses config PAIRS if None)
        days:          days of historical data to download
        interval:      candle interval
        epochs:        training epochs
        batch_size:    batch size
        lookback:      LSTM lookback window
        use_features:  if True, train on multiple features (close + indicators)
    """
    if not TF_AVAILABLE:
        print("=" * 60)
        print("❌ TensorFlow is NOT installed.")
        print("   LSTM training requires TensorFlow (Python 3.10-3.12).")
        print()
        print("   Your bot will still work with 7 AI models!")
        print("   To enable LSTM, install Python 3.12 from:")
        print("   https://www.python.org/downloads/release/python-31210/")
        print("=" * 60)
        return False
    
    pairs = pairs or PAIRS
    downloader = HistoricalDownloader()
    technical = TechnicalAnalyzer()

    # -----------------------------------------------------------
    # 1. Download data for all pairs
    # -----------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Downloading historical data")
    print("=" * 60)
    all_data = downloader.download_all(pairs, days=days, interval=interval)

    if not all_data:
        print("❌ No data downloaded. Cannot train.")
        return False

    # -----------------------------------------------------------
    # 2. Prepare combined training dataset
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Preparing training data")
    print("=" * 60)

    all_prices = []
    all_features = []

    for pair, df in all_data.items():
        if df is None or len(df) < lookback + 50:
            print(f"  ⚠️  Skipping {pair} – insufficient data ({len(df) if df is not None else 0} candles)")
            continue

        # Add indicators
        df_enriched = technical.calculate_indicators(df)

        # Normalise prices relative to mean (so different pairs can be combined)
        mean_price = df_enriched['close'].mean()
        normalised_close = df_enriched['close'].values / mean_price

        all_prices.append(normalised_close)
        print(f"  ✅ {pair}: {len(normalised_close)} samples (mean price: {mean_price:.5f})")

        if use_features:
            # Multi-feature: close, rsi, macd, bb_position, atr, volume_ratio
            features = pd.DataFrame()
            features['close_norm'] = normalised_close
            features['rsi'] = df_enriched['rsi'].values / 100.0  # Scale 0-1
            features['macd'] = df_enriched['macd'].values / mean_price  # Relative
            features['bb_pos'] = (
                (df_enriched['close'] - df_enriched['bb_lower']) /
                (df_enriched['bb_upper'] - df_enriched['bb_lower'] + 1e-8)
            ).values
            features['atr_norm'] = df_enriched['atr'].values / mean_price
            vol_sma = df_enriched['volume'].rolling(20).mean()
            features['vol_ratio'] = (df_enriched['volume'] / (vol_sma + 1)).values

            features = features.fillna(0).replace([np.inf, -np.inf], 0)
            all_features.append(features.values)

    if not all_prices:
        print("❌ No valid training data. Aborting.")
        return False

    # Concatenate all pairs
    combined_prices = np.concatenate(all_prices)
    print(f"\nTotal training samples: {len(combined_prices)}")

    # -----------------------------------------------------------
    # 3. Train LSTM
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Training LSTM model")
    print("=" * 60)

    if use_features and all_features:
        success = _train_multi_feature(all_features, lookback, epochs, batch_size)
    else:
        # Single-feature training (close price only)
        predictor = LSTMPredictor(lookback_window=lookback)
        success = predictor.train(combined_prices, epochs=epochs, batch_size=batch_size)

    if success:
        print("\n" + "=" * 60)
        print("✅ LSTM MODEL TRAINED SUCCESSFULLY!")
        print(f"   Model saved to: {LSTM_MODEL_PATH}")
        print(f"   Scaler saved to: {SCALER_PATH}")
        print("=" * 60)
    else:
        print("\n❌ Training failed. Check logs for details.")

    return success


def _train_multi_feature(all_features, lookback, epochs, batch_size):
    """Train LSTM on multiple features for better predictions."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    combined = np.concatenate(all_features, axis=0)
    num_features = combined.shape[1]

    print(f"  Features per sample: {num_features}")
    print(f"  Total rows: {len(combined)}")

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(combined)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i + lookback])
        y.append(scaled[i + lookback, 0])  # Predict normalised close

    X = np.array(X)
    y = np.array(y)

    print(f"  Training sequences: {len(X)}")
    print(f"  Input shape: {X.shape}")

    # Split train/validation
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build enhanced model
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

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    model.save(LSTM_MODEL_PATH)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # Save training metadata
    meta = {
        'num_features': num_features,
        'lookback': lookback,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
        'feature_names': ['close_norm', 'rsi', 'macd', 'bb_pos', 'atr_norm', 'vol_ratio'],
    }

    import json
    meta_path = LSTM_MODEL_PATH.replace('.h5', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Final train loss: {meta['final_train_loss']:.6f}")
    print(f"  Final val loss:   {meta['final_val_loss']:.6f}")
    print(f"  Epochs trained:   {meta['epochs_trained']}")

    return True


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM model for forex prediction')
    parser.add_argument('--pairs', nargs='+', default=None, help='Currency pairs to train on')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    parser.add_argument('--interval', type=str, default='1h', help='Candle interval')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lookback', type=int, default=60, help='LSTM lookback window')
    parser.add_argument('--simple', action='store_true', help='Use single-feature (close only)')

    args = parser.parse_args()

    train_lstm(
        pairs=args.pairs,
        days=args.days,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lookback=args.lookback,
        use_features=not args.simple,
    )
