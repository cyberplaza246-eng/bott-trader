#!/usr/bin/env python3
"""
Train both AI models (LSTM + RL) using local CSV data.
No broker connection needed — uses existing data/ files.

Usage:
    python scripts/train_ai.py                 # Both models
    python scripts/train_ai.py --lstm-only     # LSTM only
    python scripts/train_ai.py --rl-only       # RL only
"""
import os, sys, argparse, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  LSTM Training (local CSV data)
# ═══════════════════════════════════════════════════════════════════
def train_lstm_local(epochs=30, batch_size=32, lookback=60):
    """Train LSTM on local 5m/1h CSV data."""
    try:
        import tensorflow as tf
        print(f"  TensorFlow {tf.__version__} loaded")
    except ImportError:
        print("❌ TensorFlow not installed — skipping LSTM training")
        return False

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    from src.ai.technical_analyzer import TechnicalAnalyzer
    from config.strategy_config import LSTM_MODEL_PATH, SCALER_PATH

    ta = TechnicalAnalyzer()

    print("\n" + "=" * 60)
    print("  LSTM TRAINING (local data)")
    print("=" * 60)

    # Load all available CSV data
    all_features = []
    data_dir = 'data'

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('_5m.csv'):
            continue
        pair = fname.replace('_5m.csv', '').replace('_', '/')
        fpath = os.path.join(data_dir, fname)

        df = pd.read_csv(fpath)
        if len(df) < lookback + 100:
            print(f"  ⚠️  {pair}: only {len(df)} candles — skipping")
            continue

        # Limit to last 10000 candles for speed
        df = df.tail(10000).reset_index(drop=True)
        df = ta.calculate_indicators(df)

        mean_price = df['close'].mean()
        features = pd.DataFrame()
        features['close_norm'] = df['close'].values / mean_price
        features['rsi'] = df['rsi'].values / 100.0 if 'rsi' in df.columns else 0.5
        features['macd'] = df['macd'].values / mean_price if 'macd' in df.columns else 0
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            features['bb_pos'] = (
                (df['close'] - df['bb_lower']) /
                (df['bb_upper'] - df['bb_lower'] + 1e-8)
            ).values
        else:
            features['bb_pos'] = 0.5
        features['atr_norm'] = df['atr'].values / mean_price if 'atr' in df.columns else 0
        vol_sma = df['volume'].rolling(20).mean() if 'volume' in df.columns else pd.Series(1, index=df.index)
        features['vol_ratio'] = (df['volume'] / (vol_sma + 1)).values if 'volume' in df.columns else 1.0
        features = features.fillna(0).replace([np.inf, -np.inf], 0)

        all_features.append(features.values)
        print(f"  ✅ {pair}: {len(features)} samples")

    if not all_features:
        print("❌ No training data found")
        return False

    combined = np.concatenate(all_features, axis=0)
    num_features = combined.shape[1]
    print(f"\n  Total samples: {len(combined)}, features: {num_features}")

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(combined)

    # Sequences
    X, y = [], []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i + lookback])
        y.append(scaled[i + lookback, 0])  # predict close_norm
    X = np.array(X)
    y = np.array(y)
    print(f"  Sequences: {len(X)}, shape: {X.shape}")

    # Split
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build model
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
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ]

    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0

    # Save
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'num_features': num_features,
        'lookback': lookback,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
        'feature_names': ['close_norm', 'rsi', 'macd', 'bb_pos', 'atr_norm', 'vol_ratio'],
        'training_time_sec': round(elapsed, 1),
    }
    meta_path = LSTM_MODEL_PATH.replace('.h5', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ LSTM trained in {elapsed:.0f}s")
    print(f"     Train loss: {meta['final_train_loss']:.6f}")
    print(f"     Val loss:   {meta['final_val_loss']:.6f}")
    print(f"     Epochs:     {meta['epochs_trained']}")
    print(f"     Saved: {LSTM_MODEL_PATH}")
    return True


# ═══════════════════════════════════════════════════════════════════
#  RL Agent Training (local CSV data)
# ═══════════════════════════════════════════════════════════════════
def train_rl_local(epochs=3, balance=50):
    """Train RL agent on local 5m CSV data."""
    from src.ai.rl_agent import RLTradingAgent
    from src.ai.technical_analyzer import TechnicalAnalyzer
    from src.risk.position_manager import RiskManager

    SPREAD_SIM = {'EUR/USD': 0.00012, 'GBP/USD': 0.00016, 'USD/JPY': 0.020}
    SL_ATR_MULT = 0.8
    TP_ATR_MULT = 1.6   # 2R target (matched to sweep TP ratio)
    MAX_HOLD_BARS = 30
    STEP_SIZE = 10
    COOLDOWN_BARS = 6
    WINDOW_SIZE = 200

    print("\n" + "=" * 60)
    print("  RL AGENT TRAINING (local data)")
    print("=" * 60)

    # Load data
    ta = TechnicalAnalyzer()
    pair_data = {}
    for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY']:
        fpath = f'data/{pair.replace("/", "_")}_5m.csv'
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=['datetime'])
            df = df.tail(5000).reset_index(drop=True)  # limit for speed
            df = ta.calculate_indicators(df)
            pair_data[pair] = df
            print(f"  ✅ {pair}: {len(df)} bars loaded")

    if not pair_data:
        print("❌ No data. Exiting.")
        return False

    agent = RLTradingAgent(
        learning_rate=5e-4, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
        min_experiences=64,
    )
    risk_mgr = RiskManager(initial_balance=balance)

    print(f"\n  Mode: {agent.mode.upper()}")
    print(f"  Epochs: {epochs}")
    print(f"  Balance: ${balance}")

    for epoch in range(epochs):
        e_stats = {'signals': 0, 'taken': 0, 'skipped': 0,
                   'wins': 0, 'losses': 0, 'pips': 0.0, 'rewards': []}

        for pair, df in pair_data.items():
            pip_size = 0.01 if 'JPY' in pair else 0.0001
            spread = SPREAD_SIM.get(pair, 0.00015)
            atr_med = float(df['atr'].median()) if 'atr' in df.columns else 0.001
            last_trade = -COOLDOWN_BARS

            eval_pts = list(range(WINDOW_SIZE, len(df) - MAX_HOLD_BARS - 1, STEP_SIZE))
            for idx in eval_pts:
                if idx - last_trade < COOLDOWN_BARS:
                    continue

                row = df.iloc[idx]
                rsi = float(row.get('rsi', 50))
                adx = float(row.get('adx', 20))
                atr = float(row.get('atr', 0.001))
                close = float(row['close'])
                ema_200 = float(row.get('ema_200', close))
                ema_50 = float(row.get('ema_50', close))
                macd = float(row.get('macd', 0))
                vol_ratio = float(row.get('volume_ratio', 1.0))

                # Quick direction scoring
                buy_s, sell_s = 0.0, 0.0
                if close > ema_200: buy_s += 0.2
                else: sell_s += 0.2
                if ema_50 > ema_200: buy_s += 0.15
                else: sell_s += 0.15
                if rsi < 35: buy_s += 0.15
                elif rsi > 65: sell_s += 0.15
                if macd > 0: buy_s += 0.15
                elif macd < 0: sell_s += 0.15
                if adx > 25:
                    (buy_s if buy_s > sell_s else sell_s)  # just for variable reference
                    if buy_s > sell_s: buy_s += 0.1
                    else: sell_s += 0.1

                conf = max(buy_s, sell_s)
                if conf < 0.3:
                    continue

                direction = 'BUY' if buy_s > sell_s else 'SELL'
                agreement = 3 if conf > 0.45 else 2

                e_stats['signals'] += 1
                ema200_dist = (close - ema_200) / close if ema_200 else 0
                hour = pd.to_datetime(row['datetime']).hour if 'datetime' in row.index else 12

                state = agent.build_state(
                    ensemble_confidence=conf, model_agreement=agreement,
                    total_models=6,
                    regime='trending' if adx > 25 else ('volatile' if atr > atr_med * 1.5 else 'ranging'),
                    rsi=rsi, adx=adx, atr=atr, atr_median=atr_med,
                    ema200_dist=ema200_dist, hour=hour,
                    spread=spread, volume_ratio=vol_ratio,
                    daily_trades=risk_mgr.open_trades,
                    max_daily_trades=3, current_drawdown=0.0,
                )

                action = agent.select_action(state, training=True)

                # Simulate trade
                sl_d = atr * SL_ATR_MULT
                tp_d = atr * TP_ATR_MULT
                entry = close + (spread/2 if direction == 'BUY' else -spread/2)
                won, t_pips = False, 0.0
                exit_type = 'TIMEOUT'
                for h in range(1, MAX_HOLD_BARS + 1):
                    bi = idx + h
                    if bi >= len(df): break
                    bar = df.iloc[bi]
                    if direction == 'BUY':
                        if float(bar['low']) <= entry - sl_d:
                            t_pips = -sl_d / pip_size; exit_type = 'SL'; break
                        if float(bar['high']) >= entry + tp_d:
                            t_pips = tp_d / pip_size; won = True; exit_type = 'TP'; break
                    else:
                        if float(bar['high']) >= entry + sl_d:
                            t_pips = -sl_d / pip_size; exit_type = 'SL'; break
                        if float(bar['low']) <= entry - tp_d:
                            t_pips = tp_d / pip_size; won = True; exit_type = 'TP'; break
                else:
                    ep = float(df.iloc[min(idx + MAX_HOLD_BARS, len(df)-1)]['close'])
                    t_pips = ((ep - entry) / pip_size) if direction == 'BUY' else ((entry - ep) / pip_size)
                    won = t_pips > 0

                trade_result = {'won': won, 'pips': round(t_pips, 1), 'exit_type': exit_type,
                                'rr': round(t_pips * pip_size / sl_d, 2) if sl_d > 0 else 0}

                if action == 0:
                    reward = agent.compute_reward(action, would_have_won=won)
                    e_stats['skipped'] += 1
                else:
                    reward = agent.compute_reward(action, trade_result=trade_result)
                    lot_mult = agent.get_lot_multiplier(action)
                    e_stats['taken'] += 1
                    if won: e_stats['wins'] += 1
                    else: e_stats['losses'] += 1
                    e_stats['pips'] += t_pips * lot_mult
                    last_trade = idx

                e_stats['rewards'].append(reward)
                next_state = state.copy()
                next_state[0] = 0.0
                agent.record_outcome(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=False,
                    trade_info={'won': won, 'pips': t_pips, 'rr': trade_result['rr']} if action > 0 else None,
                )

        wr = e_stats['wins'] / max(e_stats['taken'], 1) * 100
        skip_r = e_stats['skipped'] / max(e_stats['signals'], 1) * 100
        avg_r = np.mean(e_stats['rewards']) if e_stats['rewards'] else 0

        print(f"\n  Epoch {epoch+1}/{epochs}:")
        print(f"    Signals: {e_stats['signals']} | Taken: {e_stats['taken']} | Skipped: {e_stats['skipped']}")
        print(f"    WR: {wr:.1f}% ({e_stats['wins']}W/{e_stats['losses']}L) | Pips: {e_stats['pips']:.1f}")
        print(f"    AvgReward: {avg_r:.3f} | ε: {agent.epsilon:.3f}")

        agent.save_state()

    print(f"\n  ✅ RL Agent trained — step {agent.training_step}, ε={agent.epsilon:.3f}")
    print(f"     Saved: models/rl_agent.json + models/rl_dqn.pt")
    return True


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AI models')
    parser.add_argument('--lstm-only', action='store_true')
    parser.add_argument('--rl-only', action='store_true')
    parser.add_argument('--lstm-epochs', type=int, default=30)
    parser.add_argument('--rl-epochs', type=int, default=3)
    args = parser.parse_args()

    t_start = time.time()

    if not args.rl_only:
        train_lstm_local(epochs=args.lstm_epochs)

    if not args.lstm_only:
        train_rl_local(epochs=args.rl_epochs)

    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE ({time.time()-t_start:.0f}s)")
    print(f"{'='*60}\n")
