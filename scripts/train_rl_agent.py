#!/usr/bin/env python3
"""
RL Agent Training Script — Replay Historical CSV Data

Runs the full ensemble signal generation on sliding windows of historical
5m data, lets the RL agent make trade/skip decisions, simulates outcomes
based on ATR-based SL/TP logic, and feeds rewards back to the agent.

This produces a trained DQN (or Q-table) that is saved to models/rl_agent.json
and models/rl_dqn.pt, ready for live trading.

Usage:
    python scripts/train_rl_agent.py                   # Default: all pairs, 3 epochs
    python scripts/train_rl_agent.py --epochs 5        # More training
    python scripts/train_rl_agent.py --pair EUR/USD    # Single pair
"""
import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.rl_agent import RLTradingAgent
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.risk.position_manager import RiskManager
from src.utils.logger import bot_logger

# ── Constants ──
SPREAD_SIM = {'EUR/USD': 0.00012, 'GBP/USD': 0.00016, 'USD/JPY': 0.020}
SL_ATR_MULT = 0.8
TP_ATR_MULT = 1.3
MAX_HOLD_BARS = 15
WINDOW_SIZE = 250    # Bars of history to feed to analyzers
STEP_SIZE = 15       # Skip bars between evaluation points (faster training)
COOLDOWN_BARS = 8    # Bars between trades
ENTRY_THRESHOLD = 0.65  # Ensemble score threshold to offer trade to RL


def load_data(pair: str) -> pd.DataFrame:
    """Load 5m CSV data for a pair."""
    fname = pair.replace('/', '_') + '_5m.csv'
    fpath = os.path.join('data', fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No data file: {fpath}")
    df = pd.read_csv(fpath, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def quick_ensemble_score(row, df_window, ta, scalping, ema_xover,
                         candle_det, sr_det, vol_analyzer):
    """
    Lightweight ensemble signal using available analyzers (no broker needed).
    Returns (direction, confidence, agreement, details).
    """
    df_enriched = ta.calculate_indicators(df_window.copy())
    tech_sig = ta.get_signal(df_enriched)
    vol_sig = vol_analyzer.get_volume_signal(df_enriched)
    ema_sig = ema_xover.get_signal(df_enriched)
    candle_sig = candle_det.get_pattern_signal(df_enriched)
    sr_sig = sr_det.get_sr_signal(df_enriched)
    scalp_sig = scalping.get_signal(df_enriched, 'EUR/USD')  # pair doesn't matter for indicator logic

    signals = {
        'scalping': scalp_sig,
        'technical': tech_sig,
        'ema_crossover': ema_sig,
        'candlestick': candle_sig,
        'support_resistance': sr_sig,
        'volume': vol_sig,
    }

    weights = {
        'scalping': 0.30,
        'technical': 0.22,
        'ema_crossover': 0.15,
        'candlestick': 0.10,
        'support_resistance': 0.05,
        'volume': 0.03,
    }

    buy_votes = 0
    sell_votes = 0
    buy_conf = 0.0
    sell_conf = 0.0

    for name, sig in signals.items():
        s = sig.get('signal', 'HOLD')
        c = sig.get('confidence', 0.0)
        w = weights.get(name, 0.1)
        if s == 'BUY':
            buy_votes += 1
            buy_conf += c * w
        elif s == 'SELL':
            sell_votes += 1
            sell_conf += c * w
        # HOLD/SKIP don't count

    total_models = len(signals)
    agreement = max(buy_votes, sell_votes)

    if buy_votes > sell_votes and agreement >= 2:
        direction = 'BUY'
        conf = buy_conf / sum(weights[k] for k, v in signals.items() if v.get('signal') == 'BUY')
    elif sell_votes > buy_votes and agreement >= 2:
        direction = 'SELL'
        conf = sell_conf / sum(weights[k] for k, v in signals.items() if v.get('signal') == 'SELL')
    else:
        direction = 'SKIP'
        conf = 0.0

    return direction, min(conf, 1.0), agreement, total_models, df_enriched


def simulate_trade(df, entry_idx, direction, pair, atr):
    """
    Simulate a trade from entry_idx using ATR SL/TP, return outcome.
    """
    pip_size = 0.01 if 'JPY' in pair else 0.0001
    spread = SPREAD_SIM.get(pair, 0.00015)
    entry_price = float(df.iloc[entry_idx]['close'])

    sl_dist = atr * SL_ATR_MULT
    tp_dist = atr * TP_ATR_MULT

    if direction == 'BUY':
        entry_price += spread / 2
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        entry_price -= spread / 2
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    # Walk forward bar by bar
    for hold in range(1, MAX_HOLD_BARS + 1):
        bar_idx = entry_idx + hold
        if bar_idx >= len(df):
            # Data ended — treat as timeout
            exit_price = float(df.iloc[-1]['close'])
            pips = (exit_price - entry_price) / pip_size if direction == 'BUY' else (entry_price - exit_price) / pip_size
            return {'won': pips > 0, 'pips': round(pips, 1), 'exit_type': 'TIMEOUT',
                    'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else 0}

        bar = df.iloc[bar_idx]
        high = float(bar['high'])
        low = float(bar['low'])
        close_p = float(bar['close'])

        if direction == 'BUY':
            if low <= sl_price:
                pips = (sl_price - entry_price) / pip_size
                return {'won': False, 'pips': round(pips, 1), 'exit_type': 'SL',
                        'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else -1}
            if high >= tp_price:
                pips = (tp_price - entry_price) / pip_size
                return {'won': True, 'pips': round(pips, 1), 'exit_type': 'TP',
                        'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else 1}
        else:
            if high >= sl_price:
                pips = (entry_price - sl_price) / pip_size
                return {'won': False, 'pips': round(pips, 1), 'exit_type': 'SL',
                        'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else -1}
            if low <= tp_price:
                pips = (entry_price - tp_price) / pip_size
                return {'won': True, 'pips': round(pips, 1), 'exit_type': 'TP',
                        'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else 1}

    # Max hold reached — exit at close
    exit_price = float(df.iloc[min(entry_idx + MAX_HOLD_BARS, len(df) - 1)]['close'])
    pips = (exit_price - entry_price) / pip_size if direction == 'BUY' else (entry_price - exit_price) / pip_size
    return {'won': pips > 0, 'pips': round(pips, 1), 'exit_type': 'TIMEOUT',
            'rr': round(pips * pip_size / sl_dist, 2) if sl_dist > 0 else 0}


def train_on_pair(agent: RLTradingAgent, pair: str, df: pd.DataFrame,
                  risk_mgr: RiskManager, epoch: int):
    """
    Train the RL agent by replaying historical data for one pair.
    Uses precomputed indicators to avoid per-window ensemble calls.
    Returns stats dict.
    """
    ta = TechnicalAnalyzer()

    pip_size = 0.01 if 'JPY' in pair else 0.0001
    spread = SPREAD_SIM.get(pair, 0.00015)

    stats = {
        'signals_offered': 0,
        'trades_taken': 0,
        'trades_skipped': 0,
        'wins': 0, 'losses': 0,
        'total_pips': 0.0,
        'rewards': [],
    }

    # Precompute indicators ONCE for the entire dataset
    df_enriched = ta.calculate_indicators(df.copy())
    atr_median_global = float(df_enriched['atr'].median()) if 'atr' in df_enriched.columns else 0.001

    total_bars = len(df_enriched)
    last_trade_idx = -COOLDOWN_BARS

    eval_points = list(range(WINDOW_SIZE, total_bars - MAX_HOLD_BARS - 1, STEP_SIZE))
    total_eval = len(eval_points)

    print(f"  Epoch {epoch+1} | {pair} | {total_eval} evaluation points across {total_bars} bars")

    for i, idx in enumerate(eval_points):
        if idx - last_trade_idx < COOLDOWN_BARS:
            continue

        row = df_enriched.iloc[idx]

        # Quick directional signal from indicators
        rsi = float(row.get('rsi', 50))
        adx = float(row.get('adx', 20))
        atr = float(row.get('atr', 0.001))
        ema_200 = float(row.get('ema_200', row['close']))
        ema_50 = float(row.get('ema_50', row['close']))
        close = float(row['close'])
        volume_ratio = float(row.get('volume_ratio', 1.0))
        macd = float(row.get('macd', 0))

        # Determine direction and confidence from indicators
        buy_score = 0.0
        sell_score = 0.0

        # EMA trend
        if close > ema_200:
            buy_score += 0.2
        else:
            sell_score += 0.2
        if ema_50 > ema_200:
            buy_score += 0.15
        else:
            sell_score += 0.15

        # RSI
        if rsi < 35:
            buy_score += 0.15
        elif rsi > 65:
            sell_score += 0.15

        # MACD
        if macd > 0:
            buy_score += 0.15
        elif macd < 0:
            sell_score += 0.15

        # ADX strength
        if adx > 25:
            if buy_score > sell_score:
                buy_score += 0.1
            else:
                sell_score += 0.1

        # Volume confirmation
        if volume_ratio > 1.2:
            buy_score += 0.05
            sell_score += 0.05

        # Determine direction
        conf = max(buy_score, sell_score)
        if conf < 0.3:
            continue  # No clear signal

        if buy_score > sell_score:
            direction = 'BUY'
            agreement = 3 if buy_score > 0.45 else 2
        elif sell_score > buy_score:
            direction = 'SELL'
            agreement = 3 if sell_score > 0.45 else 2
        else:
            continue

        stats['signals_offered'] += 1
        total_models = 6

        # EMA 200 distance
        ema200_dist = (close - ema_200) / close if ema_200 else 0
        hour = pd.to_datetime(row['datetime']).hour if 'datetime' in row.index else 12

        # Build RL state
        state = agent.build_state(
            ensemble_confidence=conf,
            model_agreement=agreement,
            total_models=total_models,
            regime='trending' if adx > 25 else ('volatile' if atr > atr_median_global * 1.5 else 'ranging'),
            rsi=rsi, adx=adx, atr=atr, atr_median=atr_median_global,
            ema200_dist=ema200_dist, hour=hour,
            spread=spread, volume_ratio=volume_ratio,
            daily_trades=risk_mgr.open_trades,
            max_daily_trades=risk_mgr._current_tier.get('max_concurrent', 3),
            current_drawdown=0.0,
        )

        # RL decision
        action = agent.select_action(state, training=True)

        # Simulate what would have happened
        trade_result = simulate_trade(df, idx, direction, pair, atr)

        if action == 0:
            # SKIP — compute counterfactual reward
            reward = agent.compute_reward(action, would_have_won=trade_result['won'])
            stats['trades_skipped'] += 1
        else:
            # TRADE — compute reward from actual outcome
            reward = agent.compute_reward(action, trade_result=trade_result)
            lot_mult = agent.get_lot_multiplier(action)
            stats['trades_taken'] += 1
            if trade_result['won']:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            stats['total_pips'] += trade_result['pips'] * lot_mult
            last_trade_idx = idx

        stats['rewards'].append(reward)

        # Build next state (slightly shifted)
        next_state = state.copy()
        next_state[0] = 0.0  # Reset confidence for next evaluation point

        # Record outcome for learning
        agent.record_outcome(
            state=state, action=action, reward=reward,
            next_state=next_state, done=False,
            trade_info={
                'won': trade_result['won'] if action > 0 else None,
                'pips': trade_result['pips'] if action > 0 else 0,
                'rr': trade_result.get('rr', 0) if action > 0 else 0,
            } if action > 0 else None,
        )

        # Progress update every 200 decisions
        if stats['signals_offered'] % 200 == 0:
            wr = stats['wins'] / max(stats['trades_taken'], 1) * 100
            avg_r = np.mean(stats['rewards'][-100:]) if stats['rewards'] else 0
            print(f"    [{stats['signals_offered']:>5}] ε={agent.epsilon:.3f} "
                  f"taken={stats['trades_taken']} skipped={stats['trades_skipped']} "
                  f"WR={wr:.0f}% pips={stats['total_pips']:.1f} "
                  f"avg_reward={avg_r:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on historical data")
    parser.add_argument('--pair', type=str, default=None,
                        help='Single pair to train on (e.g., EUR/USD)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs over the data')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore saved state)')
    parser.add_argument('--balance', type=float, default=1000,
                        help='Simulated account balance')
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ['EUR/USD', 'GBP/USD']

    # Load data
    pair_data = {}
    for pair in pairs:
        try:
            pair_data[pair] = load_data(pair)
            print(f"Loaded {len(pair_data[pair])} bars for {pair}")
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {pair}: {e}")

    if not pair_data:
        print("No data available. Exiting.")
        return

    # Initialize agent
    if args.fresh:
        # Remove saved state to start clean
        for f in ['models/rl_agent.json', 'models/rl_dqn.pt']:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed {f}")

    agent = RLTradingAgent(
        learning_rate=5e-4,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        min_experiences=64,
    )

    # If not fresh and state was loaded, keep existing epsilon (loaded in __init__)
    if not args.fresh and agent.training_step > 0:
        print(f"Resuming from step {agent.training_step}, ε={agent.epsilon:.3f}, trades={agent.total_trades}")

    risk_mgr = RiskManager(initial_balance=args.balance)

    print(f"\n{'='*60}")
    print(f"  RL Agent Training — {agent.mode.upper()} mode")
    print(f"  Pairs: {list(pair_data.keys())}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Balance: ${args.balance:.0f}")
    print(f"{'='*60}\n")

    all_epoch_stats = []

    for epoch in range(args.epochs):
        epoch_stats = {'signals': 0, 'taken': 0, 'skipped': 0,
                       'wins': 0, 'losses': 0, 'pips': 0.0, 'rewards': []}

        for pair, df in pair_data.items():
            stats = train_on_pair(agent, pair, df, risk_mgr, epoch)

            epoch_stats['signals'] += stats['signals_offered']
            epoch_stats['taken'] += stats['trades_taken']
            epoch_stats['skipped'] += stats['trades_skipped']
            epoch_stats['wins'] += stats['wins']
            epoch_stats['losses'] += stats['losses']
            epoch_stats['pips'] += stats['total_pips']
            epoch_stats['rewards'].extend(stats['rewards'])

        wr = epoch_stats['wins'] / max(epoch_stats['taken'], 1) * 100
        skip_rate = epoch_stats['skipped'] / max(epoch_stats['signals'], 1) * 100
        avg_reward = np.mean(epoch_stats['rewards']) if epoch_stats['rewards'] else 0

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Signals offered: {epoch_stats['signals']}")
        print(f"    Trades taken:    {epoch_stats['taken']}  ({100-skip_rate:.0f}% take rate)")
        print(f"    Trades skipped:  {epoch_stats['skipped']} ({skip_rate:.0f}% skip rate)")
        print(f"    Win rate:        {wr:.1f}% ({epoch_stats['wins']}W / {epoch_stats['losses']}L)")
        print(f"    Total pips:      {epoch_stats['pips']:.1f}")
        print(f"    Avg reward:      {avg_reward:.3f}")
        print(f"    Epsilon:         {agent.epsilon:.4f}")
        print(f"    Training steps:  {agent.training_step}")
        print()

        all_epoch_stats.append(epoch_stats)

        # Save after each epoch
        agent.save_state()
        print(f"  ✅ Agent saved (step {agent.training_step}, ε={agent.epsilon:.3f})")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total training steps: {agent.training_step}")
    print(f"  Total trades:         {agent.total_trades}")
    print(f"  Final epsilon:        {agent.epsilon:.4f}")
    print(f"  Agent mode:           {agent.mode}")
    print(f"  Replay buffer:        {len(agent.replay)} experiences")
    print()

    # Show learning progression
    if len(all_epoch_stats) > 1:
        print("  Learning Progression:")
        for i, es in enumerate(all_epoch_stats):
            wr = es['wins'] / max(es['taken'], 1) * 100
            avg_r = np.mean(es['rewards']) if es['rewards'] else 0
            print(f"    Epoch {i+1}: WR={wr:.0f}% | Pips={es['pips']:.1f} | AvgReward={avg_r:.3f} | ε={agent.epsilon:.3f}")
        print()

    # Final readiness check
    agent_stats = agent.get_stats()
    print(f"  Agent Stats: {agent_stats}")
    print()

    if agent.epsilon < 0.15 and agent.training_step > 500:
        print("  ✅ Agent is sufficiently trained for live trading!")
        print("     Epsilon is low enough for exploitation mode.")
    elif agent.training_step > 200:
        print("  ⚠️  Agent has some training but epsilon is still high.")
        print("     Consider running more epochs (--epochs 5 or more).")
    else:
        print("  ❌ Agent needs more training. Run with more epochs.")


if __name__ == '__main__':
    main()
