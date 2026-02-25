"""
Adaptive Learning System

Learns from past trades to improve future decisions:
  - Tracks which model combinations produce winning trades
  - Adjusts model weights dynamically based on recent performance
  - Records market conditions during trades for pattern learning
  - Saves/loads trade history for persistent learning
"""
import os
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from src.utils.logger import bot_logger

LEARNING_DB_PATH = 'data/adaptive_learning.json'


class AdaptiveLearner:
    """
    Learns from trade outcomes and adjusts strategy parameters dynamically.

    Key features:
      - Model weight optimization based on accuracy history
      - Win-rate tracking per pair, per session, per market condition
      - Confidence threshold auto-adjustment
      - Cool-down tuning after losing streaks
    """

    def __init__(self, initial_weights: dict = None):
        self.model_weights = initial_weights or {
            'lstm': 0.35,
            'sentiment': 0.30,
            'technical': 0.20,
            'volume': 0.15,
        }

        self.trade_history = []
        self.model_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.pair_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        self.session_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        
        # Use config threshold, not hardcoded
        from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD
        self.confidence_threshold = ENSEMBLE_CONFIDENCE_THRESHOLD
        
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        self._load()

    # ------------------------------------------------------------------
    # Trade Recording
    # ------------------------------------------------------------------
    def record_trade(self, trade_result: dict):
        """
        Record a completed trade and update learning data.

        Args:
            trade_result: {
                'pair': 'EUR/USD',
                'signal': 'BUY' | 'SELL',
                'profit_loss': float,
                'model_signals': {
                    'lstm': {'signal': 'BUY', 'confidence': 0.6},
                    'sentiment': {...}, 'technical': {...}, 'volume': {...}
                },
                'entry_price': float,
                'exit_price': float,
                'exit_type': 'TAKE_PROFIT' | 'STOP_LOSS',
                'entry_time': str,
                'market_conditions': {  # Optional
                    'atr': float,
                    'rsi': float,
                    'volume_ratio': float,
                    'session': 'london' | 'ny' | 'asian',
                }
            }
        """
        is_win = trade_result.get('profit_loss', 0) > 0
        pair = trade_result.get('pair', 'UNKNOWN')
        signal = trade_result.get('signal', 'UNKNOWN')

        # Update pair stats
        if is_win:
            self.pair_stats[pair]['wins'] += 1
            self.consecutive_losses = 0
        else:
            self.pair_stats[pair]['losses'] += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

        self.pair_stats[pair]['total_pnl'] += trade_result.get('profit_loss', 0)

        # Update model accuracy
        model_signals = trade_result.get('model_signals', {})
        for model_name, model_signal in model_signals.items():
            if model_name not in self.model_accuracy:
                self.model_accuracy[model_name] = {'correct': 0, 'total': 0}

            self.model_accuracy[model_name]['total'] += 1

            # Model was correct if it agreed with the trade direction AND trade was profitable
            if model_signal.get('signal') == signal and is_win:
                self.model_accuracy[model_name]['correct'] += 1
            elif model_signal.get('signal') != signal and not is_win:
                self.model_accuracy[model_name]['correct'] += 1

        # Session tracking
        hour = datetime.now().hour
        if 8 <= hour < 16:
            session = 'london'
        elif 13 <= hour < 21:
            session = 'new_york'
        else:
            session = 'asian'

        if is_win:
            self.session_stats[session]['wins'] += 1
        else:
            self.session_stats[session]['losses'] += 1

        # Store trade
        trade_record = {
            **trade_result,
            'timestamp': datetime.now().isoformat(),
            'session': session,
            'is_win': is_win,
        }
        # Convert datetime objects to strings for JSON serialisation
        for key in ('entry_time', 'exit_time'):
            if key in trade_record and hasattr(trade_record[key], 'isoformat'):
                trade_record[key] = trade_record[key].isoformat()

        self.trade_history.append(trade_record)

        # Adapt
        self._adapt_weights()
        self._adapt_confidence_threshold()

        # Persist
        self._save()

        bot_logger.info(
            f"📚 Trade recorded: {pair} {signal} → {'WIN' if is_win else 'LOSS'} "
            f"(${trade_result.get('profit_loss', 0):+.2f})"
        )

    # ------------------------------------------------------------------
    # Weight Adaptation
    # ------------------------------------------------------------------
    def _adapt_weights(self):
        """Adjust model weights based on recent accuracy (exponential decay)."""
        min_trades = 10  # Need at least 10 trades before adapting

        total_trades = sum(m['total'] for m in self.model_accuracy.values())
        if total_trades < min_trades:
            return

        # Calculate accuracy for each model
        accuracies = {}
        for model, stats in self.model_accuracy.items():
            if stats['total'] > 0:
                accuracies[model] = stats['correct'] / stats['total']
            else:
                accuracies[model] = 0.5  # Default

        if not accuracies:
            return

        # Normalise accuracies to sum to 1.0
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            new_weights = {
                model: acc / total_accuracy
                for model, acc in accuracies.items()
            }

            # Blend with current weights (slow adaptation: 90% old, 10% new)
            blend = 0.1
            for model in self.model_weights:
                if model in new_weights:
                    self.model_weights[model] = (
                        (1 - blend) * self.model_weights[model] +
                        blend * new_weights[model]
                    )

            # Re-normalise
            weight_sum = sum(self.model_weights.values())
            self.model_weights = {k: v / weight_sum for k, v in self.model_weights.items()}

            bot_logger.info(
                f"📊 Adapted weights: " +
                " | ".join(f"{k}: {v:.2f}" for k, v in self.model_weights.items())
            )

    def _adapt_confidence_threshold(self):
        """
        Adjust confidence threshold:
          - Raise after losing streaks (more cautious)
          - Lower after winning streaks (more aggressive)
        """
        if self.consecutive_losses >= 3:
            # Increase threshold (more cautious)
            self.confidence_threshold = min(0.90, self.confidence_threshold + 0.02)
            bot_logger.info(
                f"⚠️  Raising confidence threshold to {self.confidence_threshold:.2f} "
                f"after {self.consecutive_losses} consecutive losses"
            )
        elif self.consecutive_losses == 0:
            # Recent wins: can be slightly more aggressive
            recent_wins = sum(
                1 for t in self.trade_history[-10:]
                if t.get('is_win', False)
            )
            if recent_wins >= 7:
                self.confidence_threshold = max(0.65, self.confidence_threshold - 0.01)
                bot_logger.info(
                    f"✅ Lowering confidence threshold to {self.confidence_threshold:.2f} "
                    f"(winning streak)"
                )

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------
    def get_adjusted_weights(self) -> dict:
        """Get the current adapted model weights."""
        return dict(self.model_weights)

    def get_adjusted_threshold(self) -> float:
        """Get the current adapted confidence threshold."""
        return self.confidence_threshold

    def get_pair_win_rate(self, pair: str) -> float:
        """Get win rate for a specific pair."""
        stats = self.pair_stats.get(pair, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.0

    def should_skip_pair(self, pair: str) -> bool:
        """
        Suggest skipping a pair if it has a very poor win rate
        (after sufficient trades).
        """
        stats = self.pair_stats.get(pair, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        if total < 15:
            return False  # Not enough data
        win_rate = stats['wins'] / total
        return win_rate < 0.35  # Skip if winning less than 35%

    def get_performance_summary(self) -> dict:
        """Get overall performance summary."""
        total_trades = len(self.trade_history)
        total_wins = sum(1 for t in self.trade_history if t.get('is_win', False))
        total_pnl = sum(t.get('profit_loss', 0) for t in self.trade_history)

        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_trades - total_wins,
            'win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'model_weights': dict(self.model_weights),
            'model_accuracy': dict(self.model_accuracy),
            'confidence_threshold': self.confidence_threshold,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'pair_stats': dict(self.pair_stats),
            'session_stats': dict(self.session_stats),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        """Save learning data to disk."""
        os.makedirs(os.path.dirname(LEARNING_DB_PATH), exist_ok=True)

        data = {
            'model_weights': self.model_weights,
            'model_accuracy': dict(self.model_accuracy),
            'pair_stats': dict(self.pair_stats),
            'session_stats': dict(self.session_stats),
            'confidence_threshold': self.confidence_threshold,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trade_history': self.trade_history[-500:],  # Keep last 500 trades
            'last_updated': datetime.now().isoformat(),
        }

        with open(LEARNING_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load learning data from disk."""
        if not os.path.exists(LEARNING_DB_PATH):
            return

        try:
            with open(LEARNING_DB_PATH, 'r') as f:
                data = json.load(f)

            self.model_weights = data.get('model_weights', self.model_weights)
            self.confidence_threshold = data.get('confidence_threshold', 0.75)
            self.consecutive_losses = data.get('consecutive_losses', 0)
            self.max_consecutive_losses = data.get('max_consecutive_losses', 0)
            self.trade_history = data.get('trade_history', [])

            # Restore defaultdicts
            for k, v in data.get('model_accuracy', {}).items():
                self.model_accuracy[k] = v
            for k, v in data.get('pair_stats', {}).items():
                self.pair_stats[k] = v
            for k, v in data.get('session_stats', {}).items():
                self.session_stats[k] = v

            bot_logger.info(
                f"📚 Loaded learning data: {len(self.trade_history)} past trades, "
                f"threshold={self.confidence_threshold:.2f}"
            )

        except Exception as e:
            bot_logger.warning(f"Could not load learning data: {e}")
