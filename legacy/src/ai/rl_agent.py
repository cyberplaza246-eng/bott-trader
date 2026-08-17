"""
Reinforcement Learning Agent for Trade Decision Optimization

Implements a Deep Q-Network (DQN) agent that learns optimal:
  - Trade entry/skip decisions (given ensemble signal)
  - Position sizing (micro, small, standard lot)
  - Early exit timing

The RL agent acts as an additional gate on top of the ensemble:
  ensemble_signal → RL agent → final_decision

Uses experience replay and target network for stable training.

Requirements:
  scikit-learn (required)
  torch (optional — uses simple Q-table fallback if unavailable)
"""
import os
import json
import random
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from src.utils.logger import bot_logger, error_logger

# Try PyTorch for DQN
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────
RL_MODEL_PATH = 'models/rl_agent.json'
RL_DQN_PATH = 'models/rl_dqn.pt'

# Actions the RL agent can take
ACTIONS = {
    0: 'SKIP',         # Don't trade even though ensemble says go
    1: 'TRADE_SMALL',  # Enter with 50% of calculated lot size
    2: 'TRADE_FULL',   # Enter with full calculated lot size
    3: 'TRADE_LARGE',  # Enter with 120% of calculated lot size (high conviction)
}

# State features (normalized)
STATE_DIM = 16
ACTION_DIM = len(ACTIONS)


# ── DQN Neural Network (PyTorch) ─────────────────────────────────
class DQNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Small DQN for trade decision making."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=64):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Experience Replay Buffer ─────────────────────────────────────
class ReplayBuffer:
    """Fixed-size ring buffer for experience replay."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── RL Trading Agent ──────────────────────────────────────────────
class RLTradingAgent:
    """
    DQN-based trading agent that sits on top of the ensemble.

    State vector (16 features):
      [0]  ensemble_confidence
      [1]  model_agreement_ratio
      [2]  regime_encoded (trending=1, volatile=0.5, ranging=0)
      [3]  rsi_normalized (0-1)
      [4]  adx_normalized (0-1)
      [5]  atr_percentile (0-1)
      [6]  ema200_distance_normalized
      [7]  hour_encoded (sin)
      [8]  hour_encoded (cos)
      [9]  recent_win_rate (last 20 trades, 0-1)
      [10] recent_avg_rr (risk-reward of last 20, normalized)
      [11] drawdown_depth (0-1)
      [12] consecutive_losses (0-1, capped at 5)
      [13] spread_ratio (spread / atr, 0-1)
      [14] volume_ratio (current / avg, 0-1 capped)
      [15] daily_trades_ratio (today's trades / max, 0-1)

    Reward design:
      - Win:  +profit_pips * lot_multiplier
      - Loss: -loss_pips * lot_multiplier * 1.5 (asymmetric)
      - Skip on would-have-lost: +2.0
      - Skip on would-have-won: -0.5 (small regret)
      - Timeout exit: -1.0
    """

    def __init__(self, learning_rate=1e-3, gamma=0.95, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.995, min_experiences=100):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.min_experiences = min_experiences
        self.training_step = 0
        self.total_trades = 0

        # Stats
        self.episode_rewards = []
        self.action_counts = defaultdict(int)

        # Experience replay
        self.replay = ReplayBuffer(capacity=50000)
        self.last_state = None
        self.last_action = None

        # Trade history for state computation
        self.recent_trades = deque(maxlen=50)

        if TORCH_AVAILABLE:
            self.device = torch.device('cpu')
            self.q_network = DQNetwork().to(self.device)
            self.target_network = DQNetwork().to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.mode = 'dqn'
            bot_logger.info("🤖 RL Agent: DQN mode (PyTorch)")
        else:
            # Fallback: simple Q-table with discretized states
            self.q_table = defaultdict(lambda: np.zeros(ACTION_DIM))
            self.lr = learning_rate
            self.mode = 'qtable'
            bot_logger.info("🤖 RL Agent: Q-table mode (no PyTorch)")

        self._load_state()

    # ──────────────────────────────────────────────────────────────
    #  State Construction
    # ──────────────────────────────────────────────────────────────
    def build_state(self, ensemble_confidence: float, model_agreement: int,
                    total_models: int, regime: str, rsi: float, adx: float,
                    atr: float, atr_median: float, ema200_dist: float,
                    hour: int, spread: float, volume_ratio: float,
                    daily_trades: int, max_daily_trades: int,
                    current_drawdown: float) -> np.ndarray:
        """
        Build normalized state vector from market and account features.
        """
        regime_map = {'trending': 1.0, 'volatile': 0.5, 'ranging': 0.0}

        # Recent trade metrics
        recent_wins = sum(1 for t in self.recent_trades if t.get('won', False))
        recent_total = max(len(self.recent_trades), 1)
        recent_wr = recent_wins / recent_total

        recent_rrs = [t.get('rr', 0) for t in self.recent_trades]
        recent_avg_rr = np.mean(recent_rrs) if recent_rrs else 0.0

        consec_losses = 0
        for t in reversed(list(self.recent_trades)):
            if not t.get('won', True):
                consec_losses += 1
            else:
                break

        state = np.array([
            np.clip(ensemble_confidence, 0, 1),
            model_agreement / max(total_models, 1),
            regime_map.get(regime, 0.5),
            np.clip(rsi / 100.0, 0, 1) if rsi else 0.5,
            np.clip(adx / 50.0, 0, 1) if adx else 0.5,
            np.clip(atr / max(atr_median, 1e-6), 0, 2) / 2.0 if atr else 0.5,
            np.clip(ema200_dist, -1, 1),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            recent_wr,
            np.clip(recent_avg_rr / 3.0, -1, 1),
            np.clip(current_drawdown, 0, 1),
            min(consec_losses / 5.0, 1.0),
            np.clip(spread / max(atr, 1e-6), 0, 1) if spread and atr else 0.5,
            np.clip(volume_ratio / 2.0, 0, 1) if volume_ratio else 0.5,
            daily_trades / max(max_daily_trades, 1),
        ], dtype=np.float32)

        return state

    # ──────────────────────────────────────────────────────────────
    #  Action Selection
    # ──────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Returns: action index (0=SKIP, 1=SMALL, 2=FULL, 3=LARGE)
        """
        if training and random.random() < self.epsilon:
            action = random.randint(0, ACTION_DIM - 1)
        else:
            if self.mode == 'dqn':
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_t)
                    action = q_values.argmax(dim=1).item()
            else:
                key = self._discretize_state(state)
                action = int(np.argmax(self.q_table[key]))

        self.action_counts[action] += 1
        self.last_state = state
        self.last_action = action
        return action

    def get_lot_multiplier(self, action: int) -> float:
        """Convert action to lot size multiplier."""
        return {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.2}.get(action, 1.0)

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        return ACTIONS.get(action, 'UNKNOWN')

    # ──────────────────────────────────────────────────────────────
    #  Learning
    # ──────────────────────────────────────────────────────────────
    def record_outcome(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool = False,
                       trade_info: dict = None):
        """
        Record a transition and trigger learning.

        Args:
            state: State when action was taken.
            action: Action index taken.
            reward: Reward received.
            next_state: State after outcome.
            done: Whether episode ended.
            trade_info: Optional dict with {won, pips, rr} for history.
        """
        self.replay.push(state, action, reward, next_state, done)

        if trade_info:
            self.recent_trades.append(trade_info)
            self.total_trades += 1

        self.episode_rewards.append(reward)

        # Train if enough experiences
        if len(self.replay) >= self.min_experiences:
            self._train_step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def compute_reward(self, action: int, trade_result: dict = None,
                       would_have_won: bool = None) -> float:
        """
        Compute reward for a given action and outcome.

        Args:
            action: The action taken (0=skip, 1-3=trade with varying size).
            trade_result: If traded, dict with {pips, exit_type}.
            would_have_won: If skipped, whether trade would have won.
        """
        if action == 0:
            # Agent chose to skip
            if would_have_won is True:
                return -0.5  # Small regret for missing a win
            elif would_have_won is False:
                return 2.0   # Good skip — avoided a loss
            return 0.0       # Neutral

        # Agent chose to trade
        if trade_result is None:
            return 0.0

        pips = trade_result.get('pips', 0)
        exit_type = trade_result.get('exit_type', '')
        lot_mult = self.get_lot_multiplier(action)

        if pips > 0:
            reward = pips * lot_mult * 0.2  # Scale down for stability
        else:
            reward = pips * lot_mult * 0.3  # Asymmetric: losses hurt more

        # Timeout penalty
        if exit_type == 'TIMEOUT':
            reward -= 1.0

        return float(np.clip(reward, -10, 10))

    def _train_step(self, batch_size=64):
        """One step of Q-learning."""
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        if self.mode == 'dqn':
            self._train_dqn(states, actions, rewards, next_states, dones)
        else:
            self._train_qtable(states, actions, rewards, next_states, dones)

        self.training_step += 1

        # Update target network every 100 steps
        if self.mode == 'dqn' and self.training_step % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_dqn(self, states, actions, rewards, next_states, dones):
        """DQN training step."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_network(states_t)
        q_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN: use main net to select, target net to evaluate)
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_action, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def _train_qtable(self, states, actions, rewards, next_states, dones):
        """Simple Q-table update."""
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            key = self._discretize_state(s)
            next_key = self._discretize_state(ns)

            if d:
                target = r
            else:
                target = r + self.gamma * np.max(self.q_table[next_key])

            self.q_table[key][a] += self.lr * (target - self.q_table[key][a])

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for Q-table (4 bins per feature)."""
        bins = np.array([0.25, 0.5, 0.75])
        discrete = np.digitize(np.clip(state, 0, 1), bins)
        return tuple(discrete)

    # ──────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────
    def save_state(self):
        """Save agent state to disk."""
        try:
            state = {
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'total_trades': self.total_trades,
                'action_counts': dict(self.action_counts),
                'recent_trades': list(self.recent_trades),
                'episode_rewards_summary': {
                    'mean': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
                    'total': len(self.episode_rewards),
                },
            }

            if self.mode == 'qtable':
                state['q_table'] = {
                    str(k): v.tolist() for k, v in self.q_table.items()
                }

            os.makedirs(os.path.dirname(RL_MODEL_PATH), exist_ok=True)
            with open(RL_MODEL_PATH, 'w') as f:
                json.dump(state, f, indent=2)

            if self.mode == 'dqn':
                torch.save(self.q_network.state_dict(), RL_DQN_PATH)

            bot_logger.info(
                f"RL agent saved: ε={self.epsilon:.3f}, "
                f"steps={self.training_step}, trades={self.total_trades}"
            )
        except Exception as e:
            error_logger.error(f"Failed to save RL agent: {e}")

    def _load_state(self):
        """Load agent state from disk if available."""
        try:
            if os.path.exists(RL_MODEL_PATH):
                with open(RL_MODEL_PATH, 'r') as f:
                    state = json.load(f)

                self.epsilon = state.get('epsilon', self.epsilon)
                self.training_step = state.get('training_step', 0)
                self.total_trades = state.get('total_trades', 0)
                raw_counts = state.get('action_counts', {})
                self.action_counts = defaultdict(int, {int(k): v for k, v in raw_counts.items()})
                self.recent_trades = deque(
                    state.get('recent_trades', []), maxlen=50
                )

                if self.mode == 'qtable' and 'q_table' in state:
                    for k, v in state['q_table'].items():
                        self.q_table[eval(k)] = np.array(v)

                bot_logger.info(
                    f"RL agent loaded: ε={self.epsilon:.3f}, "
                    f"steps={self.training_step}, trades={self.total_trades}"
                )

            if self.mode == 'dqn' and os.path.exists(RL_DQN_PATH):
                self.q_network.load_state_dict(
                    torch.load(RL_DQN_PATH, map_location=self.device,
                               weights_only=True)
                )
                self.target_network.load_state_dict(self.q_network.state_dict())
                bot_logger.info("RL DQN weights loaded")

        except Exception as e:
            error_logger.error(f"Failed to load RL agent state: {e}")

    # ──────────────────────────────────────────────────────────────
    #  Info / Debug
    # ──────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return agent statistics."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        return {
            'mode': self.mode,
            'epsilon': round(self.epsilon, 4),
            'training_steps': self.training_step,
            'total_trades': self.total_trades,
            'replay_size': len(self.replay),
            'action_distribution': {
                ACTIONS.get(int(k), f'ACTION_{k}'): v for k, v in self.action_counts.items()
            },
            'recent_avg_reward': round(float(np.mean(recent_rewards)), 4) if recent_rewards else 0,
            'recent_win_rate': round(
                sum(1 for t in self.recent_trades if t.get('won'))
                / max(len(self.recent_trades), 1), 4
            ),
        }
