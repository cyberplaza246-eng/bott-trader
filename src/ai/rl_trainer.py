"""
Ultimate RL Training Module
Combines TD3 with multi-source data for the greatest trading AI
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gym
from gym import spaces
from collections import deque
import random
import os
from datetime import datetime, timedelta
import logging

class TradingEnvironment(gym.Env):
    """Ultimate trading environment combining all data sources"""

    def __init__(self, symbols, initial_balance=50000, max_steps=1000):
        super(TradingEnvironment, self).__init__()

        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: [position_size, stop_loss_mult, take_profit_mult] for each asset
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5, 1.0] * self.n_assets),
            high=np.array([1.0, 2.0, 3.0] * self.n_assets),
            dtype=np.float32
        )

        # State space: comprehensive market state
        state_dims = self.n_assets * 50 + 10  # Market data + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dims,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.trades = []
        self.done = False

        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Reshape action for multiple assets
        action = action.reshape(self.n_assets, 3)
        rewards = []

        for i, symbol in enumerate(self.symbols):
            asset_action = action[i]
            position_size = asset_action[0]
            stop_loss_mult = asset_action[1]
            take_profit_mult = asset_action[2]

            reward = self._execute_trade(symbol, position_size, stop_loss_mult, take_profit_mult)
            rewards.append(reward)

        # Combined reward
        total_reward = np.mean(rewards)

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        return self._get_state(), total_reward, self.done, {}

    def _execute_trade(self, symbol, position_size, stop_loss_mult, take_profit_mult):
        """Execute trade for a single asset"""
        # Simplified simulation - in real implementation, use actual price data
        price_change = np.random.normal(0, 0.01)  # Random price movement

        if abs(position_size) > 0.1:  # Minimum position threshold
            # Calculate position value
            position_value = position_size * self.portfolio_value * 0.1  # Max 10% per asset

            # Simulate trade outcome
            pnl = position_value * price_change * np.sign(position_size)

            # Apply stop loss / take profit logic
            if abs(price_change) > stop_loss_mult * 0.02:  # 2% base stop
                pnl = -abs(position_value) * stop_loss_mult * 0.02
            elif abs(price_change) > take_profit_mult * 0.05:  # 5% base target
                pnl = abs(position_value) * take_profit_mult * 0.05

            self.balance += pnl
            self.portfolio_value = self.balance + np.sum(np.abs(self.positions))

            return pnl / self.initial_balance  # Normalized reward

        return 0

    def _get_state(self):
        """Get comprehensive market state"""
        state = []

        # Simulate market data for each asset
        for symbol in self.symbols:
            # Price data (OHLCV)
            prices = np.random.normal(100, 5, 5)  # [Open, High, Low, Close, Volume]

            # Technical indicators
            rsi = np.random.uniform(30, 70)
            macd = np.random.normal(0, 0.5)
            bb_upper = prices[3] * 1.02
            bb_lower = prices[3] * 0.98

            # Volume indicators
            volume_sma = np.random.normal(1000000, 100000)

            # Sentiment data
            news_sentiment = np.random.normal(0, 0.3)

            # Macro data
            macro_score = np.random.normal(0, 0.2)

            asset_state = np.concatenate([
                prices,  # 5 dims
                [rsi, macd, bb_upper, bb_lower],  # 4 dims
                [volume_sma],  # 1 dim
                [news_sentiment],  # 1 dim
                [macro_score]  # 1 dim
            ])
            state.extend(asset_state)

        # Portfolio state
        portfolio_state = [
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            np.mean(self.positions),  # Average position
            len([p for p in self.positions if abs(p) > 0]) / self.n_assets,  # Position concentration
            self.current_step / self.max_steps  # Episode progress
        ]

        state.extend(portfolio_state)

        return np.array(state, dtype=np.float32)

class ReplayBuffer:
    """Experience replay buffer for TD3"""

    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

class TD3Trainer:
    """TD3 Training with multi-source data integration"""

    def __init__(self, state_dim, action_dim, max_action, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        # Select action according to policy and add clipped noise
        noise = (
            torch.randn_like(action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (
            self.actor_target(next_state) + noise
        ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor")
        torch.save(self.critic.state_dict(), f"{filename}_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic"))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

def train_ultimate_model(symbols, episodes=1000, save_path="models/ultimate_td3"):
    """Train the ultimate TD3 model"""

    # Initialize environment and agent
    env = TradingEnvironment(symbols)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Trainer(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Training loop
    rewards = []
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            if episode < 100:  # Exploration
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Store transition
            replay_buffer.add(state, action, next_state, reward, done)

            # Train agent
            if replay_buffer.size > 256:
                agent.train(replay_buffer)

            state = next_state

        rewards.append(episode_reward)
        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.4f}")

            # Save model
            agent.save(f"{save_path}_episode_{episode + 1}")

    return agent, episode_rewards

if __name__ == "__main__":
    # Train the ultimate model
    symbols = ['MES', 'NQ', 'SPY', 'QQQ']
    agent, rewards = train_ultimate_model(symbols, episodes=1000)

    print("Training completed!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.4f}")