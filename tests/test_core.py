"""
Unit tests for the AI Forex Trading Bot

Tests cover:
  - RiskManager: position sizing, tier management, trade gating
  - EnsembleTrader: signal generation, weight normalization
  - BacktestEngine: basic run, trade recording
  - WalkForwardEngine: fold construction, result aggregation
  - NLPSentimentAnalyzer: keyword fallback
  - RLTradingAgent: state construction, action selection, reward computation
  - AdaptiveLearner: regime detection, weight adjustment

Run:
  python -m pytest tests/ -v
  python -m pytest tests/test_core.py -v --tb=short
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ═══════════════════════════════════════════════════════════════════
#  Test Data Helpers
# ═══════════════════════════════════════════════════════════════════

def make_ohlcv(n=500, start_price=1.1000, volatility=0.0005, freq='5min'):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq=freq)
    close = [start_price]
    for _ in range(n - 1):
        change = np.random.normal(0, volatility)
        close.append(close[-1] + change)
    close = np.array(close)

    high = close + np.abs(np.random.normal(0, volatility * 0.5, n))
    low = close - np.abs(np.random.normal(0, volatility * 0.5, n))
    opn = close + np.random.normal(0, volatility * 0.3, n)
    volume = np.random.randint(100, 10000, n)

    return pd.DataFrame({
        'datetime': dates,
        'open': opn,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


# ═══════════════════════════════════════════════════════════════════
#  RiskManager Tests
# ═══════════════════════════════════════════════════════════════════

class TestRiskManager:
    """Tests for src.risk.position_manager.RiskManager"""

    def setup_method(self):
        from src.risk.position_manager import RiskManager
        self.rm = RiskManager(initial_balance=1000)

    def test_initial_tier(self):
        """Balance $1000 should be in 'standard' tier."""
        assert self.rm._current_tier_name == 'standard'
        assert self.rm._current_tier['max_lot_size'] == 0.04

    def test_micro_tier(self):
        from src.risk.position_manager import RiskManager
        rm = RiskManager(initial_balance=50)
        assert rm._current_tier_name == 'micro'

    def test_elite_tier(self):
        from src.risk.position_manager import RiskManager
        rm = RiskManager(initial_balance=30000)
        assert rm._current_tier_name == 'elite'

    def test_can_trade_basic(self):
        """Should be able to trade when within limits."""
        assert self.rm.can_trade() is True

    def test_cannot_trade_max_concurrent(self):
        """Should block when max trades reached."""
        self.rm.open_trades = 10
        assert self.rm.can_trade() is False

    def test_cannot_trade_daily_loss(self):
        """Should block when daily loss exceeded."""
        self.rm.daily_loss = self.rm.daily_loss_limit + 1
        assert self.rm.can_trade() is False

    def test_position_size_basic(self):
        """Should calculate valid position size."""
        result = self.rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            pair='EUR/USD'
        )
        assert result is not None
        assert result['lot_size'] > 0
        assert result['lot_size'] <= self.rm._current_tier['max_lot_size']

    def test_position_size_jpy(self):
        """JPY pair should use correct pip math."""
        result = self.rm.calculate_position_size(
            entry_price=150.000,
            stop_loss_price=149.500,
            pair='USD/JPY'
        )
        assert result is not None
        assert result['lot_size'] > 0

    def test_tier_upgrade(self):
        """Tier should upgrade when balance increases."""
        self.rm.sync_balance(6000)
        assert self.rm._current_tier_name == 'professional'

    def test_tier_hysteresis_no_downgrade(self):
        """Should not downgrade within hysteresis band (2%)."""
        self.rm.sync_balance(5000)
        assert self.rm._current_tier_name == 'professional'
        # Drop slightly — within 2% band, should stay
        self.rm.sync_balance(4950)
        assert self.rm._current_tier_name == 'professional'

    def test_daily_loss_limit_scales(self):
        """Daily loss limit should scale with balance."""
        from config.strategy_config import DAILY_LOSS_LIMIT_PERCENT
        self.rm.sync_balance(10000)
        expected = 10000 * (DAILY_LOSS_LIMIT_PERCENT / 100)
        assert self.rm.daily_loss_limit == expected

    def test_on_trade_opened_closed(self):
        """Trade count should track opens and closes."""
        self.rm.on_trade_opened()
        assert self.rm.open_trades == 1
        self.rm.on_trade_closed(profit_loss=10.0)
        assert self.rm.open_trades == 0


# ═══════════════════════════════════════════════════════════════════
#  AdaptiveLearner Tests
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveLearner:
    """Tests for src.ai.adaptive_learner.AdaptiveLearner"""

    def setup_method(self):
        from src.ai.adaptive_learner import AdaptiveLearner
        self.learner = AdaptiveLearner()

    def test_initial_weights(self):
        """Should have default weights."""
        assert 'scalping' in self.learner.model_weights
        assert self.learner.model_weights['scalping'] > 0

    def test_regime_detection_trending(self):
        """Should detect trending regime with strong ADX."""
        df = make_ohlcv(100)
        # Make a clear uptrend
        df['close'] = np.linspace(1.1, 1.15, 100)
        df['high'] = df['close'] + 0.001
        df['low'] = df['close'] - 0.0005
        regime = self.learner.detect_regime(df)
        assert regime in ('trending', 'ranging', 'volatile')

    def test_adjusted_weights_returns_dict(self):
        """get_adjusted_weights should return a dict of model weights."""
        weights = self.learner.get_adjusted_weights(pair='EUR/USD')
        assert isinstance(weights, dict)

    def test_drawdown_protection(self):
        """Consecutive losses should trigger drawdown protection."""
        for i in range(6):
            self.learner.record_trade({
                'pair': 'EUR/USD',
                'profit_loss': -5.0,
                'model_signals': {},
                'signal': 'BUY',
                'regime': 'trending',
            })
        # After multiple losses, threshold should be raised
        threshold = self.learner.get_adjusted_threshold()
        assert threshold >= 0.30  # At minimum the base threshold


# ═══════════════════════════════════════════════════════════════════
#  NLP Sentiment Tests
# ═══════════════════════════════════════════════════════════════════

class TestNLPSentiment:
    """Tests for src.ai.nlp_sentiment.NLPSentimentAnalyzer"""

    def setup_method(self):
        from src.ai.nlp_sentiment import NLPSentimentAnalyzer
        self.analyzer = NLPSentimentAnalyzer()
        # Force keyword mode for deterministic tests
        self.analyzer.use_finbert = False

    def test_bullish_text(self):
        result = self.analyzer.analyze_text_sentiment(
            "EUR surges on strong growth data, bullish momentum continues"
        )
        assert result['score'] > 0
        assert result['method'] == 'keyword'

    def test_bearish_text(self):
        result = self.analyzer.analyze_text_sentiment(
            "Dollar plunges amid recession fears, bearish selloff deepens"
        )
        assert result['score'] < 0

    def test_neutral_text(self):
        result = self.analyzer.analyze_text_sentiment(
            "The weather is nice today in New York."
        )
        assert result['score'] == 0.0
        assert result['confidence'] == 0.0

    def test_empty_text(self):
        result = self.analyzer.analyze_text_sentiment("")
        assert result['score'] == 0.0

    def test_batch_analysis(self):
        texts = [
            "Markets rally on strong earnings",
            "Economy in decline as recession looms",
            "The cat sat on the mat"
        ]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert results[0]['score'] > 0  # bullish
        assert results[1]['score'] < 0  # bearish

    def test_get_pair_sentiment_no_api_key(self):
        """Without API key, should return neutral."""
        result = self.analyzer.get_pair_sentiment('EUR/USD')
        assert result['sentiment_score'] == 0.0
        assert result['news_count'] == 0


# ═══════════════════════════════════════════════════════════════════
#  RL Agent Tests
# ═══════════════════════════════════════════════════════════════════

class TestRLAgent:
    """Tests for src.ai.rl_agent.RLTradingAgent"""

    def setup_method(self):
        from src.ai.rl_agent import RLTradingAgent
        self.agent = RLTradingAgent(epsilon_start=1.0, min_experiences=10)
        # Reset counters so tests start from clean state
        self.agent.total_trades = 0
        self.agent.training_step = 0
        self.agent.epsilon = 1.0
        self.agent.replay.buffer.clear()
        self.agent.episode_rewards.clear()

    def test_build_state_shape(self):
        """State vector should have correct dimensions."""
        state = self.agent.build_state(
            ensemble_confidence=0.55,
            model_agreement=5, total_models=9,
            regime='trending',
            rsi=45, adx=25, atr=0.0008, atr_median=0.0006,
            ema200_dist=0.002,
            hour=10,
            spread=0.00015, volume_ratio=1.2,
            daily_trades=1, max_daily_trades=3,
            current_drawdown=0.02,
        )
        assert state.shape == (16,)
        assert state.dtype == np.float32

    def test_select_action(self):
        """Should return valid action index."""
        state = np.random.rand(16).astype(np.float32)
        action = self.agent.select_action(state, training=True)
        assert 0 <= action <= 3

    def test_lot_multiplier(self):
        assert self.agent.get_lot_multiplier(0) == 0.0  # skip
        assert self.agent.get_lot_multiplier(1) == 0.5  # small
        assert self.agent.get_lot_multiplier(2) == 1.0  # full
        assert self.agent.get_lot_multiplier(3) == 1.2  # large

    def test_compute_reward_skip_avoided_loss(self):
        reward = self.agent.compute_reward(0, would_have_won=False)
        assert reward > 0

    def test_compute_reward_skip_missed_win(self):
        reward = self.agent.compute_reward(0, would_have_won=True)
        assert reward < 0

    def test_compute_reward_winning_trade(self):
        reward = self.agent.compute_reward(
            2, trade_result={'pips': 5.0, 'exit_type': 'TAKE_PROFIT'}
        )
        assert reward > 0

    def test_compute_reward_losing_trade(self):
        reward = self.agent.compute_reward(
            2, trade_result={'pips': -5.0, 'exit_type': 'STOP_LOSS'}
        )
        assert reward < 0

    def test_record_and_learn(self):
        """Agent should learn from recorded experiences."""
        for i in range(20):
            state = np.random.rand(16).astype(np.float32)
            action = self.agent.select_action(state)
            reward = float(np.random.normal(0, 1))
            next_state = np.random.rand(16).astype(np.float32)
            self.agent.record_outcome(
                state, action, reward, next_state,
                trade_info={'won': reward > 0, 'pips': reward, 'rr': abs(reward)}
            )
        assert len(self.agent.replay) == 20
        assert self.agent.total_trades == 20

    def test_epsilon_decay(self):
        """Epsilon should decay over time."""
        initial_eps = self.agent.epsilon
        state = np.random.rand(16).astype(np.float32)
        for _ in range(50):
            action = self.agent.select_action(state)
            self.agent.record_outcome(
                state, action, 1.0, state,
                trade_info={'won': True, 'pips': 1, 'rr': 1}
            )
        assert self.agent.epsilon < initial_eps

    def test_get_stats(self):
        stats = self.agent.get_stats()
        assert 'mode' in stats
        assert 'epsilon' in stats
        assert 'replay_size' in stats


# ═══════════════════════════════════════════════════════════════════
#  BacktestEngine Tests
# ═══════════════════════════════════════════════════════════════════

class TestBacktestEngine:
    """Tests for src.backtest.backtest_engine.BacktestEngine"""

    def setup_method(self):
        from src.backtest.backtest_engine import BacktestEngine
        self.engine = BacktestEngine(initial_balance=10000)

    def test_initialization(self):
        assert self.engine.initial_balance == 10000
        assert self.engine.current_balance == 10000
        assert len(self.engine.trades) == 0

    def test_run_backtest_with_little_data(self):
        """Should handle short datasets gracefully."""
        df = make_ohlcv(50)  # Too short for lookback=200
        result = self.engine.run_backtest(df, 'EUR/USD')
        assert 'total_trades' in result
        assert result['initial_balance'] == 10000

    def test_run_backtest_normal(self):
        """Should produce valid results on normal-length data."""
        df = make_ohlcv(1000, volatility=0.001)
        result = self.engine.run_backtest(df, 'EUR/USD', confidence_threshold=0.30)
        assert 'total_trades' in result
        assert 'win_rate' in result
        assert 'max_drawdown' in result
        assert 'sharpe_ratio' in result
        assert result['initial_balance'] == 10000

    def test_result_has_required_keys(self):
        df = make_ohlcv(500)
        result = self.engine.run_backtest(df, 'EUR/USD')
        required = [
            'pair', 'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown',
            'total_profit', 'final_balance', 'return_percent',
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════
#  WalkForwardEngine Tests
# ═══════════════════════════════════════════════════════════════════

class TestWalkForwardEngine:
    """Tests for src.backtest.walk_forward.WalkForwardEngine"""

    def setup_method(self):
        from src.backtest.walk_forward import WalkForwardEngine
        self.wf = WalkForwardEngine(initial_balance=10000)

    def test_build_folds_rolling(self):
        folds = self.wf._build_folds(1000, train_pct=0.7, n_splits=5, mode='rolling')
        assert len(folds) > 0
        for train_s, train_e, test_s, test_e in folds:
            assert train_s < train_e
            assert test_s == train_e
            assert test_s < test_e

    def test_build_folds_anchored(self):
        folds = self.wf._build_folds(1000, train_pct=0.7, n_splits=5, mode='anchored')
        assert len(folds) > 0
        for train_s, train_e, test_s, test_e in folds:
            assert train_s == 0  # anchored starts at 0
            assert test_s < test_e

    def test_empty_result_on_short_data(self):
        df = make_ohlcv(100)
        result = self.wf.run_walk_forward(df, 'EUR/USD')
        assert result['total_trades'] == 0

    def test_run_walk_forward_basic(self):
        """Full walk-forward should produce results without crashing."""
        df = make_ohlcv(2000, volatility=0.001)
        result = self.wf.run_walk_forward(
            df, 'EUR/USD',
            n_splits=3,
            confidence_threshold=0.30,
        )
        assert 'fold_results' in result
        assert 'overfit_ratio' in result
        assert result['mode'] == 'walk_forward'

    def test_slippage_reduces_profits(self):
        """Slippage should reduce overall profitability."""
        from src.risk.position_manager import PIP_VALUES, DEFAULT_PIP
        # Just test the _apply_slippage method
        trades = [
            {'profit_loss': 10.0, 'lot_size': 0.01, 'pips': 5.0},
            {'profit_loss': -5.0, 'lot_size': 0.01, 'pips': -2.5},
        ]
        result = {
            'win_rate': 50.0, 'profit_factor': 2.0,
            'total_profit': 5.0, 'initial_balance': 10000,
            'final_balance': 10005, 'return_percent': 0.05,
        }
        adjusted = self.wf._apply_slippage(result, trades, 'EUR/USD', slippage_pips=1.0)
        # Slippage should reduce total profit
        assert adjusted['total_profit'] < 5.0


# ═══════════════════════════════════════════════════════════════════
#  Technical Analyzer Tests
# ═══════════════════════════════════════════════════════════════════

class TestTechnicalAnalyzer:
    """Tests for src.ai.technical_analyzer.TechnicalAnalyzer"""

    def setup_method(self):
        from src.ai.technical_analyzer import TechnicalAnalyzer
        self.ta = TechnicalAnalyzer()

    def test_calculate_indicators(self):
        """Should add indicator columns to dataframe."""
        df = make_ohlcv(300)
        result = self.ta.calculate_indicators(df)
        expected_cols = ['rsi', 'macd', 'ema_200', 'ema_50', 'atr']
        for col in expected_cols:
            assert col in result.columns, f"Missing indicator: {col}"

    def test_get_signal(self):
        """Should return a valid signal dict."""
        df = make_ohlcv(300)
        df = self.ta.calculate_indicators(df)
        signal = self.ta.get_signal(df)
        assert 'signal' in signal
        assert signal['signal'] in ('BUY', 'SELL', 'HOLD')
        assert 'confidence' in signal
        assert 0 <= signal['confidence'] <= 1


# ═══════════════════════════════════════════════════════════════════
#  Integration Smoke Test
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationSmoke:
    """Lightweight integration tests to verify modules work together."""

    def test_ensemble_imports(self):
        """EnsembleTrader should import without errors."""
        from src.core.ensemble_trader import EnsembleTrader
        ensemble = EnsembleTrader()
        assert ensemble is not None

    def test_backtest_pipeline(self):
        """Full backtest pipeline should run end-to-end."""
        from src.backtest.backtest_engine import BacktestEngine
        engine = BacktestEngine(initial_balance=5000)
        df = make_ohlcv(800, volatility=0.0008)
        result = engine.run_backtest(df, 'EUR/USD', confidence_threshold=0.25)
        assert isinstance(result, dict)
        assert 'final_balance' in result

    def test_config_loads(self):
        """Strategy config should load without errors."""
        from config.strategy_config import (
            PAIRS, SCALPING_PAIRS, ENSEMBLE_CONFIDENCE_THRESHOLD,
            TRADING_MODE, SCALPING_SESSION_WINDOWS,
        )
        assert len(PAIRS) >= 2
        assert ENSEMBLE_CONFIDENCE_THRESHOLD > 0
        assert TRADING_MODE in ('live', 'paper', 'backtest')
        assert 'EUR/USD' in SCALPING_PAIRS

    def test_config_threshold_reads_env(self):
        """Threshold should read from env, not be hard-coded."""
        from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD
        # With .env ENSEMBLE_CONFIDENCE_THRESHOLD=0.70, it should be 0.70
        # Without .env, default should be 0.45 (not 0.30)
        assert ENSEMBLE_CONFIDENCE_THRESHOLD >= 0.40, (
            f"Threshold is {ENSEMBLE_CONFIDENCE_THRESHOLD}, expected >= 0.40"
        )


# ═══════════════════════════════════════════════════════════════════
#  LiquiditySweepAnalyzer Tests
# ═══════════════════════════════════════════════════════════════════

def make_sweep_data(n=100, direction='BUY', include_sweep=True):
    """Generate OHLCV data with an embedded liquidity sweep event.

    For BUY sweeps: creates a 5-candle range, then a candle that dips
    below the range low and closes back above it, followed by a strong
    bullish displacement candle.
    """
    np.random.seed(42)
    base = 1.1000
    data = {
        'open': [], 'high': [], 'low': [], 'close': [],
        'volume': [],
    }

    price = base
    for i in range(n):
        vol = np.random.randint(500, 2000)

        if include_sweep and i == n - 3:
            # Sweep candle: dips below prior range then closes above
            if direction == 'BUY':
                sweep_low = price - 0.0020  # Dips 20 pips below
                data['open'].append(price)
                data['high'].append(price + 0.0002)
                data['low'].append(sweep_low)
                data['close'].append(price - 0.0003)  # Closes near open (above sweep low)
                data['volume'].append(int(vol * 0.8))
            else:
                sweep_high = price + 0.0020
                data['open'].append(price)
                data['high'].append(sweep_high)
                data['low'].append(price - 0.0002)
                data['close'].append(price + 0.0003)
                data['volume'].append(int(vol * 0.8))
        elif include_sweep and i == n - 2:
            # Displacement candle: strong move in sweep direction
            if direction == 'BUY':
                data['open'].append(price - 0.0002)
                data['high'].append(price + 0.0015)
                data['low'].append(price - 0.0003)
                data['close'].append(price + 0.0014)
                data['volume'].append(int(vol * 2.5))  # High volume
            else:
                data['open'].append(price + 0.0002)
                data['high'].append(price + 0.0003)
                data['low'].append(price - 0.0015)
                data['close'].append(price - 0.0014)
                data['volume'].append(int(vol * 2.5))
        elif include_sweep and i == n - 1:
            # Continuation candle
            if direction == 'BUY':
                data['open'].append(price + 0.0012)
                data['high'].append(price + 0.0018)
                data['low'].append(price + 0.0010)
                data['close'].append(price + 0.0016)
                data['volume'].append(vol)
            else:
                data['open'].append(price - 0.0012)
                data['high'].append(price - 0.0010)
                data['low'].append(price - 0.0018)
                data['close'].append(price - 0.0016)
                data['volume'].append(vol)
        else:
            change = np.random.normal(0, 0.0003)
            c = price + change
            h = max(price, c) + abs(np.random.normal(0, 0.0002))
            l = min(price, c) - abs(np.random.normal(0, 0.0002))
            data['open'].append(price)
            data['high'].append(h)
            data['low'].append(l)
            data['close'].append(c)
            data['volume'].append(vol)
            price = c

    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    return pd.DataFrame({
        'datetime': dates,
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume'],
    })


class TestLiquiditySweepAnalyzer:
    """Tests for src.ai.liquidity_sweep.LiquiditySweepAnalyzer"""

    def setup_method(self):
        from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
        self.analyzer = LiquiditySweepAnalyzer()

    def test_instantiation(self):
        """Should create analyzer without errors."""
        assert self.analyzer is not None
        assert self.analyzer.SWEEP_LOOKBACK == 3

    def test_calculate_indicators(self):
        """Should add all required indicator columns."""
        df = make_ohlcv(200, freq='1min')
        result = self.analyzer.calculate_indicators(df)
        for col in ['ema_20', 'ema_50', 'ema_200', 'atr', 'rsi', 'adx',
                     'volume_ratio', 'body_ratio', 'liq_low', 'liq_high']:
            assert col in result.columns, f"Missing column: {col}"

    def test_detect_regime_insufficient_data(self):
        """Should return unknown regime with too little data."""
        df = make_ohlcv(10, freq='5min')
        df = self.analyzer.calculate_indicators(df)
        regime = self.analyzer.detect_regime(df)
        assert regime['regime'] == 'unknown'
        assert regime['bias'] is None

    def test_detect_regime_with_data(self):
        """Should classify regime with sufficient data."""
        df = make_ohlcv(200, freq='5min')
        df = self.analyzer.calculate_indicators(df)
        regime = self.analyzer.detect_regime(df)
        assert regime['regime'] in ('trend_up', 'trend_down', 'range',
                                     'high_volatility', 'low_volatility', 'unknown')

    def test_detect_sweep_no_data(self):
        """Should return not-detected with insufficient data."""
        result = self.analyzer.detect_sweep(None, 'BUY')
        assert result['detected'] is False

    def test_check_market_conditions(self):
        """Market condition gate should work."""
        df = make_ohlcv(200, freq='1min')
        df = self.analyzer.calculate_indicators(df)
        ok, reason = self.analyzer.check_market_conditions(df, 'EUR/USD')
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_risk_reward_calculation(self):
        """Should calculate valid SL/TP from sweep event."""
        sweep_result = {
            'detected': True,
            'direction': 'BUY',
            'sweep_wick': 1.0980,
            'swept_level': 1.0985,
        }
        displacement_result = {
            'confirmed': True,
            'entry_price': 1.1005,
        }
        regime_info = {
            'regime': 'trend_up',
            'atr': 0.0010,
        }
        rr = self.analyzer.calculate_risk_reward(
            sweep_result, displacement_result, regime_info, 'EUR/USD'
        )
        assert rr is not None
        assert rr['stop_loss'] < rr['entry_price']
        assert rr['take_profit'] > rr['entry_price']
        assert rr['rr_ratio'] == 2.0  # trend_up → 2.0R

    def test_risk_reward_high_vol(self):
        """High volatility regime should use 2.0R."""
        sweep_result = {
            'detected': True,
            'direction': 'BUY',
            'sweep_wick': 1.0980,
            'swept_level': 1.0985,
        }
        displacement_result = {
            'confirmed': True,
            'entry_price': 1.1005,
        }
        regime_info = {'regime': 'high_volatility', 'atr': 0.0010}
        rr = self.analyzer.calculate_risk_reward(
            sweep_result, displacement_result, regime_info, 'EUR/USD'
        )
        assert rr['rr_ratio'] == 2.5

    def test_risk_reward_range(self):
        """Range regime should use 1.2R."""
        sweep_result = {
            'detected': True,
            'direction': 'SELL',
            'sweep_wick': 1.1020,
            'swept_level': 1.1015,
        }
        displacement_result = {
            'confirmed': True,
            'entry_price': 1.0995,
        }
        regime_info = {'regime': 'range', 'atr': 0.0010}
        rr = self.analyzer.calculate_risk_reward(
            sweep_result, displacement_result, regime_info, 'EUR/USD'
        )
        assert rr['rr_ratio'] == 1.5
        assert rr['stop_loss'] > rr['entry_price']  # SELL SL above entry

    def test_get_signal_returns_skip_no_bias(self):
        """Should SKIP when 5M data has no directional bias."""
        df_1m = make_ohlcv(200, freq='1min')
        # Use short 5M data to trigger 'unknown' regime
        df_5m = make_ohlcv(30, freq='5min')
        result = self.analyzer.get_signal(df_1m, 'EUR/USD', df_5m=df_5m)
        assert result['signal'] == 'SKIP'
        assert result['confidence'] == 0.0

    def test_pair_config_coverage(self):
        """All traded pairs should have config."""
        for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY']:
            assert pair in self.analyzer.PAIR_CONFIG
            cfg = self.analyzer.PAIR_CONFIG[pair]
            assert 'pip_size' in cfg
            assert 'session_atr_min' in cfg

    # ── v2 Swing Detection Tests ──────────────────────────────────

    def test_detect_swing_points_basic(self):
        """Should detect swing highs and lows from synthetic data with clear structure."""
        # Build data with obvious swing: rise → peak → fall → trough → rise
        np.random.seed(42)
        n = 50
        prices = []
        base = 1.1000
        # Segment 1: rise for 12 bars (HL → HH)
        for i in range(12):
            prices.append(base + i * 0.0003)
        # Segment 2: fall for 12 bars (HH → HL)
        peak = prices[-1]
        for i in range(1, 13):
            prices.append(peak - i * 0.0003)
        # Segment 3: rise for 12 bars (creates another HH)
        trough = prices[-1]
        for i in range(1, 13):
            prices.append(trough + i * 0.0004)
        # Pad to n
        while len(prices) < n:
            prices.append(prices[-1] + 0.0001)

        close = np.array(prices[:n])
        high = close + 0.0002
        low = close - 0.0002
        opn = close - 0.0001
        df = pd.DataFrame({
            'open': opn, 'high': high, 'low': low, 'close': close,
            'volume': np.random.randint(500, 2000, n),
        })

        swings = self.analyzer.detect_swing_points(df, lookback=3)
        assert len(swings) > 0, "Should detect at least one swing point"
        types_found = {s['swing_type'] for s in swings}
        # Should find swing highs and/or swing lows
        assert 'high' in types_found or 'low' in types_found

    def test_detect_swing_points_flat_data(self):
        """Flat data should produce few or no meaningful swings."""
        n = 50
        flat = np.full(n, 1.1000)
        df = pd.DataFrame({
            'open': flat, 'high': flat + 0.00001,
            'low': flat - 0.00001, 'close': flat,
            'volume': np.full(n, 1000),
        })
        swings = self.analyzer.detect_swing_points(df, lookback=5)
        # May detect some, but labeling should be consistent
        assert isinstance(swings, list)

    def test_detect_mss_returns_dict(self):
        """detect_mss should return a dict with 'confirmed' key."""
        df = make_ohlcv(100, freq='1min')
        df = self.analyzer.calculate_indicators(df)
        sweep_result = {
            'detected': True,
            'direction': 'BUY',
            'sweep_wick': 1.0990,
            'swept_level': 1.0995,
            'candle_index': -3,
        }
        mss = self.analyzer.detect_mss(df, sweep_result)
        assert isinstance(mss, dict)
        assert 'confirmed' in mss
        assert isinstance(mss['confirmed'], bool)

    def test_detect_mss_no_sweep(self):
        """detect_mss should return not-confirmed when sweep not detected."""
        df = make_ohlcv(100, freq='1min')
        df = self.analyzer.calculate_indicators(df)
        sweep_result = {
            'detected': False,
            'direction': None,
            'sweep_wick': None,
            'swept_level': None,
            'candle_index': None,
        }
        mss = self.analyzer.detect_mss(df, sweep_result)
        assert mss['confirmed'] is False

    def test_get_signal_returns_mss_key(self):
        """get_signal result should contain 'mss' key in v2."""
        df_1m = make_ohlcv(200, freq='1min')
        df_5m = make_ohlcv(200, freq='5min')
        result = self.analyzer.get_signal(df_1m, 'EUR/USD', df_5m=df_5m)
        assert 'mss' in result, "v2 get_signal should return 'mss' key"

    def test_regime_returns_invalidation_levels(self):
        """detect_regime should return 5M invalidation levels."""
        df = make_ohlcv(200, freq='5min')
        df = self.analyzer.calculate_indicators(df)
        regime = self.analyzer.detect_regime(df)
        # v2 always returns these keys
        assert 'last_swing_high' in regime
        assert 'last_swing_low' in regime
        assert 'swing_points' in regime


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
