"""Adaptive skip must not block hybrid entries unless ADAPTIVE_SKIP_ENABLED=true."""

import os
from unittest.mock import MagicMock, patch

import pytest

from config.strategy_config import adaptive_skip_enabled


class _FakeLearner:
    current_regime = "volatile"

    def should_skip_trade(self, pair, regime, hour):
        return True, f"regime {regime} has <20% win rate"

    def should_skip_loss_pattern(self, pair):
        return True


def _blocks_entry(learner, *, verbose=False):
    """Mirror start_live_mtf_scalping.MTFScalpingBot._adaptive_blocks_entry logic."""
    from datetime import datetime, timezone

    if not learner:
        return None
    if not adaptive_skip_enabled():
        if verbose:
            hour = datetime.now(timezone.utc).hour
            skip, reason = learner.should_skip_trade(
                "MNQ", learner.current_regime, hour,
            )
            if skip:
                print(f"advisory: {reason}")
        return None
    hour = datetime.now(timezone.utc).hour
    skip, reason = learner.should_skip_trade("MNQ", learner.current_regime, hour)
    if skip:
        return f"Adaptive skip: {reason}"
    if learner.should_skip_loss_pattern("MNQ"):
        return "Adaptive loss-pattern block on MNQ"
    return None


def test_adaptive_skip_default_off():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ADAPTIVE_SKIP_ENABLED", None)
        assert adaptive_skip_enabled() is False


def test_adaptive_skip_explicit_true():
    with patch.dict(os.environ, {"ADAPTIVE_SKIP_ENABLED": "true"}):
        assert adaptive_skip_enabled() is True


def test_regime_skip_does_not_block_when_disabled():
    with patch.dict(os.environ, {"ADAPTIVE_SKIP_ENABLED": "false"}):
        assert _blocks_entry(_FakeLearner()) is None


def test_regime_skip_blocks_when_enabled():
    with patch.dict(os.environ, {"ADAPTIVE_SKIP_ENABLED": "true"}):
        msg = _blocks_entry(_FakeLearner())
        assert msg is not None
        assert "regime volatile" in msg


def test_loss_pattern_does_not_block_when_disabled():
    learner = _FakeLearner()
    with patch.dict(os.environ, {"ADAPTIVE_SKIP_ENABLED": "false"}):
        assert _blocks_entry(learner) is None


def test_loss_pattern_blocks_when_enabled():
    learner = _FakeLearner()
    with patch.dict(os.environ, {"ADAPTIVE_SKIP_ENABLED": "true"}):
        msg = _blocks_entry(learner)
        assert msg is not None
        assert "Adaptive skip" in msg or "loss-pattern" in msg
