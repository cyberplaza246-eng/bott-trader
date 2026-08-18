from src.strategy.mnq_profit_defaults import (
    apply_profit_env_defaults,
    should_lock_profit_mode,
)


PROFIT_CFG = {"scalp_hybrid": {"enabled": True, "profit_mode": True}}


def test_locks_when_profit_mode_and_no_strategy_env():
    assert should_lock_profit_mode(PROFIT_CFG, {}) is True


def test_does_not_lock_when_operator_picks_full_adaptive():
    env = {"STRATEGY_MODE": "full_adaptive"}
    assert should_lock_profit_mode(PROFIT_CFG, env) is False
    assert apply_profit_env_defaults(PROFIT_CFG, env) == []


def test_fills_only_unset_keys():
    env = {"LOSS_COOLDOWN_MINUTES": "7"}
    applied = apply_profit_env_defaults(PROFIT_CFG, env)
    assert "SCALP_MODE" in applied
    assert env["SCALP_MODE"] == "hybrid"
    assert env["SESSION_MODE"] == "rth"
    assert env["OVERNIGHT_TRADING"] == "false"
    assert env["LOSS_COOLDOWN_MINUTES"] == "7"
    assert "LOSS_COOLDOWN_MINUTES" not in applied


def test_skips_when_hybrid_block_disabled():
    assert should_lock_profit_mode({"scalp_hybrid": {"enabled": False}}, {}) is False
