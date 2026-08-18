from src.strategy.mnq_profit_defaults import (
    apply_profit_env_defaults,
    should_lock_profit_mode,
)


PROFIT_CFG = {"ema15_eod": {"enabled": True, "profit_mode": True}}


def test_locks_when_profit_mode_and_no_strategy_env():
    assert should_lock_profit_mode(PROFIT_CFG, {}) is True


def test_does_not_lock_when_operator_picks_full_adaptive():
    env = {"STRATEGY_MODE": "full_adaptive"}
    assert should_lock_profit_mode(PROFIT_CFG, env) is False
    assert apply_profit_env_defaults(PROFIT_CFG, env) == []


def test_upgrades_leftover_hybrid():
    env = {"SCALP_MODE": "hybrid", "STRATEGY_MODE": "scalp_hybrid"}
    assert should_lock_profit_mode(PROFIT_CFG, env) is True
    applied = apply_profit_env_defaults(PROFIT_CFG, env)
    assert env["SCALP_MODE"] == "ema15_eod"
    assert env["STRATEGY_MODE"] == "ema15_eod"
    assert env["USE_30S_BARS"] == "false"
    assert env["MAX_TRADES_PER_DAY"] == "6"
    assert env["SCALP_BREAKEVEN_ENABLED"] == "false"
    assert env["SCALP_TRAIL_AFTER_BE"] == "false"
    assert env["EMA15_REQUIRE_60M"] == "true"
    assert "SCALP_MODE" in applied


def test_fills_only_unset_non_force_keys():
    env = {"LOSS_COOLDOWN_MINUTES": "7"}
    applied = apply_profit_env_defaults(PROFIT_CFG, env)
    assert env["SCALP_MODE"] == "ema15_eod"
    assert env["SESSION_MODE"] == "rth"
    assert env["OVERNIGHT_TRADING"] == "false"
    assert env["LOSS_COOLDOWN_MINUTES"] == "7"
    assert "LOSS_COOLDOWN_MINUTES" not in applied
    assert "SCALP_MODE" in applied


def test_forces_off_leftover_breakeven_trail():
    env = {"SCALP_BREAKEVEN_ENABLED": "true", "SCALP_TRAIL_AFTER_BE": "true"}
    apply_profit_env_defaults(PROFIT_CFG, env)
    assert env["SCALP_BREAKEVEN_ENABLED"] == "false"
    assert env["SCALP_TRAIL_AFTER_BE"] == "false"


def test_skips_when_ema15_block_disabled():
    assert should_lock_profit_mode({"ema15_eod": {"enabled": False}}, {}) is False
