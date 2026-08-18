"""Flow-aware 30s trigger relax when bar color blocks but tape confirms direction."""

from src.strategy.scalp_hybrid import (
    _trigger_eval,
    default_trigger_flow_cfg,
)


def _relax_cfg(**overrides):
    cfg = default_trigger_flow_cfg()
    cfg["trigger_relax_flow"] = True
    cfg.update(overrides)
    return cfg


def test_user_scenario1_red_bar_close_above_prev_close():
    """Live log: red 30s bar, Δ+21 buy%=64%, close > prev close → trigger fires."""
    flow = {"delta": 21, "buy_pct": 0.64}
    fired, reason = _trigger_eval(
        30226.50, 30225.00,
        prev_high=30226.75, prev_low=30224.00, prev_close=30224.50,
        direction=1,
        h=30226.75, l=30224.00,
        aggressive=True,
        setup_mode="burst",
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is True
    assert reason == "flow relax: close > prev close"


def test_user_scenario2_red_bar_close_above_prev_close():
    """Live log: red 30s bar, Δ+30 buy%=64%, close > prev close → trigger fires."""
    flow = {"delta": 30, "buy_pct": 0.64}
    fired, reason = _trigger_eval(
        30228.00, 30225.25,
        prev_high=30228.25, prev_low=30224.75, prev_close=30224.00,
        direction=1,
        h=30228.25, l=30224.75,
        aggressive=True,
        setup_mode="burst",
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is True
    assert reason == "flow relax: close > prev close"


def test_red_bar_blocked_without_flow_relax():
    flow = {"delta": 21, "buy_pct": 0.64}
    fired, reason = _trigger_eval(
        30226.50, 30225.00,
        prev_high=30226.75, prev_low=30224.00, prev_close=30224.50,
        direction=1,
        h=30226.75, l=30224.00,
        flow_snap=flow,
        trigger_flow_cfg={"trigger_relax_flow": False},
    )
    assert fired is False
    assert reason == "not bullish (close <= open)"


def test_red_bar_blocked_when_flow_weak():
    flow = {"delta": 5, "buy_pct": 0.52}
    fired, reason = _trigger_eval(
        30226.50, 30225.00,
        prev_high=30226.75, prev_low=30224.00, prev_close=30224.50,
        direction=1,
        h=30226.75, l=30224.00,
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is False
    assert reason == "not bullish (close <= open)"


def test_flow_relax_close_above_prev_high_no_green():
    flow = {"delta": 40, "buy_pct": 0.65}
    fired, reason = _trigger_eval(
        30226.00, 30227.00,
        prev_high=30226.50, prev_low=30224.00, prev_close=30225.00,
        direction=1,
        h=30227.25, l=30225.50,
        aggressive=True,
        setup_mode="burst",
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is True
    assert "prev high" in reason


def test_flow_relax_upper_half_of_bar():
    flow = {"delta": 25, "buy_pct": 0.62}
    # close at bar mid or above, but not above prev close
    fired, reason = _trigger_eval(
        30226.00, 30226.50,
        prev_high=30227.00, prev_low=30225.00, prev_close=30226.75,
        direction=1,
        h=30227.00, l=30226.00,
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is True
    assert reason == "flow relax: close in upper half"


def test_short_flow_relax_close_below_prev_close():
    flow = {"delta": -35, "buy_pct": 0.35}
    fired, reason = _trigger_eval(
        30227.00, 30226.60,
        prev_high=30228.00, prev_low=30226.50, prev_close=30226.75,
        direction=-1,
        h=30227.25, l=30226.40,
        aggressive=True,
        setup_mode="burst",
        flow_snap=flow,
        trigger_flow_cfg=_relax_cfg(),
    )
    assert fired is True
    assert "prev close" in reason
