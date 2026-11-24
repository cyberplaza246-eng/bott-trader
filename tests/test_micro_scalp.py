"""Unit tests for micro-scalp $20 TP / 10s min-hold logic."""
from datetime import datetime

import pytz

from src.strategy.micro_scalp import (
    MicroScalpState,
    check_quote_exit,
    is_rth_scan_et,
    tp_pts_from_dollars,
    unrealized_pnl_usd,
)

ET = pytz.timezone("US/Eastern")


def test_tp_pts_twenty_dollars():
    assert tp_pts_from_dollars(20.0) == 10.0


def test_unrealized_long_short():
    assert unrealized_pnl_usd("long", 20000.0, 20010.0) == 20.0
    assert unrealized_pnl_usd("short", 20000.0, 19990.0) == 20.0


def test_target_not_before_min_hold():
    pos = {
        "direction": "long",
        "entry": 20000.0,
        "sl": 19990.0,
        "tp_dollars": 20.0,
        "min_hold_sec": 10.0,
        "max_hold_sec": 180.0,
    }
    # +10 pts = $20, but only 5s held → no target yet
    assert check_quote_exit(pos, 20010.0, held_sec=5.0) is None
    hit = check_quote_exit(pos, 20010.0, held_sec=10.0)
    assert hit is not None
    assert hit[0] == "target"


def test_stop_always_live():
    pos = {
        "direction": "long",
        "entry": 20000.0,
        "sl": 19990.0,
        "tp_dollars": 20.0,
        "min_hold_sec": 10.0,
        "max_hold_sec": 180.0,
    }
    hit = check_quote_exit(pos, 19990.0, held_sec=1.0)
    assert hit is not None
    assert hit[0] == "stop"


def test_time_stop():
    pos = {
        "direction": "short",
        "entry": 20000.0,
        "sl": 20010.0,
        "tp_dollars": 20.0,
        "min_hold_sec": 10.0,
        "max_hold_sec": 180.0,
    }
    hit = check_quote_exit(pos, 20001.0, held_sec=180.0)
    assert hit is not None
    assert hit[0] == "time"


def test_rth_scan_window():
    # Wednesday 10:00 ET
    now = ET.localize(datetime(2026, 9, 16, 10, 0, 0))
    assert is_rth_scan_et(now) is True
    late = ET.localize(datetime(2026, 9, 16, 15, 55, 0))
    assert is_rth_scan_et(late) is False


def test_state_defaults():
    s = MicroScalpState()
    assert s.open_side == 0
    assert s.trades_today == 0
