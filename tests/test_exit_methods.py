from scripts.backtest_exit_methods import manage_bar, should_flatten


def test_overnight_flatten_is_morning_only():
    assert should_flatten(9 * 60 + 25, "overnight") is True
    assert should_flatten(18 * 60 + 5, "overnight") is False
    assert should_flatten(15 * 60 + 50, "day") is True
    assert should_flatten(10 * 60, "day") is False


def test_trail_after_1r_long_moves_stop():
    p = {
        "dir": 1,
        "entry": 100.0,
        "sl_pts": 10.0,
        "sl": 90.0,
        "tp": 600.0,
        "armed": False,
        "extreme": None,
    }
    assert manage_bar(p, "trail_after_1r", 109.0, 99.0, 108.0, False) is None
    hit = manage_bar(p, "trail_after_1r", 116.0, 112.0, 115.0, False)
    assert hit is None
    assert p["armed"] is True
    assert p["sl"] == 111.0  # extreme 116 - 5
    sl_hit = manage_bar(p, "trail_after_1r", 114.0, 110.5, 111.0, False)
    assert sl_hit == ("SL", 111.0)
