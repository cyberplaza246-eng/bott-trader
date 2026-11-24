import numpy as np

from scripts.backtest_1m_entries import try_fill
from src.strategy.mnq_15m_ema_eod import quiet_1m_ok


def _bars():
    # 4 one-minute prints. Long signal at i=0 close 100.
    op = np.array([99.0, 100.2, 99.4, 99.8])
    cl = np.array([100.0, 99.6, 99.2, 100.4])
    hi = np.maximum(op, cl) + 0.2
    lo = np.array([98.8, 99.3, 98.9, 99.6])
    ema = np.array([99.5, 99.5, 99.4, 99.5])
    mins = np.array([9 * 60 + 35, 9 * 60 + 36, 9 * 60 + 37, 9 * 60 + 38])
    return hi, lo, op, cl, ema, mins


def test_now_uses_signal_close():
    hi, lo, op, cl, ema, mins = _bars()
    hit = try_fill("now", 1, 0, hi, lo, op, cl, ema, mins, book="day", level=None, max_i=3)
    assert hit == (0, 100.0)


def test_pullback_then_go_waits():
    hi, lo, op, cl, ema, mins = _bars()
    hit = try_fill("pullback_go", 1, 0, hi, lo, op, cl, ema, mins, book="day", level=None, max_i=3)
    assert hit is not None
    assert hit[0] == 3
    assert hit[1] == 100.4


def test_live_quiet_uses_real_helper():
    hi, lo, op, cl, ema, mins = _bars()
    assert quiet_1m_ok(1, 99.0, 100.0) is True
    hit = try_fill("live_quiet", 1, 0, hi, lo, op, cl, ema, mins, book="day", level=None, max_i=3)
    assert hit == (0, 100.0)
    op2 = op.copy()
    cl2 = cl.copy()
    op2[0], cl2[0] = 90.0, 100.0
    hit2 = try_fill("live_quiet", 1, 0, hi, lo, op2, cl2, ema, mins, book="day", level=None, max_i=3)
    assert hit2 is not None
    assert hit2[0] > 0
