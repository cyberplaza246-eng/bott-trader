"""REVAMP per-bar readout (display only): formatter + no-side-effect checks."""
import numpy as np
import pandas as pd
import pytest

from src.research_ensemble import live_display as ld
from src.research_ensemble.live_signals import classify_last_sleeve


def _snap(**over):
    s = {
        "time": pd.Timestamp("2026-10-06 14:00", tz="UTC"), "time_et": "10:00 ET", "in_rth": True,
        "close": 31548.5, "ema_fast": 31540.0, "ema_slow": 31520.0, "ema_trend": 31210.25,
        "fast_n": 20, "slow_n": 50, "trend_n": 200, "ema_bars_since_cross": 14,
        "ema_cross_up": False, "ema_cross_down": False,
        "adx": 22.4, "adx_trend_min": 25.0, "adx_range_max": 20.0,
        "trending": False, "ranging": False, "breakout_allowed": True,
        "don_hi": 31560.5, "don_lo": 31480.0, "prev_beyond_hi": False, "prev_beyond_lo": False,
        "next_hi": 31560.5, "next_lo": 31480.0,
        "don_n": 20, "vol_ratio": 1.12, "vol_mult": 1.5,
        "bb_up": 31570.0, "bb_mid": 31525.0, "bb_lo": 31480.0, "rsi": 58.3,
        "rsi_os": 30.0, "rsi_ob": 70.0,
        "trend_entry": 0, "mr_entry": 0, "bo_entry": 0,
        "routed_entry": 0, "routed_sleeve": "", "final_entry": 0, "rth_only": True,
    }
    s.update(over)
    return s


def _gates(**over):
    g = dict(daily_pnl=0.0, daily_loss_limit=1500.0, daily_trades=0, max_trades=20, open_count=0,
             max_positions=2, cooldown=0, loss_cooldown_min=0, consec_pause_min=0, consec_losses=0,
             session_block=None)
    g.update(over)
    return g


def test_block_shape_and_ascii():
    lines = ld.format_revamp_block(_snap(), _gates())
    assert len(lines) == 6
    heads = ["Bias", "Regime", "Trend", "Breakout", "MeanRev", "Decision"]
    for ln, h in zip(lines, heads):
        assert ln.startswith("   " + h)
        ln.encode("ascii")  # safe for any Windows console code page


def test_bias_uses_buy_sell_neutral_wording():
    assert "Bias    : BUY (price 31548.50 above EMA200" in ld.format_revamp_block(_snap())[0]
    sell = ld.format_revamp_block(_snap(close=31000.0, ema_fast=31010.0, ema_slow=31050.0))[0]
    assert sell.strip().startswith("Bias    : SELL (price 31000.00 below EMA200")
    assert "EMA20 < EMA50" in sell
    neu = ld.format_revamp_block(_snap(close=31000.0))[0]
    assert "Bias    : NEUTRAL" in neu
    for ln in (ld.format_revamp_block(_snap())[0], sell, neu):
        assert "bullish" not in ln.lower() and "bearish" not in ln.lower()


def test_transition_reason_and_breakout_distance():
    lines = ld.format_revamp_block(_snap(), _gates())
    assert "TRANSITION (20-25)" in lines[1]
    assert lines[-1] == ("   Decision: NO TRADE: ADX 22.4 in transition (trend/MR off); "
                         "no breakout: high 31560.50 is 12.00 pts away, vol 1.12x <= 1.50x")


def test_outside_rth_reason():
    d = ld.format_revamp_block(_snap(in_rth=False, time_et="18:05 ET"), _gates())[-1]
    assert "NO TRADE: outside RTH (18:05 ET" in d


@pytest.mark.parametrize("over,needle", [
    ({"cooldown": 3}, "cooldown (3 bars left)"),
    ({"open_count": 2}, "max concurrent reached (2/2 open)"),
    ({"daily_pnl": -1600.0}, "daily loss limit hit"),
    ({"loss_cooldown_min": 7}, "loss cooldown (7 min left)"),
    ({"session_block": "bad hour filter (21:00 UTC)"}, "bad hour filter"),
])
def test_gate_reasons(over, needle):
    d = ld.format_revamp_block(_snap(), _gates(**over))[-1]
    assert d.startswith("   Decision: NO TRADE: ") and needle in d


def test_setup_blocked_by_gate_and_signal_decision():
    s = _snap(adx=30.0, trending=True, bo_entry=1, routed_entry=1, routed_sleeve="breakout", final_entry=1,
              close=31570.0)
    d = ld.format_revamp_block(s, _gates(cooldown=2), signal=None, checked=False)[-1]
    assert d == "   Decision: NO TRADE: BUY setup (breakout) blocked by cooldown (2 bars left)"
    sig = {"ensemble_signal": "SELL", "direction": "short", "entry": 31570.0, "sl": 31650.0, "tp": 0.0,
           "exit_sleeve": "trend"}
    d2 = ld.format_revamp_block(s, _gates(), signal=sig)[-1]
    assert d2 == "   Decision: SELL (trend) @ 31570.00 | SL 31650.00 | TP ATR trail"


def test_verbose_levels(monkeypatch):
    monkeypatch.delenv("REVAMP_VERBOSE", raising=False)
    assert ld.revamp_verbose_level() == 1 and ld.revamp_verbose_enabled() is True  # default = compact
    for v, lvl in (("0", 0), ("false", 0), ("off", 0), ("1", 1), ("on", 1), ("true", 1),
                   ("2", 2), ("detailed", 2), ("3", 2)):
        monkeypatch.setenv("REVAMP_VERBOSE", v)
        assert ld.revamp_verbose_level() == lvl, v
        assert ld.revamp_verbose_enabled() is (lvl > 0)


# ----------------------------------------------------------------- compact

def _c(snap=None, gates=None, signal=None):
    lines = ld.format_revamp_compact(snap or _snap(), gates if gates is not None else _gates(), signal)
    assert 3 <= len(lines) <= 6
    for ln in lines:
        assert ln.startswith("   ") and len(ln) <= 90, ln
        ln.encode("ascii")
    assert "ADX" not in lines[0] and "EMA" not in lines[0]  # line 1 is plain words
    return [ln.strip() for ln in lines]


TREND_BUY = dict(adx=30.0, trending=True, close=31584.0, ema_fast=31570.0, ema_slow=31550.0,
                 next_hi=31616.5, don_hi=31616.5, next_lo=31480.0, don_lo=31480.0, vol_ratio=0.6)


def test_compact_trending_no_trade():
    assert _c(_snap(**TREND_BUY)) == [
        "Bias: BUY | Trending | No trade",
        "Price: 31584 | needs above 31617 to BUY (33 pts) | below 31480 to SELL (104 pts)",
        "Volume: 0.6x | needs 1.5x",
        # a SELL cross would also need price under the 200 EMA (price is above it now)
        "EMA cross: 20 above 50 (20 pts) | needs cross down + price under 200 EMA to SELL",
        "RSI: 58 (neutral) | needs under 30 to BUY dip, over 70 to SELL pop [off now: trending]",
    ]


def test_compact_bias_side_first_and_volume_ok():
    s = _snap(**dict(TREND_BUY, close=31500.0, ema_trend=31600.0, ema_fast=31520.0, ema_slow=31540.0,
                     vol_ratio=2.04))
    assert _c(s)[:4] == [
        "Bias: SELL | Trending | No trade",
        "Price: 31500 | needs below 31480 to SELL (20 pts) | above 31617 to BUY (117 pts)",
        "Volume: 2.0x | needs 1.5x (OK)",
        "EMA cross: 20 below 50 (20 pts) | needs cross up + price over 200 EMA to BUY",
    ]
    assert _c(_snap(**dict(TREND_BUY, vol_ratio=1.47)))[2] == "Volume: 1.47x | needs 1.5x"


def test_compact_fresh_cross_line_when_filter_passes():
    s = _snap(**dict(TREND_BUY, close=31200.0, ema_trend=31300.0, ema_fast=31204.0, ema_slow=31200.0,
                     next_hi=31260.0, don_hi=31260.0, next_lo=31150.0, don_lo=31150.0))
    l = _c(s)
    assert l[0].startswith("Bias: NEUTRAL | Trending")
    assert l[3] == "EMA cross: 20 above 50 by 4 pts (no new cross) | needs a fresh cross down to SELL"


def test_compact_already_broken_out():
    s = _snap(**dict(TREND_BUY, close=31630.0, don_hi=31616.5, next_hi=31640.0))
    # broke out on this bar -> a new BUY break can't fire next bar (needs a pullback first)
    assert _c(s)[1] == "Price: 31630 | already above 31617 | needs below 31480 to SELL (150 pts)"
    s2 = _snap(**dict(TREND_BUY, close=31600.0, don_hi=31616.5, next_hi=31616.5, don_lo=31605.0, next_lo=31590.0))
    assert _c(s2)[1] == "Price: 31600 | needs above 31617 to BUY (17 pts) | already below 31605"


def test_compact_in_between_shows_breakout_only():
    l = _c(_snap())
    assert l[0] == "Bias: BUY | In between | No trade"
    assert l[1].startswith("Price: 31549 | needs above 31561 to BUY (12 pts) | below 31480 to SELL")
    assert l[2] == "Volume: 1.1x | needs 1.5x"
    assert not any(x.startswith("EMA cross") for x in l)
    assert l[-1].endswith("[off now: in between]") and l[-1].startswith("RSI: 58")


def test_compact_choppy_shows_mean_reversion_only():
    s = _snap(adx=15.0, ranging=True, breakout_allowed=False, bb_lo=31480.0, bb_up=31634.0)
    l = _c(s)
    assert l == ["Bias: BUY | Choppy | No trade",
                 "Price: 31549 | needs dip to 31480 to BUY (69 pts) | pop to 31634 to SELL (86 pts)",
                 "RSI: 58 (neutral) | needs under 30 to BUY dip, over 70 to SELL pop"]


def test_compact_fired_trade():
    sig = {"ensemble_signal": "BUY", "direction": "long", "entry": 31584.5, "sl": 31480.0, "tp": 31790.0,
           "exit_sleeve": "breakout"}
    l = _c(_snap(**TREND_BUY), signal=sig)
    assert l[0] == "Bias: BUY | Trending | BUY (breakout)"
    assert l[1] == "Entry 31584.50 | Stop 31480.00 | Target 31790.00"
    sig2 = dict(sig, ensemble_signal="SELL", direction="short", tp=0.0, exit_sleeve="trend")
    l = _c(_snap(**TREND_BUY), signal=sig2)
    assert l[0].endswith("| SELL (trend)") and l[1].endswith("Target trailing stop (no fixed target)")


@pytest.mark.parametrize("gates,result", [
    ({"open_count": 2}, "Blocked: 2 trades already open"),
    ({"cooldown": 3}, "Paused: cooldown 3 bars"),
    ({"daily_pnl": -1600.0}, "Stopped: daily loss limit"),
])
def test_compact_gates(gates, result):
    l = _c(_snap(**TREND_BUY), _gates(**gates))
    assert l[0] == "Bias: BUY | Trending | " + result
    assert l[1].startswith("Price: ")  # conditions still shown


def test_compact_blocked_setup_and_closed():
    s = _snap(**dict(TREND_BUY, close=31380.0, ema_fast=31390.0, ema_slow=31420.0, ema_trend=31500.0,
                     bo_entry=-1, routed_entry=-1, routed_sleeve="breakout", final_entry=-1))
    l = _c(s, _gates(open_count=2))
    assert l[0] == "Bias: SELL | Trending | Blocked: 2 trades already open"
    assert l[1] == "SELL breakout skipped because 2 trades are already open"
    l = _c(_snap(in_rth=False, time_et="18:05 ET"))
    assert l[0] == "Bias: BUY | In between | Closed: outside trading hours (18:05 ET)"


@pytest.mark.parametrize("rsi,txt", [(25.0, "RSI: 25 (oversold)"), (29.6, "RSI: 29.6 (oversold)"),
                                     (30.0, "RSI: 30.0 (neutral)"), (57.4, "RSI: 57 (neutral)"),
                                     (70.0, "RSI: 70.0 (neutral)"), (78.2, "RSI: 78 (overbought)")])
def test_compact_rsi_line(rsi, txt):
    line = _c(_snap(rsi=rsi, adx=15.0, ranging=True, breakout_allowed=False))[-1]
    assert line.startswith(txt + " | needs under 30 to BUY dip, over 70 to SELL pop")
    off = _c(_snap(rsi=rsi, **TREND_BUY))[-1]  # MR sleeve inactive -> tagged, label kept
    assert off.startswith(txt + " | needs under 30 to BUY") and off.endswith("[off now: trending]")


def test_render_levels():
    df = _synthetic_5m(n=300)
    assert 3 <= len(ld.render_revamp_block(df, gates=_gates(), level=1)) <= 6
    assert len(ld.render_revamp_block(df, gates=_gates(), level=2)) == 6


def _synthetic_5m(n=420, seed=7):
    rng = np.random.default_rng(seed)
    # RTH-only 5m bars (09:30-16:00 ET = 13:30-20:00 UTC in Sept/Oct) over consecutive weekdays
    days = pd.bdate_range("2026-09-01", periods=n // 78 + 2)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=13, minutes=30) + pd.Timedelta(minutes=5 * k)
                            for d in days for k in range(78)]).tz_localize("UTC")[:n]
    steps = rng.normal(0, 6, n) + np.sin(np.arange(n) / 25.0) * 4
    close = 31000 + np.cumsum(steps)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 5, n)
    vol = rng.integers(200, 900, n).astype(float)
    vol[rng.integers(0, n, 25)] *= 4  # some volume spikes so breakouts can confirm
    return pd.DataFrame({"datetime": idx, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_display_has_no_effect_on_strategy(monkeypatch):
    monkeypatch.setenv("REVAMP_EXITS", "1")
    full = _synthetic_5m()
    seen = 0
    hits = 0
    for end in range(240, len(full)):
        w = full.iloc[max(0, end - 320):end].reset_index(drop=True)
        before = w.copy(deep=True)
        off = classify_last_sleeve(w)
        snap = ld.compute_revamp_snapshot(w)
        lines = ld.format_revamp_block(snap, _gates())
        compact = ld.format_revamp_compact(snap, _gates())
        on = classify_last_sleeve(w)
        pd.testing.assert_frame_equal(w, before)
        assert off == on
        assert (snap["final_entry"] != 0) == bool(on)
        assert len(lines) == 6 and 3 <= len(compact) <= 6
        if on:
            hits += 1
            assert lines[-1].startswith("   Decision: NO TRADE: " + on["ensemble_signal"] + " setup")
        seen += 1
    assert seen > 150
    assert hits > 0  # the fixture really exercises signal bars
