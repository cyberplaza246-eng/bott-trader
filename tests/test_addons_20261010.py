"""2026-10-10: backtested add-on setups live in both bots (ONL reclaim LONG + E1 6 PM momentum LONG).

Fixture = real MNQ 5m bars (Aug 28 - Sep 25 2026) from the box backtest file bars5_feat.parquet, with the
backtest's own signal bars, ATR, SL/TP and label-A P&L, so these tests prove the live code fires on exactly
the backtested bars with the backtested stop / target / time limit.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core import addon_setups as A

FIX = Path(__file__).parent / "fixtures"
BARS = pd.read_csv(FIX / "addon_bt_bars_20261010.csv")
BARS["datetime"] = pd.to_datetime(BARS["datetime"], utc=True)          # bar START (UTC)
EXP = json.loads((FIX / "addon_bt_expected_20261010.json").read_text())
START_ET = pd.DatetimeIndex(BARS["datetime"]).tz_convert(A.ET)


def _end_frame(upto_et: str, n: int = 400) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Live-style Rithmic frame: last n bars up to and including the bar starting at upto_et,
    stamped by bar END with attrs bar_time='end'. Returns (frame, now just after that bar closed)."""
    t = pd.Timestamp(upto_et)
    i = int(np.flatnonzero(START_ET == t)[0])
    d = BARS.iloc[max(0, i + 1 - n): i + 1].copy()
    d["datetime"] = d["datetime"] + pd.Timedelta(minutes=5)
    d.attrs["bar_time"] = "end"
    now = d["datetime"].iloc[-1] + pd.Timedelta(seconds=30)
    return d.reset_index(drop=True), now


def _feat(upto_et: str, n: int = 400):
    d, now = _end_frame(upto_et, n)
    return A.add_features(A.closed_bars(d, now=now)), now


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("ADDON_ONL_RECLAIM", raising=False)
    monkeypatch.delenv("ADDON_E1_6PM", raising=False)


# ---------------- rules match the backtest ----------------

def test_frozen_constants_match_backtest():
    r = EXP["rules"]
    assert (A.STOP_ATR, A.STOP_MIN_PTS, A.TARGET_R, A.MAX_BARS) == (r["stop_atr"], r["stop_min_pts"], r["target_r"], r["max_bars"])
    assert (A.E1_RSI_MIN, A.E1_MOVE_ATR_MIN) == (r["e1_rsi_min"], r["e1_move_atr_min"])
    assert (A.ONL_FIRST_MIN, A.ONL_LAST_MIN, A.E1_FIRST_MIN, A.E1_LAST_MIN) == (570, 685, 1080, 1135)
    assert A.enabled("onl") and A.enabled("e1")                        # default ON


@pytest.mark.parametrize("case", EXP["onl_first"], ids=lambda c: c["bar_start_et"])
def test_onl_fires_on_backtest_bar_with_backtest_bracket(case):
    b, now = _feat(case["bar_start_et"])
    ev = A.evaluate(b, "onl", now=now)
    assert ev.signal, ev.status
    assert ev.atr == pytest.approx(case["atr"], abs=1e-3)
    assert float(b.onl.iloc[-1]) == pytest.approx(case["onl"])
    sig = A.make_signal(ev, "MNQ")
    assert (sig["sl"], sig["tp"]) == (case["sl"], case["tp"])
    assert sig["fixed_contracts"] == 1 and sig["direction"] == "long" and sig["addon"] == "onl"
    # time limit = 24 bars after the signal bar = signal bar end + 2h
    bar_end = pd.Timestamp(case["bar_start_et"]).tz_convert("UTC") + pd.Timedelta(minutes=5)
    assert pd.Timestamp(sig["addon_exit_at"]) == bar_end + pd.Timedelta(hours=2)


@pytest.mark.parametrize("case", EXP["onl_later"], ids=lambda c: c["bar_start_et"])
def test_onl_only_first_of_day(case):
    b, now = _feat(case["bar_start_et"])
    assert bool(b.onl_hit.iloc[-1])                     # the bar itself qualifies...
    ev = A.evaluate(b, "onl", now=now)
    assert not ev.signal and "done for today" in ev.status   # ...but it isn't the day's first


@pytest.mark.parametrize("case", EXP["e1"][::4], ids=lambda c: c["bar_start_et"])
def test_e1_fires_on_backtest_bar_with_backtest_bracket(case):
    b, now = _feat(case["bar_start_et"])
    ev = A.evaluate(b, "e1", now=now)
    assert ev.signal, ev.status
    assert float(b.rsi.iloc[-1]) == pytest.approx(case["rsi"], abs=1e-3)
    assert float(b.ret12.iloc[-1]) == pytest.approx(case["ret12"], abs=1e-3)
    sig = A.make_signal(ev, "MNQ")
    assert (sig["sl"], sig["tp"]) == (case["sl"], case["tp"])


def test_full_scan_signal_bars_equal_backtest():
    """Walk every bar in both windows (as the live loop would) and collect where each setup fires."""
    want_onl = {c["bar_start_et"] for c in EXP["onl_first"]}
    want_e1 = {c["bar_start_et"] for c in EXP["e1"]}
    first_ok = START_ET[0] + pd.Timedelta(days=3)
    got_onl, got_e1 = set(), set()
    for t in START_ET:
        if t < first_ok:
            continue
        m = t.hour * 60 + t.minute
        kinds = (["onl"] if 570 <= m <= 700 else []) + (["e1"] if 1080 <= m <= 1145 else [])
        if not kinds:
            continue
        b, now = _feat(t.isoformat())
        for k in kinds:
            if A.evaluate(b, k, now=now).signal:
                (got_onl if k == "onl" else got_e1).add(t.isoformat())
    assert got_onl == want_onl
    assert got_e1 == want_e1


def _label_a_pnl(i_sig: int, atr: float) -> tuple[float, int]:
    """Re-run the backtest trade model (build.py label A) from the fixture bars with the live constants."""
    o, h, l, c = (BARS[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    br = A.bracket(0.0, atr)                       # distance only
    sd = br["stop_dist"]
    e = o[i_sig + 1] + 0.25
    stp, tgt = e - sd, e + A.TARGET_R * sd
    k = i_sig + 1
    for j in range(i_sig + 1, min(len(o), i_sig + 1 + A.MAX_BARS)):
        k = j
        if l[j] <= stp:
            return (min(stp, o[j]) - 0.25 - e) * 2.0 - 3.0, j - i_sig
        if h[j] >= tgt:
            return (tgt - 0.25 - e) * 2.0 - 3.0, j - i_sig
    return (c[k] - 0.25 - e) * 2.0 - 3.0, k - i_sig


@pytest.mark.parametrize("case", EXP["onl_first"] + EXP["e1"][::4], ids=lambda c: c["bar_start_et"])
def test_stop_target_time_reproduce_backtest_pnl(case):
    i = int(np.flatnonzero(START_ET == pd.Timestamp(case["bar_start_et"]))[0])
    pnl, held = _label_a_pnl(i, case["atr"])
    assert pnl == pytest.approx(case["pnl_label_A"], abs=1e-3)   # fixture ATR rounded to 1e-6
    assert held == case["bars_held"]


def test_onl_needs_bars_back_to_6pm():
    c = EXP["onl_first"][0]
    b, now = _feat(c["bar_start_et"], n=150)          # 150 bars only reaches back to ~ midnight
    ev = A.evaluate(b, "onl", now=now)
    assert not ev.signal and "overnight low unknown" in ev.status


def test_stale_or_forming_bar_never_signals():
    c = EXP["onl_first"][0]
    d, now = _end_frame(c["bar_start_et"])
    b = A.add_features(A.closed_bars(d, now=now - pd.Timedelta(minutes=1)))   # newest bar not finished yet
    assert b.index[-1] < pd.Timestamp(c["bar_start_et"])
    b2 = A.add_features(A.closed_bars(d, now=now))
    assert not A.evaluate(b2, "onl", now=now + pd.Timedelta(minutes=10)).signal  # 10 min late = stale


def test_start_stamped_frame_drops_forming_bar():
    c = EXP["e1"][0]
    d, now = _end_frame(c["bar_start_et"])
    s = d.copy(); s["datetime"] = s["datetime"] - pd.Timedelta(minutes=5); s.attrs = {}
    b = A.closed_bars(s, now=now)
    assert b.index[-1] == pd.Timestamp(c["bar_start_et"])
    b_mid = A.closed_bars(s, now=now - pd.Timedelta(minutes=2))      # newest bar still forming
    assert b_mid.index[-1] < pd.Timestamp(c["bar_start_et"])


def test_toggles_off(monkeypatch):
    monkeypatch.setenv("ADDON_ONL_RECLAIM", "0")
    monkeypatch.setenv("ADDON_E1_6PM", "0")
    c = EXP["onl_first"][0]
    b, now = _feat(c["bar_start_et"])
    ev = A.evaluate(b, "onl", now=now)
    assert not ev.signal and "off" in ev.status
    c = EXP["e1"][0]
    b, now = _feat(c["bar_start_et"])
    assert not A.evaluate(b, "e1", now=now).signal


def test_windows():
    et = lambda s: pd.Timestamp(s, tz=A.ET)
    assert A.in_display_window(et("2026-10-12 10:00")) == ["onl"]       # Mon
    assert A.in_display_window(et("2026-10-12 18:30")) == ["e1"]
    assert A.in_display_window(et("2026-10-11 18:30")) == ["e1"]        # Sunday open
    assert A.in_display_window(et("2026-10-09 18:30")) == []            # Friday: no 6 PM open
    assert A.in_display_window(et("2026-10-12 13:00")) == []


def test_status_line_wording():
    c = EXP["onl_first"][0]
    i = int(np.flatnonzero(START_ET == pd.Timestamp(c["bar_start_et"]))[0])
    b, now = _feat(START_ET[i - 1].isoformat())
    ev = A.evaluate(b, "onl", now=now)
    assert ev.status.startswith("ONL reclaim: overnight low ") and "9:30-11:30" in ev.status
    # inside the window, before the setup: "needs dip under X then close back above"
    c2 = next(x for x in EXP["onl_first"] if not x["bar_start_et"].endswith("09:30:00-04:00"))
    j = int(np.flatnonzero(START_ET == pd.Timestamp(c2["bar_start_et"]))[0])
    b, now = _feat(START_ET[j - 1].isoformat())
    st = A.evaluate(b, "onl", now=now).status
    assert "price" in st and ("needs dip under" in st or "needs a 5m close back above" in st) and st.endswith("(9:30-11:30)")
    c = EXP["e1"][0]
    i = int(np.flatnonzero(START_ET == pd.Timestamp(c["bar_start_et"]))[0])
    b, now = _feat(c["bar_start_et"])
    st = A.evaluate(b, "e1", now=now).status
    assert st.startswith("6PM long: up ") and "needs 1.48; RSI" in st and "needs 59.5" in st


# ---------------- live trader wiring (both bots) ----------------

import start_live_rithmic as slr  # noqa: E402


@pytest.fixture(params=[True, False], ids=["revamp", "ensemble"])
def trader(request, monkeypatch):
    monkeypatch.setattr(slr._reentry, "REENTRY_ENABLED", False)
    monkeypatch.setattr(slr, "trades_logger", MagicMock())
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    monkeypatch.setattr(slr, "REVAMP_EXITS_ON", request.param)
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.broker.place_order.return_value = {"ticket": "bot_addon", "entry_price": 30000.0}
    t._broker_can_update_stop = lambda r: True
    t._write_board_status = lambda **k: None
    t.max_risk_per_trade_usd = 500.0
    # Ensemble bat sizing (conf bands 72/78/85, max 4) - add-ons must still be 1 lot
    t._size_cfg = dict(t._size_cfg, base=1, max_contracts=4, conf_2=0.40, conf_3=0.42, conf_4=0.44,
                       win_streak_min=0, require_day_green=False)
    t._session_ledger = None
    t._effective_open_count = lambda: len(t.positions)
    t.get_price = lambda symbol=None: {"bid": 30010.0, "last": 30010.0}
    return t


def _wire(t, case):
    b, now = _feat(case["bar_start_et"])
    t._addon_frame = lambda symbol: b
    return now.to_pydatetime()


def test_trader_fires_when_flat_and_places_one_lot(trader):
    c = EXP["onl_first"][0]
    now = _wire(trader, c)
    sig, lines = trader.check_addon_signals("MNQ", now=now)
    assert sig and sig["addon"] == "onl" and any("-> LONG 1 MNQ" in x for x in lines)
    assert trader.place_order(sig) is True
    kw = trader.broker.place_order.call_args.kwargs
    assert kw["size"] == 1 and kw["stop_loss"] == c["sl"] and kw["take_profit"] == c["tp"] and kw["order_type"] == "buy"
    pos = trader.positions["bot_addon"]
    assert pos.addon == "onl" and pos.trailing_enabled is False and pos.addon_exit_at is not None
    # same bar again (e.g. loop repeat) -> no second order
    sig2, _ = trader.check_addon_signals("MNQ", now=now)
    assert sig2 is None


def test_trader_skips_when_not_flat_and_day_is_used(trader):
    c = EXP["onl_first"][0]
    now = _wire(trader, c)
    trader.positions["x"] = slr.Position("x", "MNQ", "short", 30000.0, 1, 30050.0, 29900.0, datetime.now(timezone.utc))
    sig, lines = trader.check_addon_signals("MNQ", now=now)
    assert sig is None and any("SKIPPED: bot already has an open trade" in x for x in lines)
    # next bar, flat again: the day's ONL chance is gone (first-of-day only)
    trader.positions.clear()
    i = int(np.flatnonzero(START_ET == pd.Timestamp(c["bar_start_et"]))[0])
    b, now2 = _feat(START_ET[i + 1].isoformat())
    trader._addon_frame = lambda symbol: b
    sig, lines = trader.check_addon_signals("MNQ", now=now2.to_pydatetime())
    assert sig is None and any("done for today" in x for x in lines)


def test_trader_outside_window_does_nothing(trader):
    trader._addon_frame = MagicMock()
    sig, lines = trader.check_addon_signals("MNQ", now=datetime(2026, 10, 12, 17, 0, tzinfo=timezone.utc))  # 1 PM ET
    assert sig is None and lines == [] and not trader._addon_frame.called


def test_trader_session_limit_blocks(trader):
    c = EXP["e1"][0]
    now = _wire(trader, c)
    trader._session_limit_block = lambda: "overnight loss limit -$300 reached"
    sig, lines = trader.check_addon_signals("MNQ", now=now)
    assert sig is None and any("SKIPPED: overnight loss limit" in x for x in lines)


def test_bad_hour_filter_does_not_block_e1(trader, monkeypatch):
    assert 22 not in [0, 4, 21] and 23 not in [0, 4, 21]               # 6 PM ET = 22 UTC (EDT) / 23 UTC (EST)
    trader.bad_hours = {"MNQ": list(range(24))}                          # even if every hour were "bad"
    monkeypatch.setitem(slr.SCALPING_PAIRS, "MNQ", dict(slr.SCALPING_PAIRS.get("MNQ", {}), bad_hours_utc=list(range(24))))
    c = EXP["e1"][0]
    now = _wire(trader, c)
    sig, _ = trader.check_addon_signals("MNQ", now=now)
    assert sig and sig["addon"] == "e1"


def test_main_signal_takes_priority(trader):
    c = EXP["e1"][0]
    now = _wire(trader, c)
    sig, lines = trader.check_addon_signals("MNQ", main_signal={"direction": "long"}, now=now)
    assert sig is None and any("main bot signal" in x for x in lines)


def test_risk_cap_skips_addon(trader):
    c = EXP["e1"][0]
    now = _wire(trader, c)
    sig, _ = trader.check_addon_signals("MNQ", now=now)
    sig = dict(sig, sl=sig["entry"] - 300.0)                             # $600 risk on 1 MNQ
    assert trader.place_order(sig) is False
    trader.broker.place_order.assert_not_called()


def _addon_pos(exit_at):
    return slr.Position("a1", "MNQ", "long", 30000.0, 1, 29950.0, 30100.0, datetime.now(timezone.utc),
                        trailing_enabled=False, addon="e1", addon_exit_at=exit_at)


def test_time_exit_flattens_when_alone(trader):
    trader.positions["a1"] = _addon_pos(datetime.now(timezone.utc) - timedelta(seconds=1))
    trader._broker_net_qty = lambda s: 1
    trader.broker.flatten_symbol.return_value = {"flat": True}
    trader._cancel_orders_for_closed_position = lambda *a, **k: 0
    assert trader.check_addon_time_exits("MNQ") == 1
    trader.broker.flatten_symbol.assert_called_once_with("MNQ")
    assert "a1" not in trader.positions and trader.trades[-1]["exit_type"] == "TIME_LIMIT"


def test_time_exit_waits_before_limit_and_never_flattens_other_trades(trader):
    trader.positions["a1"] = _addon_pos(datetime.now(timezone.utc) + timedelta(minutes=5))
    assert trader.check_addon_time_exits("MNQ") == 0
    trader.positions["a1"].addon_exit_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    trader.positions["b"] = slr.Position("b", "MNQ", "long", 30000.0, 1, 29950.0, 30100.0, datetime.now(timezone.utc))
    trader._broker_net_qty = lambda s: 2
    assert trader.check_addon_time_exits("MNQ") == 0
    trader.broker.flatten_symbol.assert_not_called()
    assert "a1" in trader.positions


def test_addon_stop_never_moved_by_locks_or_trail(trader):
    p = _addon_pos(datetime.now(timezone.utc) + timedelta(hours=1))
    p.entry_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    trader.positions["a1"] = p
    trader._apply_stop_update = MagicMock(return_value=True)
    trader._cancel_position_take_profit = MagicMock(return_value=True)
    trader.get_price = lambda symbol=None: {"last": 30200.0, "bid": 30200.0}   # +$400 open, past TP
    df = pd.DataFrame([{"datetime": pd.Timestamp.now(tz="UTC"), "open": 30150.0, "high": 30210.0,
                        "low": 30140.0, "close": 30200.0, "atr": 20.0}])
    trader.update_past_tp_to_trail("MNQ", df)
    trader.update_dollar_profit_locks("MNQ", df)
    trader.update_trailing_stops("MNQ", df)
    trader._apply_stop_update.assert_not_called()
    trader._cancel_position_take_profit.assert_not_called()
    assert p.sl == 29950.0 and p.tp == 30100.0


def test_addon_never_arms_reentry(trader):
    p = _addon_pos(None)
    assert trader._maybe_arm_reentry(p, "STOP_LOSS", 50.0, 30025.0) is False


def test_entry_line_format(trader):
    c = EXP["onl_first"][0]
    now = _wire(trader, c)
    sig, _ = trader.check_addon_signals("MNQ", now=now)
    line = A.entry_line(sig, 1, 30000.0)
    assert line.startswith("LONG 1 MNQ at 30,000.00. SL ") and "(ONL reclaim, conf 45%" in line
