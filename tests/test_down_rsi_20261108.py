"""Saved daily dip: three lower closes and RSI(2) < 15. Clock and signal only."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.core.down_rsi import (
    DipState,
    add_weekdays,
    completed_session,
    entry_moment,
    entry_session,
    is_signal,
    load_state,
    make_order,
    plan,
    rsi2,
    save_state,
    weekdays_inclusive,
)

ET = ZoneInfo("America/New_York")

# NQ=F closes already spike-filtered, 2026-09-14 through 2026-11-06.
# Research mask on this stretch: 2026-10-14, 10-15, 10-16, 11-02.
CLOSES = """
2026-09-14,29152.25
2026-09-15,28955.00
2026-09-16,28963.50
2026-09-17,29446.75
2026-09-18,29613.6797
2026-09-21,30784.75
2026-09-22,31028.50
2026-09-23,30764.75
2026-09-24,30766.75
2026-09-25,30889.25
2026-09-28,30566.25
2026-09-29,30613.25
2026-09-30,30698.75
2026-10-01,30760.50
2026-10-02,31061.75
2026-10-05,31317.75
2026-10-06,31483.25
2026-10-07,31402.25
2026-10-08,30969.50
2026-10-09,31108.75
2026-10-12,30898.25
2026-10-13,30791.50
2026-10-14,30182.75
2026-10-15,30143.75
2026-10-16,29391.50
2026-10-19,29722.50
2026-10-20,29675.25
2026-10-21,30007.00
2026-10-22,29142.00
2026-10-23,29117.75
2026-10-26,29437.25
2026-10-27,29812.00
2026-10-28,29979.50
2026-10-29,29562.00
2026-10-30,28534.00
2026-11-02,28127.50
2026-11-03,28305.75
2026-11-04,28566.75
2026-11-05,26752.00
2026-11-06,27085.25
"""


def _bars(last: str | None = None):
    out = []
    for line in CLOSES.strip().splitlines():
        d, c = line.split(",")
        if last and d > last:
            break
        out.append((date.fromisoformat(d), float(c)))
    return out


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=ET)


def test_recent_signal_days_match_the_research_mask():
    flags = []
    bars = _bars()
    closes = [c for _, c in bars]
    for i in range(20, len(bars)):
        if is_signal(closes[: i + 1]):
            flags.append(bars[i][0].isoformat())
    assert "2026-11-02" in flags
    assert "2026-11-05" not in flags
    assert "2026-11-06" not in flags
    assert abs(float(rsi2(closes[:36])[-1]) - 5.141) < 0.02


def test_sunday_after_a_non_signal_friday_does_not_trade():
    now = _at("2026-11-08T12:35:00")
    decided = plan(DipState(), now, _bars(), flat=True)
    assert decided.action == "none"
    assert decided.state.phase == "idle"
    assert "off" in decided.status


def test_monday_signal_arms_for_the_6pm_open_and_buys_only_when_flat():
    bars = _bars("2026-11-02")
    armed = plan(DipState(), _at("2026-11-02T17:05:00"), bars, flat=True)
    assert armed.action == "arm"
    assert armed.state.signal_day == "2026-11-02"
    assert armed.state.entry_session == "2026-11-03"
    assert armed.state.force_session == "2026-11-10"
    waiting = plan(armed.state, _at("2026-11-02T17:30:00"), bars, flat=True)
    assert waiting.action == "none"
    blocked = plan(armed.state, _at("2026-11-02T18:05:00"), bars, flat=False)
    assert blocked.action == "none"
    buy = plan(armed.state, _at("2026-11-02T18:05:00"), bars, flat=True)
    assert buy.action == "enter"
    late = plan(armed.state, _at("2026-11-02T18:20:00"), bars, flat=True)
    assert late.action == "expire"


def test_friday_signal_waits_until_sunday_6pm():
    assert entry_moment(date(2026, 11, 6)) == _at("2026-11-08T18:00:00")
    assert entry_session(date(2026, 11, 6)) == date(2026, 11, 9)
    assert add_weekdays(date(2026, 11, 9), 5) == date(2026, 11, 16)
    assert weekdays_inclusive(date(2026, 11, 3), date(2026, 11, 10)) == 6
    assert completed_session(_at("2026-11-08T12:00:00")) == date(2026, 11, 6)
    assert completed_session(_at("2026-11-02T16:00:00")) == date(2026, 10, 30)


def test_exit_on_rsi_and_on_the_sixth_session():
    state = DipState(phase="open", signal_day="2026-11-02", entry_session="2026-11-03", force_session="2026-11-10")
    # A close far above the prior close pushes RSI(2) through 65 on the entry session.
    bars = _bars("2026-11-02") + [(date(2026, 11, 3), 32000.0)]
    early = plan(state, _at("2026-11-03T16:58:00"), bars, flat=False)
    assert early.action == "exit"
    assert "RSI" in early.reason
    held = plan(state, _at("2026-11-10T16:58:00"), _bars("2026-11-06"), flat=False)
    assert held.action == "exit"
    assert "session 6" in held.reason
    # Sunday night is not a close. Do not sell a trade whose first close is Monday.
    sunday = DipState(phase="open", entry_session="2026-11-09", signal_day="2026-11-06")
    assert plan(sunday, _at("2026-11-08T19:00:00"), _bars(), flat=False).action == "none"


def test_same_session_does_not_rearm_after_an_exit():
    state = DipState(last_exit_day="2026-11-02")
    decided = plan(state, _at("2026-11-02T17:10:00"), _bars("2026-11-02"), flat=True)
    assert decided.action == "none"
    assert "already exited" in decided.status


def test_state_roundtrip_and_bracket(tmp_path):
    path = tmp_path / "down_rsi_state.json"
    st = DipState(phase="pending", signal_day="2026-11-02", entry_session="2026-11-03", force_session="2026-11-10")
    save_state(st, path)
    assert load_state(path).signal_day == "2026-11-02"
    order = make_order("MNQ", 28127.5, _at("2026-11-10T16:59:00"))
    assert order["fixed_contracts"] == 1
    assert order["addon"] == "dip"
    assert order["ignore_risk_cap"] is True
    assert order["trailing_enabled"] is False
    assert order["entry"] - order["sl"] == 2500
    assert order["tp"] - order["entry"] == 3000
