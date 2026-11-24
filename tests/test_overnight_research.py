"""Overnight paper research: zones, journal tagging, cues. Never live RTH."""
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import pytz

from src.ai.overnight_journal import (
    OvernightPaperJournal,
    build_overnight_suggestions,
    is_overnight_research_row,
    write_zones,
    read_zones,
)
from src.ai.trade_review import load_closes
from src.dashboard.mtf_actions import (
    bot_run_state,
    build_status_card,
    is_bot_ema15_close,
    is_overnight_research_close,
    todays_live_trade_count,
)
from src.strategy.overnight_research import (
    CUE_BREAK_ONH,
    CUE_FADE_ONH,
    MAX_FILLS_PER_NIGHT,
    PAPER_RECIPE,
    SESSION_NAME,
    TP_R,
    ZONE_ONH,
    allow_new_overnight_entry,
    check_paper_exit,
    compute_zones,
    day_bot_owns_lucid,
    enable_overnight_lucid_sim_orders,
    evaluate_overnight_cue,
    globex_session_start_et,
    in_rth_hard_idle,
    is_overnight_research_hours,
    overnight_cap_reached,
    overnight_entry_blocked_reason,
    overnight_tp_r,
    overnight_tp_text,
    paper_test_fill_from_env,
    paper_test_fill_signal,
    recipe_session_stats,
    resolve_overnight_research,
    should_flatten_before_rth,
    skip_local_fake_fill,
)
from src.data.paper_csv_feed import paper_rithmic_brackets_enabled, paper_uses_rithmic
from src.strategy.overnight_paper_runner import BLOCKED_ORDER_METHODS, QuotesOnlyBroker
from src.broker.rithmic_connector import (
    apply_lucid_ticker_order_plants,
    history_pnl_skipped,
    is_forced_logout_text,
    paper_rithmic_brackets_enabled as connector_brackets_flag,
)

ET = pytz.timezone("US/Eastern")


@pytest.fixture(autouse=True)
def _no_day_bot_process(monkeypatch):
    monkeypatch.setattr(
        "src.strategy.overnight_research.ema15_day_bot_process_running",
        lambda: False,
    )


def _bars(
    start_et: str,
    n: int,
    *,
    base=24700.0,
    high_at=None,
    high_px=None,
    last_wick=None,
    last_close=None,
    last_high=None,
    last_low=None,
):
    start = ET.localize(datetime.strptime(start_et, "%Y-%m-%d %H:%M"))
    rows = []
    for i in range(n):
        ts = start + pd.Timedelta(minutes=i)
        hi = base + 2
        lo = base - 2
        close = base
        if high_at is not None and i == high_at:
            hi = high_px
            close = base
        if last_wick is not None and i == n - 1:
            hi = last_wick
            close = base - 1
        if last_close is not None and i == n - 1:
            close = last_close
            if last_high is not None:
                hi = last_high
            if last_low is not None:
                lo = last_low
        rows.append({
            "datetime": ts.astimezone(pytz.UTC),
            "open": base,
            "high": hi,
            "low": lo,
            "close": close,
            "volume": 10,
        })
    return pd.DataFrame(rows)


def test_live_flag_never_enables_overnight():
    assert resolve_overnight_research(live=True, flag=True, environ={"PAPER_OVERNIGHT": "true"}) is False
    assert resolve_overnight_research(live=False, flag=True) is True
    assert resolve_overnight_research(live=False, flag=False, environ={"PAPER_OVERNIGHT": "true"}) is True
    assert resolve_overnight_research(live=False, flag=False, environ={}) is False


def test_research_hours_include_rth_when_cme_open():
    rth = ET.localize(datetime(2026, 8, 17, 10, 0))
    globex = ET.localize(datetime(2026, 8, 17, 23, 10))
    friday_halt = ET.localize(datetime(2026, 8, 21, 17, 30))
    saturday = ET.localize(datetime(2026, 8, 22, 12, 0))
    assert is_overnight_research_hours(rth) is True
    assert is_overnight_research_hours(globex) is True
    assert is_overnight_research_hours(friday_halt) is False
    assert is_overnight_research_hours(saturday) is False
    assert should_flatten_before_rth(ET.localize(datetime(2026, 8, 18, 9, 25))) is False
    assert should_flatten_before_rth(globex) is False
    assert in_rth_hard_idle(rth) is False


def test_rth_1049_allows_entry_when_day_bot_not_owning():
    now = ET.localize(datetime(2026, 8, 18, 10, 49))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    assert in_rth_hard_idle(now) is False
    assert day_bot_owns_lucid(now) is False
    assert allow_new_overnight_entry(now) is True
    assert is_overnight_research_hours(now) is True
    sig = evaluate_overnight_cue(df, now, already_filled=False)
    assert sig is not None
    assert sig["cue"] == CUE_BREAK_ONH
    assert overnight_entry_blocked_reason(now) is None


def test_day_bot_env_blocks_rth_entry(monkeypatch):
    monkeypatch.setenv("DAY_BOT_OWNS_LUCID", "true")
    now = ET.localize(datetime(2026, 8, 18, 10, 49))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    assert day_bot_owns_lucid(now) is True
    assert allow_new_overnight_entry(now) is False
    assert evaluate_overnight_cue(df, now) is None
    reason = overnight_entry_blocked_reason(now) or ""
    assert "Lucid" in reason or "ema15" in reason


def test_cme_closed_friday_evening_no_entry():
    now = ET.localize(datetime(2026, 8, 21, 17, 30))
    df = _bars(
        "2026-08-21 09:30", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    assert in_rth_hard_idle(now) is False
    assert day_bot_owns_lucid(now) is False
    assert allow_new_overnight_entry(now) is False
    assert evaluate_overnight_cue(df, now) is None
    reason = overnight_entry_blocked_reason(now) or ""
    assert "CME closed" in reason or "closed" in reason.lower()


def test_no_flatten_at_0925():
    pos = {"side": "long", "stop": 24600.0, "target": 24800.0}
    at_flat = ET.localize(datetime(2026, 8, 18, 9, 25))
    before = ET.localize(datetime(2026, 8, 18, 9, 24))
    overnight = ET.localize(datetime(2026, 8, 17, 23, 10))
    assert should_flatten_before_rth(at_flat) is False
    assert should_flatten_before_rth(before) is False
    assert check_paper_exit(pos, high=24700.0, low=24690.0, last=24695.0, now=at_flat) is None
    assert check_paper_exit(pos, high=24700.0, low=24690.0, last=24695.0, now=overnight) is None


def test_zone_file_roundtrip(tmp_path: Path):
    now = ET.localize(datetime(2026, 8, 17, 23, 10))
    # Prior RTH Monday 9:30-16:00 plus overnight from 18:00.
    rth_start = ET.localize(datetime(2026, 8, 17, 9, 30))
    rows = []
    for i in range(90):
        ts = rth_start + pd.Timedelta(minutes=i)
        rows.append({
            "datetime": ts.astimezone(pytz.UTC),
            "open": 24600.0, "high": 24650.0, "low": 24550.0, "close": 24620.0, "volume": 5,
        })
    on = _bars("2026-08-17 18:00", 80, base=24700.0)
    df = pd.concat([pd.DataFrame(rows), on], ignore_index=True)
    payload = compute_zones(df, now, last_price=24705.0)
    path = tmp_path / "overnight_zones.json"
    write_zones(payload, str(path))
    loaded = read_zones(str(path))
    names = {z["name"] for z in loaded["zones"]}
    assert ZONE_ONH in names
    assert "overnight_low" in names
    assert "prior_rth_high" in names
    assert "round_number" in names
    assert "ema15_slow" in names
    assert loaded["session"] == SESSION_NAME
    assert loaded["writes_live_config"] is False
    onh = next(z for z in loaded["zones"] if z["name"] == ZONE_ONH)
    assert onh["price"] is not None


def test_journal_tags_overnight_not_ema15(tmp_path: Path):
    path = tmp_path / "paper_trade_journal.jsonl"
    journal = OvernightPaperJournal(str(path))
    sig = {
        "symbol": "MNQ",
        "side": "short",
        "direction": "short",
        "qty": 1,
        "entry_price": 24710.0,
        "stop": 24734.0,
        "target": 24686.0,
        "atr_stop_pts": 24.0,
        "cue": CUE_FADE_ONH,
        "zone_name": ZONE_ONH,
        "zone_price": 24712.0,
        "why": "fade first touch of overnight high",
    }
    journal.log_entry(sig, trade_id="ON_TEST")
    rec = journal.log_close({
        "trade_id": "ON_TEST",
        "symbol": "MNQ",
        "side": "short",
        "entry_price": 24710.0,
        "exit_price": 24686.0,
        "pts": 24.0,
        "pnl_usd": 46.76,
        "exit_reason": "TP",
        "cue": CUE_FADE_ONH,
        "zone_name": ZONE_ONH,
        "zone_price": 24712.0,
        "mae_pts": 4.0,
        "mfe_pts": 24.0,
        "why": {"primary_reason": "fade first touch", "facts": []},
    }, gemini_note="Wick fade worked; do not change RTH windows.")
    assert rec["session"] == SESSION_NAME
    assert is_overnight_research_row(rec)
    assert is_overnight_research_close(rec)
    assert is_bot_ema15_close(rec) is False
    ema15_closes = load_closes(str(path))
    assert ema15_closes == []
    overnight = journal.closed_trades()
    assert len(overnight) == 1
    assert overnight[0]["gemini_note"].startswith("Wick fade")
    mixed = {
        "event": "close",
        "session": "overnight_research",
        "window_name": "09:35",
        "atr_stop_pts": 40,
        "entry_snapshot": {"window": 0},
    }
    assert is_bot_ema15_close(mixed) is False


def test_fade_does_not_fire_after_settle():
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars("2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0, last_wick=24740.25)
    assert evaluate_overnight_cue(df, now, already_filled=False) is None


def test_break_onh_after_settle():
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    sig = evaluate_overnight_cue(df, now, already_filled=False)
    assert sig is not None
    assert sig["session"] == SESSION_NAME
    assert sig["recipe"] == PAPER_RECIPE
    assert sig["cue"] == CUE_BREAK_ONH
    assert sig["side"] == "long"
    assert sig["qty"] == 1
    assert not sig.get("flatten_et")
    assert TP_R == 0.65
    assert overnight_tp_r() == 0.65
    sl = float(sig["atr_stop_pts"])
    assert 12.0 <= sl <= 40.0
    assert sig["target"] == round(float(sig["entry_price"]) + sl * overnight_tp_r(), 2)


def test_overnight_tp_r_env(monkeypatch):
    monkeypatch.setenv("OVERNIGHT_TP_R", "0.85")
    assert overnight_tp_r() == 0.85
    assert overnight_tp_text() == "0.85R TP"
    monkeypatch.setenv("OVERNIGHT_TP_R", "1")
    assert overnight_tp_r() == 1.0
    assert overnight_tp_text() == "1R TP"
    monkeypatch.delenv("OVERNIGHT_TP_R", raising=False)
    assert overnight_tp_r() == 0.65
    assert overnight_tp_text() == "0.65R TP"


def test_overnight_macd_gate_and_confidence(monkeypatch):
    from src.strategy.overnight_research import (
        allow_overnight_macd_entry,
        entry_confidence_pct,
        overnight_macd_ok,
        overnight_stack_gate,
    )

    monkeypatch.setenv("OVERNIGHT_MACD", "true")
    monkeypatch.setenv("OVERNIGHT_MACD_REQUIRE", "true")
    assert overnight_macd_ok("long", 1) is True
    assert overnight_macd_ok("long", -1) is False
    assert overnight_macd_ok("short", -1) is True
    assert allow_overnight_macd_entry("long", -1) is False
    assert allow_overnight_macd_entry("long", 0) is True

    aligned = entry_confidence_pct(
        "long", overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1, 1),
    )
    opposed = entry_confidence_pct(
        "long", overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1, -1),
    )
    assert aligned["pct"] > opposed["pct"]
    assert any("MACD aligned" in x for x in aligned["breakdown"])
    assert any("MACD opposed" in x for x in opposed["breakdown"])
    assert "MACD aligned" in aligned["summary"]
    assert "MACD opposed" in opposed["summary"]

    monkeypatch.setenv("OVERNIGHT_MACD_REQUIRE", "false")
    assert allow_overnight_macd_entry("long", -1) is True


def test_break_onl_after_settle():
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=10, high_px=24705.0,
        last_close=24657.0, last_high=24658.0, last_low=24656.0,
    )
    # Force a settled overnight low on an early bar.
    df.loc[20, "low"] = 24660.0
    sig = evaluate_overnight_cue(df, now, already_filled=False)
    assert sig is not None
    assert sig["cue"] == "break_onl"
    assert sig["side"] == "short"


def test_no_cue_before_warmup():
    now = ET.localize(datetime(2026, 8, 17, 18, 20))
    df = _bars("2026-08-17 18:00", 15, base=24700.0, last_wick=24740.0)
    assert evaluate_overnight_cue(df, now) is None


def test_suggestions_never_write_live_config():
    closes = [
        {
            "event": "close",
            "session": SESSION_NAME,
            "cue": CUE_FADE_ONH,
            "zone_name": ZONE_ONH,
            "pnl_usd": 20.0,
        }
        for _ in range(12)
    ] + [
        {
            "event": "close",
            "session": SESSION_NAME,
            "cue": CUE_FADE_ONH,
            "zone_name": ZONE_ONH,
            "pnl_usd": -10.0,
        }
        for _ in range(10)
    ]
    payload = build_overnight_suggestions(closes, min_trades=20)
    assert payload["ready"] is True
    assert payload["auto_apply"] is False
    assert payload["writes_live_config"] is False
    assert payload["writes_mnq_profit_config"] is False
    assert "mnq_profit_config" in payload["note"]


def test_desk_overnight_pill_and_tonight_count():
    now = datetime(2026, 8, 18, 3, 10, tzinfo=pytz.UTC)
    status = {
        "running": True,
        "paper_mode": True,
        "mode": "paper",
        "overnight_research": True,
        "strategy": SESSION_NAME,
        "session": SESSION_NAME,
        "live_trades_today": 2,
        "daily_pnl": 12.5,
        "open_positions": [{"symbol": "MNQ", "direction": "LONG", "cue": "test_fill"}],
        "last_price": 24705.5,
        "quote_age_seconds": 2,
        "quote_source": "rithmic",
        "updated_at": now.isoformat(),
        "last_paper_fill": {"side": "short", "cue": CUE_FADE_ONH, "exit_price": 24700.0},
    }
    run = bot_run_state(status, now=now)
    assert run["pill"] == "PAPER OVERNIGHT RESEARCH"
    assert run["bot_running"] is True
    card = build_status_card(run, status, {"empty": True}, {}, levels={"quote_age_min": 40})
    rows = dict(card["rows"])
    assert rows["Bot"] == "PAPER OVERNIGHT RESEARCH"
    assert rows["Ticker last"] == "24705.50"
    assert rows["Target"] == "0.65R TP"
    assert rows["Open"] == "MNQ LONG"
    assert rows["Fills today (ET)"] == "2"
    assert "Live trades" not in rows
    assert todays_live_trade_count(status, [{}] * 1080, now=now) == 2
    assert "rithmic" in rows["Quote source"]


def test_quotes_only_blocks_orders():
    class Inner:
        connected = True

        def place_order(self, *a, **k):
            return {"id": "should-not-run"}

        def get_latest_price(self, symbol):
            return {"last": 1.0}

    wrap = QuotesOnlyBroker(Inner())
    assert wrap.connected is True
    assert wrap.get_latest_price("MNQ")["last"] == 1.0
    assert wrap.get_candles("MNQ", 1, 240) is None
    try:
        wrap.place_order("MNQ", "buy", 1, 1, 1, 1)
        assert False, "place_order must be blocked"
    except RuntimeError as exc:
        assert "never sends" in str(exc)
    assert "place_order" in BLOCKED_ORDER_METHODS


def test_paper_test_fill_defaults_long_20pt_stop():
    sig = paper_test_fill_signal(None, 25000.0)
    assert sig["side"] == "long"
    assert sig["cue"] == "test_fill"
    assert sig["session"] == SESSION_NAME
    assert "Lucid TEST011" in sig["why"]
    assert sig["entry_price"] == 25000.0
    assert sig["atr_stop_pts"] == 20.0
    assert sig["stop"] == 24980.0
    assert sig["paper"] is True
    assert paper_test_fill_from_env({"PAPER_TEST_FILL": "true"}) is True
    assert paper_test_fill_from_env({}) is False
    leftover = {"PAPER_TEST_FILL": "true", "PAPER_RITHMIC_BRACKETS": "false"}
    enable_overnight_lucid_sim_orders(leftover, keep_test_fill=False)
    assert leftover["PAPER_TEST_FILL"] == "false"
    assert leftover["PAPER_RITHMIC_BRACKETS"] == "true"
    assert paper_test_fill_from_env(leftover) is False
    assert skip_local_fake_fill(leftover) is True


def test_paper_test_fill_follows_15m_ema_short():
    start = ET.localize(datetime(2026, 8, 17, 18, 0))
    rows = []
    px = 25000.0
    for i in range(400):
        ts = start + pd.Timedelta(minutes=i)
        px -= 1.0
        rows.append({
            "datetime": ts.astimezone(pytz.UTC),
            "open": px + 1,
            "high": px + 2,
            "low": px - 2,
            "close": px,
            "volume": 10,
        })
    df = pd.DataFrame(rows)
    last = float(df.iloc[-1]["close"])
    sig = paper_test_fill_signal(df, last)
    assert sig["side"] == "short"
    assert 20.0 <= float(sig["atr_stop_pts"]) <= 60.0
    assert sig["stop"] == round(last + float(sig["atr_stop_pts"]), 2)


def test_quotes_only_connector_skips_history_candles():
    from src.broker.rithmic_connector import RithmicConnector

    broker = RithmicConnector(live_mode=False, quotes_only=True)
    broker._connected = True
    assert broker.get_candles("MNQ", 1, 50) is None
    assert broker.get_candles_seconds("MNQ", 30, 20) is None


def test_skip_history_pnl_skips_candles_without_second_bar_retry(monkeypatch):
    from src.broker.rithmic_connector import RithmicConnector

    monkeypatch.setenv("RITHMIC_SKIP_HISTORY_PNL", "true")
    broker = RithmicConnector(live_mode=True, quotes_only=False)
    broker._connected = True
    assert history_pnl_skipped(quotes_only=False) is True
    assert broker.get_candles("MNQ", 1, 400) is None
    assert broker.get_candles_seconds("MNQ", 30, 120) is None
    assert broker.get_candles_deep("MNQ", 1, 400) is None
    assert broker.aggregate_1m_from_30s("MNQ", num_candles=100) is None


def test_lucid_ticker_order_plants_skip_history():
    env = {
        "RITHMIC_QUOTES_ONLY": "true",
        "RITHMIC_ORDER_PLANT_ONLY": "true",
        "PAPER_RITHMIC_BRACKETS": "true",
    }
    apply_lucid_ticker_order_plants(env)
    assert env["RITHMIC_QUOTES_ONLY"] == "false"
    assert env["RITHMIC_SKIP_HISTORY_PNL"] == "true"
    assert env["RITHMIC_ORDER_PLANT_ONLY"] == "false"
    assert env["RITHMIC_ALLOW_SIMULATOR"] == "true"
    assert paper_rithmic_brackets_enabled(env) is True
    assert connector_brackets_flag(env) is True
    assert paper_uses_rithmic(env) is True
    assert paper_rithmic_brackets_enabled({}) is False
    assert is_forced_logout_text("ForcedLogout from plant") is True
    assert is_forced_logout_text("ok") is False


def test_overnight_lucid_sim_defaults_not_test_fill():
    env = {"PAPER_TEST_FILL": "true"}
    enable_overnight_lucid_sim_orders(env, keep_test_fill=False)
    assert env["PAPER_RITHMIC_BRACKETS"] == "true"
    assert env["PAPER_TEST_FILL"] == "false"
    assert env["OVERNIGHT_TRADING"] == "false"
    assert env["RITHMIC_ALLOW_SIMULATOR"] == "true"
    assert env["RITHMIC_SKIP_HISTORY_PNL"] == "true"
    assert paper_test_fill_from_env(env) is False
    assert paper_rithmic_brackets_enabled(env) is True
    assert skip_local_fake_fill(env) is True
    keep = {"PAPER_TEST_FILL": "true"}
    enable_overnight_lucid_sim_orders(keep, keep_test_fill=True)
    assert paper_test_fill_from_env(keep) is True
    assert keep["PAPER_RITHMIC_BRACKETS"] == "true"


def test_overnight_bat_sends_lucid_sim_not_test_fill():
    root = Path(__file__).resolve().parents[1]
    overnight = (root / "start_mnq_overnight_paper.bat").read_text(encoding="utf-8")
    live = (root / "start_mnq_live.bat").read_text(encoding="utf-8")
    assert "PAPER_RITHMIC_BRACKETS=true" in overnight
    assert "PAPER_TEST_FILL=false" in overnight
    assert "OVERNIGHT_TRADING=false" in overnight
    assert "RITHMIC_ALLOW_SIMULATOR=true" in overnight
    assert "--overnight-research" in overnight
    assert "Flatten 09:25" not in overnight
    assert "HARD idle" not in overnight
    assert "PAPER_MAX_FILLS" in overnight or "no cap" in overnight.lower()
    assert "PAPER_OVERNIGHT" not in live
    assert "set OVERNIGHT_TRADING=true" not in live
    assert "--overnight-research" not in live


def test_unlimited_fills_by_default(monkeypatch):
    monkeypatch.delenv("PAPER_MAX_FILLS", raising=False)
    from src.strategy.overnight_research import (
        explain_overnight_wait,
        overnight_cap_reached,
        overnight_fill_limit,
        paper_max_fills_base,
    )

    assert paper_max_fills_base() is None
    assert overnight_fill_limit(25, session_pnl=200.0) is None
    assert overnight_cap_reached(25, session_pnl=200.0) is False
    assert overnight_cap_reached(100) is False
    now = ET.localize(datetime(2026, 8, 18, 21, 10))
    line = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, last=29500.0, now=now,
        fills_tonight=25, session_pnl=200.0,
    )
    assert "Already took" not in line
    assert "25 fills" not in line
    assert "LONG:" in line


def test_paper_max_fills_env_caps_when_set(monkeypatch):
    monkeypatch.setenv("PAPER_MAX_FILLS", "10")
    from src.strategy.overnight_research import (
        overnight_cap_reached,
        overnight_fill_limit,
        paper_max_fills_base,
    )

    assert paper_max_fills_base() == 10
    assert overnight_fill_limit(9, session_pnl=-5.0) == 10
    assert overnight_fill_limit(10, session_pnl=50.0) == 15
    assert overnight_cap_reached(10, session_pnl=-5.0) is True
    assert overnight_cap_reached(10, session_pnl=50.0) is False
    assert overnight_cap_reached(15, session_pnl=50.0) is True


def test_globex_start_sunday_open():
    mon_0100 = ET.localize(datetime(2026, 8, 18, 1, 0))
    start = globex_session_start_et(mon_0100)
    assert start.hour == 18
    assert start.day == 17


def test_restore_skips_test_fill():
    from src.strategy.overnight_paper_runner import should_restore_open

    assert should_restore_open(None) is False
    assert should_restore_open({"cue": "test_fill"}) is False
    assert should_restore_open({"cue": "break_onl"}) is True


def test_explain_overnight_wait_first_hour():
    from src.strategy.overnight_research import explain_overnight_wait

    now = ET.localize(datetime(2026, 8, 17, 18, 20))
    line = explain_overnight_wait(bars=15, now=now)
    assert "First hour after 18:00" in line
    assert "Scanning starts around 19:00" in line


def test_late_start_can_settle_after_1900():
    from src.strategy.overnight_research import overnight_completed_bars, settled_overnight_range

    now = ET.localize(datetime(2026, 8, 18, 20, 5))
    df = _bars(
        "2026-08-18 19:40", 25, base=24700.0, high_at=0, high_px=24740.0,
        last_close=24741.0, last_high=24742.0,
    )
    rng = settled_overnight_range(overnight_completed_bars(df, now), now)
    assert rng["onh_settled"] is True
    sig = evaluate_overnight_cue(df, now, already_filled=False)
    assert sig is not None
    assert sig["cue"] == CUE_BREAK_ONH


def test_late_start_still_watching_after_1900():
    from src.strategy.overnight_research import explain_overnight_wait, wait_reason_from_zones

    now = ET.localize(datetime(2026, 8, 18, 20, 5))
    line = explain_overnight_wait(bars=16, now=now, onh=29550.0, onl=29474.0, last=29500.0)
    assert "Not looking for a trade yet" not in line
    assert "LONG:" in line
    assert "needs to rise" in line
    payload = {
        "overnight_bars": 16,
        "price": 29500.0,
        "zones": [
            {"name": "overnight_high", "price": 29550.0, "settled": False},
            {"name": "overnight_low", "price": 29474.0, "settled": False},
        ],
    }
    assert "LONG:" in wait_reason_from_zones(payload, now=now)


def test_explain_overnight_wait_watching_for_break():
    from src.strategy.overnight_research import explain_overnight_wait

    now = ET.localize(datetime(2026, 8, 18, 21, 10))
    line = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, onh_settled=True, onl_settled=True,
        last=29500.0, now=now,
    )
    assert "LONG:" in line
    assert "SHORT:" in line
    assert "29550.00" in line
    assert "29474.00" in line
    assert "too low" in line
    assert "needs to rise" in line
    assert "too high" in line
    assert "needs to fall" in line


def test_last_print_fires_before_1m_close():
    now = ET.localize(datetime(2026, 8, 18, 22, 40))
    df = _bars("2026-08-18 18:00", 80, base=24700.0)
    df.loc[5, "high"] = 24800.0
    assert evaluate_overnight_cue(df, now, already_filled=False) is None
    sig = evaluate_overnight_cue(df, now, already_filled=False, last_price=24710.0)
    assert sig is not None
    assert sig["side"] == "long"
    assert sig["entry_price"] == 24710.0


def test_roll20_fires_inside_wide_session_range():
    now = ET.localize(datetime(2026, 8, 18, 22, 40))
    df = _bars("2026-08-18 18:00", 80, base=24700.0)
    df.loc[5, "high"] = 24800.0
    df.loc[79, "high"] = 24708.0
    df.loc[79, "close"] = 24707.5
    sig = evaluate_overnight_cue(df, now, already_filled=False)
    assert sig is not None
    assert sig["side"] == "long"
    assert float(sig["zone_price"]) < 24750.0


def test_twenty_fills_cap_and_same_cue_still_fires(monkeypatch):
    monkeypatch.setenv("PAPER_MAX_FILLS", "20")
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    sig = evaluate_overnight_cue(df, now, already_filled=False, used_cues={CUE_BREAK_ONH})
    assert sig is not None
    assert sig["cue"] == CUE_BREAK_ONH
    assert evaluate_overnight_cue(df, now, already_filled=True) is None
    assert MAX_FILLS_PER_NIGHT == 20
    assert overnight_cap_reached(1) is False
    assert overnight_cap_reached(6) is False
    assert overnight_cap_reached(19) is False
    assert overnight_cap_reached(20) is True
    assert overnight_cap_reached(20, session_pnl=0) is True
    assert overnight_cap_reached(20, session_pnl=-12.5) is True


def test_green_session_allows_five_extra_after_twenty(monkeypatch):
    monkeypatch.setenv("PAPER_MAX_FILLS", "20")
    from src.strategy.overnight_research import (
        EXTRA_FILLS_IF_GREEN,
        MAX_FILLS_GREEN,
        explain_overnight_wait,
        overnight_fill_limit,
        recipe_session_stats,
    )

    assert EXTRA_FILLS_IF_GREEN == 5
    assert MAX_FILLS_GREEN == 25
    assert overnight_fill_limit(19, session_pnl=80.0) == 20
    assert overnight_cap_reached(19, session_pnl=80.0) is False
    assert overnight_cap_reached(20, session_pnl=80.0) is False
    assert overnight_cap_reached(24, session_pnl=80.0) is False
    assert overnight_cap_reached(25, session_pnl=80.0) is True
    assert overnight_cap_reached(22, session_pnl=-1.0) is True

    now = ET.localize(datetime(2026, 8, 18, 21, 10))
    red = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, last=29500.0, now=now,
        fills_tonight=20, session_pnl=-40.0,
    )
    assert "20 fills today, ET-day red" in red
    green = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, last=29500.0, now=now,
        fills_tonight=20, session_pnl=40.0,
    )
    assert "Fill 21 of 25" in green
    assert "session green — 5 extra allowed" in green
    done = explain_overnight_wait(
        already_filled=True, fills_tonight=25, session_pnl=40.0, now=now,
    )
    assert "today" in done.lower() or "25 fills" in done

    start = pd.Timestamp("2026-08-18 22:00:00", tz="UTC")
    rows = [
        {"cue": "break_onh", "exit_time": "2026-08-18T23:00:00+00:00", "pnl_usd": 40.0},
        {"cue": "test_fill", "exit_time": "2026-08-18T23:05:00+00:00", "pnl_usd": 999.0},
        {"cue": "break_onl", "exit_time": "2026-08-18T23:10:00+00:00", "pnl_usd": -12.0},
        {"cue": "break_onh", "exit_time": "2026-08-17T12:00:00+00:00", "pnl_usd": 80.0},
    ]
    n, pnl = recipe_session_stats(rows, start)
    assert n == 2
    assert pnl == 28.0


def test_fill_cap_window_is_et_midnight_not_globex():
    from src.strategy.overnight_research import fill_window_start_et

    after_midnight = ET.localize(datetime(2026, 8, 18, 0, 30))
    evening = ET.localize(datetime(2026, 8, 17, 22, 0))
    start = fill_window_start_et(after_midnight)
    assert start.hour == 0
    assert start.day == 18
    evening_start = fill_window_start_et(evening)
    assert evening_start.day == 17
    assert evening_start.hour == 0
    rows = [
        {"cue": "break_onh", "exit_time": "2026-08-18T03:50:00+00:00", "pnl_usd": 10.0},  # 23:50 ET 17th
        {"cue": "break_onl", "exit_time": "2026-08-18T04:30:00+00:00", "pnl_usd": 12.0},  # 00:30 ET 18th
    ]
    n, _ = recipe_session_stats(rows, pd.Timestamp(start).tz_convert("UTC"))
    assert n == 1


def test_stack_qty_two_only_when_last_close_and_htf_agree():
    from src.strategy.overnight_research import (
        overnight_stack_qty,
        overnight_stack_gate,
        should_defer_entry_for_stack,
        STACK_QTY,
    )

    assert overnight_stack_qty("long", 24710.0, 24700.0, 24705.0, 1, 1) == 1
    assert overnight_stack_qty("long", 24710.0, 24710.0, 24705.0, 0, 1) == 1
    assert overnight_stack_qty("long", 24710.0, 24710.0, 24705.0, 1, 0) == 1
    assert overnight_stack_qty("long", 24710.0, 24710.0, 24705.0, 1, 1) == STACK_QTY
    assert overnight_stack_qty("short", 24690.0, 24690.0, 24700.0, -1, -1) == STACK_QTY
    assert overnight_stack_qty("short", 24690.0, 24690.0, 24700.0, 1, -1) == 1

    gate = overnight_stack_gate("long", 24710.0, 24700.0, 24705.0, 1, 1)
    assert gate["last_ok"] is True
    assert gate["close_ok"] is False
    assert gate["trend_ok"] is True
    assert should_defer_entry_for_stack("long", 24710.0, 24700.0, 24705.0, 1, 1) is True
    assert should_defer_entry_for_stack("long", 24710.0, 24710.0, 24705.0, 1, 1) is False

    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    one = evaluate_overnight_cue(df, now, already_filled=False, trend_15m=0, trend_60m=0)
    assert one is not None
    assert one["qty"] == 1
    two = evaluate_overnight_cue(df, now, already_filled=False, trend_15m=1, trend_60m=1)
    assert two is not None
    assert two["qty"] == 2
    tick_only = evaluate_overnight_cue(
        df, now, already_filled=False, last_price=24750.0, trend_15m=1, trend_60m=1,
    )
    assert tick_only is not None
    # last through but completed close is 24740.75 vs zone from bars; still through on this fixture
    # Force a last-only fire with a close that is NOT through:
    df2 = _bars("2026-08-17 18:00", 80, base=24700.0)
    df2.loc[5, "high"] = 24800.0
    last_only = evaluate_overnight_cue(
        df2, now, already_filled=False, last_price=24710.0, trend_15m=1, trend_60m=1,
    )
    assert last_only is not None
    assert last_only["qty"] == 1


def test_entry_confidence_full_stack_highest():
    from src.strategy.overnight_research import (
        entry_confidence_pct,
        format_entry_confidence_line,
        overnight_stack_gate,
        STACK_QTY,
    )

    gate = overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1)
    assert gate["qty"] == STACK_QTY
    conf = entry_confidence_pct("long", gate)
    assert conf["pct"] >= 85
    assert "full stack +8" in conf["breakdown"]
    line = format_entry_confidence_line(conf)
    assert line.startswith("Confidence: ")
    assert "HTF aligned" in line or "1m close" in line


def test_entry_confidence_last_only_lower_than_close():
    from src.strategy.overnight_research import entry_confidence_pct, overnight_stack_gate

    close_pending = overnight_stack_gate("long", 24710.0, 24700.0, 24705.0, 1, 1)
    close_ok = overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1)
    pending = entry_confidence_pct("long", close_pending)
    confirmed = entry_confidence_pct("long", close_ok)
    assert confirmed["pct"] > pending["pct"]
    assert "1m close pending" in " ".join(pending["breakdown"])


def test_entry_confidence_opposed_htf_penalized():
    from src.strategy.overnight_research import entry_confidence_pct, overnight_stack_gate

    aligned = entry_confidence_pct(
        "long", overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1),
    )
    opposed = entry_confidence_pct(
        "long", overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, -1, -1),
    )
    assert aligned["pct"] > opposed["pct"]
    assert any("opposed" in x for x in opposed["breakdown"])


def test_entry_confidence_journal_nudge():
    from src.strategy.overnight_research import (
        entry_confidence_pct,
        overnight_stack_gate,
        recipe_historical_win_pct,
    )

    gate = overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 1)
    rows = [
        {"cue": "break_onh", "pnl_usd": 10, "exit_reason": "TP"},
        {"cue": "break_onh", "pnl_usd": 10, "exit_reason": "TP"},
        {"cue": "break_onh", "pnl_usd": -5, "exit_reason": "SL"},
    ] * 4
    hist = recipe_historical_win_pct(rows, "break_onh", min_trades=10)
    assert hist["historical_n"] == 12
    assert hist["historical_win_pct"] == 66.7
    boosted = entry_confidence_pct("long", gate, **hist)
    base = entry_confidence_pct("long", gate)
    assert boosted["pct"] == min(92, base["pct"] + 2)
    assert boosted["journal_delta"] == 2


def test_entry_confidence_journal_cannot_alone_force_skip(monkeypatch):
    """Losing journal soft-nudge must not push a stack that clears min below it."""
    from src.strategy.overnight_research import entry_confidence_pct, overnight_stack_gate

    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "65")
    # close confirmed + one HTF aligned + one flat = 50+15+12+3 = 80 stack
    gate = overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 1, 0)
    assert gate.get("close_ok")
    assert not gate.get("trend_ok")
    base = entry_confidence_pct("long", gate)
    assert base["pct"] >= 65
    bad = entry_confidence_pct(
        "long",
        gate,
        historical_win_pct=39.0,
        historical_n=12,
    )
    assert bad["journal_delta"] == -2
    assert bad["pct"] >= 65
    assert bad["pct"] >= base["stack_pct"] - 2 or "journal floor" in " ".join(bad["breakdown"])
    # With soft -2, 80-2=78 still clears 65; floor only matters near threshold.
    near = entry_confidence_pct(
        "long",
        overnight_stack_gate("long", 24710.0, 24710.0, 24705.0, 0, 0),
        historical_win_pct=30.0,
        historical_n=12,
    )
    # close + both flat = 50+15+3+3 = 71; -2 would be 69 still >= 65
    assert near["pct"] >= 65


def test_entry_confidence_journal_floor_at_min(monkeypatch):
    """When stack exactly meets min, journal penalty floors at min (does not skip)."""
    from src.strategy.overnight_research import (
        allow_overnight_entry_confidence,
        entry_confidence_pct,
    )

    monkeypatch.setenv("OVERNIGHT_MACD", "false")
    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "65")
    # Craft gate-like mapping: close + one opposed + one flat = 50+15-6+3 = 62
    # Actually need stack >= 65. Use: close + both flat = 71, then force min=70
    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "70")
    gate = {
        "close_ok": True,
        "last_ok": True,
        "trend_ok": False,
        "trend_15m": 0,
        "trend_60m": 0,
        "qty": 1,
    }
    # 50+15+3+3 = 71 stack; journal -2 → 69, floor back to 70
    floored = entry_confidence_pct(
        "long",
        gate,
        historical_win_pct=25.0,
        historical_n=15,
    )
    assert floored["stack_pct"] == 71
    assert floored["pct"] == 70
    assert allow_overnight_entry_confidence(floored["pct"]) is True
    assert any("journal floor" in x for x in floored["breakdown"])


def test_entry_confidence_clamped():
    from src.strategy.overnight_research import entry_confidence_pct, overnight_stack_gate

    bad = entry_confidence_pct(
        "long",
        overnight_stack_gate("long", 24710.0, 24700.0, 24705.0, -1, -1),
        historical_win_pct=25.0,
        historical_n=12,
    )
    assert 42 <= bad["pct"] <= 92
    assert bad["journal_delta"] == -2


def test_trade_placed_banner_readable():
    from src.strategy.overnight_research import (
        entry_confidence_pct,
        format_trade_placed_banner,
        format_trade_placed_breakdown,
        overnight_stack_gate,
        STACK_QTY,
    )

    gate = overnight_stack_gate("short", 29528.0, 29528.0, 29529.25, -1, -1)
    assert gate["qty"] == STACK_QTY
    conf = entry_confidence_pct("short", gate)
    sig = {
        "side": "short",
        "qty": STACK_QTY,
        "entry_price": 29528.0,
        "entry_confidence": conf,
        "entry_confidence_pct": conf["pct"],
    }
    breakdown = format_trade_placed_breakdown(sig)
    assert "15m+60m aligned" in breakdown
    assert "1m close confirmed" in breakdown
    assert "qty 2 stack" in breakdown
    banner = format_trade_placed_banner(sig, color=False)
    assert "📉" in banner or "📈" in banner
    assert "ENTRY SHORT 2 MNQ" in banner or "ENTRY SHORT @ 29528.00" in banner
    assert f"Win {conf['pct']}%" in banner
    assert "15m+60m aligned" in banner


def test_trade_placed_banner_test_fill():
    from src.strategy.overnight_research import format_trade_placed_banner

    banner = format_trade_placed_banner(
        {"side": "long", "qty": 1, "entry_price": 29500.0, "cue": "test_fill"},
        color=False,
    )
    assert "TRADE PLACED — LONG 1 MNQ" in banner or "ENTRY LONG" in banner
    assert "test fill" in banner.lower()


def test_evaluate_overnight_cue_includes_confidence():
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
    )
    sig = evaluate_overnight_cue(
        df, now, already_filled=False, last_price=24750.0, trend_15m=1, trend_60m=1,
    )
    assert sig is not None
    assert sig.get("entry_confidence_pct") is not None
    assert isinstance(sig.get("entry_confidence"), dict)


def test_overnight_defers_entry_when_stack_waiting_for_close(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[])
    eng = _overnight_open_engine(rithmic)
    sig = _overnight_open_sig(zone_price=24740.0)
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24700.0, last_high=24750.0,
    )
    now = ET.localize(datetime(2026, 8, 17, 19, 20))
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 24750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (1, 1),
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, df, now)
    rithmic.place_order.assert_not_called()
    assert eng.position is None
    eng._emit_throttled.assert_called()


def test_five_loss_streak_cools_down_thirty_minutes():
    from src.strategy.overnight_research import (
        recipe_cooldown_until,
        explain_overnight_wait,
        in_overnight_cooldown,
    )

    start = pd.Timestamp("2026-08-18 22:00:00", tz="UTC")
    now = ET.localize(datetime(2026, 8, 18, 18, 20))
    rows = []
    for i in range(5):
        rows.append({
            "cue": "break_onh",
            "exit_time": f"2026-08-18T22:{10 + i:02d}:00+00:00",
            "pnl_usd": -20.0,
        })
    rows.append({
        "cue": "test_fill",
        "exit_time": "2026-08-18T22:16:00+00:00",
        "pnl_usd": -999.0,
    })
    until = recipe_cooldown_until(rows, start, now)
    assert until is not None
    assert until.strftime("%H:%M") == "18:44"  # 22:14 UTC + 30m = 22:44 UTC = 18:44 ET
    assert in_overnight_cooldown(until, now) is True
    line = explain_overnight_wait(now=now, cooldown_until=until, fills_tonight=5)
    assert "cooling down until 18:44 ET" in line

    later = ET.localize(datetime(2026, 8, 18, 18, 50))
    assert recipe_cooldown_until(rows, start, later) is None

    # After cooldown, same five losses do not re-trigger; a new 5-loss streak does.
    more = list(rows)
    for i in range(5):
        more.append({
            "cue": "break_onl",
            "exit_time": f"2026-08-18T23:{20 + i:02d}:00+00:00",
            "pnl_usd": -8.0,
        })
    now2 = ET.localize(datetime(2026, 8, 18, 19, 40))
    until2 = recipe_cooldown_until(more, start, now2)
    assert until2 is not None
    assert until2.strftime("%H:%M") == "19:54"

    # A winner in the middle resets the streak — 4 losses + win + 4 losses is not 5.
    mixed = [
        {"cue": "break_onh", "exit_time": f"2026-08-18T22:{10 + i:02d}:00+00:00", "pnl_usd": -20.0}
        for i in range(4)
    ]
    mixed.append({"cue": "break_onl", "exit_time": "2026-08-18T22:15:00+00:00", "pnl_usd": 40.0})
    mixed.extend(
        {"cue": "break_onh", "exit_time": f"2026-08-18T22:{16 + i:02d}:00+00:00", "pnl_usd": -20.0}
        for i in range(4)
    )
    assert recipe_cooldown_until(mixed, start, now) is None


def test_overnight_skip_cooldown_override(monkeypatch):
    from src.strategy.overnight_research import (
        recipe_cooldown_until,
        explain_overnight_wait,
        in_overnight_cooldown,
        overnight_cooldown_override_message,
        overnight_skip_cooldown_from_env,
    )

    start = pd.Timestamp("2026-08-18 22:00:00", tz="UTC")
    now = ET.localize(datetime(2026, 8, 18, 18, 20))
    rows = [
        {
            "cue": "break_onh",
            "exit_time": f"2026-08-18T22:{10 + i:02d}:00+00:00",
            "pnl_usd": -20.0,
        }
        for i in range(5)
    ]
    until = recipe_cooldown_until(rows, start, now)
    assert until is not None
    assert in_overnight_cooldown(until, now) is True

    monkeypatch.setenv("OVERNIGHT_SKIP_COOLDOWN", "1")
    assert overnight_skip_cooldown_from_env() is True
    assert overnight_cooldown_override_message() == (
        "Loss cooldown OVERRIDDEN (OVERNIGHT_SKIP_COOLDOWN=1)"
    )
    assert recipe_cooldown_until(rows, start, now) is None
    assert in_overnight_cooldown(until, now) is False
    line = explain_overnight_wait(now=now, cooldown_until=until, fills_tonight=5)
    assert "cooling down" not in line

    monkeypatch.delenv("OVERNIGHT_SKIP_COOLDOWN", raising=False)
    assert overnight_skip_cooldown_from_env() is False


def test_overnight_cooldown_losses_zero_disables_trigger(monkeypatch):
    from src.strategy.overnight_research import (
        recipe_cooldown_until,
        in_overnight_cooldown,
        overnight_cooldown_override_message,
        loss_streak_for_cooldown_from_env,
    )

    start = pd.Timestamp("2026-08-18 22:00:00", tz="UTC")
    now = ET.localize(datetime(2026, 8, 18, 18, 20))
    rows = [
        {
            "cue": "break_onh",
            "exit_time": f"2026-08-18T22:{10 + i:02d}:00+00:00",
            "pnl_usd": -20.0,
        }
        for i in range(5)
    ]
    monkeypatch.setenv("OVERNIGHT_COOLDOWN_LOSSES", "0")
    assert loss_streak_for_cooldown_from_env() == 0
    assert overnight_cooldown_override_message() == (
        "Loss cooldown DISABLED (OVERNIGHT_COOLDOWN_LOSSES=0)"
    )
    assert recipe_cooldown_until(rows, start, now) is None
    fake_until = ET.localize(datetime(2026, 8, 18, 19, 0))
    assert in_overnight_cooldown(fake_until, now) is True

    monkeypatch.delenv("OVERNIGHT_COOLDOWN_LOSSES", raising=False)
    assert loss_streak_for_cooldown_from_env() == 5


def test_wait_copy_after_first_fill_still_looking(monkeypatch):
    monkeypatch.setenv("PAPER_MAX_FILLS", "20")
    from src.strategy.overnight_research import explain_overnight_wait

    now = ET.localize(datetime(2026, 8, 18, 21, 10))
    line = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, last=29500.0, now=now, fills_tonight=1,
    )
    assert "Fill 2 of 20" in line
    assert "LONG:" in line
    cap = explain_overnight_wait(already_filled=True, now=now)
    assert "20 fills" in cap
    mid = explain_overnight_wait(
        bars=80, onh=29550.0, onl=29474.0, last=29500.0, now=now, fills_tonight=6,
    )
    assert "Fill 7 of 20" in mid
    assert "Already took" not in mid


def test_explain_says_new_high_resets_clock():
    from src.strategy.overnight_research import explain_overnight_wait

    now = ET.localize(datetime(2026, 8, 18, 21, 10))
    line = explain_overnight_wait(
        bars=40, onh=29578.0, onl=29451.75, last=29592.0, now=now,
    )
    assert "last through" in line
    assert "needs to fall" in line


def test_persist_live_1m_keeps_this_globex_session(tmp_path):
    from src.strategy.overnight_paper_runner import load_persisted_live_1m, save_persisted_live_1m

    now = ET.localize(datetime(2026, 8, 18, 20, 5))
    path = tmp_path / "live_1m.csv"
    rows = []
    for i, ts in enumerate([
        "2026-08-18 16:00:00-04:00",
        "2026-08-18 18:01:00-04:00",
        "2026-08-18 19:44:00-04:00",
    ]):
        px = 29500.0 + i
        rows.append({"datetime": ts, "open": px, "high": px, "low": px, "close": px, "volume": 0})
    save_persisted_live_1m(pd.DataFrame(rows), str(path), now)
    got = load_persisted_live_1m(str(path), now)
    assert got is not None
    assert len(got) == 2


class _FakeRithmic:
    connected = True

    def __init__(self, quote=None):
        self._quote = quote

    def get_latest_price(self, symbol):
        return self._quote


class _FakeFeed:
    def __init__(self, rithmic=None):
        self.rithmic = rithmic
        self.source = "rithmic" if rithmic is not None else "databento"

    def get_latest_price(self):
        return {"last": 29506.5}


def test_last_price_ignores_stale_csv_when_rithmic_plant_exists():
    from src.strategy.overnight_paper_runner import _last_price

    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-08-19 05:00:00", tz="UTC")],
        "close": [29506.5],
    })
    feed = _FakeFeed(rithmic=_FakeRithmic(quote=None))
    assert _last_price(feed, df) is None


def test_last_price_uses_rithmic_not_csv():
    from src.strategy.overnight_paper_runner import _last_price

    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-08-19 05:00:00", tz="UTC")],
        "close": [29506.5],
    })
    feed = _FakeFeed(rithmic=_FakeRithmic(quote={"last": 29718.0, "updated_at": time.time()}))
    assert _last_price(feed, df) == 29718.0


def test_manage_open_does_not_sl_restored_long_on_stale_csv():
    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    eng = OvernightPaperEngine.__new__(OvernightPaperEngine)
    eng.feed = _FakeFeed(rithmic=_FakeRithmic(quote=None))
    eng.position = {
        "side": "long",
        "entry_price": 29734.0,
        "stop": 29694.0,
        "target": 29774.0,
        "mae_pts": 0.0,
        "mfe_pts": 0.0,
    }
    eng._emit_throttled = lambda *a, **k: None
    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-08-19 05:00:00", tz="UTC")],
        "high": [29510.0],
        "low": [29500.0],
        "close": [29506.5],
    })
    eng._manage_open(df, datetime.now(ET))
    assert eng.position is not None
    assert float(eng.position["stop"]) == 29694.0


def _overnight_close_engine(rithmic):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    eng = OvernightPaperEngine.__new__(OvernightPaperEngine)
    eng.feed = MagicMock()
    eng.feed.rithmic = rithmic
    eng.feed.brackets = True
    eng.feed.maybe_resubscribe_ticker = MagicMock()
    eng.journal = MagicMock()
    eng.journal.log_close = MagicMock(side_effect=lambda rec, gemini_note="": rec)
    eng.journal.refresh_suggestions = MagicMock()
    rithmic.list_working_orders.return_value = []
    eng._last_leftover_sweep = 99.0
    eng.daily_pnl = 0.0
    eng.fills_tonight = 1
    eng._count_tonight = MagicMock(return_value=1)
    eng._gemini_note = MagicMock(return_value="")
    eng._emit = MagicMock()
    eng._emit_dev = MagicMock()
    eng.position = {
        "trade_id": "bot_x",
        "lucid_ticket": "bot_x",
        "side": "long",
        "qty": 1,
        "entry_price": 29734.0,
        "stop": 29694.0,
        "target": 29774.0,
        "entry_time": datetime.now(ET),
        "cue": "break_onh",
        "zone_name": "overnight_high",
        "zone_price": 29714.25,
        "mae_pts": 0.0,
        "mfe_pts": 0.0,
        "atr_stop_pts": 40.0,
        "why": "break",
    }
    return eng


def test_overnight_sl_cancels_leftover_orders_not_flatten(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    OvernightPaperEngine._close_position(eng, 29694.0, "SL", pd.DataFrame())
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", "bot_x", reason="overnight_sl_fill", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()
    eng.feed.maybe_resubscribe_ticker.assert_called()
    assert eng.position is None


def test_overnight_tp_cancels_leftover_stop_not_flatten(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    OvernightPaperEngine._close_position(eng, 29774.0, "TP", pd.DataFrame())
    rithmic.cancel_working_orders_after_exit.assert_any_call(
        "MNQ", "bot_x", reason="overnight_tp_fill", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()
    assert eng.position is None


def test_overnight_short_sl_cancels_buy_limit_not_flatten(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position["side"] = "short"
    eng.position["stop"] = 29774.0
    eng.position["target"] = 29694.0
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    OvernightPaperEngine._close_position(eng, 29774.0, "SL", pd.DataFrame())
    rithmic.cancel_working_orders_after_exit.assert_any_call(
        "MNQ", "bot_x", reason="overnight_sl_fill", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()
    assert eng.position is None


def test_overnight_sl_cancels_leftovers_without_lucid_ticket(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position["lucid_ticket"] = ""
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    OvernightPaperEngine._close_position(eng, 29694.0, "SL", pd.DataFrame())
    rithmic.cancel_working_orders_after_exit.assert_any_call(
        "MNQ", "", reason="overnight_sl_fill", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_startup_flat_cancels_leftover_working():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position = None
    eng.positions = []
    n = OvernightPaperEngine._cancel_leftover_working_if_flat(
        eng, "overnight_startup_flat",
    )
    assert n == 1
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_startup_flat", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_periodic_flat_sweeps_leftover_buy_limit():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position = None
    eng.positions = []
    eng._last_leftover_sweep = 0.0
    rithmic.list_working_orders = MagicMock(return_value=[{
        "user_tag": "bot_orphan_tp",
        "transaction_type": "BUY",
        "price_type": "LIMIT",
        "status": "open",
        "basket_id": "551990",
    }])
    OvernightPaperEngine._maybe_sweep_leftover_working(eng)
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_periodic_flat", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_periodic_flat_leaves_manual_working_alone():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position = None
    eng.positions = []
    eng._last_leftover_sweep = 0.0
    rithmic.list_working_orders = MagicMock(return_value=[{
        "user_tag": "",
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "open",
        "basket_id": "999001",
    }])
    OvernightPaperEngine._maybe_sweep_leftover_working(eng)
    rithmic.cancel_working_orders_after_exit.assert_not_called()
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_wipe_cancels_bot_preserves_manual():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng.position = None
    eng.positions = []
    listed = [
        {
            "user_tag": "bot_leftover_tp",
            "transaction_type": "BUY",
            "price_type": "LIMIT",
            "status": "open",
            "basket_id": "551990",
        },
        {
            "user_tag": "",
            "transaction_type": "SELL",
            "price_type": "STOP",
            "status": "open",
            "basket_id": "999001",
        },
    ]
    rithmic.list_working_orders = MagicMock(side_effect=[listed, [listed[1]]])
    n = OvernightPaperEngine._aggressive_remnant_wipe(eng, "overnight_periodic_flat")
    assert n == 1
    rithmic.cancel_working_orders_after_exit.assert_called_once()
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_entry_allows_manual_working_orders(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    manual = [{
        "user_tag": "",
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "999001",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=manual)
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=0)
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_new", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.increment_zone_touch",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.write_zones",
        lambda *a, **k: None,
    )
    OvernightPaperEngine._open_paper_position(eng, _overnight_open_sig())
    rithmic.cancel_working_orders_after_exit.assert_not_called()
    rithmic.place_order.assert_called_once()
    assert eng.position is not None


def test_overnight_ctrlc_flattens_then_cancels_leftovers(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    OvernightPaperEngine._close_position(eng, 29700.0, "shutdown", pd.DataFrame())
    rithmic.flatten_symbol.assert_called_once()
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", "bot_x", reason="overnight_shutdown_fill", already_flat=True,
    )
    assert eng.position is None


def test_overnight_periodic_skips_leftover_cancel_when_position_open():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_close_engine(rithmic)
    eng._last_leftover_sweep = 0.0
    OvernightPaperEngine._maybe_sweep_leftover_working(eng)
    rithmic.cancel_working_orders_after_exit.assert_not_called()
    rithmic.flatten_symbol.assert_not_called()
    assert eng.position is not None


def test_overnight_restore_if_broker_flat_cancels_not_flatten(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.flatten_symbol = MagicMock()
    rithmic.get_symbol_list_positions_net = MagicMock(return_value=0)
    eng = _overnight_close_engine(rithmic)
    OvernightPaperEngine._reconcile_restored_if_broker_flat(eng)
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_restore_broker_flat", already_flat=True,
    )
    rithmic.flatten_symbol.assert_not_called()
    eng.journal.log_close.assert_not_called()
    assert eng.position is None


def test_overnight_skip_restore_when_lucid_flat(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.get_symbol_list_positions_net = MagicMock(return_value=0)
    rithmic.flatten_symbol = MagicMock()
    eng = OvernightPaperEngine.__new__(OvernightPaperEngine)
    eng.position = None
    eng.positions = []
    eng.feed = MagicMock()
    eng.feed.rithmic = rithmic
    eng.feed.brackets = True
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.load_last_open_overnight",
        lambda: {
            "cue": "break_onh",
            "trade_id": "bot_stale",
            "side": "long",
            "entry_price": 29591.75,
        },
    )
    OvernightPaperEngine._restore_open_position(eng)
    assert eng.position is None
    rithmic.flatten_symbol.assert_not_called()


def test_overnight_skip_restore_when_lucid_net_unknown(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.get_symbol_list_positions_net = MagicMock(return_value=None)
    eng = OvernightPaperEngine.__new__(OvernightPaperEngine)
    eng.position = None
    eng.positions = []
    eng.feed = MagicMock()
    eng.feed.rithmic = rithmic
    eng.feed.brackets = True
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.load_last_open_overnight",
        lambda: {
            "cue": "break_onh",
            "trade_id": "bot_stale",
            "side": "long",
            "entry_price": 29591.75,
        },
    )
    OvernightPaperEngine._restore_open_position(eng)
    assert eng.position is None


def _overnight_open_sig(**overrides):
    sig = {
        "side": "long",
        "qty": 1,
        "entry_price": 29750.0,
        "stop": 29710.0,
        "target": 29790.0,
        "atr_stop_pts": 40.0,
        "cue": "break_onh",
        "zone_name": "overnight_high",
        "zone_price": 29740.0,
        "why": "break",
    }
    sig.update(overrides)
    return sig


def _overnight_open_engine(rithmic):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    eng = OvernightPaperEngine.__new__(OvernightPaperEngine)
    eng.feed = MagicMock()
    eng.feed.rithmic = rithmic
    eng.feed.brackets = True
    eng.journal = MagicMock()
    eng.zones = {}
    eng.positions = []
    eng.used_cues = set()
    eng.fills_tonight = 0
    eng.session_pnl = 0.0
    eng.cooldown_until = None
    eng._session_key = "2026-08-19"
    eng.filled_session = None
    eng.last_fill = None
    eng._emit = MagicMock()
    eng._emit_dev = MagicMock()
    eng._emit_throttled = MagicMock()
    eng._last_entry_fingerprint = ""
    eng._last_entry_at = 0.0
    eng._entry_placed_this_loop = False
    return eng


def test_overnight_blocks_entry_while_leftover_sell_stop(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    leftover = [{
        "user_tag": "bot_orphan_sl",
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551910",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=leftover)
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=0)
    rithmic.place_order = MagicMock()
    rithmic.flatten_symbol = MagicMock()
    eng = _overnight_open_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    OvernightPaperEngine._open_paper_position(eng, _overnight_open_sig())
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_pre_entry", already_flat=True,
    )
    rithmic.place_order.assert_not_called()
    rithmic.flatten_symbol.assert_not_called()
    assert eng.position is None


def test_overnight_blocks_entry_while_leftover_buy_stop(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    leftover = [{
        "user_tag": "bot_orphan_sl",
        "transaction_type": "BUY",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551911",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=leftover)
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=0)
    rithmic.place_order = MagicMock()
    eng = _overnight_open_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    OvernightPaperEngine._open_paper_position(eng, _overnight_open_sig(side="short"))
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_pre_entry", already_flat=True,
    )
    rithmic.place_order.assert_not_called()
    assert eng.position is None


def test_overnight_no_new_entry_until_working_book_empty(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    leftover = [{
        "user_tag": "bot_orphan_sl",
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551910",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(side_effect=[leftover, [], []])
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=1)
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_new", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.increment_zone_touch",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.write_zones",
        lambda *a, **k: None,
    )
    OvernightPaperEngine._open_paper_position(eng, _overnight_open_sig())
    rithmic.cancel_working_orders_after_exit.assert_called_once_with(
        "MNQ", reason="overnight_pre_entry", already_flat=True,
    )
    rithmic.place_order.assert_called_once()
    assert eng.position is not None
    assert eng.position.get("lucid_ticket") == "bot_new"


def test_overnight_maybe_enter_skips_place_while_working_stop(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    leftover = [{
        "user_tag": "bot_orphan_sl",
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551910",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=leftover)
    rithmic.cancel_working_orders_after_exit = MagicMock(return_value=0)
    rithmic.place_order = MagicMock()
    eng = _overnight_open_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: _overnight_open_sig(),
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()
    assert eng.position is None


def test_overnight_sync_lucid_bracket_exit_closes_on_sl_fill(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.confirm_bracket_exit_fill = MagicMock(return_value={
        "confirmed": True,
        "exit_price": 29558.25,
        "leg": "SL",
        "fill_from_broker": True,
    })
    rithmic.acknowledge_flat_position = MagicMock()
    rithmic._broker_symbol_has_position = MagicMock(return_value=False)
    eng = _overnight_close_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    df = pd.DataFrame()
    assert OvernightPaperEngine._sync_lucid_bracket_exit(eng, df, eng.position) is True
    rithmic.confirm_bracket_exit_fill.assert_called()
    rithmic.acknowledge_flat_position.assert_called_once_with("bot_x", "MNQ")
    eng.journal.log_close.assert_called_once()
    assert eng.position is None


def test_overnight_sync_lucid_bracket_exit_waits_when_broker_open():
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.confirm_bracket_exit_fill = MagicMock(return_value=None)
    rithmic._broker_symbol_has_position = MagicMock(return_value=True)
    eng = _overnight_close_engine(rithmic)
    assert OvernightPaperEngine._sync_lucid_bracket_exit(eng, pd.DataFrame(), eng.position) is False
    assert eng.position is not None
    assert eng._broker_flat_scans == 0


def test_manage_open_prefers_lucid_sync_over_ticker_sl(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.confirm_bracket_exit_fill = MagicMock(return_value={
        "confirmed": True,
        "exit_price": 29558.25,
        "leg": "SL",
    })
    rithmic.acknowledge_flat_position = MagicMock()
    rithmic._broker_symbol_has_position = MagicMock(return_value=False)
    eng = _overnight_close_engine(rithmic)
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.attach_mae_mfe",
        lambda rec, *a, **k: rec,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._rithmic_last",
        lambda feed: 29540.0,
    )
    OvernightPaperEngine._manage_open(eng, pd.DataFrame(), datetime.now(ET))
    assert eng.position is None
    eng.journal.log_close.assert_called_once()


def test_overnight_addon_confidence_helpers(monkeypatch):
    from src.strategy.overnight_research import (
        allow_overnight_addon,
        allow_overnight_entry_confidence,
        overnight_addon_confidence_min,
        overnight_max_open,
        overnight_min_confidence,
        overnight_open_slots_full,
    )

    monkeypatch.delenv("OVERNIGHT_ADDON_CONFIDENCE_MIN", raising=False)
    monkeypatch.delenv("OVERNIGHT_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("OVERNIGHT_MAX_OPEN", raising=False)
    assert overnight_min_confidence() == 0
    assert overnight_addon_confidence_min() == 80
    assert overnight_max_open() == 2
    assert allow_overnight_entry_confidence(42) is True
    assert allow_overnight_entry_confidence(0) is True
    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "65")
    assert allow_overnight_entry_confidence(64) is False
    assert allow_overnight_entry_confidence(65) is True
    assert allow_overnight_entry_confidence(80) is True
    monkeypatch.delenv("OVERNIGHT_MIN_CONFIDENCE", raising=False)
    assert allow_overnight_addon(0, 50) is True
    assert allow_overnight_addon(1, 79) is False
    assert allow_overnight_addon(1, 80) is True
    assert overnight_open_slots_full(1) is False
    assert overnight_open_slots_full(2) is True


def test_overnight_min_confidence_blocks_first_entry(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[])
    rithmic.place_order = MagicMock()
    eng = _overnight_open_engine(rithmic)
    sig = _overnight_open_sig(
        entry_confidence_pct=64,
        entry_confidence={"pct": 64, "summary": "partial HTF"},
    )
    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "65")
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (0, 0),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 64, "summary": "partial HTF"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.should_defer_entry_for_stack",
        lambda *a, **k: False,
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()
    assert eng._open_count() == 0
    eng._emit_throttled.assert_called()
    assert any(
        "overnight-min-confidence" in str(c)
        for c in eng._emit_throttled.call_args_list
    ) or any(
        "skip entry" in str(c) for c in eng._emit_throttled.call_args_list
    )
    assert any(
        "64%" in str(c) and "65%" in str(c)
        for c in eng._emit_throttled.call_args_list
    )


def test_overnight_min_confidence_allows_at_threshold(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[])
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_conf65", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    sig = _overnight_open_sig(
        entry_confidence_pct=65,
        entry_confidence={"pct": 65, "summary": "HTF aligned"},
        zone_price=29740.0,
    )
    monkeypatch.setenv("OVERNIGHT_MIN_CONFIDENCE", "65")
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (1, 1),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 65, "summary": "HTF aligned"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.should_defer_entry_for_stack",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_stack_gate",
        lambda *a, **k: {"qty": 1, "last_ok": True, "close_ok": True, "trend_ok": True,
                         "trend_15m": 1, "trend_60m": 1},
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_called()
    assert eng._open_count() == 1


def test_overnight_addon_blocked_when_confidence_low(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[{
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
    }])
    eng = _overnight_open_engine(rithmic)
    eng.position = {
        "trade_id": "bot_first",
        "lucid_ticket": "bot_first",
        "side": "long",
        "qty": 1,
        "entry_price": 29700.0,
        "stop": 29660.0,
        "target": 29740.0,
        "cue": "break_onh",
    }
    sig = _overnight_open_sig(
        entry_confidence_pct=65,
        entry_confidence={"pct": 65, "summary": "partial HTF"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (0, 0),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 65, "summary": "partial HTF"},
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()
    assert eng._open_count() == 1
    eng._emit_throttled.assert_called()


def test_overnight_addon_allowed_when_confidence_high(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    bracket = [{
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551910",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=bracket)
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_second", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    eng.position = {
        "trade_id": "bot_first",
        "lucid_ticket": "bot_first",
        "side": "long",
        "qty": 1,
        "entry_price": 29700.0,
        "stop": 29660.0,
        "target": 29740.0,
        "cue": "break_onh",
    }
    sig = _overnight_open_sig(
        cue="break_onl",
        side="short",
        zone_name="overnight_low",
        zone_price=29680.0,
        entry_confidence_pct=85,
        entry_confidence={"pct": 85, "summary": "HTF aligned"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29750.0,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (1, 1),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 85, "summary": "HTF aligned"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.increment_zone_touch",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.write_zones",
        lambda *a, **k: None,
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.cancel_working_orders_after_exit.assert_not_called()
    rithmic.place_order.assert_called_once()
    assert eng._open_count() == 2
    tickets = {p.get("lucid_ticket") for p in eng.positions}
    assert tickets == {"bot_first", "bot_second"}


def test_overnight_max_open_blocks_third_entry(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    eng = _overnight_open_engine(rithmic)
    eng.positions = [
        {"trade_id": "bot_a", "lucid_ticket": "bot_a", "side": "long", "qty": 1,
         "entry_price": 1.0, "stop": 0.0, "target": 2.0, "cue": "break_onh"},
        {"trade_id": "bot_b", "lucid_ticket": "bot_b", "side": "short", "qty": 1,
         "entry_price": 1.0, "stop": 0.0, "target": 2.0, "cue": "break_onl"},
    ]
    monkeypatch.setenv("OVERNIGHT_MAX_OPEN", "2")
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()
    assert eng._open_count() == 2


def test_overnight_same_cue_addon_blocked_at_99_confidence(monkeypatch):
    """Reproduce 3:47 PM bug: two break_onl add-ons at 92%+ must not stack."""
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    bracket = [{
        "transaction_type": "SELL",
        "price_type": "STOP",
        "status": "STOP",
        "basket_id": "551910",
    }]
    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=bracket)
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_dup", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    eng.positions = [{
        "trade_id": "bot_first",
        "lucid_ticket": "bot_first",
        "side": "short",
        "qty": 2,
        "entry_price": 29513.5,
        "stop": 29553.5,
        "target": 29473.5,
        "cue": "break_onl",
        "zone_name": "overnight_low",
        "zone_price": 29523.5,
    }]
    sig = _overnight_open_sig(
        cue="break_onl",
        side="short",
        zone_name="overnight_low",
        zone_price=29523.5,
        qty=2,
        entry_confidence_pct=99,
        entry_confidence={"pct": 99, "summary": "full stack"},
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29508.25,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (1, 1),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 99, "summary": "full stack"},
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()
    assert eng._open_count() == 1
    eng._emit_throttled.assert_called()


def test_overnight_dedupe_blocks_rapid_same_cue(monkeypatch):
    from unittest.mock import MagicMock
    import time

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[])
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_dup", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    eng._last_entry_fingerprint = "break_onl|short|29523.5"
    eng._last_entry_at = time.time()
    sig = _overnight_open_sig(
        cue="break_onl",
        side="short",
        zone_name="overnight_low",
        zone_price=29523.5,
    )
    monkeypatch.setenv("OVERNIGHT_ENTRY_DEDUPE_SEC", "120")
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_cap_reached",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.in_overnight_cooldown",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner._last_price",
        lambda *a, **k: 29508.25,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.evaluate_overnight_cue",
        lambda *a, **k: sig,
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_htf_sides",
        lambda df: (1, 1),
    )
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.entry_confidence_pct",
        lambda *a, **k: {"pct": 99, "summary": "full stack"},
    )
    eng._reset_session_if_needed = lambda now: None
    OvernightPaperEngine._maybe_enter(eng, pd.DataFrame(), datetime.now(ET))
    rithmic.place_order.assert_not_called()


def test_overnight_open_paper_position_hard_blocks_at_cap(monkeypatch):
    from unittest.mock import MagicMock

    from src.strategy.overnight_paper_runner import OvernightPaperEngine

    rithmic = MagicMock()
    rithmic.list_working_orders = MagicMock(return_value=[])
    rithmic.place_order = MagicMock(
        return_value={"ticket": "bot_extra", "native_stop_attached": True},
    )
    eng = _overnight_open_engine(rithmic)
    eng.positions = [
        {"trade_id": "bot_a", "lucid_ticket": "bot_a", "side": "short", "qty": 2,
         "entry_price": 1.0, "stop": 2.0, "target": 0.0, "cue": "break_onl"},
        {"trade_id": "bot_b", "lucid_ticket": "bot_b", "side": "short", "qty": 2,
         "entry_price": 1.0, "stop": 2.0, "target": 0.0, "cue": "break_onh"},
    ]
    monkeypatch.setenv("OVERNIGHT_MAX_OPEN", "2")
    monkeypatch.setattr(
        "src.strategy.overnight_paper_runner.overnight_entry_blocked_reason",
        lambda now=None: None,
    )
    OvernightPaperEngine._open_paper_position(eng, _overnight_open_sig())
    rithmic.place_order.assert_not_called()
    assert eng._open_count() == 2

