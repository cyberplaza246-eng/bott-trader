"""Overnight paper research: zones, journal tagging, cues. Never live RTH."""
from datetime import datetime
from pathlib import Path

import pandas as pd
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
    overnight_entry_blocked_reason,
    paper_test_fill_from_env,
    paper_test_fill_signal,
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


def test_research_hours_exclude_rth():
    rth = ET.localize(datetime(2026, 8, 17, 10, 0))
    globex = ET.localize(datetime(2026, 8, 17, 23, 10))
    assert is_overnight_research_hours(rth) is False
    assert is_overnight_research_hours(globex) is True
    assert should_flatten_before_rth(ET.localize(datetime(2026, 8, 18, 9, 25))) is True
    assert should_flatten_before_rth(globex) is False


def test_rth_1049_hard_idle_no_entry():
    now = ET.localize(datetime(2026, 8, 18, 10, 49))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    assert in_rth_hard_idle(now) is True
    assert day_bot_owns_lucid(now) is True
    assert allow_new_overnight_entry(now) is False
    assert is_overnight_research_hours(now) is False
    assert evaluate_overnight_cue(df, now, already_filled=False) is None
    reason = overnight_entry_blocked_reason(now) or ""
    assert "9:30" in reason or "RTH" in reason


def test_pre_globex_1644_no_entry():
    now = ET.localize(datetime(2026, 8, 18, 16, 44))
    df = _bars(
        "2026-08-17 18:00", 80, base=24700.0, high_at=40, high_px=24740.0,
        last_close=24740.75, last_high=24741.0,
    )
    assert in_rth_hard_idle(now) is False
    assert day_bot_owns_lucid(now) is True
    assert allow_new_overnight_entry(now) is False
    assert evaluate_overnight_cue(df, now) is None
    reason = overnight_entry_blocked_reason(now) or ""
    assert "18:00" in reason or "Globex" in reason or "Pre-Globex" in reason


def test_flatten_at_0925_not_0924():
    pos = {"side": "long", "stop": 24600.0, "target": 24800.0}
    at_flat = ET.localize(datetime(2026, 8, 18, 9, 25))
    before = ET.localize(datetime(2026, 8, 18, 9, 24))
    overnight = ET.localize(datetime(2026, 8, 17, 23, 10))
    assert should_flatten_before_rth(at_flat) is True
    assert should_flatten_before_rth(before) is False
    hit = check_paper_exit(pos, high=24700.0, low=24690.0, last=24695.0, now=at_flat)
    assert hit is not None
    assert hit["reason"] == "flatten_before_rth"
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
    assert sig["flatten_et"] == "09:25"
    assert TP_R == 1.0
    sl = float(sig["atr_stop_pts"])
    assert 12.0 <= sl <= 40.0
    assert sig["target"] == round(float(sig["entry_price"]) + sl, 2)


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
    assert rows["Target"] == "1R TP, flatten 09:25 ET"
    assert rows["Open"] == "MNQ LONG"
    assert rows["Live trades tonight"] == "2"
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
    assert "PAPER_OVERNIGHT" not in live
    assert "set OVERNIGHT_TRADING=true" not in live
    assert "--overnight-research" not in live


def test_globex_start_sunday_open():
    mon_0100 = ET.localize(datetime(2026, 8, 18, 1, 0))
    start = globex_session_start_et(mon_0100)
    assert start.hour == 18
    assert start.day == 17
