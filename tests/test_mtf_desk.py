"""Desk UX helpers: bot running vs desk-only, Gemini on/off, collapsed asks."""
from datetime import datetime, timedelta, timezone

from src.dashboard.mtf_actions import (
    ask_looks_successful,
    bot_run_state,
    build_plain_english,
    build_profit_card,
    build_signal_card,
    build_status_card,
    collapse_gemini_questions,
    entry_windows_display,
    gemini_desk_status,
    last_ask_ok_from_history,
    todays_live_trade_count,
)


def test_desk_only_when_status_missing():
    run = bot_run_state({})
    assert run["state"] == "desk_only"
    assert run["bot_running"] is False
    assert "desk only" in run["detail"]


def test_paper_bot_when_status_fresh():
    now = datetime(2026, 8, 18, 2, 10, tzinfo=timezone.utc)
    run = bot_run_state(
        {
            "running": True,
            "paper_mode": True,
            "mode": "paper",
            "updated_at": now.isoformat(),
        },
        now=now,
    )
    assert run["state"] == "paper"
    assert run["bot_running"] is True
    assert run["pill"] == "PAPER BOT"


def test_stale_after_30s_is_not_running():
    now = datetime(2026, 8, 18, 2, 10, tzinfo=timezone.utc)
    old = (now - timedelta(seconds=31)).isoformat()
    run = bot_run_state(
        {"running": True, "paper_mode": True, "mode": "paper", "updated_at": old},
        now=now,
    )
    assert run["bot_running"] is False
    assert run["pill"] == "not running"
    assert run["updated_label"] == "status stale"


def test_stale_status_is_desk_only():
    now = datetime(2026, 8, 18, 2, 10, tzinfo=timezone.utc)
    old = (now - timedelta(minutes=10)).isoformat()
    run = bot_run_state(
        {"paper_mode": False, "mode": "live", "updated_at": old},
        now=now,
    )
    assert run["state"] == "desk_only"
    assert run["bot_running"] is False


def test_gemini_advisory_on_after_successful_ask():
    st = gemini_desk_status(enabled=True, mode="advisory", last_ask_ok=True)
    assert st["label"] == "advisory / on"
    assert st["on"] is True


def test_gemini_off_when_disabled():
    st = gemini_desk_status(enabled=False, mode="advisory", last_ask_ok=True)
    assert st["label"] == "off"
    assert st["on"] is False


def test_ask_success_and_unavailable():
    assert ask_looks_successful("In-sample PF 1.41 / +$5,643")
    assert not ask_looks_successful("Gemini/LLM is off. Set LLM_ENABLED=true")
    assert not ask_looks_successful("Gemini unavailable: 404 Client Error")


def test_collapse_consecutive_identical_questions():
    rows = [
        {"kind": "gemini_chat", "reason": "is it profitable?", "answer": "a", "ts": "1"},
        {"kind": "gemini_chat", "reason": "is it profitable?", "answer": "b", "ts": "2"},
        {"kind": "entry", "reason": "ema15"},
        {"kind": "gemini_chat", "reason": "Are we profitable?", "answer": "c", "ts": "3"},
    ]
    out = collapse_gemini_questions(rows, n=6)
    assert len(out) == 2
    assert out[0]["repeat"] == 2
    assert out[0]["answer"] == "b"
    assert out[1]["question"] == "Are we profitable?"


def test_last_ask_from_history_when_memory_empty():
    ok = last_ask_ok_from_history(
        [{"answer": "Based on the locked backtest"}],
        None,
    )
    assert ok is True


def test_plain_english_mentions_both_recipes():
    plain = build_plain_english(
        run={"state": "desk_only", "label": "not trading yet", "detail": "desk only / no live bot", "bot_running": False},
        status={},
        snapshot={"n_all": 231, "all": {"overall": {"n": 231, "pf": 1.557}}},
        live_empty=True,
        suggestions={"ready": False},
        gemini={"label": "advisory / on", "on": True},
        facts={"is_pf": 1.41, "is_pnl_usd": 5643, "oos_pf": 1.59, "oos_pnl_usd": 5371},
    )
    assert "Not trading yet" in plain["live"]
    assert "9:35" in plain["backtest"]
    assert "13:30 loses" in plain["backtest"]
    assert "30 real closes" in plain["suggestions"]
    assert "6-window" in plain["recipe_note"]
    assert "3-window" in plain["recipe_note"]
    assert "1.41" in plain["recipe_note"]
    assert "231" in plain["recipe_note"]


def test_profit_card_uses_3_window_not_6():
    snap = {
        "n_all": 231,
        "all": {"overall": {"n": 231, "pf": 1.557, "total_pnl_usd": 9820.7}},
        "oos": {"overall": {"pf": 1.531}},
    }
    card = build_profit_card(snap, {"is_pf": 1.41, "oos_pf": 1.59})
    assert card["yes"] is True
    assert card["trades"] == 231
    assert card["pf"] == 1.56
    assert card["oos_pf"] == 1.53
    assert "1.41" in card["footnote"]
    assert "1.59" in card["footnote"]


def test_signal_overnight_leads_with_paper_long_not_rth_wait():
    packet = {
        "levels": {
            "callout": "WAIT",
            "waiting": True,
            "first_line": "WAIT — last MNQ 30215.00 as of 2026-08-16 19:59 ET (file, not live). Next window 09:35 ET.",
            "last_price": 30215.0,
            "stop": 30175.0,
            "next_window": "09:35",
        }
    }
    status = {
        "overnight_research": True,
        "session": "overnight_research",
        "last_price": 29920.25,
        "quote_source": "rithmic",
        "last_quote_ts_et": "2026-08-18 00:36 ET",
        "open_positions": [{
            "symbol": "MNQ",
            "direction": "LONG",
            "entry": 29909.75,
            "sl": 29849.75,
            "size": 1,
        }],
    }
    sig = build_signal_card(packet, [{"answer": "WAIT — last MNQ 30215.00"}], status=status, run={"state": "overnight_research"})
    assert sig["overnight"] is True
    assert "PAPER LONG" in sig["callout"]
    assert "29920.25" in sig["headline"]
    assert "30215" not in sig["headline"]
    assert "09:35" not in sig["headline"]
    assert sig["stop"] == 29849.75
    assert sig["stop_pts"] == 60.0
    assert sig["take_profit"] == "1R TP, flatten 09:25 ET"
    assert sig["gemini_callout"] is None
    assert "Day recipe" in sig["rth_recipe_note"]


def test_signal_card_shows_price_stop_window():
    packet = {
        "price_stale": True,
        "session": "closed",
        "levels": {
            "callout": "WAIT",
            "waiting": True,
            "first_line": "WAIT — last MNQ 24812.25 as of 16:00 ET (file, not live). Next window 09:35 ET.",
            "last_price": 24812.25,
            "stop": 24770.25,
            "stop_pts": 42.0,
            "next_window": "09:35",
            "last_bar_ts_et": "2026-08-14 16:00 ET",
        },
    }
    sig = build_signal_card(packet, [])
    assert sig["last_price"] == 24812.25
    assert sig["stop"] == 24770.25
    assert sig["next_window"] == "09:35"
    assert "24812.25" in sig["headline"]
    wins = entry_windows_display()
    assert any("9:35" in w for w in wins)
    assert any("1:30" in w for w in wins)
    assert len(wins) == 6


def test_status_card_overnight_stop_and_target():
    card = build_status_card(
        {"state": "overnight_research", "bot_running": True},
        {
            "daily_pnl": 0,
            "open_positions": [{
                "symbol": "MNQ",
                "direction": "LONG",
                "entry": 29909.75,
                "sl": 29849.75,
                "size": 1,
            }],
            "last_price": 29897.5,
            "quote_source": "rithmic",
            "quote_age_seconds": 1,
        },
        {"empty": True},
        {},
        levels={"quote_age_min": 0, "refresh_ok": True},
    )
    rows = dict(card["rows"])
    assert rows["Stop"] == "Stop 29,849.75 (60 pts)"
    assert rows["Target"] == "1R TP, flatten 09:25 ET"


def test_status_card_has_quote_age_not_desk_only_banner():
    card = build_status_card(
        {"state": "desk_only", "bot_running": False},
        {},
        {"empty": True},
        {"label": "advisory / ready"},
        levels={"quote_age_min": 47, "refresh_ok": True},
    )
    labels = [r[0] for r in card["rows"]]
    assert labels == ["Bot", "Open", "Daily P&L", "Live trades", "Quote age"]
    assert card["headline"] == "not running"
    assert "Desk Only" not in card["headline"]
    assert "47 min (stale)" in dict(card["rows"])["Quote age"]
    assert dict(card["rows"])["Live trades"] == "0"


def test_status_card_shows_rithmic_note():
    card = build_status_card(
        {"state": "paper", "bot_running": True},
        {"last_note": "Rithmic: connected", "daily_pnl": 0, "open_positions": []},
        {"empty": True},
        {},
        levels={"quote_age_min": 1, "refresh_ok": True},
    )
    assert dict(card["rows"])["Note"] == "Rithmic: connected"


def test_live_trades_uses_today_count_not_lifetime_journal():
    now = datetime(2026, 8, 18, 2, 10, tzinfo=timezone.utc)
    run = {"state": "paper", "bot_running": True}
    card = build_status_card(
        run,
        {"daily_pnl": 0, "live_trades_today": 0, "open_positions": []},
        {"empty": False, "overall": {"n": 1080}},
        {},
        levels={"quote_age_min": 3, "refresh_ok": True},
    )
    assert dict(card["rows"])["Live trades"] == "0"
    assert dict(card["rows"])["Bot"] == "paper bot running"
    assert todays_live_trade_count({"live_trades_today": 0}, [{"event": "close", "source": "x"}] * 1080, now=now) == 0


def test_status_card_download_failed():
    card = build_status_card(
        {"state": "desk_only", "bot_running": False},
        {},
        {"empty": True},
        {},
        levels={"refresh_ok": False, "refresh_error": "Databento download failed: timeout"},
        refresh={"ok": False, "error": "Databento download failed: timeout"},
    )
    age = dict(card["rows"])["Quote age"]
    assert "download failed" in age
    assert "not current" in age


def test_signal_hides_file_print_when_refresh_fails():
    packet = {
        "levels": {
            "callout": "WAIT",
            "waiting": True,
            "first_line": "WAIT — Databento download failed. Last file print is not current.",
            "last_price": 30215.0,
            "refresh_ok": False,
            "next_window": "09:35",
        }
    }
    sig = build_signal_card(packet, [])
    assert sig["last_price"] is None
    assert sig["callout"] == "WAIT"
    assert "not current" in sig["headline"]
