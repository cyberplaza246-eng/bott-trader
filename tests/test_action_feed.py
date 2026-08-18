"""Human activity feed: emoji/title/lines, today split, throttle."""
import os
import tempfile
import time
from datetime import datetime, timezone

from src.ai.action_log import (
    ActionLog,
    build_today_activity,
    format_human_event,
    humanize_action,
    looks_like_dev,
    open_position_card,
)


def test_humanize_overnight_paper_short():
    rec = humanize_action({
        "ts": "2026-08-18T04:17:00+00:00",
        "kind": "entry",
        "symbol": "MNQ",
        "direction": "short",
        "entry": 29899.25,
        "stop": 29959.25,
        "qty": 1,
        "session": "overnight_research",
        "reason": "test_fill — paper only, not sent to Lucid",
        "cue": "test_fill",
    })
    assert rec["emoji"] == "🟢"
    assert rec["clock"] == "12:17 AM"
    assert "PAPER SHORT" in rec["title"]
    assert rec["layer"] == "human"
    joined = " ".join(rec["lines"])
    assert "29,899.25" in joined
    assert "29,959.25" in joined
    assert "60 pts" in joined
    assert "1R TP, flatten 09:25 ET" in joined
    assert "Paper only" in joined


def test_humanize_day_entry_shows_sl_tp_placed():
    rec = humanize_action({
        "kind": "entry",
        "symbol": "MNQ",
        "direction": "long",
        "entry": 25000.0,
        "stop": 24960.0,
        "tp": 25500.0,
        "tp_cap": 25500.0,
        "session": "ema15_eod",
        "qty": 1,
    })
    joined = " ".join(rec["lines"])
    assert "SL placed @ 24,960.00" in joined
    assert "TP cap @ 25,500.00" in joined


def test_humanize_upgrades_old_lines_with_flatten_tp():
    rec = humanize_action({
        "kind": "entry",
        "symbol": "MNQ",
        "direction": "long",
        "entry": 29909.75,
        "stop": 29849.75,
        "session": "overnight_research",
        "paper": True,
        "lines": ["MNQ ×1 @ 29,909.75", "Stop 29,849.75", "🧪 Paper only"],
    })
    joined = " ".join(rec["lines"])
    assert "Stop 29,849.75 (60 pts)" in joined
    assert "1R TP, flatten 09:25 ET" in joined


def test_humanize_profit_and_loss():
    win = humanize_action({
        "ts": "2026-08-18T05:42:00+00:00",
        "kind": "profit",
        "symbol": "MNQ",
        "direction": "short",
        "entry": 29899.25,
        "exit": 29828.00,
        "pnl": 142.50,
        "paper": True,
    })
    assert win["emoji"] == "💰"
    assert "PROFITABLE" in win["title"]
    assert any("+$142.50" in x for x in win["lines"])
    lose = humanize_action({
        "kind": "loss",
        "symbol": "MNQ",
        "direction": "short",
        "entry": 29899.25,
        "exit": 29959.25,
        "pnl": -120.0,
        "paper": True,
    })
    assert lose["emoji"] == "🔴"
    assert "LOSING" in lose["title"]


def test_skip_shows_human_reason():
    rec = humanize_action({
        "kind": "skip",
        "reason": "15m and Daily disagree",
    })
    assert rec["emoji"] == "❌"
    assert rec["title"] == "TRADE SKIPPED"
    assert rec["lines"] == ["15m and Daily disagree"]


def test_forced_logout_raw_is_developer_connection_is_human():
    raw = {"kind": "dev", "reason": "ForcedLogout rpCode 13 GET /api/desk"}
    assert looks_like_dev(raw) is True
    conn = humanize_action({
        "kind": "connection",
        "title": "CONNECTION",
        "lines": ["Reconnected successfully"],
        "reason": "ForcedLogout recovered",
    })
    assert conn["layer"] == "human"
    assert conn["emoji"] == "🔄"


def test_format_human_event_multiline():
    text = format_human_event({
        "ts": "2026-08-18T04:18:00+00:00",
        "kind": "monitoring",
        "lines": ["Short still open · MNQ 29,898.00"],
    })
    assert "🟡" in text
    assert "MONITORING" in text
    assert "Short still open" in text


def test_open_card_from_status():
    card = open_position_card({
        "paper_mode": True,
        "overnight_research": True,
        "updated_at": "2026-08-18T04:22:00+00:00",
        "open_positions": [{
            "symbol": "MNQ",
            "direction": "SHORT",
            "entry": 29899.25,
            "sl": 29959.25,
            "size": 1,
            "cue": "test_fill",
        }],
    })
    assert card is not None
    assert card["emoji"] == "🟢"
    assert "PAPER TRADE" in card["title"]
    joined = " ".join(card["lines"])
    assert "SHORT / MNQ · 1" in joined
    assert "Entry 29,899.25" in joined
    assert "29,959.25" in joined
    assert "60 pts" in joined
    assert "1R TP, flatten 09:25 ET" in joined
    assert "not sent to Rithmic" in joined


def test_today_activity_splits_and_injects_monitor():
    now = datetime(2026, 8, 18, 4, 22, tzinfo=timezone.utc)
    actions = [
        {
            "ts": "2026-08-18T04:17:00+00:00",
            "kind": "entry",
            "symbol": "MNQ",
            "direction": "short",
            "entry": 29899.25,
            "session": "overnight_research",
        },
        {
            "ts": "2026-08-18T04:20:00+00:00",
            "kind": "dev",
            "reason": "Traceback (most recent call last): ForcedLogout rpCode 13",
            "layer": "dev",
        },
        {"ts": "2026-08-18T04:21:00+00:00", "kind": "gemini_chat", "reason": "hi"},
    ]
    status = {
        "paper_mode": True,
        "overnight_research": True,
        "open": "SHORT",
        "last_price": 29890.0,
        "updated_at": "2026-08-18T04:22:00+00:00",
        "open_positions": [{
            "symbol": "MNQ", "direction": "SHORT", "entry": 29899.25, "sl": 29959.25, "size": 1,
        }],
    }
    feed = build_today_activity(actions, status, now=now)
    assert feed["card"]["title"].startswith("PAPER TRADE")
    kinds = [e["kind"] for e in feed["events"]]
    assert "entry" in kinds
    assert "monitoring" in kinds
    assert all(e.get("kind") != "dev" for e in feed["events"])
    assert any(e.get("kind") == "dev" for e in feed["developer"])
    assert all(e.get("kind") != "gemini_chat" for e in feed["events"])


def test_record_throttled_skips_heartbeats():
    with tempfile.TemporaryDirectory() as td:
        log = ActionLog(path=os.path.join(td, "bot_actions.jsonl"))
        a = log.record_throttled("monitoring", key="open", interval_sec=60, lines=["a"])
        b = log.record_throttled("monitoring", key="open", interval_sec=60, lines=["b"])
        assert a is not None
        assert b is None
        time.sleep(0.05)
        c = log.record_throttled("monitoring", key="open", interval_sec=0.01, lines=["c"])
        assert c is not None
        assert len(log.recent(10)) == 2
