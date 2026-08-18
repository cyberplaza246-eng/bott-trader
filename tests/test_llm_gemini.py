"""Gemini advisor URL/key resolution and action-log append (no network)."""
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.action_log import ActionLog, read_status, write_status
from src.ai.llm_advisor import (
    DEFAULT_GEMINI_BASE,
    LLMTradeAdvisor,
    format_locked_backtest_answer,
)


def test_gemini_resolves_key_url_and_model():
    env = {
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-gemini-key",
        "LLM_MODEL": "",
        "LLM_BASE_URL": "",
        "GOOGLE_API_KEY": "",
        "LLM_API_KEY": "",
        "OPENAI_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        for key in ("LLM_MODEL", "LLM_BASE_URL", "GOOGLE_API_KEY", "LLM_API_KEY"):
            os.environ.pop(key, None)
        adv = LLMTradeAdvisor()
        assert adv.provider == "gemini"
        assert adv.api_key == "test-gemini-key"
        assert adv.model == "gemini-3.5-flash"
        assert adv.base_url == DEFAULT_GEMINI_BASE
        assert "/openai" not in adv.base_url
        url = adv._gemini_generate_url()
        assert url == f"{DEFAULT_GEMINI_BASE}/models/gemini-3.5-flash:generateContent"
        assert "/openai/chat/completions" not in url


def test_gemini_falls_back_to_google_api_key():
    env = {
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "google",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "from-google",
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
    }
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("LLM_BASE_URL", None)
        adv = LLMTradeAdvisor()
        assert adv.api_key == "from-google"
        assert adv.base_url == DEFAULT_GEMINI_BASE
        assert "/openai" not in adv.base_url


def test_gemini_ignores_openai_compat_base_url():
    env = {
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "k",
        "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
    }
    with patch.dict(os.environ, env, clear=False):
        adv = LLMTradeAdvisor()
        assert adv.base_url == DEFAULT_GEMINI_BASE
        assert "/openai" not in adv._gemini_generate_url()


def test_gemini_uses_native_generate_content():
    env = {
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "k",
        "LLM_MODEL": "gemini-3.5-flash",
        "LLM_BASE_URL": "",
    }
    captured = {}

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"action":"allow"}'}]}}]}

    def fake_post(url, headers=None, json=None, timeout=None, params=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers or {}
        captured["params"] = params
        return _Resp()

    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("LLM_BASE_URL", None)
        adv = LLMTradeAdvisor()
        with patch("src.ai.llm_advisor.requests.post", side_effect=fake_post):
            out = adv._call_api("ping")
    assert out == '{"action":"allow"}'
    assert captured["url"].endswith("/models/gemini-3.5-flash:generateContent")
    assert "/openai/" not in captured["url"]
    assert "response_format" not in (captured["payload"] or {})
    assert captured["headers"].get("x-goog-api-key") == "k"
    assert "contents" in captured["payload"]


def test_ask_profit_question_falls_back_without_leaking_key():
    env = {
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "AQ.secret-test-key-value",
        "LLM_MODEL": "gemini-2.5-flash",
    }

    class _Resp:
        status_code = 404
        reason = "Not Found"
        text = '{"error":{"message":"no longer available"}}'
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key=AQ.secret-test-key-value"

    def fake_post(url, headers=None, json=None, timeout=None, params=None):
        return _Resp()

    with patch.dict(os.environ, env, clear=False):
        adv = LLMTradeAdvisor()
        with patch("src.ai.llm_advisor.requests.post", side_effect=fake_post):
            answer = adv.ask("is it profitable?")
    assert "BACKTEST" in answer
    assert "1.41" in answer
    assert "5,643" in answer or "5643" in answer
    assert "1.59" in answer
    assert "AQ.secret" not in answer
    assert "openai/chat/completions" not in answer or "generateContent" in answer


def test_locked_backtest_answer_flags_not_live():
    text = format_locked_backtest_answer()
    assert "BACKTEST" in text
    assert "not live" in text.lower()
    assert "1.41" in text
    assert "1.59" in text


def test_recipe_levels_first_line_has_real_prices():
    from src.ai.llm_advisor import compute_recipe_levels, ensure_levels_in_answer

    packet = {
        "price_1m": 24812.25,
        "atr_stop_pts": 42.0,
        "trend_15m": 1,
        "daily_permission": 1,
        "trend_60m": 1,
        "sep15": 0.6,
        "in_window": False,
        "window_index": None,
        "session": "closed",
        "price_stale": True,
        "last_bar_ts_et": "2026-08-14 16:00 ET",
        "next_window_name": "09:35",
        "ema15_fast": 24810.0,
        "ema15_slow": 24790.0,
        "daily_permission_txt": "long",
    }
    lv = compute_recipe_levels(packet)
    assert lv["callout"] == "WAIT"
    assert lv["last_price"] == 24812.25
    assert lv["entry"] == 24812.25
    assert lv["stop"] == 24770.25
    line = lv["first_line"]
    assert "24812.25" in line
    assert "24770.25" in line
    assert "09:35" in line
    assert "16:00" in line
    generic = "WAIT — enter at market when the window opens if EMA8 > EMA21."
    fixed = ensure_levels_in_answer(generic, lv)
    assert "24812.25" in fixed
    assert fixed.startswith("WAIT")


def test_ask_enter_price_offline_uses_snapshot_numbers():
    env = {
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-not-used",
    }
    market = {
        "price_1m": 24812.25,
        "atr_stop_pts": 42.0,
        "trend_15m": -1,
        "daily_permission": -1,
        "trend_60m": -1,
        "sep15": 0.5,
        "in_window": False,
        "session": "closed",
        "price_stale": True,
        "last_bar_ts_et": "2026-08-14 16:00 ET",
        "next_window_name": "09:35",
        "ema15_fast": 24700.0,
        "ema15_slow": 24750.0,
    }
    with patch.dict(os.environ, env, clear=False):
        adv = LLMTradeAdvisor()
        answer = adv.ask("at what price do I enter?", {"market": market})
    assert "24812.25" in answer
    assert "WAIT" in answer
    assert "I bought" not in answer.lower()
    assert "order sent" not in answer.lower()


def test_stale_quote_first_line_leads_with_age():
    from src.ai.llm_advisor import compute_recipe_levels

    packet = {
        "price_1m": 25100.5,
        "atr_stop_pts": 40.0,
        "trend_15m": 1,
        "daily_permission": 1,
        "in_window": False,
        "session": "closed",
        "price_stale": True,
        "last_bar_ts_et": "2026-08-17 22:00 ET",
        "last_bar_age_sec": 47 * 60,
        "next_window_name": "09:35",
    }
    lv = compute_recipe_levels(packet)
    assert lv["first_line"].startswith("WAIT — quote is 47 min old — last MNQ 25100.50")
    assert "231" not in lv["first_line"]
    assert lv["first_line"].index("25100.50") > lv["first_line"].index("47 min")


def test_overnight_packet_uses_live_ticker_not_file():
    from src.ai.llm_advisor import apply_overnight_live_quote, overlay_overnight_levels

    status = {
        "overnight_research": True,
        "session": "overnight_research",
        "last_price": 29920.25,
        "quote_source": "rithmic",
        "quote_age_seconds": 0,
        "last_quote_ts_et": "2026-08-18 00:36 ET",
        "open_positions": [{
            "direction": "LONG",
            "entry": 29909.75,
            "sl": 29849.75,
        }],
    }
    stale = {
        "price_1m": 30215.0,
        "price_stale": True,
        "refresh_ok": False,
        "data_source": "file",
        "levels": {
            "callout": "WAIT",
            "first_line": "WAIT — last MNQ 30215.00 as of 2026-08-16 19:59 ET. Next window 09:35.",
            "last_price": 30215.0,
        },
    }
    out = apply_overnight_live_quote(stale, status)
    assert out["price_1m"] == 29920.25
    assert out["refresh_ok"] is True
    assert out["price_stale"] is False
    line = out["levels"]["first_line"]
    assert "PAPER LONG" in line
    assert "29920.25" in line
    assert "29849.75" in line
    assert "1R TP, flatten 09:25 ET" in line
    assert "30215" not in line
    assert "09:35" not in line
    ov = overlay_overnight_levels(stale["levels"], status, price=29920.25, source="rithmic")
    assert ov["stop"] == 29849.75
    assert ov["take_profit"] == "1R TP, flatten 09:25 ET"


def test_refresh_failed_hides_stale_file_price():
    from src.ai.llm_advisor import compute_recipe_levels

    packet = {
        "price_1m": 30215.0,
        "atr_stop_pts": 40.0,
        "refresh_ok": False,
        "refresh_error": "Databento download failed: timeout",
        "session": "closed",
        "price_stale": True,
        "in_window": False,
        "next_window_name": "09:35",
    }
    lv = compute_recipe_levels(packet)
    assert lv["last_price"] is None
    assert lv["first_line"].startswith("WAIT —")
    assert "30215" not in lv["first_line"]
    assert "not current" in lv["first_line"]


def test_action_log_appends_and_status_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        log_path = os.path.join(td, "bot_actions.jsonl")
        status_path = os.path.join(td, "bot_status.json")
        log = ActionLog(path=log_path)
        rec = log.record("entry", symbol="MNQ", direction="long", reason="ema15_eod")
        assert rec["kind"] == "entry"
        rows = log.recent(10)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "MNQ"
        write_status({"strategy": "ema15_eod", "paper_mode": True, "daily_pnl": 12.5}, path=status_path)
        status = read_status(status_path)
        assert status["strategy"] == "ema15_eod"
        assert status["paper_mode"] is True
        assert "updated_at" in status
