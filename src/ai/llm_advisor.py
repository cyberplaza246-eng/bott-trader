"""
LLM trade advisor — DeepSeek / OpenAI-compatible API.

Used as a context filter on top of rule-based MTF signals.
Does NOT replace risk management or entry rules.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from src.ai.mnq_context import build_mnq_context, compute_setup_score
from src.ai.economic_calendar import EconomicCalendar
from src.utils.logger import bot_logger

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


class LLMTradeAdvisor:
    """Ask an LLM whether a proposed MTF trade aligns with macro/context."""

    def __init__(self):
        self.enabled = os.getenv("LLM_ENABLED", "false").lower() == "true"
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        self.min_confidence = float(os.getenv("LLM_MIN_CONFIDENCE", "0.60"))
        self.min_setup_score = float(os.getenv("LLM_MIN_SETUP_SCORE", "75"))
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "12"))
        self.cache_ttl_sec = int(os.getenv("LLM_CACHE_TTL_SEC", "900"))  # 15 min
        self.model = os.getenv("LLM_MODEL", self._default_model())
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

        if self.enabled and not self.api_key:
            bot_logger.warning("LLM advisor enabled but no API key — advisor disabled")
            self.enabled = False
        elif self.enabled:
            bot_logger.info(
                f"LLM advisor active: provider={self.provider} model={self.model} "
                f"min_confidence={self.min_confidence}"
            )

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4o-mini"
        return "deepseek-chat"

    def _resolve_api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "").strip()
        return os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")).strip()

    def _resolve_base_url(self) -> str:
        explicit = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        if explicit:
            return explicit
        if self.provider == "openai":
            return DEFAULT_OPENAI_BASE
        return DEFAULT_DEEPSEEK_BASE

    def _cache_key(self, symbol: str, direction: str) -> str:
        hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        return f"{symbol}:{direction}:{hour_bucket}"

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _cache_key_historical(self, symbol: str, direction: str, dt_iso: str) -> str:
        return f"{symbol}:{direction}:{dt_iso}"

    def _build_prompt(self, signal: Dict[str, Any], context: Dict[str, Any]) -> str:
        direction = signal.get("direction", "").upper()
        symbol = signal.get("symbol", "MNQ")
        return f"""You are an expert MNQ micro futures scalping advisor.

A rule-based MTF system proposes {direction} on {symbol}.
Review the setup using session awareness, market structure, VWAP, volume, MTF alignment, and news risk.

Respond ONLY with JSON:
{{
  "action": "allow" or "skip",
  "bias": "bullish" or "bearish" or "neutral",
  "confidence": 0.0 to 1.0,
  "setup_score": 0 to 100,
  "market_type": "trend" or "range" or "chop",
  "position_size_pct": 0 or 50 or 100,
  "reason": "one short sentence",
  "warnings": ["optional list"]
}}

Guidelines:
- SKIP midday chop (11am-2pm ET) unless setup_score would be 90+.
- SKIP within 15-30 min of high-impact US news (CPI, NFP, FOMC).
- SKIP if 1m direction fights 5m/15m trend.
- SKIP chop/volatile_chop unless exceptional confluence.
- ALLOW ny_open and power_hour when trend-aligned, above/below VWAP, strong volume.
- setup_score: 95-100 full size, 80-94 normal, 60-79 half, below 60 skip.
- confidence = certainty in allow/skip decision (not just bullish/bearish).

Context:
{json.dumps(context, default=str, indent=2)}
"""

    def _call_api(self, prompt: str) -> Optional[str]:
        url = f"{self.base_url}/chat/completions"
        if self.base_url.endswith("/v1"):
            url = f"{self.base_url}/chat/completions"
        elif "/v1" not in self.base_url:
            url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You output strict JSON only for trade risk decisions.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _decide_from_parsed(self, parsed: Dict[str, Any], direction: str) -> Dict[str, Any]:
        action = str(parsed.get("action", "allow")).lower()
        bias = str(parsed.get("bias", "neutral")).lower()
        confidence = float(parsed.get("confidence", 0.5))
        setup_score = float(parsed.get("setup_score", confidence * 100))
        market_type = str(parsed.get("market_type", "range"))
        size_pct = int(parsed.get("position_size_pct", 100))
        reason = str(parsed.get("reason", "LLM response"))
        warnings = parsed.get("warnings") or []

        wants_long = direction.lower() in ("long", "buy")
        bias_conflicts = (wants_long and bias == "bearish") or (
            (not wants_long) and bias == "bullish"
        )

        allowed = (
            action == "allow"
            and confidence >= self.min_confidence
            and setup_score >= self.min_setup_score
            and not bias_conflicts
        )
        if action == "skip" or setup_score < self.min_setup_score:
            allowed = False

        return {
            "allowed": allowed,
            "action": action,
            "bias": bias,
            "confidence": confidence,
            "setup_score": setup_score,
            "market_type": market_type,
            "position_size_pct": size_pct if allowed else 0,
            "reason": reason,
            "warnings": warnings,
            "source": "llm",
        }

    def evaluate_trade(
        self,
        signal: Dict[str, Any],
        ctx_5m: Dict[str, Any],
        row_1m: Optional[Any] = None,
        rich_context: Optional[Dict[str, Any]] = None,
        df_1m=None,
        df_5m=None,
        df_15m=None,
        dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Review proposed trade; on API error returns allow (rules-only fallback)."""
        if not self.enabled:
            return {
                "allowed": True,
                "action": "allow",
                "confidence": 1.0,
                "setup_score": 100,
                "reason": "LLM disabled",
                "source": "fallback",
                "position_size_pct": 100,
            }

        symbol = signal.get("symbol", "MNQ")
        direction = signal.get("direction", "long")
        when = dt or datetime.now(timezone.utc)
        dt_iso = when.strftime("%Y%m%d%H%M")
        cache_key = self._cache_key_historical(symbol, direction, dt_iso)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        if rich_context is None and row_1m is not None and dt is not None:
            check_dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            nb, ne = EconomicCalendar().is_event_blocked("USD/JPY", check_dt)
            rich_context = build_mnq_context(
                dt, row_1m, ctx_5m, df_1m, df_5m, df_15m,
                news_blocked=nb, news_event=ne,
            )
            rich_context["proposed_direction"] = direction
            rich_context["entry"] = signal.get("entry")
            rich_context["stop_loss"] = signal.get("sl")
            rich_context["take_profit"] = signal.get("tp")
            rich_context["rule_setup_score"] = compute_setup_score(direction, rich_context)["setup_score"]

        context = rich_context or {
            "utc_time": when.isoformat(),
            "symbol": symbol,
            "proposed_direction": direction,
            "entry": signal.get("entry"),
            "stop_loss": signal.get("sl"),
            "take_profit": signal.get("tp"),
            "trend_5m": ctx_5m.get("trend"),
            "adx_5m": ctx_5m.get("adx"),
        }
        if row_1m is not None and "rule_setup_score" not in context:
            context["rsi_1m"] = float(row_1m.get("rsi", 50))
            context["volume_ratio_1m"] = float(row_1m.get("volume_ratio", 1))

        try:
            raw = self._call_api(self._build_prompt(signal, context))
            parsed = self._parse_json_response(raw) or {}
            result = self._decide_from_parsed(parsed, direction)
            self._cache[cache_key] = (now, result)
            bot_logger.info(
                f"🤖 LLM {symbol} {direction}: "
                f"{'ALLOW' if result['allowed'] else 'SKIP'} "
                f"conf={result['confidence']:.0%} score={result.get('setup_score', 0):.0f} — "
                f"{result['reason']}"
            )
            return result
        except Exception as e:
            bot_logger.warning(f"LLM advisor error (rules-only fallback): {e}")
            return {
                "allowed": True,
                "action": "allow",
                "confidence": 0.0,
                "setup_score": 0,
                "reason": f"LLM unavailable: {e}",
                "source": "fallback",
                "position_size_pct": 100,
            }
