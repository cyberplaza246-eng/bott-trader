"""
Policy scorer — headline → categorize → score (-5..+5) → confirm with TA.

Richer macro/policy read than news_bias (categories, institutional view, second-order).
Never trade on news alone; TA must align before block mode rejects entries.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from src.ai.news_bias import (
    DEFAULT_DEEPSEEK_BASE,
    DEFAULT_OPENAI_BASE,
    _headline_llm_source,
    fetch_headlines_for_symbol,
    headline_providers_label,
    resolve_use_tiingo_news,
)
from src.utils.logger import bot_logger

POLICY_CATEGORIES = [
    "Trade/Tariffs",
    "Fed",
    "Taxes",
    "AI/Technology",
    "Geopolitics",
    "Other",
]

IMPACT_SCORE_MAP = {
    "strong bullish": 5,
    "bullish": 2,
    "neutral": 0,
    "bearish": -2,
    "strong bearish": -5,
}


class PolicyScorer:
    """Score policy/headline impact on NQ/MNQ and gate entries against MTF context."""

    def __init__(self):
        self.enabled = os.getenv("USE_POLICY_SCORER", "false").lower() == "true"
        self.mode = os.getenv("POLICY_SCORER_MODE", "advisory").lower()  # advisory | block | boost
        self.min_score = int(os.getenv("POLICY_SCORER_MIN_SCORE", "3"))
        self.min_confidence = float(os.getenv("POLICY_SCORER_MIN_CONFIDENCE", "0.70"))
        self.boost_pct = int(os.getenv("POLICY_SCORER_BOOST_PCT", "25"))
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "12"))
        self.cache_ttl_sec = int(os.getenv("POLICY_SCORER_CACHE_TTL_SEC", "1800"))
        self.model = os.getenv("LLM_MODEL", self._default_model())
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "").strip()
        self.tiingo_key = os.getenv("TIINGO_API_KEY", "").strip()
        self.use_tiingo_news = resolve_use_tiingo_news(self.tiingo_key)
        self.newsapi_cooldown_min = int(os.getenv("NEWSAPI_RATE_LIMIT_COOLDOWN_MINUTES", "20"))
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

        if self.enabled and not self.api_key:
            bot_logger.warning("USE_POLICY_SCORER=true but no DeepSeek/OpenAI API key — policy scorer disabled")
            self.enabled = False
        elif self.enabled:
            src = headline_providers_label(
                self.newsapi_key, self.tiingo_key, self.use_tiingo_news,
            )
            if src == "LLM-only":
                src = "LLM only (no live headlines)"
            else:
                src = f"{src}+LLM"
            bot_logger.info(
                f"Policy scorer active: mode={self.mode} provider={self.provider} "
                f"source={src} min_score={self.min_score} min_conf>={self.min_confidence:.0%}"
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

    def _cache_key(self, symbol: str, as_of: Optional[datetime] = None) -> str:
        when = as_of or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        bucket = when.astimezone(timezone.utc).strftime("%Y%m%d%H")
        return f"policy_scorer:{symbol}:{bucket}"

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _normalize_confidence(raw: Any) -> float:
        conf = float(raw or 0)
        if conf > 1.0:
            conf /= 100.0
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _impact_to_score(impact: str) -> int:
        return IMPACT_SCORE_MAP.get(str(impact or "").strip().lower(), 0)

    def _build_prompt(self, symbol: str, headlines: List[Dict[str, str]]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if headlines:
            news_block = json.dumps(headlines[:12], indent=2)
            news_section = f"Recent headlines (primary evidence):\n{news_block}"
        else:
            news_section = (
                "No live headlines available. Use general US policy/macro context for Nasdaq/NQ "
                "but keep confidence moderate (<=55) since headlines are missing."
            )

        cats = ", ".join(POLICY_CATEGORIES)
        return f"""You are a US policy/macro analyst for {symbol} (Nasdaq-100 futures) scalping (1–4 hour horizon).

Time: {now}
{news_section}

Summarize headline/policy impact for NQ/MNQ. Categorize themes and score directional bias.

Categories (pick all that apply): {cats}

Bullish signals: AI investment, tax cuts, lower regulation, infrastructure spending, positive China trade news.
Bearish signals: tariffs, China restrictions, Fed attacks, geopolitical threats, semiconductor sanctions.

Respond ONLY with JSON:
{{
  "summary": "one sentence for traders",
  "categories": ["Trade/Tariffs", "Fed"],
  "impact": "Strong Bearish|Bearish|Neutral|Bullish|Strong Bullish",
  "score": -4,
  "confidence": 75,
  "institutional_view": "how institutions likely react",
  "second_order": "e.g. yields up → NQ headwind"
}}

Scoring guide:
- Strong Bullish = +5, Bullish = +2, Neutral = 0, Bearish = -2, Strong Bearish = -5
- confidence: 0–100 (percent certainty)
- Without headlines, score should usually be 0 with confidence <= 55
- Never trade on news alone — this score confirms or conflicts with technical alignment
"""

    def _call_api(self, prompt: str) -> Optional[str]:
        base = self.base_url.rstrip("/")
        url = f"{base}/v1/chat/completions" if "/v1" not in base else f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You output strict JSON only for futures policy scoring."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 320,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def get_score(
        self,
        symbol: str = "MNQ",
        force_refresh: bool = False,
        as_of: Optional[datetime] = None,
        headlines: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Return cached or fresh policy score for a symbol."""
        neutral = {
            "summary": "Policy scorer disabled",
            "categories": [],
            "impact": "Neutral",
            "score": 0,
            "confidence": 0.0,
            "institutional_view": "",
            "second_order": "",
            "headline_count": 0,
            "source": "disabled",
        }
        if not self.enabled:
            return neutral

        cache_key = self._cache_key(symbol, as_of=as_of)
        now = time.time()
        cached = self._cache.get(cache_key)
        if not force_refresh and cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        if headlines is None:
            headlines = fetch_headlines_for_symbol(
                symbol,
                self.newsapi_key,
                cooldown_min=self.newsapi_cooldown_min,
                tiingo_key=self.tiingo_key,
                use_tiingo=self.use_tiingo_news,
            )

        try:
            raw = self._call_api(self._build_prompt(symbol, headlines))
            parsed = self._parse_json_response(raw) or {}
            impact = str(parsed.get("impact", "Neutral"))
            score = parsed.get("score")
            if score is None:
                score = self._impact_to_score(impact)
            else:
                score = int(max(-5, min(5, int(score))))
            confidence = self._normalize_confidence(parsed.get("confidence", 0.5))
            categories = parsed.get("categories") or []
            if not isinstance(categories, list):
                categories = [str(categories)]
            categories = [str(c) for c in categories[:6]]

            result = {
                "summary": str(parsed.get("summary", "Policy assessment")),
                "categories": categories,
                "impact": impact,
                "score": score,
                "confidence": confidence,
                "institutional_view": str(parsed.get("institutional_view", "")),
                "second_order": str(parsed.get("second_order", "")),
                "headline_count": len(headlines),
                "source": _headline_llm_source(
                    headlines,
                    self.newsapi_key,
                    self.tiingo_key,
                    self.use_tiingo_news,
                ),
            }
            self._cache[cache_key] = (now, result)
            log_line = (
                f"🏛️ Policy {symbol}: score={score:+d} [{','.join(categories)}] "
                f"conf={confidence:.0%} — {result['summary']}"
            )
            if result.get("second_order"):
                log_line += f" | 2nd: {result['second_order']}"
            if result.get("institutional_view"):
                log_line += f" | inst: {result['institutional_view']}"
            bot_logger.info(log_line)
            return result
        except Exception as e:
            bot_logger.warning(f"Policy scorer error (neutral fallback): {e}")
            return {
                **neutral,
                "summary": f"Policy scorer unavailable: {e}",
                "source": "fallback",
            }

    @staticmethod
    def _mtf_aligned_with_trade(direction: str, ctx_5m: Dict, ctx_15m: Dict) -> bool:
        """True when 5M and 15M trends support the proposed trade direction."""
        wants_long = direction.lower() in ("long", "buy")
        t5 = (ctx_5m or {}).get("trend")
        t15 = (ctx_15m or {}).get("trend")
        if wants_long:
            return t5 == "bullish" and t15 == "bullish"
        return t5 == "bearish" and t15 == "bearish"

    @staticmethod
    def _direction_fights_score(direction: str, score: int, min_score: int) -> bool:
        wants_long = direction.lower() in ("long", "buy")
        if wants_long and score <= -min_score:
            return True
        if not wants_long and score >= min_score:
            return True
        return False

    @staticmethod
    def _direction_aligned_with_score(direction: str, score: int, min_score: int) -> bool:
        wants_long = direction.lower() in ("long", "buy")
        if wants_long and score >= min_score:
            return True
        if not wants_long and score <= -min_score:
            return True
        return False

    def evaluate_entry(
        self,
        direction: str,
        score: int,
        confidence: float,
        ctx_5m: Dict,
        ctx_15m: Dict,
        mode: Optional[str] = None,
        volume_confirmed: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply policy score to a proposed entry.

        Block mode: skip when |score|>=min, conf>=min_conf, direction fights score,
        and MTF is not aligned with the trade (never trade news alone).
        Advisory mode: log only, never block.
        """
        mode = (mode or self.mode).lower()
        conf = self._normalize_confidence(confidence)
        fights = self._direction_fights_score(direction, score, self.min_score)
        aligned_score = self._direction_aligned_with_score(direction, score, self.min_score)
        mtf_ok = self._mtf_aligned_with_trade(direction, ctx_5m, ctx_15m)

        parts = [f"Policy score {score:+d}"]
        if fights:
            parts.append(f"conflicts with {direction.upper()}")
        elif aligned_score:
            parts.append(f"aligned with {direction.upper()}")
        if not mtf_ok:
            parts.append("MTF not fully aligned")
        elif mtf_ok and aligned_score:
            parts.append("MTF confirms policy tilt")
        if not volume_confirmed:
            parts.append("volume unconfirmed")

        advisory_note = " | ".join(parts)
        allowed = True
        reason = "advisory only" if mode == "advisory" else "passed"
        size_adjust = 0

        if mode == "block":
            if (
                abs(score) >= self.min_score
                and conf >= self.min_confidence
                and fights
                and not mtf_ok
            ):
                allowed = False
                reason = (
                    f"policy {score:+d} fights {direction.upper()} "
                    f"and 5M/15M not aligned (conf {conf:.0%})"
                )
        elif mode == "boost" and aligned_score and conf >= self.min_confidence and mtf_ok:
            size_adjust = self.boost_pct
            reason = f"policy + MTF aligned (score {score:+d})"

        return {
            "allowed": allowed,
            "reason": reason,
            "advisory_note": advisory_note,
            "size_adjust_pct": size_adjust,
            "mtf_aligned": mtf_ok,
            "fights_score": fights,
        }

    @staticmethod
    def console_enabled() -> bool:
        return os.getenv("POLICY_CONSOLE", "false").lower() == "true"

    def format_scan_line(
        self,
        symbol: str,
        policy: Optional[Dict[str, Any]] = None,
        compact: Optional[bool] = None,
    ) -> str:
        """Summary for scan output (compact console vs verbose detail)."""
        if not self.enabled or not self.console_enabled():
            return ""
        if compact is None:
            compact = os.getenv("COMPACT_NEWS", "true").lower() == "true"
        p = policy or self.get_score(symbol)
        score = int(p.get("score", 0))
        cats = p.get("categories") or []
        cat_str = cats[0] if cats else "—"
        conf = self._normalize_confidence(p.get("confidence", 0))
        if compact:
            return f"🏛️ {score:+d} [{cat_str}]"
        cat_str = "/".join(cats[:2]) if cats else "—"
        line = f"🏛️ Policy ({symbol}): score={score:+d} [{cat_str}] conf={conf:.0%}"
        if p.get("summary"):
            line += f" — {p['summary']}"
        if p.get("second_order"):
            line += f" | 2nd: {p['second_order']}"
        return line
