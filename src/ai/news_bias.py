"""
News bias advisor — macro/futures headlines + DeepSeek summary.

Fetches recent NQ/Nasdaq/Fed/tech headlines (Tiingo and/or NewsAPI when configured)
and asks an LLM for a short-lived directional bias for micro futures scalping.

Optional filter on top of rule-based MTF signals; does NOT replace TA or risk.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import bot_logger

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
TIINGO_NEWS_URL = "https://api.tiingo.com/tiingo/news"

# Symbol → NewsAPI search focus
_SYMBOL_QUERIES = {
    "NQ": '("Nasdaq" OR "NQ futures" OR "tech stocks" OR "Magnificent Seven" OR "Federal Reserve" OR "FOMC")',
    "MNQ": '("Nasdaq" OR "NQ futures" OR "tech stocks" OR "Magnificent Seven" OR "Federal Reserve" OR "FOMC")',
    "MES": '("S&P 500" OR "ES futures" OR "stock market" OR "Federal Reserve" OR "FOMC")',
    "MGC": '("gold" OR "GC futures" OR "Federal Reserve" OR "inflation" OR "dollar")',
}

# Symbol → Tiingo tickers (Nasdaq-100 proxies + mega-cap) and macro tags
_SYMBOL_TIINGO_TICKERS = {
    "NQ": ["qqq", "nvda", "aapl", "msft", "meta", "googl", "amzn"],
    "MNQ": ["qqq", "nvda", "aapl", "msft", "meta", "googl", "amzn"],
    "MES": ["spy", "aapl", "msft", "nvda"],
    "MGC": ["gld", "gdx", "slv"],
}
_SYMBOL_TIINGO_TAGS = {
    "NQ": ["Earnings", "Federal Reserve", "Interest Rates"],
    "MNQ": ["Earnings", "Federal Reserve", "Interest Rates"],
    "MES": ["Earnings", "Federal Reserve", "Interest Rates"],
    "MGC": ["Federal Reserve", "Inflation", "Interest Rates"],
}

# Shared NewsAPI rate-limit state (news_bias + policy_scorer)
_NEWSAPI_BLOCKED_UNTIL: Optional[datetime] = None

# Set after Tiingo 403 permission error to avoid repeated failed requests per session
_TIINGO_NEWS_BLOCKED_REASON: Optional[str] = None


def resolve_use_tiingo_news(tiingo_key: Optional[str] = None) -> bool:
    """True when Tiingo news should be fetched (default on if TIINGO_API_KEY is set)."""
    if _TIINGO_NEWS_BLOCKED_REASON:
        return False
    key = (tiingo_key if tiingo_key is not None else os.getenv("TIINGO_API_KEY", "")).strip()
    if not key:
        return False
    env_val = os.getenv("USE_TIINGO_NEWS", "").strip().lower()
    if env_val in ("false", "0", "no"):
        return False
    return True


def _tiingo_response_hint(resp: requests.Response) -> str:
    """Extract a short, safe error hint from Tiingo error bodies (no secrets)."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            for field in ("detail", "message", "error"):
                val = data.get(field)
                if val:
                    return str(val)[:200]
    except (ValueError, json.JSONDecodeError):
        pass
    text = (resp.text or "").strip()
    return text[:200] if text else f"HTTP {resp.status_code}"


def _log_tiingo_error(resp: requests.Response) -> None:
    """Log Tiingo fetch failure; disable further fetches on permission/subscription 403."""
    global _TIINGO_NEWS_BLOCKED_REASON
    hint = _tiingo_response_hint(resp)
    if resp.status_code == 403:
        _TIINGO_NEWS_BLOCKED_REASON = hint or "forbidden"
        bot_logger.warning(
            "Tiingo 403: check API key or news subscription at tiingo.com "
            f"({hint}). Set USE_TIINGO_NEWS=false to disable, or add News access at "
            "https://www.tiingo.com/products/news"
        )
        return
    if resp.status_code == 401:
        _TIINGO_NEWS_BLOCKED_REASON = hint or "unauthorized"
        bot_logger.warning(
            "Tiingo 401: invalid or missing API key "
            f"({hint}). Set USE_TIINGO_NEWS=false or update TIINGO_API_KEY"
        )
        return
    bot_logger.warning(
        f"Tiingo news error {resp.status_code} — continuing without Tiingo headlines ({hint})"
    )


def headline_providers_label(
    newsapi_key: str = "",
    tiingo_key: str = "",
    use_tiingo: Optional[bool] = None,
) -> str:
    """Human-readable headline source label for startup logs."""
    tiingo_key = tiingo_key or os.getenv("TIINGO_API_KEY", "").strip()
    use_tiingo = resolve_use_tiingo_news(tiingo_key) if use_tiingo is None else use_tiingo
    has_tiingo = use_tiingo and bool(tiingo_key)
    has_newsapi = bool(newsapi_key)
    if has_tiingo and has_newsapi:
        return "Tiingo+NewsAPI"
    if has_tiingo:
        return "Tiingo"
    if has_newsapi:
        return "NewsAPI"
    return "LLM-only"


def _headline_llm_source(
    headlines: List[Dict[str, str]],
    newsapi_key: str = "",
    tiingo_key: str = "",
    use_tiingo: Optional[bool] = None,
) -> str:
    if not headlines:
        return "llm_only"
    tiingo_key = tiingo_key or os.getenv("TIINGO_API_KEY", "").strip()
    use_tiingo = resolve_use_tiingo_news(tiingo_key) if use_tiingo is None else use_tiingo
    has_tiingo = use_tiingo and bool(tiingo_key)
    has_newsapi = bool(newsapi_key)
    if has_tiingo and has_newsapi:
        return "tiingo+newsapi+llm"
    if has_tiingo:
        return "tiingo+llm"
    return "newsapi+llm"


def _normalize_headline(
    title: str,
    description: str,
    source: str,
    published: str,
) -> Dict[str, str]:
    return {
        "title": title,
        "description": (description or "")[:300],
        "source": source,
        "published": published,
        "publishedAt": published,
    }


def _dedupe_headlines(headlines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for h in headlines:
        key = (h.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)
    return out


def _fetch_tiingo_headlines(
    symbol: str,
    tiingo_key: str,
    *,
    limit: int = 12,
    lookback_days: int = 3,
) -> List[Dict[str, str]]:
    sym = symbol.upper()
    tickers = _SYMBOL_TIINGO_TICKERS.get(sym, _SYMBOL_TIINGO_TICKERS["NQ"])
    tags = _SYMBOL_TIINGO_TAGS.get(sym, _SYMBOL_TIINGO_TAGS["NQ"])
    start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    params = {
        "tickers": ",".join(tickers),
        "tags": ",".join(tags),
        "startDate": start_date,
        "limit": min(limit, 100),
        "sortBy": "publishedDate",
    }
    headers = {
        "Authorization": f"Token {tiingo_key}",
        "Content-Type": "application/json",
    }
    if _TIINGO_NEWS_BLOCKED_REASON:
        return []

    try:
        resp = requests.get(TIINGO_NEWS_URL, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            articles = resp.json()
            if not isinstance(articles, list):
                return []
            return [
                _normalize_headline(
                    a.get("title", ""),
                    a.get("description") or "",
                    str(a.get("source") or ""),
                    a.get("publishedDate", ""),
                )
                for a in articles
                if a.get("title")
            ]
        _log_tiingo_error(resp)
    except Exception as e:
        bot_logger.warning(f"Tiingo news fetch failed: {e}")
    return []


def _fetch_newsapi_headlines(
    symbol: str,
    newsapi_key: str,
    *,
    limit: int = 12,
    cooldown_min: int = 20,
) -> List[Dict[str, str]]:
    global _NEWSAPI_BLOCKED_UNTIL
    if not newsapi_key:
        return []
    if _NEWSAPI_BLOCKED_UNTIL and datetime.now() < _NEWSAPI_BLOCKED_UNTIL:
        return []

    sym = symbol.upper()
    query = _SYMBOL_QUERIES.get(sym, _SYMBOL_QUERIES["NQ"])
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": limit,
        "apiKey": newsapi_key,
    }
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=8,
        )
        if resp.status_code == 200:
            _NEWSAPI_BLOCKED_UNTIL = None
            articles = resp.json().get("articles", [])
            return [
                _normalize_headline(
                    a.get("title", ""),
                    a.get("description") or "",
                    (a.get("source") or {}).get("name", ""),
                    a.get("publishedAt", ""),
                )
                for a in articles
                if a.get("title")
            ]
        if resp.status_code == 429:
            _NEWSAPI_BLOCKED_UNTIL = datetime.now() + timedelta(minutes=cooldown_min)
            bot_logger.warning("NewsAPI rate-limited — headline fetch paused")
        else:
            bot_logger.warning(f"NewsAPI error {resp.status_code} — continuing without headlines")
    except Exception as e:
        bot_logger.warning(f"NewsAPI fetch failed: {e}")
    return []


def fetch_headlines_for_symbol(
    symbol: str,
    newsapi_key: str = "",
    *,
    limit: int = 12,
    cooldown_min: int = 20,
    tiingo_key: Optional[str] = None,
    use_tiingo: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """Fetch recent headlines for a futures symbol (shared by news_bias and policy_scorer).

    Merges Tiingo + NewsAPI when both are configured (deduped by title, Tiingo first).
    Falls back to whichever provider is available.
    """
    tiingo_key = (tiingo_key if tiingo_key is not None else os.getenv("TIINGO_API_KEY", "")).strip()
    use_tiingo = resolve_use_tiingo_news(tiingo_key) if use_tiingo is None else use_tiingo

    tiingo_headlines: List[Dict[str, str]] = []
    newsapi_headlines: List[Dict[str, str]] = []

    if use_tiingo and tiingo_key:
        tiingo_headlines = _fetch_tiingo_headlines(symbol, tiingo_key, limit=limit)

    newsapi_headlines = _fetch_newsapi_headlines(
        symbol, newsapi_key, limit=limit, cooldown_min=cooldown_min,
    )

    merged = _dedupe_headlines(tiingo_headlines + newsapi_headlines)
    return merged[:limit]


class NewsBiasAdvisor:
    """Fetch/summarize macro news and produce a trade bias signal."""

    def __init__(self):
        self.enabled = os.getenv("USE_LLM_NEWS", "false").lower() == "true"
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        self.mode = os.getenv("LLM_NEWS_MODE", "block").lower()  # block | advisory | boost
        self.min_confidence = float(os.getenv("LLM_NEWS_MIN_CONFIDENCE", "0.65"))
        self.block_confidence = float(os.getenv("LLM_NEWS_BLOCK_CONFIDENCE", "0.70"))
        self.boost_pct = int(os.getenv("LLM_NEWS_BOOST_PCT", "25"))
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "12"))
        self.cache_ttl_sec = int(os.getenv("LLM_NEWS_CACHE_TTL_SEC", "1800"))
        self.model = os.getenv("LLM_MODEL", self._default_model())
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "").strip()
        self.tiingo_key = os.getenv("TIINGO_API_KEY", "").strip()
        self.use_tiingo_news = resolve_use_tiingo_news(self.tiingo_key)
        self.newsapi_cooldown_min = int(os.getenv("NEWSAPI_RATE_LIMIT_COOLDOWN_MINUTES", "20"))
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

        if self.enabled and not self.api_key:
            bot_logger.warning("USE_LLM_NEWS=true but no DeepSeek/OpenAI API key — news bias disabled")
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
                f"News bias active: mode={self.mode} provider={self.provider} "
                f"source={src} block_conf>={self.block_confidence:.0%}"
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
        return f"news_bias:{symbol}:{bucket}"

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

    def _fetch_headlines(self, symbol: str, limit: int = 12) -> List[Dict[str, str]]:
        return fetch_headlines_for_symbol(
            symbol,
            self.newsapi_key,
            limit=limit,
            cooldown_min=self.newsapi_cooldown_min,
            tiingo_key=self.tiingo_key,
            use_tiingo=self.use_tiingo_news,
        )

    def _build_prompt(self, symbol: str, headlines: List[Dict[str, str]]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if headlines:
            news_block = json.dumps(headlines[:12], indent=2)
            news_section = f"Recent headlines (use these as primary evidence):\n{news_block}"
        else:
            news_section = (
                "No live headlines available. Use general macro context for US index futures "
                "(Nasdaq/NQ, Fed policy, mega-cap tech, geopolitical risk) but keep confidence "
                "moderate (<=0.55) since headlines are missing."
            )

        return f"""You are a US index futures macro advisor for {symbol} micro scalping (1–4 hour horizon).

Time: {now}
{news_section}

Assess near-term directional bias for {symbol} (Nasdaq-100 linked) considering:
- Fed / rates / inflation narrative
- Mega-cap tech / earnings sentiment
- Risk-on vs risk-off (geopolitics, yields, dollar)
- Index futures positioning context

Respond ONLY with JSON:
{{
  "bias": "bullish" or "bearish" or "neutral",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence for traders",
  "key_themes": ["up to 3 short tags"],
  "risk_note": "optional caution or empty string"
}}

Guidelines:
- bullish = favor longs / avoid aggressive shorts
- bearish = favor shorts / avoid aggressive longs
- neutral = no strong macro tilt; TA dominates
- High confidence only when headlines/themes align clearly
- Without headlines, bias should usually be neutral with confidence <= 0.55
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
                {"role": "system", "content": "You output strict JSON only for futures macro bias."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def get_bias(
        self,
        symbol: str = "MNQ",
        force_refresh: bool = False,
        as_of: Optional[datetime] = None,
        headlines: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Return cached or fresh news bias for a symbol."""
        neutral = {
            "bias": "neutral",
            "confidence": 0.0,
            "reason": "News bias disabled",
            "key_themes": [],
            "risk_note": "",
            "headline_count": 0,
            "source": "disabled",
            "allowed": True,
            "size_adjust_pct": 0,
        }
        if not self.enabled:
            return neutral

        cache_key = self._cache_key(symbol, as_of=as_of)
        now = time.time()
        cached = self._cache.get(cache_key)
        if not force_refresh and cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        if headlines is None:
            headlines = self._fetch_headlines(symbol)
        try:
            raw = self._call_api(self._build_prompt(symbol, headlines))
            parsed = self._parse_json_response(raw) or {}
            bias = str(parsed.get("bias", "neutral")).lower()
            if bias not in ("bullish", "bearish", "neutral"):
                bias = "neutral"
            confidence = float(parsed.get("confidence", 0.5))
            result = {
                "bias": bias,
                "confidence": confidence,
                "reason": str(parsed.get("reason", "LLM news summary")),
                "key_themes": parsed.get("key_themes") or [],
                "risk_note": str(parsed.get("risk_note", "")),
                "headline_count": len(headlines),
                "source": _headline_llm_source(
                    headlines,
                    self.newsapi_key,
                    self.tiingo_key,
                    self.use_tiingo_news,
                ),
                "allowed": True,
                "size_adjust_pct": 0,
            }
            self._cache[cache_key] = (now, result)
            themes = result.get("key_themes") or []
            top_titles = [h.get("title", "") for h in headlines[:3] if h.get("title")]
            log_parts = [
                f"📰 News bias {symbol}: {bias.upper()} "
                f"conf={confidence:.0%} ({len(headlines)} headlines) — {result['reason']}",
            ]
            if themes:
                log_parts.append(f"themes={', '.join(themes[:3])}")
            if top_titles:
                log_parts.append(f"headlines={' | '.join(top_titles[:3])}")
            if result.get("risk_note"):
                log_parts.append(f"risk={result['risk_note']}")
            bot_logger.info(" | ".join(log_parts))
            return result
        except Exception as e:
            bot_logger.warning(f"News bias error (neutral fallback): {e}")
            fallback = {
                **neutral,
                "reason": f"News bias unavailable: {e}",
                "source": "fallback",
            }
            return fallback

    def evaluate_direction(
        self,
        direction: str,
        symbol: str = "MNQ",
        bias: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply news bias to a proposed trade direction.

        Returns allowed, size_adjust_pct, and advisory text for scan logs.
        """
        if not self.enabled:
            return {
                "allowed": True,
                "size_adjust_pct": 0,
                "advisory": "",
                "bias": "neutral",
                "confidence": 0.0,
                "reason": "News bias disabled",
                "source": "disabled",
            }

        bias = bias or self.get_bias(symbol)
        b = bias.get("bias", "neutral")
        conf = float(bias.get("confidence", 0))
        wants_long = direction.lower() in ("long", "buy")
        conflicts = (wants_long and b == "bearish") or (not wants_long and b == "bullish")
        aligned = (wants_long and b == "bullish") or (not wants_long and b == "bearish")

        advisory = f"News bias: {b}"
        if bias.get("reason"):
            advisory += f" — {bias['reason']}"
        if conflicts and conf >= self.min_confidence:
            advisory += f" | conflicts with {direction.upper()}"
        elif aligned and conf >= self.min_confidence:
            advisory += f" | aligned with {direction.upper()}"

        allowed = True
        size_adjust = 0

        if self.mode == "block" and conflicts and conf >= self.block_confidence:
            allowed = False
        elif self.mode == "boost" and aligned and conf >= self.min_confidence:
            size_adjust = self.boost_pct

        return {
            "allowed": allowed,
            "size_adjust_pct": size_adjust,
            "advisory": advisory,
            "bias": b,
            "confidence": conf,
            "reason": bias.get("reason", ""),
            "key_themes": bias.get("key_themes", []),
            "headline_count": bias.get("headline_count", 0),
            "source": bias.get("source", "llm"),
        }

    @staticmethod
    def _truncate_title(title: str, max_len: int = 60) -> str:
        t = (title or "").strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 3].rstrip() + "..."

    @staticmethod
    def console_enabled() -> bool:
        return os.getenv("NEWS_CONSOLE", "true").lower() == "true"

    def format_scan_line(
        self,
        symbol: str,
        bias: Optional[Dict[str, Any]] = None,
        headlines: Optional[List[Dict[str, str]]] = None,
        compact: Optional[bool] = None,
    ) -> str:
        """Summary for scan output (compact console vs verbose detail)."""
        if not self.enabled or not self.console_enabled():
            return ""
        if compact is None:
            compact = os.getenv("COMPACT_NEWS", "true").lower() == "true"
        headline_count = max(0, int(os.getenv("NEWS_HEADLINE_COUNT", "2")))
        b = bias or self.get_bias(symbol)
        bias_label = b.get("bias", "neutral").upper()
        conf = float(b.get("confidence") or 0)
        count = b.get("headline_count") or (len(headlines) if headlines else 0)

        if compact:
            lines = [f"📰 {symbol}: {bias_label} {conf:.0%}"]
            for h in (headlines or [])[:headline_count]:
                title = self._truncate_title(h.get("title", ""))
                if title:
                    lines.append(f"   • {title}")
            return "\n".join(lines)

        line = f"📰 News bias ({symbol}): {bias_label}"
        if conf:
            line += f" {conf:.0%}"
        if b.get("reason"):
            line += f" — {b['reason']}"
        if count:
            line += f" [{count} headlines]"
        themes = b.get("key_themes") or []
        if themes:
            line += f" | {', '.join(themes[:3])}"
        return line
