"""
DeepSeek trade outcome learner — pattern analysis on closed hybrid scalp trades.

Uses DEEPSEEK_API_KEY (same as llm_advisor). Falls back to local win-rate stats
when API is unavailable. Fail-open: never blocks all trading on API errors.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.ai.llm_advisor import DEFAULT_DEEPSEEK_BASE
from src.ai.trade_journal import TradeJournal
from src.utils.logger import bot_logger

DEFAULT_ADVICE: Dict[str, Any] = {
    "boost_modes": [],
    "block_conditions": [],
    "adx_min_adjust": 0,
    "confidence": 0.0,
    "notes": "",
    "source": "none",
}

MIN_HISTORY_SESSION = 20
MIN_SAMPLES_LOCAL_BLOCK = 12
LOCAL_BLOCK_WR = 0.22


def _flow_bucket(delta: float) -> str:
    if delta > 30:
        return "positive"
    if delta < -30:
        return "negative"
    return "neutral"


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
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


class LocalPatternStats:
    """Win rate by entry mode, hour, and flow bucket — no API required."""

    def __init__(self):
        self.by_mode: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.by_hour: Dict[int, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.by_flow: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})

    def reset(self) -> None:
        self.by_mode.clear()
        self.by_hour.clear()
        self.by_flow.clear()

    def ingest(self, trades: List[Dict[str, Any]]) -> None:
        self.reset()
        for t in trades:
            pnl = float(t.get("pnl", 0) or 0)
            if pnl == 0:
                continue
            win = pnl > 0
            mode = str(t.get("entry_mode", "unknown")).lower()
            hour = int(t.get("hour", datetime.now(timezone.utc).hour))
            flow = _flow_bucket(float(t.get("flow_delta", 0) or 0))
            bucket = self.by_mode[mode]
            bucket["wins" if win else "losses"] += 1
            bucket = self.by_hour[hour]
            bucket["wins" if win else "losses"] += 1
            bucket = self.by_flow[flow]
            bucket["wins" if win else "losses"] += 1

    @staticmethod
    def _wr(stats: Dict[str, int]) -> Tuple[float, int]:
        w, l = stats.get("wins", 0), stats.get("losses", 0)
        total = w + l
        return (w / total if total else 0.5, total)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"modes": {}, "hours": {}, "flow": {}}
        for mode, stats in self.by_mode.items():
            wr, n = self._wr(stats)
            out["modes"][mode] = {"win_rate": round(wr, 3), "n": n}
        for hour, stats in self.by_hour.items():
            wr, n = self._wr(stats)
            out["hours"][str(hour)] = {"win_rate": round(wr, 3), "n": n}
        for flow, stats in self.by_flow.items():
            wr, n = self._wr(stats)
            out["flow"][flow] = {"win_rate": round(wr, 3), "n": n}
        return out

    def build_advice(self) -> Dict[str, Any]:
        """Derive block/boost hints from local stats."""
        boost_modes: List[str] = []
        block_conditions: List[str] = []
        notes: List[str] = []

        for mode, stats in self.by_mode.items():
            wr, n = self._wr(stats)
            if n >= MIN_SAMPLES_LOCAL_BLOCK and wr >= 0.55:
                boost_modes.append(mode)
                notes.append(f"{mode} WR {wr:.0%} ({n})")
            if n >= MIN_SAMPLES_LOCAL_BLOCK and wr <= LOCAL_BLOCK_WR:
                block_conditions.append(f"mode_{mode}")
                notes.append(f"avoid {mode} WR {wr:.0%} ({n})")

        for hour, stats in self.by_hour.items():
            wr, n = self._wr(stats)
            if n >= MIN_SAMPLES_LOCAL_BLOCK and wr <= LOCAL_BLOCK_WR:
                block_conditions.append(f"hour_{hour}")
                notes.append(f"hour {hour} UTC WR {wr:.0%} ({n})")

        for flow, stats in self.by_flow.items():
            wr, n = self._wr(stats)
            if n >= MIN_SAMPLES_LOCAL_BLOCK and wr <= LOCAL_BLOCK_WR:
                block_conditions.append(f"flow_{flow}")
                notes.append(f"flow {flow} WR {wr:.0%} ({n})")

        adx_bump = 0
        cont = self.by_mode.get("continuation", {})
        wr, n = self._wr(cont)
        if n >= MIN_SAMPLES_LOCAL_BLOCK and wr < 0.35:
            adx_bump = max(adx_bump, 18)

        return {
            "boost_modes": boost_modes,
            "block_conditions": block_conditions,
            "adx_min_adjust": adx_bump,
            "confidence": 0.5 if notes else 0.0,
            "notes": "; ".join(notes[:5]) if notes else "insufficient local history",
            "source": "local",
            "stats": self.summary(),
        }


class DeepSeekTradeLearner:
    """Learn win/loss patterns from trade journal; gate hybrid entries softly."""

    def __init__(
        self,
        journal: Optional[TradeJournal] = None,
        learn_every_n: int = 5,
    ):
        self.enabled = os.getenv("USE_DEEPSEEK_LEARNER", "false").lower() == "true"
        self.local_blocking = os.getenv("USE_LOCAL_PATTERN_LEARNER", "true").lower() == "true"
        self.learn_every_n = int(os.getenv("DEEPSEEK_LEARN_EVERY_N", str(learn_every_n)))
        self.min_confidence_block = float(os.getenv("DEEPSEEK_LEARN_MIN_CONF", "0.55"))
        self.timeout_sec = float(os.getenv("DEEPSEEK_LEARN_TIMEOUT_SEC", "15"))
        self.model = os.getenv("DEEPSEEK_LEARN_MODEL", os.getenv("LLM_MODEL", "deepseek-chat"))
        self.api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")).strip()
        base = os.getenv("LLM_BASE_URL", DEFAULT_DEEPSEEK_BASE).strip().rstrip("/")
        if "/v1" not in base and not base.endswith("/v1"):
            self.base_url = f"{base}/v1" if "deepseek" in base else base
        else:
            self.base_url = base

        self.journal = journal or TradeJournal()
        self.local_stats = LocalPatternStats()
        self.advice: Dict[str, Any] = dict(DEFAULT_ADVICE)
        self._trades_since_learn = 0
        self._last_learn_ts = 0.0

        if self.enabled and not self.api_key:
            bot_logger.warning("USE_DEEPSEEK_LEARNER=true but no DEEPSEEK_API_KEY — local stats only")
        elif self.enabled:
            bot_logger.info(
                f"DeepSeek trade learner ON — every {self.learn_every_n} closes, "
                f"model={self.model}"
            )

        self._refresh_local_advice()

    @property
    def blocking_active(self) -> bool:
        """Local pattern blocks work without API key; DeepSeek adds optional AI advice."""
        return self.local_blocking or self.enabled

    def _refresh_local_advice(self) -> None:
        self.local_stats.ingest(self.journal.recent())
        local = self.local_stats.build_advice()
        if (not self.enabled and self.local_blocking) or self.advice.get("source") == "none":
            self.advice = local

    def on_session_start(self) -> None:
        """Run analysis at session start when enough history exists."""
        if self.journal.count() < MIN_HISTORY_SESSION:
            bot_logger.info(
                f"Trade learner: {self.journal.count()} journal trades "
                f"(need {MIN_HISTORY_SESSION} for session analysis)"
            )
            self._refresh_local_advice()
            return
        self._run_analysis(reason="session_start")

    def record_trade(self, record: Dict[str, Any]) -> None:
        """Append to journal, refresh local stats, optionally call DeepSeek."""
        self.journal.append(record)
        self._trades_since_learn += 1
        self._refresh_local_advice()
        if not self.enabled:
            return
        if self._trades_since_learn >= self.learn_every_n:
            self._trades_since_learn = 0
            self._run_analysis(reason="periodic")

    def _run_analysis(self, reason: str = "") -> None:
        trades = self.journal.recent(50)
        if len(trades) < 5:
            return

        local = self.local_stats.build_advice()
        self.advice = local

        if not self.enabled or not self.api_key:
            bot_logger.info(
                f"Trade learner ({reason}): local advice — "
                f"boost={local.get('boost_modes')} block={local.get('block_conditions')}"
            )
            return

        try:
            ai = self._call_deepseek(trades, local.get("stats", {}))
            if ai:
                merged = dict(local)
                merged.update({k: v for k, v in ai.items() if v is not None})
                merged["source"] = "deepseek+local"
                self.advice = merged
                self._last_learn_ts = time.time()
                bot_logger.info(
                    f"DeepSeek learner ({reason}): boost={merged.get('boost_modes')} "
                    f"block={merged.get('block_conditions')} adx>={merged.get('adx_min_adjust')} — "
                    f"{merged.get('notes', '')[:120]}"
                )
        except Exception as e:
            bot_logger.warning(f"DeepSeek learner failed ({reason}) — fail-open, local only: {e}")
            self.advice = local
            self.advice["source"] = "local_fallback"

    def _build_prompt(self, trades: List[Dict], stats: Dict[str, Any]) -> str:
        wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl", 0)) < 0)
        sample = trades[-25:]
        return f"""You analyze MNQ/NQ hybrid scalp trade outcomes to improve entry selection.

Recent performance: {wins}W / {losses}L over last {len(trades)} trades.
Local stats: {json.dumps(stats, indent=2)}

Sample closed trades (newest last):
{json.dumps(sample, default=str, indent=2)}

Identify patterns in WINS vs LOSSES by entry_mode (pullback/continuation/burst/trigger),
ADX level, order flow (flow_delta, buy_pct), hour UTC, hold time, exit reason.

Respond ONLY with JSON:
{{
  "boost_modes": ["trigger","burst"],
  "block_conditions": ["flow_negative_long","mode_continuation","hour_13","adx_below_18"],
  "adx_min_adjust": 16,
  "confidence": 0.7,
  "notes": "one short sentence"
}}

block_conditions vocabulary (use these tokens):
- flow_negative_long, flow_positive_short
- low_buy_pct_long (long when buy_pct < 0.48), high_buy_pct_short
- mode_<name> e.g. mode_pullback, mode_continuation, mode_burst
- hour_<0-23> UTC
- adx_below_<N>

Only block_conditions with strong evidence. adx_min_adjust: minimum ADX floor to suggest (0 = no change).
confidence: 0-1 certainty in recommendations."""

    def _call_deepseek(self, trades: List[Dict], stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You output strict JSON only for trade pattern learning.",
                },
                {"role": "user", "content": self._build_prompt(trades, stats)},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_json_response(raw)
        if not parsed:
            return None
        return {
            "boost_modes": [str(m).lower() for m in (parsed.get("boost_modes") or [])],
            "block_conditions": [str(c).lower() for c in (parsed.get("block_conditions") or [])],
            "adx_min_adjust": int(parsed.get("adx_min_adjust") or 0),
            "confidence": float(parsed.get("confidence") or 0.5),
            "notes": str(parsed.get("notes") or ""),
        }

    def get_adx_floor(self, base_pullback: int, base_continuation: int) -> Tuple[int, int]:
        """Apply soft ADX bump from learned advice."""
        floor = int(self.advice.get("adx_min_adjust") or 0)
        if floor <= 0:
            return base_pullback, base_continuation
        return max(base_pullback, floor), max(base_continuation, floor)

    def is_mode_boosted(self, mode: str) -> bool:
        boosts = [str(m).lower() for m in self.advice.get("boost_modes") or []]
        if not boosts:
            return True
        return str(mode).lower() in boosts

    def check_entry_block(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Returns (should_block, reason). Fail-open when confidence low or no API advice.
        """
        blocks = [str(c).lower() for c in self.advice.get("block_conditions") or []]
        if not blocks:
            return False, ""

        conf = float(self.advice.get("confidence") or 0)
        if conf < self.min_confidence_block and self.advice.get("source", "").startswith("deepseek"):
            return False, ""

        for cond in blocks:
            if self._matches_block(cond, ctx):
                note = self.advice.get("notes") or cond
                return True, f"{cond} ({note})"
        return False, ""

    def _matches_block(self, cond: str, ctx: Dict[str, Any]) -> bool:
        direction = str(ctx.get("direction", "")).lower()
        flow_delta = float(ctx.get("flow_delta", 0) or 0)
        buy_pct = float(ctx.get("buy_pct", 0.5) or 0.5)
        adx = float(ctx.get("adx", 0) or 0)
        mode = str(ctx.get("entry_mode", "")).lower()
        hour = int(ctx.get("hour", datetime.now(timezone.utc).hour))

        if cond == "flow_negative_long":
            return direction == "long" and flow_delta < 0
        if cond == "flow_positive_short":
            return direction == "short" and flow_delta > 0
        if cond == "low_buy_pct_long":
            return direction == "long" and buy_pct < 0.48
        if cond == "high_buy_pct_short":
            return direction == "short" and buy_pct > 0.52
        if cond.startswith("mode_"):
            return mode == cond[5:]
        if cond.startswith("hour_"):
            try:
                return hour == int(cond[5:])
            except ValueError:
                return False
        if cond.startswith("adx_below_"):
            try:
                return adx < float(cond[10:])
            except ValueError:
                return False
        if cond.startswith("flow_"):
            bucket = _flow_bucket(flow_delta)
            return bucket == cond[5:]
        return False

    def get_status_line(self) -> str:
        src = self.advice.get("source", "none")
        boosts = ",".join(self.advice.get("boost_modes") or []) or "—"
        blocks = len(self.advice.get("block_conditions") or [])
        return f"source={src} boost=[{boosts}] blocks={blocks} journal={self.journal.count()}"
