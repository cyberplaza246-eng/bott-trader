"""
Rule-based MNQ filters — codifies session/news/structure advice (backtestable).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from src.ai.economic_calendar import EconomicCalendar
from src.ai.mnq_context import build_mnq_context, classify_session, compute_setup_score


class MNQSmartFilters:
    """Deterministic filters matching trader best-practice advice."""

    def __init__(
        self,
        min_setup_score: float = 55,
        block_midday_chop: bool = True,
        block_news: bool = True,
        block_high_volatility: bool = True,
        require_mtf_alignment: bool = True,
        require_session_scalp: bool = False,
        block_chop_structure: bool = True,
    ):
        self.min_setup_score = min_setup_score
        self.block_midday_chop = block_midday_chop
        self.block_news = block_news
        self.block_high_volatility = block_high_volatility
        self.require_mtf_alignment = require_mtf_alignment
        self.require_session_scalp = require_session_scalp
        self.block_chop_structure = block_chop_structure
        self.calendar = EconomicCalendar()

    def _news_check(self, dt: datetime) -> Tuple[bool, str]:
        if not self.block_news:
            return False, ""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        blocked, name = self.calendar.is_event_blocked("USD/JPY", dt)
        return blocked, name or ""

    def evaluate(
        self,
        direction: str,
        dt: datetime,
        row_1m,
        ctx_5m: Dict,
        df_1m,
        df_5m,
        df_15m=None,
    ) -> Dict[str, Any]:
        news_blocked, news_event = self._news_check(dt)
        ctx = build_mnq_context(
            dt, row_1m, ctx_5m, df_1m, df_5m, df_15m,
            news_blocked=news_blocked, news_event=news_event,
        )
        scoring = compute_setup_score(direction, ctx, self.min_setup_score)

        block_reasons = []
        sess = classify_session(dt)

        if self.require_session_scalp and not sess.get("allow_scalp"):
            block_reasons.append(f"session={sess['session']}")
        if self.block_midday_chop and sess["session"] == "midday_chop":
            block_reasons.append("midday chop")
        if news_blocked:
            block_reasons.append(f"news:{news_event}")
        if self.block_high_volatility and ctx["atr_ratio_vs_median"] > 2.5:
            block_reasons.append("volatility spike")
        if self.require_mtf_alignment and not ctx["mtf_aligned"]:
            block_reasons.append("MTF misalignment")
        if self.block_chop_structure and ctx["market_structure"] in ("chop", "volatile_chop"):
            block_reasons.append(ctx["market_structure"])
        if not scoring["allow"]:
            block_reasons.append(f"setup score {scoring['setup_score']}")

        allowed = len(block_reasons) == 0
        return {
            "allowed": allowed,
            "block_reasons": block_reasons,
            "setup_score": scoring["setup_score"],
            "position_size_pct": scoring["position_size_pct"] if allowed else 0,
            "market_structure": ctx["market_structure"],
            "session": ctx["session"],
            "context": ctx,
            "scoring": scoring,
        }
