"""
Overnight Globex PAPER overlay (also runs through RTH if you start it).

Paper recipe (Lucid TEST/sim — not funded live):
  After warmup, last-through last ROLL_BARS 1m high/low (also 1m close).
  Qty 1 MNQ. Qty 2 (stacked lots, same MNQ — not a second symbol) only when
  ALL of: last through the 20m extreme AND completed 1m close already through
  that level AND completed 15m EMA8/21 agrees AND completed 60m EMA8/21 agrees.
  Gemini is advisory-only and never sizes. Default QTY=1 if HTF is flat/unknown.
  No ET-day fill cap by default (PAPER_MAX_FILLS=0). Set PAPER_MAX_FILLS>0 for a
  per-ET-calendar-date cap (+5 extra if those fills are net green). Default one
  open trade; a second concurrent entry is allowed when the new signal's win
  confidence >= OVERNIGHT_ADDON_CONFIDENCE_MIN (default 80), up to OVERNIGHT_MAX_OPEN
  (default 2). Confidence floor OFF by default (OVERNIGHT_MIN_CONFIDENCE=0);
  direction filter is 15m MACD (OVERNIGHT_MACD_REQUIRE). Each entry gets its own
  Lucid bracket and bot_* ticket.
  5 consecutive closed recipe losers (pnl<=0, skip test_fill) → 30m cooldown.
  Set OVERNIGHT_SKIP_COOLDOWN=1 to bypass (restart). OVERNIGHT_COOLDOWN_LOSSES=0
  disables the trigger entirely.
  Stop = clip(completed 15m ATR14 × 1.5, 12–40). TP = OVERNIGHT_TP_R
  (default 0.65R — paper MFE often reached ~0.6–0.9R then reversed to SL).
  MACD (15m histogram, OVERNIGHT_MACD=true): aligned boosts confidence;
  opposed skips when OVERNIGHT_MACD_REQUIRE=true (default).
  No 09:25 flatten; stays open through RTH until SL/TP, Ctrl+C, or the ema15
  day bot actually holds Lucid. CME closed Fri 17:00–Sun 18:00 ET
  (plus Mon–Thu 17:00–18:00 halt). Fade is OFF.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz

from src.strategy.mnq_15m_ema_eod import (
    EMA_SLOW,
    ET,
    _as_utc_frame,
    _atr,
    _ohlc_resample,
    ema_series,
    sl_pts_from_atr as ema15_sl_pts,
    trend_15m_on_1m,
    trend_60m_on_1m,
)
from src.utils.trading_session import is_globex_session_et

SESSION_NAME = "overnight_research"
PAPER_RECIPE = "break_roll20"
POINT_VALUE = 2.0
COMMISSION = 1.24
QTY = 1
STACK_QTY = 2
LOSS_STREAK_FOR_COOLDOWN = 5
COOLDOWN_MINUTES = 30
FLATTEN_MINUTE = 9 * 60 + 25  # kept for overnight *backtests* only; paper runner does not flatten
NO_ENTRY_MINUTE = 9 * 60 + 20
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60
GLOBEX_OPEN_MINUTE = 18 * 60
WARMUP_BARS = 60
SETTLE_BARS = 0
ROLL_BARS = 20  # last 20 completed 1m high/low — session ONH/ONL was too wide to fill
MAX_FILLS_PER_NIGHT = 20  # per ET calendar date (00:00 ET), not per Globex night
EXTRA_FILLS_IF_GREEN = 5
MAX_FILLS_GREEN = MAX_FILLS_PER_NIGHT + EXTRA_FILLS_IF_GREEN
ATR_MULT = 1.5
SL_MIN = 12.0
SL_MAX = 40.0
# Default TP multiple of stop (env OVERNIGHT_TP_R overrides). 0.65R from
# overnight paper MFE: many 1R/0.8R SL losers peaked ~0.6–0.9R then reversed.
TP_R = 0.65
ROUND_INCREMENT = 50.0
TOUCH_TICK = 0.25
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def overnight_macd_enabled(environ: Optional[Dict[str, str]] = None) -> bool:
    """Use 15m MACD histogram as overnight direction filter (default on)."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("OVERNIGHT_MACD", "true")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def overnight_macd_require(environ: Optional[Dict[str, str]] = None) -> bool:
    """Skip entry when MACD is opposed to the break side (default on)."""
    if not overnight_macd_enabled(environ):
        return False
    env = environ if environ is not None else os.environ
    raw = str(env.get("OVERNIGHT_MACD_REQUIRE", "true")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def overnight_tp_r(environ: Optional[Dict[str, str]] = None) -> float:
    """TP distance as fraction of stop (OVERNIGHT_TP_R, default 0.65)."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("OVERNIGHT_TP_R", str(TP_R))).strip()
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float(TP_R)
    if v <= 0 or v > 5:
        return float(TP_R)
    return v


def overnight_tp_text(environ: Optional[Dict[str, str]] = None) -> str:
    """Human label e.g. '0.65R TP' or '1R TP'."""
    r = overnight_tp_r(environ)
    if abs(r - round(r)) < 1e-9:
        return f"{int(round(r))}R TP"
    s = f"{r:.2f}".rstrip("0").rstrip(".")
    return f"{s}R TP"

CUE_FADE_ONH = "fade_onh"
CUE_FADE_ONL = "fade_onl"
CUE_BREAK_ONH = "break_onh"
CUE_BREAK_ONL = "break_onl"

ZONE_ONH = "overnight_high"
ZONE_ONL = "overnight_low"
ZONE_RTH_HIGH = "prior_rth_high"
ZONE_RTH_LOW = "prior_rth_low"
ZONE_ROUND = "round_number"
ZONE_EMA15 = "ema15_slow"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def overnight_min_confidence() -> int:
    """Min win-confidence % for overnight entry (0 = off; live default)."""
    raw = os.getenv("OVERNIGHT_MIN_CONFIDENCE", "0").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def overnight_addon_confidence_min() -> int:
    """Min win-confidence % to stack a 2nd+ open overnight paper entry (env default 80)."""
    raw = os.getenv("OVERNIGHT_ADDON_CONFIDENCE_MIN", "80").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 80
    return max(0, min(100, n))


def overnight_max_open() -> int:
    """Max concurrent open overnight paper positions (env default 2)."""
    raw = os.getenv("OVERNIGHT_MAX_OPEN", "2").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 2
    return max(1, n)


def allow_overnight_entry_confidence(confidence_pct: Any) -> bool:
    """True when signal win-confidence meets OVERNIGHT_MIN_CONFIDENCE (0 = always)."""
    need = overnight_min_confidence()
    if need <= 0:
        return True
    try:
        pct = int(confidence_pct)
    except (TypeError, ValueError):
        return False
    return pct >= need


def allow_overnight_addon(open_count: int, confidence_pct: Any) -> bool:
    """True when flat (first entry) or new signal confidence meets add-on threshold."""
    if int(open_count or 0) <= 0:
        return True
    try:
        pct = int(confidence_pct)
    except (TypeError, ValueError):
        return False
    return pct >= overnight_addon_confidence_min()


def overnight_open_slots_full(open_count: int) -> bool:
    return int(open_count or 0) >= overnight_max_open()


def overnight_entry_dedupe_sec() -> int:
    """Min seconds between entries on the same cue+zone+side fingerprint."""
    raw = os.getenv("OVERNIGHT_ENTRY_DEDUPE_SEC", "120").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 120
    return max(0, n)


def overnight_entry_fingerprint(sig: Mapping[str, Any]) -> str:
    cue = str(sig.get("cue") or "")
    side = str(sig.get("side") or sig.get("direction") or "").lower()
    try:
        level = round(float(sig.get("zone_price") or 0), 2)
    except (TypeError, ValueError):
        level = 0.0
    return f"{cue}|{side}|{level}"


def overnight_same_cue_open(
    positions: Sequence[Any],
    cue: Any,
) -> bool:
    """True when an open position already uses this breakout cue."""
    cue_s = str(cue or "")
    if not cue_s:
        return False
    for pos in positions or []:
        if str(pos.get("cue") or "") == cue_s:
            return True
    return False


def overnight_recent_entry_blocked(
    sig: Mapping[str, Any],
    *,
    last_fingerprint: str = "",
    last_entry_at: float = 0.0,
    now_ts: Optional[float] = None,
) -> bool:
    """True when the same cue+zone+side fired too recently (rapid duplicate)."""
    dedupe = overnight_entry_dedupe_sec()
    if dedupe <= 0 or not last_fingerprint:
        return False
    fp = overnight_entry_fingerprint(sig)
    if fp != last_fingerprint:
        return False
    ts = time.time() if now_ts is None else float(now_ts)
    return (ts - float(last_entry_at or 0)) < dedupe


def paper_max_fills_base() -> Optional[int]:
    """PAPER_MAX_FILLS: 0 or unset = unlimited; positive = base ET-day cap (+5 if green)."""
    raw = os.getenv("PAPER_MAX_FILLS", "0").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def overnight_fill_limit(fills: Any = 0, session_pnl: Any = None) -> Optional[int]:
    """None = unlimited. When capped: base until base closed fills; +5 if net green."""
    base = paper_max_fills_base()
    if base is None:
        return None
    n = _as_int(fills, 0)
    pnl = _as_float(session_pnl)
    if n >= base and pnl is not None and pnl > 0:
        return base + EXTRA_FILLS_IF_GREEN
    return base


def overnight_cap_reached(fills: Any = 0, session_pnl: Any = None) -> bool:
    """True when ET-day fill cap is reached. Unlimited when PAPER_MAX_FILLS<=0."""
    limit = overnight_fill_limit(fills, session_pnl)
    if limit is None:
        return False
    n = _as_int(fills, 0)
    return n >= limit


def is_recipe_close_row(row: Optional[Mapping[str, Any]]) -> bool:
    if not row:
        return False
    if str(row.get("exit_reason") or "") == "restart_flat_reconcile":
        return False
    cue = str(row.get("cue") or "")
    if cue == "test_fill":
        return False
    return cue in (CUE_BREAK_ONH, CUE_BREAK_ONL)


def recipe_session_stats(rows: Any, start) -> Tuple[int, float]:
    """Closed break_onh/break_onl count and net USD since `start`. Skips test_fill.

    Paper runner passes ET calendar midnight so the 20/+5 cap is per local date.
    """
    n = 0
    pnl = 0.0
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    for row in rows or []:
        if not is_recipe_close_row(row):
            continue
        ts = row.get("exit_time") or row.get("timestamp")
        if not ts:
            continue
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        if t < start_ts:
            continue
        n += 1
        usd = _as_float(row.get("pnl_usd"))
        if usd is not None:
            pnl += usd
    return n, round(pnl, 4)


def overnight_stack_gate(
    side: str,
    last: float,
    close: float,
    level: float,
    trend_15m: int = 0,
    trend_60m: int = 0,
    macd_trend: int = 0,
) -> Dict[str, Any]:
    """Stack gate breakdown for qty sizing and logging. Gemini never sizes."""
    try:
        last_f = float(last)
        close_f = float(close)
        lvl = float(level)
    except (TypeError, ValueError):
        return {
            "qty": QTY,
            "last_ok": False,
            "close_ok": False,
            "trend_ok": False,
            "macd_ok": True,
            "trend_15m": int(trend_15m or 0),
            "trend_60m": int(trend_60m or 0),
            "macd_trend": int(macd_trend or 0),
        }
    long_side = str(side).lower() in ("long", "buy")
    tick = float(TOUCH_TICK)
    if long_side:
        last_ok = last_f > lvl + tick
        close_ok = close_f > lvl + tick
        trend_ok = int(trend_15m) == 1 and int(trend_60m) == 1
    else:
        last_ok = last_f < lvl - tick
        close_ok = close_f < lvl - tick
        trend_ok = int(trend_15m) == -1 and int(trend_60m) == -1
    macd_ok = overnight_macd_ok(side, int(macd_trend or 0))
    qty = STACK_QTY if (last_ok and close_ok and trend_ok) else QTY
    return {
        "qty": qty,
        "last_ok": last_ok,
        "close_ok": close_ok,
        "trend_ok": trend_ok,
        "macd_ok": macd_ok,
        "trend_15m": int(trend_15m or 0),
        "trend_60m": int(trend_60m or 0),
        "macd_trend": int(macd_trend or 0),
    }


def overnight_stack_qty(
    side: str,
    last: float,
    close: float,
    level: float,
    trend_15m: int = 0,
    trend_60m: int = 0,
) -> int:
    """Qty 2 only on extreme stack. Gemini never sizes. Else qty 1."""
    return int(overnight_stack_gate(
        side, last, close, level, trend_15m, trend_60m,
    ).get("qty") or QTY)


def should_defer_entry_for_stack(
    side: str,
    last: float,
    close: float,
    level: float,
    trend_15m: int = 0,
    trend_60m: int = 0,
) -> bool:
    """Hold briefly for completed 1m close when last+HTF agree (qty-2 stack)."""
    gate = overnight_stack_gate(side, last, close, level, trend_15m, trend_60m)
    if int(gate.get("qty") or QTY) == STACK_QTY:
        return False
    return bool(gate.get("last_ok") and gate.get("trend_ok") and not gate.get("close_ok"))


def overnight_stack_why_suffix(gate: Mapping[str, Any]) -> str:
    if int(gate.get("qty") or QTY) == STACK_QTY:
        return " qty=2 stack"
    parts: List[str] = []
    if not gate.get("last_ok"):
        parts.append("last pending")
    if not gate.get("close_ok"):
        parts.append("1m close pending")
    if not gate.get("trend_ok"):
        parts.append(
            f"HTF {gate.get('trend_15m')}/{gate.get('trend_60m')} vs side"
        )
    if not parts:
        return ""
    return f" qty=1 ({', '.join(parts)})"


def _htf_trend_points(trend: int, side_long: bool) -> Tuple[int, str]:
    """Points and label for one HTF EMA trend vs trade direction."""
    want = 1 if side_long else -1
    t = int(trend or 0)
    if t == want:
        return 12, "aligned"
    if t == 0:
        return 3, "flat"
    return -6, "opposed"


def _macd_trend_points(trend: int, side_long: bool) -> Tuple[int, str]:
    """Points and label for 15m MACD hist vs trade direction."""
    want = 1 if side_long else -1
    t = int(trend or 0)
    if t == want:
        return 10, "aligned"
    if t == 0:
        return 2, "flat"
    return -8, "opposed"


def recipe_historical_win_pct(
    closed_rows: Any,
    cue: str,
    *,
    min_trades: int = 10,
) -> Dict[str, Any]:
    """Optional journal nudge: win rate for same break cue (paper closes only)."""
    cue = str(cue or "")
    if not cue:
        return {"historical_win_pct": None, "historical_n": 0}
    wins = 0
    n = 0
    for row in closed_rows or []:
        if not is_recipe_close_row(row):
            continue
        if str(row.get("cue") or "") != cue:
            continue
        n += 1
        if float(row.get("pnl_usd") or 0) > 0:
            wins += 1
    if n < min_trades:
        return {"historical_win_pct": None, "historical_n": n}
    return {"historical_win_pct": round(100.0 * wins / n, 1), "historical_n": n}


def entry_confidence_pct(
    side: str,
    gate: Mapping[str, Any],
    *,
    historical_win_pct: Optional[float] = None,
    historical_n: int = 0,
) -> Dict[str, Any]:
    """Heuristic win-confidence at entry — not a calibrated ML probability.

    Transparent score from break_roll20 stack-gate components:
      50 base — last through 20m extreme (required to enter)
      +15 — completed 1m close also through level
       +5 — last through but 1m close still pending (weaker)
      +12/+3/-6 — 15m EMA8/21 aligned / flat / opposed
      +12/+3/-6 — 60m EMA8/21 aligned / flat / opposed
      +10/+2/-8 — 15m MACD hist aligned / flat / opposed (when OVERNIGHT_MACD)
       +8 — full stack (qty=2 gate: last + close + both HTF agree)
       ±2 — soft journal nudge when >=10 same-cue paper closes
            (win% >=55 or <=40). Never alone enough to kill a solid stack.

    Clamped 42–92%. Display as "Confidence: N%" with breakdown subtitle.
    """
    side_long = str(side).lower() in ("long", "buy")
    breakdown: List[str] = ["base break 50"]
    score = 50

    if gate.get("close_ok"):
        score += 15
        breakdown.append("1m close +15")
    elif gate.get("last_ok"):
        score += 5
        breakdown.append("1m close pending +5")

    t15_pts, t15_lbl = _htf_trend_points(int(gate.get("trend_15m") or 0), side_long)
    score += t15_pts
    breakdown.append(f"15m {t15_lbl} {t15_pts:+d}")

    t60_pts, t60_lbl = _htf_trend_points(int(gate.get("trend_60m") or 0), side_long)
    score += t60_pts
    breakdown.append(f"60m {t60_lbl} {t60_pts:+d}")

    if overnight_macd_enabled():
        macd_pts, macd_lbl = _macd_trend_points(int(gate.get("macd_trend") or 0), side_long)
        score += macd_pts
        breakdown.append(f"MACD {macd_lbl} {macd_pts:+d}")

    if int(gate.get("qty") or QTY) == STACK_QTY:
        score += 8
        breakdown.append("full stack +8")

    stack_score = score
    hist_note = ""
    hist_delta = 0
    if historical_n >= 10 and historical_win_pct is not None:
        hw = float(historical_win_pct)
        if hw >= 55.0:
            hist_delta = 2
            hist_note = f"journal {hw:.0f}% win +2"
            breakdown.append(hist_note)
        elif hw <= 40.0:
            hist_delta = -2
            hist_note = f"journal {hw:.0f}% win -2"
            breakdown.append(hist_note)
        score += hist_delta

    # Journal is advisory only: a losing streak cannot alone force a skip when
    # the break stack itself already clears OVERNIGHT_MIN_CONFIDENCE.
    min_need = overnight_min_confidence()
    if hist_delta < 0 and stack_score >= min_need and score < min_need:
        score = min_need
        breakdown.append(f"journal floor @ {min_need} (stack held)")

    pct = int(max(42, min(92, round(score))))
    summary_parts: List[str] = []
    if gate.get("close_ok"):
        summary_parts.append("1m close confirmed")
    elif gate.get("last_ok"):
        summary_parts.append("last-only break")
    if gate.get("trend_ok"):
        summary_parts.append("HTF aligned")
    elif t15_pts > 0 or t60_pts > 0:
        summary_parts.append("partial HTF")
    else:
        summary_parts.append("HTF mixed/flat")
    if overnight_macd_enabled():
        macd_t = int(gate.get("macd_trend") or 0)
        want = 1 if side_long else -1
        if macd_t == want:
            summary_parts.append("MACD aligned")
        elif macd_t == 0:
            summary_parts.append("MACD flat")
        else:
            summary_parts.append("MACD opposed")
    if int(gate.get("qty") or QTY) == STACK_QTY:
        summary_parts.append("qty-2 stack")
    if hist_note:
        summary_parts.append(
            hist_note.replace(" +2", "").replace(" -2", "")
            .replace(" +5", "").replace(" -5", "")
        )
    return {
        "pct": pct,
        "breakdown": breakdown,
        "summary": ", ".join(summary_parts),
        "stack_pct": int(max(42, min(92, round(stack_score)))),
        "journal_delta": int(hist_delta),
    }


def format_entry_confidence_line(conf: Mapping[str, Any]) -> str:
    pct = int(conf.get("pct") or 0)
    summary = str(conf.get("summary") or "").strip()
    if summary:
        return f"Confidence: {pct}% ({summary})"
    return f"Confidence: {pct}%"


def console_color_enabled() -> bool:
    """True when stdout likely supports ANSI (Windows 10+ cmd, most Unix TTYs)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.name == "nt":
        return True
    try:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        return False


def _ansi(text: str, code: str, *, enabled: bool) -> str:
    if not enabled or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _confidence_ansi_code(pct: int) -> str:
    if pct >= 75:
        return "1;92"
    if pct >= 60:
        return "1;93"
    return "1;96"


def format_trade_placed_breakdown(sig: Mapping[str, Any]) -> str:
    """Human-readable one-liner for the black console banner."""
    if str(sig.get("cue") or "") == "test_fill":
        return "test fill · desk proof only · qty 1"
    conf = sig.get("entry_confidence") or {}
    summary = str(conf.get("summary") or "").strip()
    detail = " ".join(str(x) for x in (conf.get("breakdown") or []))
    qty = int(sig.get("qty") or QTY)
    parts: List[str] = []
    if "HTF aligned" in summary or ("15m aligned" in detail and "60m aligned" in detail):
        parts.append("15m+60m aligned")
    elif "partial HTF" in summary:
        parts.append("partial HTF")
    elif "HTF mixed" in summary or "HTF mixed/flat" in summary:
        parts.append("HTF mixed/flat")
    if "1m close confirmed" in summary or "1m close +15" in detail:
        parts.append("1m close confirmed")
    elif "last-only break" in summary or "1m close pending" in detail:
        parts.append("last-only break")
    if int(qty) == STACK_QTY:
        parts.append(f"qty {qty} stack")
    else:
        parts.append(f"qty {qty}")
    for token in summary.split(", "):
        if token.startswith("journal ") and "win" in token:
            parts.append(token)
            break
    if parts:
        return " · ".join(parts)
    return summary or "break recipe"


def format_trade_placed_banner(
    sig: Mapping[str, Any],
    *,
    color: Optional[bool] = None,
    width: int = 47,
    sl_ok: str = "",
    tp_ok: str = "",
) -> str:
    """Single-line entry banner (emoji + SL/TP) for cmd.exe / PowerShell."""
    side_raw = str(sig.get("side") or sig.get("direction") or "trade").lower()
    if side_raw in ("long", "buy"):
        side_label = "LONG"
    elif side_raw in ("short", "sell"):
        side_label = "SHORT"
    else:
        side_label = side_raw.upper()
    try:
        entry = float(sig.get("entry_price") or sig.get("entry") or 0)
    except (TypeError, ValueError):
        entry = 0.0
    stop = sig.get("stop")
    target = sig.get("target")
    try:
        sl = float(stop) if stop not in (None, "") else None
    except (TypeError, ValueError):
        sl = None
    try:
        tp = float(target) if target not in (None, "") else None
    except (TypeError, ValueError):
        tp = None
    ticket = str(sig.get("lucid_ticket") or sig.get("trade_id") or "")
    conf = sig.get("entry_confidence") or {}
    pct = conf.get("pct")
    if pct is None:
        pct = sig.get("entry_confidence_pct")
    extra_parts: List[str] = []
    if pct is not None:
        try:
            extra_parts.append(f"Win {int(pct)}%")
        except (TypeError, ValueError):
            pass
    breakdown = format_trade_placed_breakdown(sig)
    if breakdown:
        extra_parts.append(breakdown)
    if str(sig.get("cue") or "") == "test_fill":
        extra_parts.append("test fill")
    from src.utils.trade_console import format_entry_line

    return format_entry_line(
        direction=side_raw,
        entry=entry,
        sl=sl,
        tp=tp,
        ticket=ticket,
        side_label=side_label,
        sl_ok=sl_ok,
        tp_ok=tp_ok,
        extra=" · ".join(extra_parts),
    )


def overnight_htf_sides(df_1m: Optional[pd.DataFrame]) -> Tuple[int, int]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return 0, 0
    try:
        t15 = int(trend_15m_on_1m(df_1m).iloc[-1])
        t60 = int(trend_60m_on_1m(df_1m).iloc[-1])
        return t15, t60
    except Exception:
        return 0, 0


def macd_trend_15m_on_1m(
    df_1m: Optional[pd.DataFrame],
    *,
    completed_only: bool = True,
) -> int:
    """Completed 15m MACD histogram direction: +1 bull, -1 bear, 0 flat/unknown."""
    if df_1m is None or getattr(df_1m, "empty", True):
        return 0
    try:
        tmp = _as_utc_frame(df_1m)
        b15 = _ohlc_resample(tmp, "15min")
        if b15 is None or len(b15) < MACD_SLOW + MACD_SIGNAL + 2:
            return 0
        close = b15["close"]
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist = macd - signal
        if completed_only:
            hist = hist.shift(1)
        hist = hist.dropna()
        if hist.empty:
            return 0
        h = float(hist.iloc[-1])
        if h > 0:
            return 1
        if h < 0:
            return -1
        return 0
    except Exception:
        return 0


def overnight_macd_ok(side: str, macd_trend: int) -> bool:
    """True when MACD is not opposed to the trade side (flat OK)."""
    want = 1 if str(side).lower() in ("long", "buy") else -1
    t = int(macd_trend or 0)
    if t == 0:
        return True
    return t == want


def allow_overnight_macd_entry(side: str, macd_trend: int) -> bool:
    """Hard gate: False only when REQUIRE is on and MACD is opposed."""
    if not overnight_macd_require():
        return True
    return overnight_macd_ok(side, macd_trend)


def _session_recipe_closes(rows: Any, start) -> List[Tuple[pd.Timestamp, float]]:
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    out: List[Tuple[pd.Timestamp, float]] = []
    for row in rows or []:
        if not is_recipe_close_row(row):
            continue
        ts = row.get("exit_time") or row.get("timestamp")
        if not ts:
            continue
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        if t < start_ts:
            continue
        usd = _as_float(row.get("pnl_usd"))
        out.append((t, 0.0 if usd is None else usd))
    out.sort(key=lambda x: x[0])
    return out


def recipe_cooldown_until(rows: Any, start, now=None) -> Optional[datetime]:
    """ET datetime until which entries are paused, or None if not cooling down.

    Each fresh streak of loss_streak_for_cooldown_from_env() closed recipe losers
    (pnl<=0) starts a COOLDOWN_MINUTES pause from that Nth close. After it expires
    the streak resets so the same losses do not re-trigger. Skip test_fill.
    OVERNIGHT_SKIP_COOLDOWN=1 or OVERNIGHT_COOLDOWN_LOSSES=0 disables cooldown.
    """
    if overnight_skip_cooldown_from_env():
        return None
    streak_threshold = loss_streak_for_cooldown_from_env()
    if streak_threshold <= 0:
        return None
    cooldown_until: Optional[pd.Timestamp] = None
    streak = 0
    for t, pnl in _session_recipe_closes(rows, start):
        if cooldown_until is not None and t < cooldown_until:
            continue
        if cooldown_until is not None and t >= cooldown_until:
            cooldown_until = None
            streak = 0
        if pnl <= 0:
            streak += 1
            if streak >= streak_threshold:
                cooldown_until = t + pd.Timedelta(minutes=COOLDOWN_MINUTES)
                streak = 0
        else:
            streak = 0
    if cooldown_until is None:
        return None
    until_et = cooldown_until.tz_convert(ET).to_pydatetime()
    if pd.Timestamp(_to_et(now)) >= pd.Timestamp(until_et):
        return None
    return until_et


def in_overnight_cooldown(cooldown_until, now=None) -> bool:
    if overnight_skip_cooldown_from_env():
        return False
    if cooldown_until is None:
        return False
    try:
        return pd.Timestamp(_to_et(now)) < pd.Timestamp(_to_et(cooldown_until))
    except (TypeError, ValueError):
        return False


def overnight_flag_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_OVERNIGHT", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def paper_test_fill_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_TEST_FILL", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_truthy(key: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get(key, "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def overnight_skip_cooldown_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Temporary override: ignore loss-streak cooldown (existing + new triggers)."""
    return _env_truthy("OVERNIGHT_SKIP_COOLDOWN", environ)


def loss_streak_for_cooldown_from_env(environ: Optional[Mapping[str, str]] = None) -> int:
    """Consecutive losers before cooldown; 0 disables the trigger (default 5)."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("OVERNIGHT_COOLDOWN_LOSSES", "")).strip()
    if not raw:
        return LOSS_STREAK_FOR_COOLDOWN
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return LOSS_STREAK_FOR_COOLDOWN
    return max(0, n)


def overnight_cooldown_override_message(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Startup log line when loss cooldown is bypassed, or None."""
    if overnight_skip_cooldown_from_env(environ):
        return "Loss cooldown OVERRIDDEN (OVERNIGHT_SKIP_COOLDOWN=1)"
    if loss_streak_for_cooldown_from_env(environ) <= 0:
        return "Loss cooldown DISABLED (OVERNIGHT_COOLDOWN_LOSSES=0)"
    return None


def enable_overnight_lucid_sim_orders(
    environ: Optional[MutableMapping[str, str]] = None,
    *,
    keep_test_fill: bool = False,
) -> MutableMapping[str, str]:
    """Overnight research submits Lucid TEST/sim orders. Not local fake fills."""
    env = environ if environ is not None else os.environ
    env["PAPER_OVERNIGHT"] = "true"
    env["PAPER_USE_RITHMIC"] = "true"
    env["PAPER_RITHMIC_BRACKETS"] = "true"
    env["OVERNIGHT_TRADING"] = "false"
    env["TRADING_MODE"] = "paper"
    env["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
    env["RITHMIC_ALLOW_SIMULATOR"] = "true"
    env["RITHMIC_SKIP_HISTORY_PNL"] = "true"
    env["RITHMIC_QUOTES_ONLY"] = "false"
    env["RITHMIC_ORDER_PLANT_ONLY"] = "false"
    if not keep_test_fill:
        env["PAPER_TEST_FILL"] = "false"
    return env


def skip_local_fake_fill(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when a fill must be booked on Lucid — never journaled locally."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("PAPER_RITHMIC_BRACKETS", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def paper_test_fill_signal(
    df_1m: Optional[pd.DataFrame],
    last_price: float,
) -> Dict[str, Any]:
    """Immediate 1 MNQ paper fill for desk testing. Never a live order.

    Side follows completed 15m EMA8 vs EMA21 when known, else long.
    Stop = clip(ATR14×2, 20–60) when ATR is known, else 20 pts.
    """
    px = round(float(last_price), 2)
    side = "long"
    sl_pts = 20.0
    if df_1m is not None and not getattr(df_1m, "empty", True):
        try:
            trend = trend_15m_on_1m(df_1m, completed_only=True)
            last_t = int(trend.iloc[-1]) if trend is not None and not trend.empty else 0
            if last_t < 0:
                side = "short"
            elif last_t > 0:
                side = "long"
            tmp = _as_utc_frame(df_1m)
            b15 = _ohlc_resample(tmp, "15min")
            atr = _atr(b15, 14).shift(1).dropna()
            if not atr.empty:
                sl_pts = float(ema15_sl_pts(float(atr.iloc[-1]), fallback=20.0))
        except Exception:
            side = "long"
            sl_pts = 20.0
    sl_pts = float(np.clip(sl_pts, 20.0, 60.0))
    tp_r = overnight_tp_r()
    if side == "short":
        stop = round(px + sl_pts, 2)
        target = round(px - sl_pts * tp_r, 2)
    else:
        stop = round(px - sl_pts, 2)
        target = round(px + sl_pts * tp_r, 2)
    return {
        "session": SESSION_NAME,
        "cue": "test_fill",
        "side": side,
        "direction": side,
        "qty": QTY,
        "symbol": "MNQ",
        "entry_price": px,
        "stop": stop,
        "target": target,
        "atr_stop_pts": round(sl_pts, 2),
        "zone_name": "test_fill",
        "zone_price": px,
        "why": "test_fill — Lucid TEST011 sim proof, does not lock today's break",
        "flatten_et": "",
        "paper": True,
    }


def resolve_overnight_research(
    *,
    live: bool = False,
    flag: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Overnight paper never runs under --live. Locked RTH recipe stays RTH."""
    if live:
        return False
    return bool(flag) or overnight_flag_from_env(environ)


def _to_et(now) -> datetime:
    if now is None:
        return datetime.now(ET)
    t = pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).to_pydatetime()


def et_minutes(now) -> int:
    et = _to_et(now)
    return int(et.hour * 60 + et.minute)


def globex_elapsed_minutes(now=None) -> int:
    """Minutes since this Globex open (18:00 ET). Late start still counts."""
    start = globex_session_start_et(now)
    return max(0, int((_to_et(now) - start).total_seconds() // 60))


def overnight_warmup_ready(bars: int = 0, now=None) -> bool:
    """60 live bars or 60 minutes after 18:00 — do not wait 60 minutes after a late start."""
    return int(bars or 0) >= WARMUP_BARS or globex_elapsed_minutes(now) >= WARMUP_BARS


def globex_session_start_et(now=None) -> datetime:
    """Last Globex open (18:00 ET Sun–Thu)."""
    et = _to_et(now)
    mins = et_minutes(et)
    start = et.replace(hour=18, minute=0, second=0, microsecond=0)
    if mins < GLOBEX_OPEN_MINUTE:
        start = start - timedelta(days=1)
    if start.weekday() == 5:  # Saturday — Sunday 18:00 is the open
        start = start - timedelta(days=1)
    return start


def fill_window_start_et(now=None) -> datetime:
    """00:00 ET of the current calendar date — 20-fill cap resets here, not at 18:00."""
    et = _to_et(now)
    return et.replace(hour=0, minute=0, second=0, microsecond=0)


def in_rth_hard_idle(now=None) -> bool:
    """Deprecated: paper research no longer idles 9:30–16:00. Always False."""
    _ = now
    return False


def is_overnight_research_hours(now=None) -> bool:
    """CME Globex hours (is_globex_session_et). RTH is allowed. Weekend/halt is not."""
    if not is_globex_session_et(now):
        return False
    if day_bot_owns_lucid(now):
        return False
    return True


def should_flatten_before_rth(now=None) -> bool:
    """Paper research does not flatten at 09:25. Always False."""
    _ = now
    return False


def _looks_like_ema15_day_bot(cmdline: str) -> bool:
    c = " ".join(str(cmdline or "").lower().split())
    if "start_live_mtf_scalping" not in c:
        return False
    if "--overnight-research" in c:
        return False
    return "--live" in c


_DAY_BOT_PROC_CACHE: Dict[str, Any] = {"at": 0.0, "val": False}


def ema15_day_bot_process_running() -> bool:
    """True if start_live_mtf_scalping.py --live is running (start_mnq_live.bat)."""
    now = time.time()
    if now - float(_DAY_BOT_PROC_CACHE.get("at") or 0) < 20.0:
        return bool(_DAY_BOT_PROC_CACHE.get("val"))
    found = False
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }",
            ],
            stderr=subprocess.DEVNULL,
            timeout=8,
            creationflags=flags,
        )
        text = out.decode("utf-8", "ignore")
        for line in text.splitlines():
            if _looks_like_ema15_day_bot(line):
                found = True
                break
    except Exception:
        found = False
    _DAY_BOT_PROC_CACHE["at"] = now
    _DAY_BOT_PROC_CACHE["val"] = found
    return found


def day_bot_owns_lucid(now=None, environ: Optional[Mapping[str, str]] = None) -> bool:
    """True only if the ema15 day bot is actually using Lucid — not a 09:20 clock.

    DAY_BOT_OWNS_LUCID=true/false overrides the process scan (tests).
    """
    _ = now
    env = environ if environ is not None else os.environ
    raw = str(env.get("DAY_BOT_OWNS_LUCID", "")).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return ema15_day_bot_process_running()


def overnight_entry_blocked_reason(now=None) -> Optional[str]:
    """Hard stop when CME is shut or the day bot holds the Lucid login."""
    if day_bot_owns_lucid(now):
        return (
            "Day ema15 bot owns Lucid — Ctrl+C start_mnq_live.bat first "
            "(one Lucid session)"
        )
    if not is_globex_session_et(now):
        return "CME closed — no new entries"
    if not allow_new_overnight_entry(now):
        return "Overnight session closed — no new entries"
    return None


def allow_new_overnight_entry(now=None) -> bool:
    return is_overnight_research_hours(now)


def _mask_rth(idx: pd.DatetimeIndex) -> np.ndarray:
    et = idx.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    return (et.weekday < 5) & (mins >= RTH_OPEN_MINUTE) & (mins < RTH_END_MINUTE)


def overnight_completed_bars(df_1m: pd.DataFrame, now=None) -> pd.DataFrame:
    """Completed 1m bars from this Globex open up to `now`, excluding an unfinished last minute."""
    if df_1m is None or getattr(df_1m, "empty", True):
        return pd.DataFrame()
    tmp = _as_utc_frame(df_1m).sort_index()
    if "datetime" in tmp.columns:
        tmp = tmp.drop(columns=["datetime"])
    start = globex_session_start_et(now)
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(_to_et(now)).tz_convert("UTC")
    out = tmp.loc[(tmp.index >= start_utc) & (tmp.index <= end)]
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if out.empty:
        return pd.DataFrame(columns=[c for c in cols if c == "datetime" or c in tmp.columns])
    last = out.index[-1]
    if (end - last) < pd.Timedelta(minutes=1) and len(out) > 1:
        out = out.iloc[:-1]
    out = out.copy()
    out.insert(0, "datetime", out.index.tz_convert("UTC"))
    return out.reset_index(drop=True)


def prior_rth_high_low(df_1m: pd.DataFrame, now=None) -> Tuple[Optional[float], Optional[float]]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None, None
    tmp = _as_utc_frame(df_1m)
    rth = tmp.loc[_mask_rth(tmp.index)]
    if rth.empty:
        return None, None
    start = globex_session_start_et(now)
    cutoff = pd.Timestamp(start).tz_convert("UTC")
    rth = rth.loc[rth.index < cutoff]
    if rth.empty:
        return None, None
    et_dates = rth.index.tz_convert(ET).date
    last_day = et_dates[-1]
    day = rth.loc[et_dates == last_day]
    if day.empty:
        return None, None
    return float(day["high"].max()), float(day["low"].min())


def _last_vs_level(last: Optional[float], level: Optional[float], *, long_side: bool) -> str:
    """How far last is from a break close. Empty if prices missing."""
    if last is None or level is None:
        return ""
    last_px = float(last)
    lvl = float(level)
    if long_side:
        need = lvl + TOUCH_TICK
        gap = need - last_px
        if gap > 1:
            return f"Last {last_px:.2f} is {gap:.1f} pts too low — needs to rise."
        if gap < -1:
            return (
                f"Last {last_px:.2f} is already {abs(gap):.1f} pts above "
                f"— should fill."
            )
        return f"Last {last_px:.2f} is on the long trigger — last through {need:.2f} fills."
    need = lvl - TOUCH_TICK
    gap = last_px - need
    if gap > 1:
        return f"Last {last_px:.2f} is {gap:.1f} pts too high — needs to fall."
    if gap < -1:
        return (
            f"Last {last_px:.2f} is already {abs(gap):.1f} pts below "
            f"— should fill."
        )
    return f"Last {last_px:.2f} is on the short trigger — last through {need:.2f} fills."


def _side_wait(
    *,
    label: str,
    level: Optional[float],
    settled: bool,
    age: Optional[int],
    last: Optional[float],
    long_side: bool,
) -> str:
    if level is None:
        return f"{label}: no level yet."
    px = float(level)
    gap = _last_vs_level(last, px, long_side=long_side)
    if settled:
        trigger = f"last through {px:.2f}" if long_side else f"last through {px:.2f} on the downside"
        return f"{label}: need {trigger}. {gap}".strip()
    left = None if age is None else max(0, SETTLE_BARS - int(age))
    printing_new = (
        last is not None
        and (
            (long_side and float(last) > px + TOUCH_TICK)
            or ((not long_side) and float(last) < px - TOUCH_TICK)
        )
    )
    if printing_new:
        return (
            f"{label}: last is making a new {'high' if long_side else 'low'}. "
            f"Clock resets — wait {SETTLE_BARS} min with no newer extreme, "
            f"then a 1m close through it. {gap}"
        ).strip()
    if left is None:
        clock = f"{px:.0f} is not 20 min old yet"
    elif left <= 0:
        clock = f"{px:.2f} is set"
    else:
        clock = f"{px:.0f} is {int(age)} min old — wait {left} more min"
    return f"{label}: {clock}. {gap}".strip()


def explain_overnight_wait(
    *,
    bars: int = 0,
    onh: Optional[float] = None,
    onl: Optional[float] = None,
    onh_settled: bool = False,
    onl_settled: bool = False,
    onh_age: Optional[int] = None,
    onl_age: Optional[int] = None,
    last: Optional[float] = None,
    now=None,
    already_filled: bool = False,
    fills_tonight: Optional[int] = None,
    session_pnl: Any = None,
    cooldown_until=None,
) -> str:
    """Plain-English why overnight is sitting. Not a trade cue."""
    n_fills = fills_tonight
    try:
        n_fills = None if n_fills is None else int(n_fills)
    except (TypeError, ValueError):
        n_fills = None
    limit = overnight_fill_limit(n_fills or 0, session_pnl)
    base = paper_max_fills_base()
    if base is not None and (already_filled or overnight_cap_reached(n_fills, session_pnl)):
        pnl = _as_float(session_pnl)
        green_cap = base + EXTRA_FILLS_IF_GREEN
        if n_fills is not None and n_fills >= green_cap:
            return f"Already took today's {green_cap} fills (ET calendar date)."
        if n_fills is not None and n_fills >= base and not (pnl is not None and pnl > 0):
            return f"{base} fills today, ET-day red — done."
        return f"Already took today's {base} fills (ET calendar date)."
    if in_overnight_cooldown(cooldown_until, now):
        hhmm = _to_et(cooldown_until).strftime("%H:%M")
        return f"cooling down until {hhmm} ET."
    n = int(bars or 0)
    if not overnight_warmup_ready(n, now):
        left = max(1, WARMUP_BARS - globex_elapsed_minutes(now))
        ready = _to_et(now) + timedelta(minutes=left)
        return (
            f"First hour after 18:00 — building the overnight range "
            f"({globex_elapsed_minutes(now)} of {WARMUP_BARS} min). "
            f"Scanning starts around {ready.strftime('%H:%M')} ET."
        )
    if SETTLE_BARS > 0 and n < SETTLE_BARS + 1:
        return (
            f"Need {SETTLE_BARS} live minutes so the high/low can settle "
            f"({n}/{SETTLE_BARS}). Then: long = 1m close above the high, "
            f"short = 1m close below the low."
        )
    if SETTLE_BARS <= 0:
        onh_settled = True
        onl_settled = True
    long_line = _side_wait(
        label="LONG", level=onh, settled=onh_settled, age=onh_age, last=last, long_side=True,
    )
    short_line = _side_wait(
        label="SHORT", level=onl, settled=onl_settled, age=onl_age, last=last, long_side=False,
    )
    extra = ""
    if n_fills is not None and n_fills > 0:
        if limit is None:
            extra = f"Fill {n_fills + 1}. "
        elif 0 < n_fills < limit:
            extra = f"Fill {n_fills + 1} of {limit}. "
            if base is not None and n_fills >= base:
                extra += "session green — 5 extra allowed. "
    return f"{extra}{long_line} {short_line}"


def wait_reason_from_zones(
    payload: Optional[Mapping[str, Any]],
    *,
    now=None,
    already_filled: bool = False,
    fills_tonight: Optional[int] = None,
    session_pnl: Any = None,
    cooldown_until=None,
) -> str:
    payload = payload or {}
    zones = {
        str(z.get("name") or ""): z
        for z in (payload.get("zones") or [])
        if isinstance(z, dict)
    }
    onh = zones.get(ZONE_ONH) or {}
    onl = zones.get(ZONE_ONL) or {}
    last = payload.get("price")
    try:
        last_f = float(last) if last is not None else None
    except (TypeError, ValueError):
        last_f = None
    onh_age = onh.get("age_bars")
    onl_age = onl.get("age_bars")
    try:
        onh_age = None if onh_age is None else int(onh_age)
        onl_age = None if onl_age is None else int(onl_age)
    except (TypeError, ValueError):
        onh_age = onl_age = None
    return explain_overnight_wait(
        bars=int(payload.get("overnight_bars") or 0),
        onh=onh.get("price"),
        onl=onl.get("price"),
        onh_settled=bool(onh.get("settled")),
        onl_settled=bool(onl.get("settled")),
        onh_age=onh_age,
        onl_age=onl_age,
        last=last_f,
        now=now,
        already_filled=already_filled,
        fills_tonight=fills_tonight,
        session_pnl=session_pnl,
        cooldown_until=cooldown_until,
    )


def settled_overnight_range(on_bars: pd.DataFrame, now=None) -> Dict[str, Any]:
    """Trade the last ROLL_BARS 1m high/low (not the full-session extreme)."""
    empty = {
        "onh": None,
        "onl": None,
        "onh_settled": False,
        "onl_settled": False,
        "bars": 0,
    }
    if on_bars is None or on_bars.empty or len(on_bars) < 2:
        return empty
    n = len(on_bars)
    prior = on_bars.iloc[:-1]
    window = prior.tail(ROLL_BARS) if len(prior) > ROLL_BARS else prior
    onh = float(window["high"].max())
    onl = float(window["low"].min())
    hi_pos = int(window["high"].values.argmax())
    lo_pos = int(window["low"].values.argmin())
    age_hi = (len(window) - 1) - hi_pos
    age_lo = (len(window) - 1) - lo_pos
    ready = overnight_warmup_ready(n, now)
    return {
        "onh": onh,
        "onl": onl,
        "onh_settled": ready and age_hi >= SETTLE_BARS,
        "onl_settled": ready and age_lo >= SETTLE_BARS,
        "bars": n,
        "onh_age_bars": age_hi,
        "onl_age_bars": age_lo,
        "roll_bars": int(len(window)),
    }


def atr_stop_pts(df_1m: pd.DataFrame) -> float:
    if df_1m is None or getattr(df_1m, "empty", True):
        return 24.0
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    atr = _atr(b15, 14).shift(1).dropna()
    if atr.empty:
        return 24.0
    return float(np.clip(float(atr.iloc[-1]) * ATR_MULT, SL_MIN, SL_MAX))


def ema15_slow_price(df_1m: pd.DataFrame) -> Optional[float]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    slow = ema_series(b15["close"], EMA_SLOW).shift(1).dropna()
    if slow.empty:
        return None
    return float(slow.iloc[-1])


def nearest_round(price: float, increment: float = ROUND_INCREMENT) -> float:
    return float(round(price / increment) * increment)


def _zone(
    name: str,
    ztype: str,
    price: Optional[float],
    *,
    extra: Optional[Dict[str, Any]] = None,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev = prev or {}
    row = {
        "name": name,
        "type": ztype,
        "price": None if price is None else round(float(price), 2),
        "touches": int(prev.get("touches") or 0),
        "last_updated": datetime.now(pytz.UTC).isoformat(),
    }
    if extra:
        row.update(extra)
    return row


def compute_zones(
    df_1m: pd.DataFrame,
    now=None,
    *,
    last_price: Optional[float] = None,
    prev_zones: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prev_map = {str(z.get("name")): z for z in (prev_zones or []) if isinstance(z, dict)}
    rng = settled_overnight_range(overnight_completed_bars(df_1m, now), now)
    rth_hi, rth_lo = prior_rth_high_low(df_1m, now)
    ema = ema15_slow_price(df_1m)
    px = last_price
    if px is None and df_1m is not None and not getattr(df_1m, "empty", True):
        px = float(df_1m.iloc[-1]["close"])
    round_px = nearest_round(float(px)) if px else None
    dist = None if ema is None or px is None else round(float(px) - float(ema), 2)
    zones = [
        _zone(
            ZONE_ONH, ZONE_ONH, rng.get("onh"),
            extra={"settled": rng.get("onh_settled"), "age_bars": rng.get("onh_age_bars")},
            prev=prev_map.get(ZONE_ONH),
        ),
        _zone(
            ZONE_ONL, ZONE_ONL, rng.get("onl"),
            extra={"settled": rng.get("onl_settled"), "age_bars": rng.get("onl_age_bars")},
            prev=prev_map.get(ZONE_ONL),
        ),
        _zone(ZONE_RTH_HIGH, ZONE_RTH_HIGH, rth_hi, prev=prev_map.get(ZONE_RTH_HIGH)),
        _zone(ZONE_RTH_LOW, ZONE_RTH_LOW, rth_lo, prev=prev_map.get(ZONE_RTH_LOW)),
        _zone(ZONE_ROUND, ZONE_ROUND, round_px, extra={"increment": ROUND_INCREMENT}, prev=prev_map.get(ZONE_ROUND)),
        _zone(ZONE_EMA15, ZONE_EMA15, ema, extra={"distance_pts": dist}, prev=prev_map.get(ZONE_EMA15)),
    ]
    return {
        "updated_at": datetime.now(pytz.UTC).isoformat(),
        "updated_at_et": pd.Timestamp(_to_et(now)).isoformat(),
        "session": SESSION_NAME,
        "globex_start_et": globex_session_start_et(now).isoformat(),
        "price": None if px is None else round(float(px), 2),
        "overnight_bars": rng.get("bars") or 0,
        "wait_reason": explain_overnight_wait(
            bars=int(rng.get("bars") or 0),
            onh=rng.get("onh"),
            onl=rng.get("onl"),
            onh_settled=bool(rng.get("onh_settled")),
            onl_settled=bool(rng.get("onl_settled")),
            onh_age=rng.get("onh_age_bars"),
            onl_age=rng.get("onl_age_bars"),
            last=px,
            now=now,
        ),
        "zones": zones,
        "advisory_only": True,
        "writes_live_config": False,
    }


def increment_zone_touch(payload: Dict[str, Any], name: str) -> None:
    for z in payload.get("zones") or []:
        if z.get("name") == name:
            z["touches"] = int(z.get("touches") or 0) + 1
            z["last_updated"] = datetime.now(pytz.UTC).isoformat()
            return


def last_completed_bar(on_bars: pd.DataFrame) -> Optional[pd.Series]:
    if on_bars is None or on_bars.empty:
        return None
    return on_bars.iloc[-1]


def evaluate_overnight_cue(
    df_1m: pd.DataFrame,
    now=None,
    *,
    already_filled: bool = False,
    used_cues: Optional[set] = None,
    last_price: Optional[float] = None,
    trend_15m: Optional[int] = None,
    trend_60m: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Paper signal or None. Last ROLL_BARS 1m high/low. Fires on last print or 1m close.

    Qty 2 only when last AND completed 1m close are both through the 20m extreme
    and completed 15m + 60m EMA8/21 agree. Otherwise qty 1. Gemini does not size.
    """
    if already_filled or not allow_new_overnight_entry(now):
        return None
    on_bars = overnight_completed_bars(df_1m, now)
    rng = settled_overnight_range(on_bars, now)
    bar = last_completed_bar(on_bars)
    if bar is None:
        return None
    _ = used_cues  # kept for callers; blocking same cue failed IS
    sl = atr_stop_pts(df_1m)
    close = float(bar["close"])
    try:
        px = close if last_price is None else float(last_price)
    except (TypeError, ValueError):
        px = close
    ts = bar["datetime"] if "datetime" in bar.index else on_bars.iloc[-1].get("datetime")

    def _sig(cue: str, side: str, zone_name: str, zone_px: float, why: str) -> Dict[str, Any]:
        t15 = trend_15m
        t60 = trend_60m
        if t15 is None or t60 is None:
            t15, t60 = overnight_htf_sides(df_1m)
        macd_t = macd_trend_15m_on_1m(df_1m) if overnight_macd_enabled() else 0
        gate = overnight_stack_gate(
            side, px, close, float(zone_px), int(t15 or 0), int(t60 or 0), macd_t,
        )
        qty = int(gate.get("qty") or QTY)
        entry = px
        tp_r = overnight_tp_r()
        if side == "short":
            stop = entry + sl
            target = entry - sl * tp_r
        else:
            stop = entry - sl
            target = entry + sl * tp_r
        stack_note = overnight_stack_why_suffix(gate)
        conf = entry_confidence_pct(side, gate)
        return {
            "session": SESSION_NAME,
            "recipe": PAPER_RECIPE,
            "cue": cue,
            "side": side,
            "direction": side,
            "qty": qty,
            "symbol": "MNQ",
            "entry_price": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "atr_stop_pts": round(sl, 2),
            "zone_name": zone_name,
            "zone_price": round(float(zone_px), 2),
            "bar_ts": str(ts),
            "why": why + stack_note,
            "flatten_et": "",
            "entry_confidence_pct": conf["pct"],
            "entry_confidence": conf,
            "macd_trend": int(macd_t),
        }

    onh, onl = rng.get("onh"), rng.get("onl")
    if rng.get("onh_settled") and onh is not None:
        if px > onh + TOUCH_TICK:
            return _sig(
                CUE_BREAK_ONH, "long", ZONE_ONH, onh,
                f"break last {ROLL_BARS}m high: last {px:.2f} through {onh:.2f}",
            )
    if rng.get("onl_settled") and onl is not None:
        if px < onl - TOUCH_TICK:
            return _sig(
                CUE_BREAK_ONL, "short", ZONE_ONL, onl,
                f"break last {ROLL_BARS}m low: last {px:.2f} through {onl:.2f}",
            )
    return None


def paper_pnl_usd(side: str, entry: float, exit_px: float, qty: int = QTY) -> Tuple[float, float]:
    si = 1 if str(side).lower() in ("long", "buy") else -1
    pts = (float(exit_px) - float(entry)) * si
    usd = pts * POINT_VALUE * int(qty) - COMMISSION * max(1, int(qty))
    return round(pts, 2), round(usd, 2)


def update_mae_mfe(
    side: str,
    entry: float,
    high: float,
    low: float,
    mae: float,
    mfe: float,
) -> Tuple[float, float]:
    si = 1 if str(side).lower() in ("long", "buy") else -1
    if si >= 0:
        mae = max(mae, max(0.0, entry - low))
        mfe = max(mfe, max(0.0, high - entry))
    else:
        mae = max(mae, max(0.0, high - entry))
        mfe = max(mfe, max(0.0, entry - low))
    return round(mae, 2), round(mfe, 2)


def check_paper_exit(
    position: Dict[str, Any],
    *,
    high: float,
    low: float,
    last: float,
    now=None,
) -> Optional[Dict[str, Any]]:
    side = str(position.get("side") or "long").lower()
    stop = float(position["stop"])
    target = float(position.get("target") or 0)
    if side in ("long", "buy"):
        if low <= stop:
            return {"reason": "SL", "exit_price": stop}
        if target and high >= target:
            return {"reason": "TP", "exit_price": target}
    else:
        if high >= stop:
            return {"reason": "SL", "exit_price": stop}
        if target and low <= target:
            return {"reason": "TP", "exit_price": target}
    return None
