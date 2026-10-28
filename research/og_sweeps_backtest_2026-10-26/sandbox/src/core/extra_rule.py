"""Rule for a 2nd/3rd concurrent trade (user 2026-10-09).

EXTRA_TRADE_MODE=same_dir_profit (default for the REVAMP bot):
    an extra trade is allowed only if it is in the SAME direction as every open
    trade AND every open trade is currently in profit (unrealized > 0 points
    vs its fill, before costs). Replaces the old confidence check.
    Mirrors box backtest second_trade2.py "same dir + open > 0.0R":
        all(q.dir == new.dir for q in open) and all(unrealized_R(q, T) > 0 for q in open)
    priced at the signal bar's close (the bot's signal 'entry').
    3rd trade uses the same rule (both open trades same direction and in profit).
    Optional, OFF by default: EXTRA_TRADE3_TP_PROGRESS=0.5 would also require every open trade
    to be that far to its own TP for a 3rd trade (no-TP trade then blocks). 0 = disabled.
EXTRA_TRADE_MODE=conf:
    old rule, extra trade needs confidence >= EXTRA_TRADE_MIN_CONF.

Default when EXTRA_TRADE_MODE is unset: same_dir_profit for BOTH bots (REVAMP and the original
Ensemble bot, user 2026-10-09). Set EXTRA_TRADE_MODE=conf to restore the old confidence rule.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Tuple

from src.core.conf_display import conf_pct, ordinal

SAME_DIR_PROFIT = "same_dir_profit"
CONF = "conf"
_ALIASES = {
    "same_dir_profit": SAME_DIR_PROFIT, "same_direction_profit": SAME_DIR_PROFIT, "samedir": SAME_DIR_PROFIT,
    "pyramid": SAME_DIR_PROFIT, "profit": SAME_DIR_PROFIT,
    "conf": CONF, "confidence": CONF, "old": CONF,
}


def extra_trade_mode(revamp_on: bool = True) -> str:
    raw = str(os.getenv("EXTRA_TRADE_MODE", "") or "").strip().lower()
    if raw in _ALIASES:
        return _ALIASES[raw]
    return SAME_DIR_PROFIT   # both bots (revamp_on kept for call-site compatibility)


def trade3_tp_progress() -> float:
    try:
        v = float(os.getenv("EXTRA_TRADE3_TP_PROGRESS", "0") or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if (v == v and v > 0) else 0.0


def describe(mode: str, min_conf: float) -> str:
    if mode == SAME_DIR_PROFIT:
        thr = trade3_tp_progress()
        third = (f"3rd trade needs both open trades {int(round(thr * 100))}% of the way to TP" if thr > 0
                 else "3rd trade needs both open trades in profit")
        return (f"same direction only; 2nd trade needs trade 1 in profit, {third} "
                "(EXTRA_TRADE_MODE=same_dir_profit; EXTRA_TRADE_MIN_CONF ignored)")
    return f"2nd/3rd trade needs confidence >= {conf_pct(min_conf)} (EXTRA_TRADE_MODE=conf)"


def _dir(pos_or_dir: Any) -> str:
    d = pos_or_dir if isinstance(pos_or_dir, str) else getattr(pos_or_dir, "direction", "")
    return "long" if str(d).lower() in ("long", "buy") else "short"


def _side(direction: str) -> str:
    return "LONG" if _dir(direction) == "long" else "SHORT"


def open_pnl(pos: Any, price: float, point_value: float) -> Tuple[float, float]:
    """(points, dollars) unrealized for an open trade at `price` (no costs)."""
    entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
    d = 1.0 if _dir(pos) == "long" else -1.0
    pts = d * (float(price) - entry)
    size = max(1, int(getattr(pos, "size", 1) or 1))
    return pts, pts * float(point_value) * size


def target_price(pos: Any) -> float:
    """Current TP, or the original TP once a past-TP trail cleared it; 0.0 if unknown."""
    tp = float(getattr(pos, "tp", 0.0) or 0.0)
    if tp > 0:
        return tp
    if getattr(pos, "tp_cleared", False):
        return float(getattr(pos, "original_tp", 0.0) or 0.0)
    return 0.0


def tp_progress(pos: Any, price: float) -> Optional[float]:
    """Fraction of the way from entry to TP in trade direction (1.0 = at TP); None if no TP."""
    tp = target_price(pos)
    entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
    d = 1.0 if _dir(pos) == "long" else -1.0
    dist = d * (tp - entry)
    if tp <= 0 or dist <= 0:
        return None
    return d * (float(price) - entry) / dist


def _pct(x: Optional[float]) -> str:
    return "no TP" if x is None else f"{int(round(x * 100))}%"


def _both(n: int) -> str:
    return "both trades" if n == 2 else f"all {n} trades"


def _money(usd: float) -> str:
    if usd > 0.005:
        return f"up ${usd:,.0f}"
    if usd < -0.005:
        return f"down ${-usd:,.0f}"
    return "flat $0"


def _now_text(positions: List[Any], price: Optional[float], point_value: float) -> str:
    if price is None or not (float(price) > 0) or not positions:
        return ""
    if len(positions) == 1:
        return f"now: {_money(open_pnl(positions[0], price, point_value)[1])}"
    parts = [f"trade {i} {_money(open_pnl(p, price, point_value)[1])}" for i, p in enumerate(positions, 1)]
    return "now: " + ", ".join(parts)


def _need_text(n_open: int) -> str:
    who = "trade 1" if n_open <= 1 else ("both trades" if n_open == 2 else f"trades 1-{n_open}")
    return f"needs {who} in profit + same direction"


def check_same_dir_profit(positions: Iterable[Any], open_n: int, direction: str, price: float,
                          point_value: float) -> Optional[str]:
    """None if the extra trade is allowed, else a plain-words reason."""
    pos = list(positions)
    if open_n < 1:
        return None
    if open_n > len(pos):
        return f"an open position the bot is not tracking ({open_n} open, {len(pos)} tracked) - can't check profit"
    want = _dir(direction)
    against = [i for i, p in enumerate(pos, 1) if _dir(p) != want]
    if against:
        return (f"{_side(want)} setup is against open trade {against[0]} "
                f"({_side(pos[against[0] - 1].direction)}) - only {_side(pos[0].direction)} allowed")
    thr = trade3_tp_progress()
    if len(pos) >= 2 and thr > 0:   # optional 3rd-trade TP-progress rule (off by default)
        for i, p in enumerate(pos, 1):
            pr = tp_progress(p, price)
            if pr is None:
                return f"trade {i} has no TP, can't check halfway (3rd trade blocked)"
            if pr < thr:
                now = " and ".join(_pct(tp_progress(q, price)) for q in pos)
                return f"trade {i} only {_pct(pr)} to TP, needs {_both(len(pos))} {_pct(thr)} to TP (now {now})"
        return None
    losing = [(i, open_pnl(p, price, point_value)[1]) for i, p in enumerate(pos, 1)
              if open_pnl(p, price, point_value)[0] <= 0]
    if losing:
        i, usd = losing[0]
        return f"trade {i} not in profit ({_money(usd)}), {_need_text(len(pos))}"
    return None


def status_line(positions: Iterable[Any], open_n: int, max_positions: int, price: Optional[float],
                point_value: float, candidate: Optional[dict] = None, note: str = "") -> Optional[str]:
    """Plain status for the next extra trade in same_dir_profit mode (None if not relevant)."""
    pos = list(positions)
    if open_n < 1 or open_n >= max_positions:
        return None
    head = f"{ordinal(open_n + 1)} trade: "
    pre = f"{note}; " if note else ""
    now = _now_text(pos, price, point_value)
    tail = f" ({now})" if now else ""
    if open_n > len(pos):
        return f"{head}{pre}blocked, open position not tracked by the bot"
    dirs = {_dir(p) for p in pos}
    if len(dirs) > 1:
        return f"{head}{pre}blocked, open trades go both ways{tail}"
    have_px = price is not None and float(price) > 0
    thr = trade3_tp_progress()
    if len(pos) >= 2 and thr > 0:   # optional 3rd-trade TP-progress rule (off by default)
        prog = [tp_progress(p, price) if have_px else None for p in pos]
        now = f" (now {' and '.join(_pct(x) for x in prog)})" if have_px else ""
        if have_px and any(tp_progress(p, price) is None for p in pos):
            i = next(i for i, p in enumerate(pos, 1) if tp_progress(p, price) is None)
            return f"{head}{pre}blocked, trade {i} has no TP"
        if not have_px or any(x < thr for x in prog):
            return f"{head}{pre}needs {_both(len(pos))} {'halfway' if abs(thr - 0.5) < 1e-9 else _pct(thr)} to TP + same direction{now}"
        line = f"{head}{pre}allowed, {_side(pos[0].direction)} setups only{now[:-1] + ' to TP)' if now else ''}"
        if candidate and not note:
            cs = _side(candidate.get("direction", ""))
            line += " - OK" if cs == _side(pos[0].direction) else f" - {cs} setup skipped"
        return line
    in_profit = have_px and all(open_pnl(p, price, point_value)[0] > 0 for p in pos)
    if not in_profit:
        return f"{head}{pre}{_need_text(len(pos))}{tail}"
    side = _side(pos[0].direction)
    line = f"{head}{pre}allowed, {side} setups only{tail}"
    if candidate and not note:
        cs = _side(candidate.get("direction", ""))
        line += " - OK" if cs == side else f" - {cs} setup skipped"
    return line
