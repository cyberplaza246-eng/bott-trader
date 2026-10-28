"""Shared console trade lines (emoji entry/exit/heartbeat) for paper bots."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def normalize_direction(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d in ("long", "buy"):
        return "long"
    if d in ("short", "sell"):
        return "short"
    return d


def trade_emoji(direction: str) -> str:
    return "📈" if normalize_direction(direction) == "long" else "📉"


def exit_emoji(reason: str) -> str:
    r = str(reason or "").lower()
    if "tp" in r or "target" in r:
        return "✅"
    if "sl" in r or "stop" in r:
        return "❌"
    if "flat" in r or "eod" in r or "shutdown" in r:
        return "🏁"
    return "🔚"


def bracket_ok_mark(attached: bool) -> str:
    return "✓" if attached else "✗"


def format_entry_line(
    *,
    direction: str,
    entry: float,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    ticket: str = "",
    label: str = "ENTRY",
    side_label: str = "",
    sl_ok: str = "",
    tp_ok: str = "",
    extra: str = "",
) -> str:
    emoji = trade_emoji(direction)
    dir_u = side_label or normalize_direction(direction).upper()
    head = f"{emoji} {label} {dir_u} @ {entry:.2f}"
    bracket_bits: List[str] = []
    if sl is not None and float(sl) > 0:
        bracket_bits.append(f"SL {float(sl):.2f}{sl_ok}")
    if tp is not None and float(tp) > 0:
        bracket_bits.append(f"TP {float(tp):.2f}{tp_ok}")
    tail: List[str] = []
    if bracket_bits:
        tail.append("| " + " | ".join(bracket_bits))
    if ticket:
        tail.append(f"| #{ticket}")
    if extra:
        tail.append(f"| {extra}")
    return head + "".join(tail)


def format_exit_line(
    direction: str,
    reason: str,
    exit_px: float,
    entry_px: float,
    *,
    leg: str = "",
) -> str:
    emoji = exit_emoji(reason)
    dir_u = normalize_direction(direction).upper()
    leg_u = (leg or dir_u).upper()
    return (
        f"{emoji} EXIT {leg_u} {str(reason).upper()} @ {exit_px:.2f} "
        f"({dir_u.lower()} from {entry_px:.2f})"
    )


def format_heartbeat_line(
    ts: str,
    price: Optional[float],
    positions: Sequence[Dict[str, Any]],
    note: str = "",
) -> str:
    px_txt = f"{price:.2f}" if price is not None else "—"
    if positions:
        parts: List[str] = []
        for pos in positions:
            d = normalize_direction(str(pos.get("direction") or pos.get("side") or ""))
            entry = float(pos.get("entry") or pos.get("entry_price") or 0)
            sl = float(pos.get("sl") or pos.get("stop") or 0)
            tp = float(pos.get("tp") or pos.get("target") or 0)
            parts.append(
                f"{trade_emoji(d)} {d[0].upper()} {entry:.0f} SL{sl:.0f} TP{tp:.0f}"
            )
        return f"💓 {ts} MNQ {px_txt} | open {len(positions)}: {' | '.join(parts)}"
    note_txt = note.strip() or "scan"
    return f"💓 {ts} MNQ {px_txt} | flat | {note_txt}"
