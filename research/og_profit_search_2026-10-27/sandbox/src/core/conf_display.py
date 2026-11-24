"""Plain confidence labels for console lines (user 2026-10-08).

conf_pct(0.74) -> '74%'; unknown / adopted / manual (confidence <= 0) -> 'n/a'.
"""
from typing import Any, Optional

_ORDINAL = {2: "2nd", 3: "3rd"}


def conf_pct(confidence: Any) -> str:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "n/a"
    if not (c > 0):  # 0 / NaN / negative = not known (adopted or manual)
        return "n/a"
    return f"{int(round(c * 100))}%"


def trade_conf_label(pos: Any) -> str:
    """'conf 74%', 'conf 74% re-entry' or 'conf n/a' (adopted/manual)."""
    if str(getattr(pos, "regime", "") or "") == "broker_adopt":
        return "conf n/a"
    label = f"conf {conf_pct(getattr(pos, 'confidence', 0.0))}"
    if int(getattr(pos, "reentry_gen", 0) or 0) > 0:
        label += " re-entry"
    return label


def trade_status_line(n: int, pos: Any) -> str:
    """'Trade 1 SHORT conf 74% | 1 MNQ @ 18500.25 SL 18520.00 TP 18460.00'"""
    side = "LONG" if str(getattr(pos, "direction", "")).lower() == "long" else "SHORT"
    size = max(1, int(getattr(pos, "size", 1) or 1))
    line = (f"Trade {n} {side} {trade_conf_label(pos)} | {size} {getattr(pos, 'symbol', '')} "
            f"@ {float(getattr(pos, 'entry_price', 0.0) or 0.0):.2f}")
    sl = float(getattr(pos, "sl", 0.0) or 0.0)
    tp = float(getattr(pos, "tp", 0.0) or 0.0)
    if sl > 0:
        line += f" SL {sl:.2f}"
    if tp > 0:
        line += f" TP {tp:.2f}"
    return line


def ordinal(n: int) -> str:
    return _ORDINAL.get(int(n), f"{int(n)}th")


def extra_trade_line(open_n: int, max_positions: int, min_conf: float,
                     candidate_conf: Optional[float] = None, note: str = "") -> Optional[str]:
    """'2nd trade: conf 66%, needs 70%' while 1..max-1 trades are open; else None."""
    if open_n < 1 or open_n >= max_positions:
        return None
    head = f"{ordinal(open_n + 1)} trade: "
    need = conf_pct(min_conf)
    if note:
        return f"{head}{note}, needs {need}"
    if candidate_conf is None:
        return f"{head}no setup, needs {need}"
    ok = " - OK" if float(candidate_conf) >= float(min_conf) else ""
    return f"{head}conf {conf_pct(candidate_conf)}, needs {need}{ok}"
