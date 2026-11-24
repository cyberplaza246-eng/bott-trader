"""
Trade alerts (sound + email) for every bot that imports src.utils.logger.

ADDITIVE and fail-safe: this module only LISTENS to log lines the bots already
write. It never places/cancels orders, never touches broker code, and every
step is wrapped so an alert problem can never block or crash trading.

Triggers (entries):
  * TradingBot log : "Rithmic order placed: <id> <side> <qty> <SYM> | ..."   (every bot that
                     sends an order through RithmicConnector.place_order)
  * Trades log     : "ENTRY <dir> <SYM> @ px SL= TP= ..." , "REENTRY ...", "ORDER_PLACED ...",
                     "ENTRY | Pair: ..."   (merged with the line above into ONE alert)
Triggers (exits, optional): Trades log "EXIT <reason> <dir> <SYM> @ px PnL=$x" / "EXIT | Pair: ..."
NOT alerted: "ADOPTED untracked ..." (a position the bot did not place).

Switches (env; default shown):
  TRADE_ALERT_SOUND=1            0 turns the PC sound off
  TRADE_ALERT_EMAIL=1            0 turns the email off
  TRADE_ALERT_EXITS=0            1 also alerts on exits
  TRADE_ALERT_TO=paraflix246@gmail.com
  TRADE_ALERT_FROM=              default: the SMTP user
  TRADE_ALERT_SMTP_USER=         default: SMTP_USER / EMAIL_USER from .env, else FROM
  TRADE_ALERT_SMTP_APP_PASSWORD= Gmail app password (falls back to SMTP_PASSWORD / EMAIL_PASS in .env)
  TRADE_ALERT_SMTP_HOST=smtp.gmail.com   TRADE_ALERT_SMTP_PORT=587
  TRADE_ALERT_MAX_PER_HOUR=30    email flood cap
  TRADE_ALERT_MERGE_SECONDS=3    window to merge the two log lines of one entry
  TRADE_ALERT_DISABLE=1          kill switch for a process
Kid / ExtremeSL learner scripts are always excluded.
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import sys
import threading
import time
from datetime import datetime
from email.mime.text import MIMEText

_LOCK = threading.Lock()
_PENDING: dict = {}
_SENT_TIMES: list = []
_WARNED: set = set()
_INSTALLED_ATTR = "_trade_alert_handler_installed"

_RE_CONNECTOR = re.compile(r"^Rithmic order placed:\s+(\S+)\s+(\w+)\s+(\d+)\s+(\S+)")
_RE_ENTRY = re.compile(r"^(?:RE)?ENTRY\s+(long|short|buy|sell)\s+(\S+)(?:.*?@\s*([0-9.]+))?", re.I)
_RE_ENTRY_PAIR = re.compile(r"^ENTRY \| Pair:\s*(\S+) \| Type:\s*(\w+)", re.I)
_RE_MT5 = re.compile(r"^ORDER_PLACED(?: \([A-Z]+\))? \| Pair:\s*(\S+) \| Type:\s*(\w+)", re.I)
_RE_EXIT = re.compile(r"^EXIT\s+(\S+)\s+(long|short)\s+(\S+)\s+@\s*([0-9.]+)\s+PnL=\$?(-?[0-9.]+)", re.I)
_RE_EXIT_PAIR = re.compile(r"^EXIT \| Pair:\s*(\S+) \| Type:\s*(\S+)", re.I)


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _script_name() -> str:
    try:
        return os.path.basename(sys.argv[0]) or "python"
    except Exception:
        return "python"


def _excluded() -> bool:
    if _flag("TRADE_ALERT_DISABLE", "0"):
        return True
    n = _script_name().lower()
    return any(k in n for k in ("kid", "extremesl", "extreme_sl"))


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        print(f"[trade-alerts] {msg}", flush=True)
    except Exception:
        pass


def _norm_side(s: str) -> str:
    s = (s or "").lower()
    return "BUY" if s in ("buy", "long") else ("SELL" if s in ("sell", "short") else s.upper())


# ---------------------------------------------------------------- sound
def _play(kind: str) -> None:
    """Beep pattern on a daemon thread. Never raises."""
    def _run():
        try:
            import winsound
            notes = ((880, 160), (1175, 160), (1480, 260)) if kind == "ENTRY" else ((1000, 200), (700, 300))
            try:
                for f, d in notes:
                    winsound.Beep(f, d)
            except Exception:
                winsound.MessageBeep()
        except Exception:
            pass
    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        pass


# ---------------------------------------------------------------- email
def _email_cfg():
    to = os.getenv("TRADE_ALERT_TO") or "paraflix246@gmail.com"
    user = (os.getenv("TRADE_ALERT_SMTP_USER") or os.getenv("SMTP_USER") or os.getenv("EMAIL_USER")
            or os.getenv("TRADE_ALERT_FROM") or os.getenv("EMAIL_FROM") or "")
    frm = os.getenv("TRADE_ALERT_FROM") or user
    pw = (os.getenv("TRADE_ALERT_SMTP_APP_PASSWORD") or os.getenv("SMTP_PASSWORD")
          or os.getenv("EMAIL_PASSWORD") or os.getenv("EMAIL_PASS") or "").replace(" ", "")
    host = os.getenv("TRADE_ALERT_SMTP_HOST") or "smtp.gmail.com"
    try:
        port = int(os.getenv("TRADE_ALERT_SMTP_PORT") or "587")
    except Exception:
        port = 587
    return to, user, frm, pw, host, port


def _send_email_blocking(subject: str, body: str) -> bool:
    to, user, frm, pw, host, port = _email_cfg()
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = frm or user
    msg["To"] = to
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=15) as s:
            s.login(user, pw)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
    return True


def _email_async(subject: str, body: str) -> None:
    if not _flag("TRADE_ALERT_EMAIL", "1"):
        return
    to, user, frm, pw, host, port = _email_cfg()
    if not (user and pw):
        _warn_once("nopw", "email skipped: no app password set (TRADE_ALERT_SMTP_APP_PASSWORD). "
                           "Sound alerts still work; trading is unaffected.")
        return
    try:
        cap = int(os.getenv("TRADE_ALERT_MAX_PER_HOUR") or "30")
    except Exception:
        cap = 30
    now = time.time()
    with _LOCK:
        _SENT_TIMES[:] = [t for t in _SENT_TIMES if now - t < 3600]
        if cap > 0 and len(_SENT_TIMES) >= cap:
            _warn_once("cap", f"email flood cap ({cap}/hour) reached; further emails skipped this hour")
            return
        _SENT_TIMES.append(now)

    def _run():
        try:
            _send_email_blocking(subject, body)
            print(f"[trade-alerts] email sent to {to}: {subject}", flush=True)
        except Exception as e:  # never affects trading
            print(f"[trade-alerts] email failed ({type(e).__name__}): {str(e)[:120]}", flush=True)
    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        pass


# ---------------------------------------------------------------- compose
def _now_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S local")


def _compose(kind: str, info: dict) -> tuple:
    side = info.get("side") or ""
    sym = info.get("symbol") or ""
    qty = info.get("qty")
    px = info.get("price")
    bits = [kind, side]
    if qty:
        bits.append(str(qty))
    bits.append(sym)
    subj = "[bott-trader] " + " ".join(b for b in bits if b)
    if px:
        subj += f" @ {px}"
    subj += f" ({_script_name()})"
    lines = [f"{kind}: {side} {qty or ''} {sym}".replace("  ", " ").strip(),
             f"Bot: {_script_name()}  mode={os.getenv('TRADING_MODE', '?')}",
             f"Time: {_now_et()}", ""]
    lines += [f"log> {l}" for l in info.get("lines", [])]
    return subj, "\n".join(lines)


def _fire(kind: str, info: dict) -> None:
    try:
        subj, body = _compose(kind, info)
        _email_async(subj, body)
    except Exception:
        pass


def _flush_pending(key) -> None:
    try:
        with _LOCK:
            info = _PENDING.pop(key, None)
        if info:
            _fire("ENTRY", info)
    except Exception:
        pass


def _entry_event(side: str, symbol: str, qty=None, price=None, line: str = "") -> None:
    side = _norm_side(side)
    key = (symbol, side)
    try:
        merge = float(os.getenv("TRADE_ALERT_MERGE_SECONDS") or "3")
    except Exception:
        merge = 3.0
    with _LOCK:
        info = _PENDING.get(key)
        if info is not None:           # second log line for the same entry -> merge
            if qty and not info.get("qty"):
                info["qty"] = qty
            if price and not info.get("price"):
                info["price"] = price
            info["lines"].append(line)
            return
        _PENDING[key] = {"side": side, "symbol": symbol, "qty": qty, "price": price, "lines": [line]}
    if _flag("TRADE_ALERT_SOUND", "1"):
        _play("ENTRY")
    t = threading.Timer(max(merge, 0.0), _flush_pending, args=(key,))
    t.daemon = True
    t.start()


def _exit_event(line: str, side: str, symbol: str, price=None, pnl=None, reason=None) -> None:
    if not _flag("TRADE_ALERT_EXITS", "0"):
        return
    if _flag("TRADE_ALERT_SOUND", "1"):
        _play("EXIT")
    info = {"side": (side or "").upper() + " closed", "symbol": symbol, "price": price, "lines": [line]}
    if pnl is not None or reason:
        info["lines"].append(f"reason={reason} pnl=${pnl}")
    _fire("EXIT", info)


def handle_message(logger_name: str, msg: str) -> None:
    """Pure dispatcher (also used by the offline test)."""
    if _excluded():
        return
    msg = (msg or "").strip()
    if not msg:
        return
    if logger_name == "TradingBot":
        m = _RE_CONNECTOR.match(msg)
        if m:
            _entry_event(m.group(2), m.group(4), qty=m.group(3), line=msg)
        return
    if logger_name == "Trades":
        m = _RE_EXIT.match(msg)
        if m:
            _exit_event(msg, m.group(2), m.group(3), price=m.group(4), pnl=m.group(5), reason=m.group(1))
            return
        m = _RE_EXIT_PAIR.match(msg)
        if m:
            _exit_event(msg, "", m.group(1), reason=m.group(2))
            return
        m = _RE_ENTRY_PAIR.match(msg) or _RE_MT5.match(msg)
        if m:
            _entry_event(m.group(2), m.group(1), line=msg)
            return
        m = _RE_ENTRY.match(msg)
        if m:
            _entry_event(m.group(1), m.group(2), price=m.group(3), line=msg)


class _TradeAlertHandler(logging.Handler):
    def emit(self, record):
        try:
            handle_message(record.name, record.getMessage())
        except Exception:
            pass

    def handleError(self, record):  # never print tracebacks into the trading console
        pass


def install(*loggers) -> None:
    """Attach the listener to the given loggers (idempotent, never raises)."""
    try:
        for lg in loggers:
            if lg is None or getattr(lg, _INSTALLED_ATTR, False):
                continue
            h = _TradeAlertHandler(level=logging.INFO)
            lg.addHandler(h)
            setattr(lg, _INSTALLED_ATTR, True)
    except Exception:
        pass