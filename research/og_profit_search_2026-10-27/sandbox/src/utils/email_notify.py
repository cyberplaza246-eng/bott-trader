"""
Minimal email notifications for trade events.

Configure via .env (preferred):
  EMAIL_ON_TRADE=true
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your_email@gmail.com
  SMTP_PASSWORD=your_gmail_app_password
  EMAIL_TO=recipient@email.com
  EMAIL_FROM=your_email@gmail.com   # optional; defaults to SMTP_USER

Legacy names (also supported): EMAIL_ENABLED, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT,
EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT, EMAIL_USER, EMAIL_PASS.

Gmail setup:
  1. Enable 2FA on your Google account
  2. https://myaccount.google.com/apppasswords → create Mail app password
  3. Use the 16-character password as SMTP_PASSWORD
"""
from __future__ import annotations

import os
import smtplib
import threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Optional, Tuple

from src.utils.logger import bot_logger

_SUBJECT_PREFIX = os.getenv('EMAIL_SUBJECT_PREFIX', '[Ai-bot]')


def _env_bool(key: str, default: str = 'false') -> bool:
    return os.getenv(key, default).lower() in ('true', '1', 'yes')


def _smtp_settings() -> Tuple[bool, str, int, str, str, str, str]:
    """Return (enabled, host, port, user, password, from_addr, to_addr)."""
    host = os.getenv('SMTP_HOST') or os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    port = int(os.getenv('SMTP_PORT') or os.getenv('EMAIL_SMTP_PORT', '587'))
    user = (
        os.getenv('SMTP_USER')
        or os.getenv('EMAIL_SENDER')
        or os.getenv('EMAIL_USER', '')
    )
    password = (
        os.getenv('SMTP_PASSWORD')
        or os.getenv('EMAIL_PASSWORD')
        or os.getenv('EMAIL_PASS', '')
    )
    to_addr = os.getenv('EMAIL_TO') or os.getenv('EMAIL_RECIPIENT', '') or user
    from_addr = os.getenv('EMAIL_FROM') or user

    wants_email = _env_bool('EMAIL_ON_TRADE', 'false') or _env_bool('EMAIL_ENABLED', 'false')
    has_creds = bool(user and password and to_addr)
    enabled = wants_email and has_creds

    return enabled, host, port, user, password, from_addr, to_addr


def _dispatch(subject: str, body: str) -> bool:
    """Queue a plain-text email on a background thread. Never raises."""
    enabled, host, port, user, password, from_addr, to_addr = _smtp_settings()
    if not enabled:
        return False

    full_subject = f"{_SUBJECT_PREFIX} {subject}"

    def _send() -> None:
        try:
            msg = MIMEText(body)
            msg['Subject'] = full_subject
            msg['From'] = from_addr
            msg['To'] = to_addr

            if port == 465:
                with smtplib.SMTP_SSL(host, port) as server:
                    server.login(user, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)

            bot_logger.info(f"📧 Email sent: {subject}")
        except Exception as e:
            bot_logger.warning(f"📧 Email failed: {e}")

    threading.Thread(target=_send, daemon=True).start()
    return True


def send_email(subject: str, body: str) -> bool:
    """Send a plain-text notification (non-blocking)."""
    return _dispatch(subject, body)


def notify_trade_placed(
    *,
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    ticket_id: str,
    mode: str,
    protection: Optional[str] = None,
) -> bool:
    """Email alert when an order is placed successfully."""
    mode_label = mode.upper()
    protection_line = f"\nProtection: {protection}" if protection else ""
    body = (
        f"Trade placed ({mode_label})\n\n"
        f"Symbol: {symbol}\n"
        f"Direction: {direction.upper()}\n"
        f"Entry: {entry:.2f}\n"
        f"Stop loss: {sl:.2f}\n"
        f"Take profit: {tp:.2f}\n"
        f"Ticket: {ticket_id}\n"
        f"Mode: {mode_label}"
        f"{protection_line}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    return _dispatch(f"Trade: {direction.upper()} {symbol} ({mode_label})", body)
