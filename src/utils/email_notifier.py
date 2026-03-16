"""
Email Notification Utility for Trade Alerts.

Configure via environment variables:
  EMAIL_ENABLED=true
  EMAIL_SMTP_SERVER=smtp.gmail.com
  EMAIL_SMTP_PORT=587
  EMAIL_SENDER=your_email@gmail.com
  EMAIL_PASSWORD=your_app_password  (use Gmail App Password, not regular password)
  EMAIL_RECIPIENT=recipient@email.com

For Gmail:
1. Enable 2FA on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Create an app password for "Mail"
4. Use that 16-character password as EMAIL_PASSWORD
"""
from __future__ import annotations

import os
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from src.utils.logger import bot_logger


class EmailNotifier:
    """Send email notifications for trade events."""

    def __init__(self):
        self.enabled = os.getenv('EMAIL_ENABLED', 'false').lower() in ('true', '1', 'yes')
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.sender = os.getenv('EMAIL_SENDER', '')
        self.password = os.getenv('EMAIL_PASSWORD', '')
        self.recipient = os.getenv('EMAIL_RECIPIENT', '') or self.sender  # Default to sender

        if self.enabled:
            if not self.sender or not self.password:
                bot_logger.warning("📧 Email notifications enabled but credentials missing — disabling")
                self.enabled = False
            else:
                bot_logger.info(f"📧 Email notifications enabled → {self.recipient}")

    def send_trade_alert(
        self,
        pair: str,
        trade_type: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        confidence: float = 0.0,
        reason: str = "",
    ) -> bool:
        """Send email notification for a trade entry.
        
        Runs in background thread to not block trading.
        """
        if not self.enabled:
            return False

        subject = f"🚀 {trade_type} {pair} @ {entry_price:.2f}"
        
        # Calculate risk/reward
        if trade_type.upper() == 'BUY':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        rr_ratio = reward / risk if risk > 0 else 0

        body = f"""
<html>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2 style="color: {'#28a745' if trade_type.upper() == 'BUY' else '#dc3545'};">
        {'📈' if trade_type.upper() == 'BUY' else '📉'} {trade_type.upper()} {pair}
    </h2>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 400px;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Entry Price</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{entry_price:.5f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Stop Loss</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; color: #dc3545;">{stop_loss:.5f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Take Profit</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; color: #28a745;">{take_profit:.5f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Position Size</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{lot_size}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Risk/Reward</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">1:{rr_ratio:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Confidence</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{confidence:.1%}</td>
        </tr>
    </table>
    
    {f'<p style="margin-top: 15px;"><strong>Reason:</strong> {reason}</p>' if reason else ''}
    
    <p style="color: #666; font-size: 12px; margin-top: 20px;">
        Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    </p>
</body>
</html>
"""

        # Send in background thread to not block trading
        thread = threading.Thread(
            target=self._send_email,
            args=(subject, body),
            daemon=True
        )
        thread.start()
        return True

    def send_trade_closed(
        self,
        pair: str,
        trade_type: str,
        profit_loss: float,
        exit_type: str = "CLOSED",
    ) -> bool:
        """Send email notification when a trade closes."""
        if not self.enabled:
            return False

        is_win = profit_loss > 0
        emoji = "✅" if is_win else "❌"
        color = "#28a745" if is_win else "#dc3545"
        
        subject = f"{emoji} {pair} {exit_type}: ${profit_loss:+.2f}"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2 style="color: {color};">
        {emoji} {pair} {trade_type} — {exit_type}
    </h2>
    
    <p style="font-size: 24px; color: {color};">
        P/L: <strong>${profit_loss:+.2f}</strong>
    </p>
    
    <p style="color: #666; font-size: 12px; margin-top: 20px;">
        Closed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    </p>
</body>
</html>
"""

        thread = threading.Thread(
            target=self._send_email,
            args=(subject, body),
            daemon=True
        )
        thread.start()
        return True

    def _send_email(self, subject: str, body: str) -> bool:
        """Actually send the email (runs in background thread)."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = self.recipient

            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipient, msg.as_string())

            bot_logger.info(f"📧 Email sent: {subject}")
            return True

        except Exception as e:
            bot_logger.warning(f"📧 Email failed: {e}")
            return False


# Global singleton
_notifier: Optional[EmailNotifier] = None


def get_email_notifier() -> EmailNotifier:
    """Get or create the global email notifier."""
    global _notifier
    if _notifier is None:
        _notifier = EmailNotifier()
    return _notifier


def notify_trade_entry(
    pair: str,
    trade_type: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    lot_size: float,
    confidence: float = 0.0,
    reason: str = "",
) -> bool:
    """Convenience function to send trade entry notification."""
    return get_email_notifier().send_trade_alert(
        pair=pair,
        trade_type=trade_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        lot_size=lot_size,
        confidence=confidence,
        reason=reason,
    )


def notify_trade_closed(
    pair: str,
    trade_type: str,
    profit_loss: float,
    exit_type: str = "CLOSED",
) -> bool:
    """Convenience function to send trade closed notification."""
    return get_email_notifier().send_trade_closed(
        pair=pair,
        trade_type=trade_type,
        profit_loss=profit_loss,
        exit_type=exit_type,
    )
