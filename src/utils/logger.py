"""
Centralized logging for the trading bot
"""
import logging
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

from src.utils.redact import redact_secrets


class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit - ensures real-time console output"""
    def emit(self, record):
        super().emit(record)
        self.flush()


class SecretRedactFilter(logging.Filter):
    """Strip broker passwords and API keys from log messages and exception text."""

    def filter(self, record):
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
        except Exception:
            try:
                record.msg = redact_secrets(record.msg)
            except Exception:
                pass
        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        elif record.exc_info:
            try:
                record.exc_text = redact_secrets(
                    logging.Formatter().formatException(record.exc_info)
                )
            except Exception:
                record.exc_info = None
        return True


def install_rithmic_log_redaction() -> None:
    """async_rithmic logs Client(password=...) on plant errors. Never print that."""
    filt = SecretRedactFilter()

    def _attach(logger: logging.Logger) -> None:
        if not any(isinstance(f, SecretRedactFilter) for f in logger.filters):
            logger.addFilter(filt)
        for handler in logger.handlers:
            if not any(isinstance(f, SecretRedactFilter) for f in handler.filters):
                handler.addFilter(filt)

    _attach(logging.getLogger())
    if logging.lastResort is not None:
        if not any(isinstance(f, SecretRedactFilter) for f in logging.lastResort.filters):
            logging.lastResort.addFilter(filt)
    _attach(logging.getLogger("rithmic"))
    _attach(logging.getLogger("async_rithmic"))
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if not str(name).startswith(("rithmic", "async_rithmic")):
            continue
        lg = obj if isinstance(obj, logging.Logger) else logging.getLogger(str(name))
        _attach(lg)


class _ConsoleNewsPolicyFilter(logging.Filter):
    """Keep news/policy INFO in file logs only — reduces duplicate console noise."""

    _SKIP_INFO_SUBSTRINGS = (
        "News bias active:",
        "📰 News bias",
        "Policy scorer active:",
        "🏛️ Policy",
    )

    def filter(self, record):
        if record.levelno != logging.INFO or record.name != "TradingBot":
            return True
        msg = record.getMessage()
        return not any(s in msg for s in self._SKIP_INFO_SUBSTRINGS)


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both console and file handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler with auto-flush for real-time output
    console_handler = FlushStreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    console_handler.addFilter(SecretRedactFilter())
    if name == "TradingBot":
        console_handler.addFilter(_ConsoleNewsPolicyFilter())
    logger.addHandler(console_handler)
    logger.addFilter(SecretRedactFilter())
    
    # File handler (JSON format for structured logging)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = jsonlogger.JsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
        file_handler.setFormatter(file_format)
        file_handler.addFilter(SecretRedactFilter())
        logger.addHandler(file_handler)
    
    return logger


# Main bot logger
bot_logger = setup_logger(
    'TradingBot',
    log_file='logs/trading_bot.log',
    level=logging.INFO
)

# Trading signals logger
signals_logger = setup_logger(
    'Signals',
    log_file='logs/signals.log',
    level=logging.INFO
)

# Trade execution logger
trades_logger = setup_logger(
    'Trades',
    log_file='logs/trades.log',
    level=logging.INFO
)

# Error logger
error_logger = setup_logger(
    'Errors',
    log_file='logs/errors.log',
    level=logging.ERROR
)


class TradeLogger:
    """Log trade execution with detailed information"""
    
    @staticmethod
    def log_signal(pair, signal_type, confidence, reason, models_agreement):
        signals_logger.info(
            f"SIGNAL | Pair: {pair} | Type: {signal_type} | "
            f"Confidence: {confidence:.2%} | Models: {models_agreement}/4 | Reason: {reason}"
        )
    
    @staticmethod
    def log_trade_entry(pair, trade_type, entry_price, stop_loss, take_profit, lot_size):
        trades_logger.info(
            f"ENTRY | Pair: {pair} | Type: {trade_type} | "
            f"Price: {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Lot Size: {lot_size}"
        )
    
    @staticmethod
    def log_trade_exit(pair, exit_type, exit_price, profit_loss, profit_loss_percent):
        trades_logger.info(
            f"EXIT | Pair: {pair} | Type: {exit_type} | "
            f"Price: {exit_price:.5f} | P/L: {profit_loss:.2f} ({profit_loss_percent:.2%})"
        )
    
    @staticmethod
    def log_error(message, exc_info=None):
        error_logger.error(message, exc_info=exc_info)
    
    @staticmethod
    def log_bot_status(status, message):
        bot_logger.info(f"BOT_STATUS | {status} | {message}")
