"""Redact broker/API secrets from logs, tracebacks, and console text."""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

_SECRET_ENV = (
    "RITHMIC_PASSWORD",
    "DATABENTO_API_KEY",
    "SMTP_PASSWORD",
    "EMAIL_PASSWORD",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "NEWSAPI_KEY",
    "TIINGO_API_KEY",
)

_PASSWORD_ASSIGN_RE = re.compile(
    r"(?i)((?:password|passwd|pwd|secret|api[_-]?key)\s*[=:]\s*)(['\"]?)([^'\"\s,)\]]+)",
)


def _secret_values(environ: Optional[dict] = None) -> Iterable[str]:
    env = environ if environ is not None else os.environ
    for key in _SECRET_ENV:
        val = str(env.get(key, "") or "").strip()
        if len(val) >= 4:
            yield val


def redact_secrets(text: object, environ: Optional[dict] = None) -> str:
    """Replace known secret values and password=... assignments. Never returns the input password."""
    out = "" if text is None else str(text)
    if not out:
        return out
    for secret in _secret_values(environ):
        out = out.replace(secret, "[REDACTED]")
    out = _PASSWORD_ASSIGN_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    return out
