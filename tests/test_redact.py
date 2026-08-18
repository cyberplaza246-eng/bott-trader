from datetime import datetime, timedelta, timezone

from src.utils.redact import redact_secrets


def test_redacts_password_env_and_assignment():
    env = {"RITHMIC_PASSWORD": "super-secret-pass", "DATABENTO_API_KEY": "db-key-zzzz"}
    dumped = "login user=lucid password=super-secret-pass key=db-key-zzzz"
    out = redact_secrets(dumped, environ=env)
    assert "super-secret-pass" not in out
    assert "db-key-zzzz" not in out
    assert "[REDACTED]" in out


def test_redact_does_not_echo_short_noise():
    env = {"RITHMIC_PASSWORD": "ab"}
    assert redact_secrets("ab leftover", environ=env) == "ab leftover"


def test_rithmic_connect_error_is_duplicate_session_not_bad_password():
    from src.broker.rithmic_connector import RithmicConnector, _is_duplicate_session_text

    blob = "ticker plant rpCode 13 permission denied infra_type 1 heartbeat_interval NoneType"
    assert _is_duplicate_session_text(blob) is True
    msg = RithmicConnector._format_connect_error(RuntimeError(blob))
    assert "super-secret" not in msg.lower()
    assert "duplicate session" in msg.lower()
    assert "not a bad password" in msg.lower()


def test_rithmic_log_filter_redacts_password_assignment():
    import logging

    from src.utils.logger import SecretRedactFilter, install_rithmic_log_redaction

    install_rithmic_log_redaction()
    filt = SecretRedactFilter()
    record = logging.LogRecord(
        name="rithmic.plant.ticker",
        level=logging.ERROR,
        pathname="",
        lineno=1,
        msg="RithmicClient(user='x', password=super-secret-pass)",
        args=(),
        exc_info=None,
    )
    assert filt.filter(record) is True
    assert "super-secret-pass" not in str(record.msg)
