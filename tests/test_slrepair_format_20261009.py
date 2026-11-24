"""2026-10-09: position-check failure must not crash (old code called missing self._format_exception)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock

import src.broker.rithmic_connector as rc
from src.broker.rithmic_connector import RithmicConnector


def _conn(fail_with):
    c = RithmicConnector.__new__(RithmicConnector)
    c._client = object()
    c._using_simulator_route = True
    calls = {"n": 0}

    def run_sync(coro, timeout=None):
        coro.close()                       # never actually run broker coroutines
        calls["n"] += 1
        if calls["n"] == 1:
            return []                      # list_positions: empty (flat / sim PnL plant empty)
        raise fail_with                    # filled-entry inference blows up
    c._run_sync = run_sync
    c._list_positions_net_qty = lambda positions, symbol: 0
    return c


def test_inference_failure_returns_unknown_not_crash(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(rc, "error_logger", log)
    c = _conn(TimeoutError("history plant slow"))
    assert c._broker_symbol_has_position("MNQ") is None
    msg = log.error.call_args[0][0]
    assert "position inference failed for MNQ" in msg and "TimeoutError" in msg


def test_no_format_exception_left():
    src = open(rc.__file__, encoding="utf-8").read()
    assert "_format_exception" not in src
