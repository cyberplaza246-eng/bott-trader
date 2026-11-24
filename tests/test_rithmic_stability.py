"""Single-flight candle fetches and Rithmic history-error classification."""

import threading
import time

import pandas as pd

from src.broker.rithmic_stability import (
    ThreadSingleFlight,
    filter_warm_history,
    is_in_progress_error,
    should_retry_send_after_reconnect,
)


def test_is_in_progress_error_matches_rithmic_text():
    msg = (
        "Cannot start another historical time bar request with the same symbol, "
        "exchange, bar type, and period while one is already in progress."
    )
    assert is_in_progress_error(RuntimeError(msg)) is True
    assert is_in_progress_error(RuntimeError("permission denied")) is False
    assert is_in_progress_error(TimeoutError()) is False


def test_replay_templates_are_not_resent_after_reconnect():
    assert should_retry_send_after_reconnect(202) is False
    assert should_retry_send_after_reconnect(206) is False
    assert should_retry_send_after_reconnect(18) is True
    assert should_retry_send_after_reconnect(None) is True


def test_singleflight_runs_one_call_for_overlapping_waiters():
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        started.set()
        assert release.wait(timeout=2)
        return "bars"

    flight = ThreadSingleFlight()
    results = []
    errors = []

    def run():
        try:
            results.append(flight.do("MNQ_5", work))
        except Exception as exc:
            errors.append(exc)

    leader = threading.Thread(target=run)
    leader.start()
    assert started.wait(timeout=2)
    follower = threading.Thread(target=run)
    follower.start()
    time.sleep(0.05)
    release.set()
    leader.join(timeout=2)
    follower.join(timeout=2)

    assert errors == []
    assert results == ["bars", "bars"]
    assert calls["n"] == 1


def test_filter_warm_history_drops_bars_older_than_six_hours():
    now = pd.Timestamp("2026-10-01 15:00:00", tz="UTC").to_pydatetime()
    df = pd.DataFrame({
        "datetime": [
            "2026-08-01 14:00:00+00:00",
            "2026-10-01 10:00:00+00:00",
            "2026-10-01 14:30:00+00:00",
        ],
        "close": [100.0, 200.0, 201.0],
    })
    fresh = filter_warm_history(df, max_age_hours=6, now=now)
    assert fresh is not None
    assert list(fresh["close"]) == [200.0, 201.0]

    stale = filter_warm_history(df.iloc[:1], max_age_hours=6, now=now)
    assert stale is None
