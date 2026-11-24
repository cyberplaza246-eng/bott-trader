from scripts.validate_hybrid_edge import filter_trades, grade


class _T:
    def __init__(self, entry_time):
        self.entry_time = entry_time


def test_grade_thresholds():
    assert grade({"trades": 10, "profit_factor": 3.0}) == "INSUFFICIENT"
    assert grade({"trades": 40, "profit_factor": 0.8}) == "FAIL"
    assert grade({"trades": 40, "profit_factor": 1.1}) == "WEAK"
    assert grade({"trades": 40, "profit_factor": 1.3}) == "PASS"


def test_filter_trades_utc_window():
    trades = [
        _T("2026-02-28 15:00:00+00:00"),
        _T("2026-03-01 15:00:00+00:00"),
        _T("2026-03-11 15:00:00+00:00"),
        _T("2026-03-12 15:00:00+00:00"),
    ]
    out = filter_trades(trades, "2026-03-01", "2026-03-12")
    assert len(out) == 2
