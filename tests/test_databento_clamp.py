from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util


def _load_dl():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_databento_latest.py"
    spec = importlib.util.spec_from_file_location("download_databento_latest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clamp_uses_available_end_when_sooner_than_lag():
    dl = _load_dl()
    now = datetime(2026, 8, 18, 2, 51, tzinfo=timezone.utc)
    avail = datetime(2026, 8, 18, 2, 40, tzinfo=timezone.utc)
    end = dl.clamp_query_end(now, available_end=avail, lag_minutes=2)
    assert end == avail


def test_clamp_falls_back_to_now_minus_two_minutes():
    dl = _load_dl()
    now = datetime(2026, 8, 18, 2, 51, tzinfo=timezone.utc)
    end = dl.clamp_query_end(now, available_end=None, lag_minutes=2)
    assert end == now - timedelta(minutes=2)


def test_parse_available_end_from_422_message():
    dl = _load_dl()
    msg = (
        "422 data_end_after_available_end The dataset GLBX.MDP3 has data "
        "available up to '2026-08-18 02:40:00+00:00'."
    )
    parsed = dl.parse_available_end_from_error(msg)
    assert parsed is not None
    assert parsed.astimezone(timezone.utc).hour == 2
    assert parsed.astimezone(timezone.utc).minute == 40
