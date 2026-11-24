"""2026-10-09: status line 'Day total' from the session ledger; shadow export keeps the newest finished Rithmic bar."""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import start_live_rithmic as slr
from src.utils import shadow_bar_export as sbe

UTC = timezone.utc


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# -- Day total ----------------------------------------------------------------------

@pytest.fixture
def trader(monkeypatch, tmp_path):
    for k in ("DAY_SESSION_LOSS_LIMIT", "OVERNIGHT_SESSION_LOSS_LIMIT", "DAILY_PROFIT_TARGET", "SESSION_LIMITS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    monkeypatch.setenv("SESSION_PNL_FILE", str(tmp_path / "s.json"))
    return slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)


def test_day_total_uses_ledger_not_since_restart(trader):
    """Oct 9 case: 7 overnight trades = +$46.50 in the ledger, this run only saw the -$76."""
    for i, pnl in enumerate((56.5, 55.0, 109.5, 17.5, -52.0, -64.0)):
        trader._session_ledger.book(_t("2026-10-09T05:00:00"), pnl, f"bot_{i}")
    trader._session_ledger.book(_t("2026-10-09T12:09:30"), -76.0, "bot_adopt")
    trader.daily_pnl = -76.0
    assert trader._day_total_text(_t("2026-10-09T12:35:00")) == "Day total: $46.50"
    # 9:30 AM day session still shows the whole trading day
    trader._session_ledger.book(_t("2026-10-09T14:00:00"), -20.0, "bot_day")
    assert trader._day_total_text(_t("2026-10-09T15:00:00")) == "Day total: $26.50"
    # 4 PM ET starts the next trading day (Mon after a Friday)
    assert trader._day_total_text(_t("2026-10-09T20:30:00")) == "Day total: $0.00"


def test_day_total_negative_format(trader):
    trader._session_ledger.book(_t("2026-10-09T05:00:00"), -76.0, "x")
    assert trader._day_total_text(_t("2026-10-09T06:00:00")) == "Day total: $-76.00"


def test_day_total_falls_back_when_session_limits_off(monkeypatch):
    monkeypatch.setenv("SESSION_LIMITS", "0")
    monkeypatch.setattr(slr, "bot_logger", MagicMock())
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.daily_pnl = -76.0
    assert t._session_ledger is None
    assert t._day_total_text() == "Daily: $-76.00"


def test_day_total_falls_back_on_ledger_error(trader):
    trader._session_ledger = MagicMock()
    trader._session_ledger.pnl.side_effect = RuntimeError("boom")
    trader.daily_pnl = 12.0
    assert trader._day_total_text() == "Daily: $12.00"


# -- shadow bar export ---------------------------------------------------------------

def _bars(ends_utc, tag=True):
    df = pd.DataFrame({
        "datetime": pd.to_datetime(ends_utc, utc=True),
        "open": [1.0 + i for i in range(len(ends_utc))],
        "high": [2.0 + i for i in range(len(ends_utc))],
        "low": [0.5 + i for i in range(len(ends_utc))],
        "close": [1.5 + i for i in range(len(ends_utc))],
        "volume": [10] * len(ends_utc),
    })
    if tag:
        df.attrs["bar_time"] = "end"
    return df


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.delenv("SHADOW_BAR_EXPORT", raising=False)
    sbe._last_written.clear()
    yield tmp_path / "bars.csv"
    sbe._last_written.clear()


ENDS = ["2026-10-09T12:25:00Z", "2026-10-09T12:30:00Z", "2026-10-09T12:35:00Z"]


def test_rithmic_newest_finished_bar_written(out):
    n = sbe.export_closed_bars(_bars(ENDS), "MNQ", path=out, now="2026-10-09T12:36:30Z")
    got = pd.read_csv(out)
    assert n == 3 and got["ts_et"].iloc[-1].startswith("2026-10-09T08:35:00")


def test_untagged_frame_still_drops_newest(out):  # Yahoo fallback / merged local history
    n = sbe.export_closed_bars(_bars(ENDS, tag=False), "MNQ", path=out, now="2026-10-09T12:36:30Z")
    assert n == 2 and pd.read_csv(out)["ts_et"].iloc[-1].startswith("2026-10-09T08:30:00")


def test_future_end_bar_not_written(out):  # bar end still ahead = forming
    n = sbe.export_closed_bars(_bars(ENDS), "MNQ", path=out, now="2026-10-09T12:34:00Z")
    assert n == 2


def test_no_duplicates_across_calls_and_restart(out):
    sbe.export_closed_bars(_bars(ENDS, tag=False), "MNQ", path=out, now="2026-10-09T12:36:30Z")
    sbe._last_written.clear()  # simulate a bot restart (reads the tip from the file)
    sbe.export_closed_bars(_bars(ENDS + ["2026-10-09T12:40:00Z"]), "MNQ", path=out, now="2026-10-09T12:40:05Z")
    sbe.export_closed_bars(_bars(ENDS + ["2026-10-09T12:40:00Z"]), "MNQ", path=out, now="2026-10-09T12:40:20Z")
    got = pd.read_csv(out)
    assert got["ts_et"].is_unique and len(got) == 4


def test_attrs_survive_live_pipeline():
    """Tag must survive tail/reset_index, the bot's df.copy()+columns, and be dropped by a merge."""
    df = _bars(ENDS)
    df2 = df.tail(2).reset_index(drop=True).copy()
    df2["ema"] = df2["close"].ewm(span=3).mean()
    assert df2.attrs.get("bar_time") == "end"
    merged = pd.concat([_bars(ENDS, tag=False), df], ignore_index=True)
    assert merged.attrs.get("bar_time") != "end"


def test_kid_reader_sees_newest_bar(out, monkeypatch):
    import importlib, sys
    from pathlib import Path
    sbe.export_closed_bars(_bars(ENDS), "MNQ", path=out, now="2026-10-09T12:36:30Z")
    kid_dir = Path(slr.__file__).resolve().parent / "kid_shadow"
    if not (kid_dir / "warmup_MNQ_5m.csv").is_file():
        pytest.skip("kid warmup file missing")
    monkeypatch.setenv("SHADOW_BARS", str(out))
    monkeypatch.syspath_prepend(str(kid_dir))
    sys.modules.pop("kid_shadow", None)
    ks = importlib.import_module("kid_shadow")
    bars, n_live, _ = ks.load_bars()
    assert n_live == 3 and bars["datetime"].iloc[-1] == pd.Timestamp("2026-10-09T12:35:00Z")
    sys.modules.pop("kid_shadow", None)
