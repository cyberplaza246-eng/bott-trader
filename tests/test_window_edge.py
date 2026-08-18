"""MAE/MFE from 1m path + window/side/alignment grouping. No live config writes."""
import json
import os
import tempfile

import pandas as pd

from src.ai.trade_review import (
    compute_mae_mfe_pts,
    desk_exit_reason,
    normalize_ema15_exit_reason,
)
from src.ai.window_edge import (
    attach_desk_fields,
    expectancy_tables,
    group_stats,
    mfe_bucket,
    mfe_mae_distribution,
    write_snapshot,
)


def _bars(rows):
    return pd.DataFrame(rows)


def test_mae_mfe_long_from_1m_path():
    entry = pd.Timestamp("2026-07-01 13:40:00", tz="UTC")
    exit_ = pd.Timestamp("2026-07-01 14:10:00", tz="UTC")
    df = _bars([
        {"datetime": entry, "open": 20000, "high": 20010, "low": 19990, "close": 20000},
        {"datetime": entry + pd.Timedelta(minutes=1), "open": 20000, "high": 20080, "low": 19970, "close": 20040},
        {"datetime": exit_, "open": 20040, "high": 20120, "low": 20000, "close": 20100},
    ])
    mae, mfe = compute_mae_mfe_pts(df, entry, exit_, "long", 20000.0)
    assert mae == 30.0
    assert mfe == 120.0


def test_mae_mfe_short_from_1m_path():
    entry = pd.Timestamp("2026-07-01 15:00:00", tz="UTC")
    exit_ = pd.Timestamp("2026-07-01 15:30:00", tz="UTC")
    df = _bars([
        {"datetime": entry, "open": 20100, "high": 20120, "low": 20080, "close": 20100},
        {"datetime": entry + pd.Timedelta(minutes=1), "open": 20100, "high": 20140, "low": 19980, "close": 20020},
        {"datetime": exit_, "open": 20020, "high": 20050, "low": 19950, "close": 19970},
    ])
    mae, mfe = compute_mae_mfe_pts(df, entry, exit_, "short", 20100.0)
    assert mae == 40.0
    assert mfe == 150.0


def test_desk_exit_reasons():
    assert desk_exit_reason("SL") == "Stop"
    assert desk_exit_reason("EOD") == "EOD flatten"
    assert desk_exit_reason("TP") == "TP cap"
    assert normalize_ema15_exit_reason("TP") == "TP cap"


def test_group_by_window_side_alignment():
    rows = [
        {
            "event": "close",
            "direction": "long",
            "window_name": "09:35",
            "pnl_usd": 80.0,
            "mae_pts": 12,
            "mfe_pts": 110,
            "exit_reason": "EOD",
            "trend_15m": 1,
            "trend_60m": 1,
            "daily_trend": 1,
        },
        {
            "event": "close",
            "direction": "long",
            "window_name": "11:00",
            "pnl_usd": -40.0,
            "mae_pts": 38,
            "mfe_pts": 8,
            "exit_reason": "SL",
            "trend_15m": 1,
            "trend_60m": -1,
            "daily_trend": 1,
        },
        {
            "event": "close",
            "direction": "short",
            "window_name": "13:30",
            "pnl_usd": 20.0,
            "mae_pts": 10,
            "mfe_pts": 90,
            "exit_reason": "EOD",
            "trend_15m": -1,
            "trend_60m": -1,
            "daily_trend": -1,
        },
    ]
    tables = expectancy_tables(rows)
    assert tables["by_window"]["09:35"]["n"] == 1
    assert tables["by_window"]["11:00"]["n"] == 1
    assert tables["by_window"]["13:30"]["n"] == 1
    assert tables["by_window"]["11:00"]["total_pnl_usd"] == -40.0
    assert tables["by_side"]["long"]["n"] == 2
    assert tables["by_side"]["short"]["n"] == 1
    assert tables["by_alignment"]["aligned"]["n"] == 2
    assert tables["by_alignment"]["not aligned"]["n"] == 1
    assert tables["writes_live_config"] is False
    assert any("11:00" in line and "09:35" in line for line in tables["advisory"])


def test_mfe_buckets_and_trend_follow_behavior():
    winners = [
        {"direction": "long", "window_name": "09:35", "pnl_usd": 100, "mfe_pts": 140, "mae_pts": 20, "exit_reason": "EOD",
         "trend_15m": 1, "trend_60m": 1, "daily_trend": 1}
        for _ in range(6)
    ]
    losers = [
        {"direction": "long", "window_name": "11:00", "pnl_usd": -50, "mfe_pts": 10, "mae_pts": 40, "exit_reason": "SL",
         "trend_15m": 1, "trend_60m": 1, "daily_trend": 1}
        for _ in range(4)
    ]
    dist = mfe_mae_distribution(winners + losers)
    assert dist["winner_mfe_buckets"]["120-200"] == 6
    assert dist["pct_winners_mfe_ge_80"] == 100.0
    assert dist["behavior"] == "trend-follow"
    assert mfe_bucket(18) == "15-20"
    assert mfe_bucket(200) == "200+"


def test_attach_desk_fields_does_not_write_config():
    rec = attach_desk_fields({
        "direction": "short",
        "exit_reason": "SL",
        "entry_snapshot": {
            "window_name": "09:35",
            "trend_15m": -1,
            "trend_60m": 1,
            "daily_permission": -1,
            "atr15": 22.0,
            "atr_stop_pts": 44.0,
        },
        "pnl_usd": -88.0,
        "mae_pts": 44,
        "mfe_pts": 5,
        "entry_price": 21000,
        "exit_price": 21044,
    })
    assert rec["desk_exit_reason"] == "Stop"
    assert rec["window_name"] == "09:35"
    assert rec["aligned"] is False
    assert rec["stop_distance"] == 44.0
    assert rec["atr_at_entry"] == 22.0
    stats = group_stats([rec])
    assert stats["n"] == 1
    assert stats["losses"] == 1
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "mnq_profit_config.json")
        path = os.path.join(tmp, "window_edge_analysis.json")
        write_snapshot({"writes_live_config": False, "advisory_only": True}, path=path)
        assert os.path.isfile(path)
        assert not os.path.isfile(cfg)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["writes_live_config"] is False
