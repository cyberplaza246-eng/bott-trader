"""Unit tests for gated confidence-based MNQ lot sizing."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.size_scale import (
    compute_sized_contracts,
    confidence_to_desired,
    format_size_log,
    load_size_scale_config,
)


def test_confidence_bands_default():
    assert confidence_to_desired(0.70, 0.72, 0.78, 0.85) == 1
    assert confidence_to_desired(0.72, 0.72, 0.78, 0.85) == 2
    assert confidence_to_desired(0.78, 0.72, 0.78, 0.85) == 3
    assert confidence_to_desired(0.85, 0.72, 0.78, 0.85) == 4
    assert confidence_to_desired(0.99, 0.72, 0.78, 0.85) == 4


def test_gate_requires_day_green_and_streak():
    # High conf but red day -> base
    n, reason = compute_sized_contracts(
        0.90, daily_pnl=-10.0, win_streak=5
    )
    assert n == 1
    assert "day_not_green" in reason

    # Green day but streak short -> base
    n, reason = compute_sized_contracts(
        0.90, daily_pnl=50.0, win_streak=1
    )
    assert n == 1
    assert "streak=" in reason

    # Green + streak >=2 -> scale to 4
    n, reason = compute_sized_contracts(
        0.90, daily_pnl=50.0, win_streak=2
    )
    assert n == 4
    assert reason.startswith("scale")


def test_mid_bands_when_gate_open():
    n, _ = compute_sized_contracts(0.75, daily_pnl=20.0, win_streak=3)
    assert n == 2
    n, _ = compute_sized_contracts(0.80, daily_pnl=20.0, win_streak=3)
    assert n == 3


def test_never_exceeds_max():
    n, _ = compute_sized_contracts(
        0.99, daily_pnl=100.0, win_streak=10, max_contracts=4
    )
    assert n == 4
    n, _ = compute_sized_contracts(
        0.99, daily_pnl=100.0, win_streak=10, max_contracts=3
    )
    assert n == 3


def test_format_size_log():
    line = format_size_log(2, 0.755, "scale day_green streak=2 conf=0.755")
    assert line.startswith("contracts=2 confidence=0.755 reason=")


def test_load_config_env(monkeypatch):
    monkeypatch.setenv("MAX_CONTRACTS", "4")
    monkeypatch.setenv("SIZE_CONF_2", "0.70")
    monkeypatch.setenv("SIZE_WIN_STREAK_MIN", "3")
    cfg = load_size_scale_config()
    assert cfg["max_contracts"] == 4
    assert cfg["conf_2"] == 0.70
    assert cfg["win_streak_min"] == 3
