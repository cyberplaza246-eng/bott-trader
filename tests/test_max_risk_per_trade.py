"""2026-10-08: MAX_RISK_PER_TRADE_USD guard (stop risk x contracts <= cap)."""
from __future__ import annotations

from unittest.mock import MagicMock

import start_live_rithmic as slr


def test_default_cap_is_500():
    assert slr.MAX_RISK_PER_TRADE_USD == 500.0


def test_helper_allows_trims_and_skips():
    # MNQ $2/pt
    assert slr.cap_contracts_by_risk(30945, 31303, 2.0, 1, 500)[0] == 0      # $716 -> skip
    assert slr.cap_contracts_by_risk(31000, 31100, 2.0, 1, 500) == (1, 200.0)
    assert slr.cap_contracts_by_risk(31000, 31100, 2.0, 4, 500)[0] == 2      # $800 -> 2 x $200
    assert slr.cap_contracts_by_risk(31000, 31125, 2.0, 2, 500)[0] == 2      # exactly $500 ok
    assert slr.cap_contracts_by_risk(31000, 31250.25, 2.0, 1, 500)[0] == 0   # $500.50 skip
    assert slr.cap_contracts_by_risk(30945, 31303, 2.0, 3, 0)[0] == 3        # 0 = off
    assert slr.cap_contracts_by_risk(31000, 31025, 20.0, 1, 500)[0] == 1     # NQ 25pt = $500
    assert slr.cap_contracts_by_risk(31000, 31124, 20.0, 1, 500)[0] == 0     # NQ ~$2480 skip
    assert slr.cap_contracts_by_risk(31000, float("nan"), 2.0, 1, 500)[0] == 0


def _trader(contracts):
    t = slr.LiveRithmicTrader(symbol="MNQ", paper_mode=False, skip_confirm=True)
    t.broker = MagicMock()
    t.broker.place_order.return_value = {"ticket": "bot_x", "entry_price": 30945.0}
    t._broker_can_update_stop = lambda r: True
    t._write_board_status = lambda **k: None
    t._contracts_for_entry = lambda sig: (contracts, 0.8, "test")
    t.max_risk_per_trade_usd = 500.0
    return t


def _sig(sl):
    return {"symbol": "MNQ", "direction": "short", "entry": 30945.0, "sl": sl, "tp": 30229.0,
            "confidence": 0.8, "atr": 51.4}


def test_place_order_skips_716_breakout():
    t = _trader(1)
    assert t.place_order(_sig(31303.0)) is False
    t.broker.place_order.assert_not_called()
    assert not t.positions


def test_place_order_trims_contracts_to_fit():
    t = _trader(4)
    assert t.place_order(_sig(31045.0)) is True       # $200/contract -> 2
    assert t.broker.place_order.call_args.kwargs["size"] == 2
    assert t.positions["bot_x"].size == 2


def test_place_order_within_cap_unchanged():
    t = _trader(1)
    assert t.place_order(_sig(31045.0)) is True
    assert t.broker.place_order.call_args.kwargs["size"] == 1


def test_cap_zero_disables_guard():
    t = _trader(1)
    t.max_risk_per_trade_usd = 0
    assert t.place_order(_sig(31303.0)) is True
    assert t.broker.place_order.call_args.kwargs["size"] == 1


def test_bats_set_cap_500():
    import pathlib
    root = pathlib.Path(slr.__file__).resolve().parent
    for name in ("start_mnq_live_revamp.bat", "START_ENSEMBLE_VISIBLE.bat"):
        txt = (root / name).read_text(encoding="utf-8", errors="replace")
        assert "MAX_RISK_PER_TRADE_USD=500" in txt, name
