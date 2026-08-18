"""Unit tests for 30s→1M candle fallback (no Rithmic connection required)."""

import pandas as pd
import pytest

from src.broker.rithmic_connector import resample_subminute_to_1m, resample_1m_to_subminute


def _make_30s_bars(n_minutes: int = 5) -> pd.DataFrame:
    """Two 30s bars per minute with distinct OHLC."""
    rows = []
    base = pd.Timestamp("2026-06-14 18:00:00", tz="UTC")
    price = 22000.0
    for m in range(n_minutes):
        t0 = base + pd.Timedelta(minutes=m)
        rows.append({
            "datetime": t0,
            "open": price,
            "high": price + 2,
            "low": price - 1,
            "close": price + 1,
            "volume": 100,
        })
        rows.append({
            "datetime": t0 + pd.Timedelta(seconds=30),
            "open": price + 1,
            "high": price + 3,
            "low": price,
            "close": price + 2,
            "volume": 80,
        })
        price += 2
    return pd.DataFrame(rows)


class TestResampleSubminuteTo1m:
    def test_aggregates_two_30s_bars_per_minute(self):
        df_30s = _make_30s_bars(5)
        df_1m = resample_subminute_to_1m(df_30s, 30)
        assert df_1m is not None
        assert len(df_1m) == 5
        assert df_1m.iloc[0]["open"] == pytest.approx(22000.0)
        assert df_1m.iloc[0]["close"] == pytest.approx(22002.0)
        assert df_1m.iloc[0]["high"] == pytest.approx(22003.0)
        assert df_1m.iloc[0]["volume"] == 180

    def test_roundtrip_with_synthetic_1m_split(self):
        df_1m = pd.DataFrame({
            "datetime": pd.date_range("2026-06-14 18:00", periods=10, freq="1min", tz="UTC"),
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
            "volume": [200] * 10,
        })
        df_30s = resample_1m_to_subminute(df_1m, 30)
        assert df_30s is not None
        df_back = resample_subminute_to_1m(df_30s, 30)
        assert df_back is not None
        assert len(df_back) == 10
        assert df_back.iloc[-1]["close"] == pytest.approx(109.0)

    def test_insufficient_bars_returns_none(self):
        assert resample_subminute_to_1m(None, 30) is None
        assert resample_subminute_to_1m(pd.DataFrame(), 30) is None


class TestRithmicConnectorFallback:
    def test_aggregate_1m_from_30s_uses_provided_df(self):
        from src.broker.rithmic_connector import RithmicConnector

        broker = RithmicConnector(live_mode=False)
        df_30s = _make_30s_bars(30)
        out = broker.aggregate_1m_from_30s("MNQ", num_candles=60, df_30s=df_30s)
        assert out is not None
        assert len(out) == 30

    def test_candle_shortfall_warning_once(self):
        from src.broker.rithmic_connector import RithmicConnector

        broker = RithmicConnector(live_mode=False)
        broker._log_candle_shortfall_once("MNQ", 1, 0)
        broker._log_candle_shortfall_once("MNQ", 1, 0)
        assert ("MNQ", 1) in broker._candle_shortfall_warned
        assert len(broker._candle_shortfall_warned) == 1
