"""Data loading + timezone verification + 1m->5m resample (research only, no broker)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLEAN = "/workspace/neat_plan_out/clean.parquet"
RAW_CSV = "/workspace/mnq_1m_massive.csv"
ET = "America/New_York"


def load_1m(path: str = CLEAN) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("datetime").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    return d


def tz_check(d: pd.DataFrame) -> dict:
    """Verify bars labelled UTC really are UTC: the cash-open volume spike must sit at
    13:30 UTC during EDT and 14:30 UTC during EST (09:30 ET in both)."""
    ny = d.index.tz_convert(ET)
    off_h = ((ny.tz_localize(None) - d.index.tz_localize(None)).total_seconds() / 3600).astype(int)
    out = {}
    for o, label in ((-4, "EDT"), (-5, "EST")):
        m = off_h == o
        sub = d.loc[m]
        v = sub.groupby(sub.index.strftime("%H:%M"))["volume"].mean()
        top = v.idxmax()
        out[label] = {
            "rows": int(m.sum()),
            "peak_volume_minute_utc": top,
            "peak_volume_minute_et": (pd.Timestamp("2000-01-01 " + top) + pd.Timedelta(hours=o)).strftime("%H:%M"),
            "vol_13:29": float(v.get("13:29", np.nan)), "vol_13:30": float(v.get("13:30", np.nan)),
            "vol_14:29": float(v.get("14:29", np.nan)), "vol_14:30": float(v.get("14:30", np.nan)),
        }
    # first bar of each Globex session should be 18:00 ET (22:00 UTC EDT / 23:00 UTC EST)
    gaps = d.index.to_series().diff() > pd.Timedelta(minutes=30)
    first_et = d.index[gaps.values].tz_convert(ET).strftime("%H:%M").value_counts().head(3).to_dict()
    out["session_first_bar_et_counts"] = first_et
    out["verdict_utc_ok"] = (out["EDT"]["peak_volume_minute_utc"] == "13:30"
                             and out["EST"]["peak_volume_minute_utc"] == "14:30")
    return out


def resample_5m(d: pd.DataFrame) -> pd.DataFrame:
    """Left-closed, left-labelled 5m bars (label = bar start, like the live 5m frame)."""
    agg = d[["open", "high", "low", "close", "volume"]].resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open", "high", "low", "close"])
