"""Pull recent NQ=F bars from Yahoo (free) to cover the live period after MNQ_1m.csv ends.

1m: Yahoo serves the last ~30 days in 7-day chunks. 5m: last ~60 days.
NQ and MNQ track the same index at the same price, so NQ=F stands in for MNQ.
"""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))


def norm(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    return df[~df.index.duplicated(keep="last")].sort_index()


now = datetime.now(timezone.utc)
parts = []
start = now - timedelta(days=29)
while start < now:
    end = min(start + timedelta(days=7), now)
    d = yf.download("NQ=F", start=start, end=end, interval="1m", progress=False, auto_adjust=False)
    if len(d):
        parts.append(norm(d))
    start = end
m1 = pd.concat(parts)
m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
m1.to_csv(os.path.join(OUT, "NQ_1m_yahoo.csv"))
print("1m", len(m1), m1.index[0], m1.index[-1])

m5 = norm(yf.download("NQ=F", period="60d", interval="5m", progress=False, auto_adjust=False))
m5.to_csv(os.path.join(OUT, "NQ_5m_yahoo.csv"))
print("5m", len(m5), m5.index[0], m5.index[-1])
