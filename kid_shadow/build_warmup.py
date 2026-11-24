"""Build warmup_MNQ_5m.csv from local MNQ_1m plus the Sep 21-30 gap file.
Local 1m matches the training series through 2026-09-21 03:58 UTC.
The 03:55 bar is incomplete locally (missing 03:59) so it is inserted from the training set.
"""
import gzip
import base64
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = base64.b64decode(Path(HERE / "gap.b64").read_text(encoding="ascii"))
Path(HERE / "gap_MNQ_5m.csv").write_bytes(gzip.decompress(raw))
gap = pd.read_csv(HERE / "gap_MNQ_5m.csv")
gap["datetime"] = pd.to_datetime(gap["datetime"], utc=True)

m = pd.read_csv(HERE.parent / "data" / "MNQ_1m.csv")
m["datetime"] = pd.to_datetime(m["datetime"], utc=True)
m["b"] = m["datetime"].dt.floor("5min")
g = m.groupby("b", sort=True)
df = pd.DataFrame(dict(
    open=g.open.first(), high=g.high.max(), low=g.low.min(),
    close=g.close.last(), volume=g.volume.sum(),
)).reset_index().rename(columns={"b": "datetime"})
df = df[df.datetime <= "2026-09-21 03:50:00+00:00"].copy()
df["no_trade"] = False
bridge = pd.DataFrame([{
    "datetime": pd.Timestamp("2026-09-21 03:55:00+00:00"),
    "open": 30097.25, "high": 30097.50, "low": 30086.00, "close": 30087.75,
    "volume": 1836, "no_trade": False,
}])
out = pd.concat([df, bridge, gap], ignore_index=True)
out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").tail(6000)
out.to_csv(HERE / "warmup_MNQ_5m.csv", index=False)
print(len(out), out.datetime.min(), out.datetime.max())
