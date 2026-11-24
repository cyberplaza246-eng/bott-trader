#!/usr/bin/env python3
"""Download latest MNQ from Databento (1m, optional 1s->30s). Key stays in .env."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUOTE_META_PATH = os.path.join(DATA_DIR, "mnq_desk_quote.json")
_SECRET_RE = re.compile(
    r"(DATABENTO_API_KEY\s*=\s*)\S+|(Bearer )\S+|(key=)[^&\s\"']+",
    re.I,
)
_AVAIL_END_RE = re.compile(r"available up to '([^']+)'", re.I)


def _public_err(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    text = _SECRET_RE.sub(lambda m: (m.group(1) or m.group(2) or m.group(3) or "") + "[REDACTED]", text)
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if key:
        text = text.replace(key, "[REDACTED]")
    return text


def parse_available_end(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def available_end_from_dataset_range(info: Dict[str, Any], schema: str = "ohlcv-1m") -> datetime | None:
    if not isinstance(info, dict):
        return parse_available_end(info)
    schema_map = info.get("schema")
    if isinstance(schema_map, dict):
        block = schema_map.get(schema)
        if isinstance(block, dict):
            parsed = parse_available_end(block.get("end"))
            if parsed is not None:
                return parsed
        parsed = parse_available_end(block)
        if parsed is not None:
            return parsed
    return parse_available_end(info.get("end"))


def parse_available_end_from_error(message: str) -> datetime | None:
    match = _AVAIL_END_RE.search(str(message) or "")
    if not match:
        return None
    return parse_available_end(match.group(1))


def clamp_query_end(
    now: datetime,
    available_end: datetime | None = None,
    lag_minutes: int = 2,
) -> datetime:
    """Keep Databento `end` at or before available_end, else now-lag (never future)."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    end = now - timedelta(minutes=max(0, int(lag_minutes)))
    if available_end is not None:
        avail = available_end
        if avail.tzinfo is None:
            avail = avail.replace(tzinfo=timezone.utc)
        else:
            avail = avail.astimezone(timezone.utc)
        end = min(end, avail)
    return end


def _dataset_available_end(client, schema: str = "ohlcv-1m") -> datetime | None:
    try:
        info = client.metadata.get_dataset_range(dataset="GLBX.MDP3")
    except Exception:
        return None
    return available_end_from_dataset_range(info, schema=schema)


def _et_stamp(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M ET")


def _client(*, strict: bool = True):
    import databento as db

    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        msg = "DATABENTO_API_KEY not set in .env"
        if strict:
            print(f"ERROR: {msg}")
            sys.exit(1)
        raise RuntimeError(msg)
    return db.Historical(key)


def _cost(client, schema: str, start: str, end: str, symbol: str = "MNQ.c.0") -> float:
    return float(
        client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema=schema,
            start=start,
            end=end,
            stype_in="continuous",
        )
    )


def _to_ohlcv(data) -> pd.DataFrame:
    df = data.to_df()
    if df.empty:
        return df
    df = df.reset_index()
    if "ts_event" in df.columns:
        df["datetime"] = pd.to_datetime(df["ts_event"], utc=True)
    elif "datetime" not in df.columns:
        raise SystemExit(f"No timestamp column: {list(df.columns)}")
    sample = float(df["open"].iloc[0])
    if sample > 1_000_000:
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] / 1e9
    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return out[out["volume"] > 0].reset_index(drop=True)


def download_range(client, schema: str, start: str, end: str, symbol: str = "MNQ.c.0"):
    return client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        schema=schema,
        start=start,
        end=end,
        stype_in="continuous",
    )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df.set_index("datetime")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out[out["volume"] > 0].reset_index(drop=True)


def merge_csv(path: str, new: pd.DataFrame) -> pd.DataFrame:
    if os.path.isfile(path):
        from src.strategy.mnq_15m_ema_eod import load_1m_seed_csv, parse_mnq_1m_datetime

        old = load_1m_seed_csv(path)
        new = new.copy()
        new["datetime"] = parse_mnq_1m_datetime(new["datetime"])
        new = new.dropna(subset=["datetime"])
        if old is None or old.empty:
            return new.reset_index(drop=True)
        both = pd.concat([old, new], ignore_index=True)
        both = both.sort_values("datetime").drop_duplicates("datetime", keep="last")
        return both.reset_index(drop=True)
    return new


def write_quote_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(DATA_DIR, exist_ok=True)
    out = dict(payload)
    with open(QUOTE_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    return out


def load_quote_meta() -> Dict[str, Any]:
    if not os.path.isfile(QUOTE_META_PATH):
        return {}
    try:
        data = json.loads(open(QUOTE_META_PATH, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _file_last_print(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["datetime"])
        if df.empty:
            return {}
        row = df.iloc[-1]
        ts = pd.to_datetime(row["datetime"], utc=True)
        return {
            "file_last_price": round(float(row["close"]), 2),
            "file_last_bar_ts_et": _et_stamp(ts),
        }
    except Exception:
        return {}


def _parse_databento_end(exc: BaseException):
    msg = str(exc)
    m2 = re.search(r"end time before ([0-9T:\.\-]+Z)", msg)
    if "dataset_unavailable_range" in msg and m2:
        ts = pd.Timestamp(m2.group(1))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return (ts.tz_convert("UTC") - pd.Timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = re.search(r"available up to '([^']+)'", msg)
    if "data_end_after_available_end" in msg and m:
        ts = pd.Timestamp(m.group(1))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _download_1m_latest(client, start: str, end: str):
    last_err = None
    for _ in range(4):
        try:
            return download_range(client, "ohlcv-1m", start, end)
        except Exception as exc:
            last_err = exc
            nxt = _parse_databento_end(exc)
            if not nxt or nxt == end:
                raise
            end = nxt
    if last_err:
        raise last_err
    raise RuntimeError("Databento 1m download failed")


def refresh_latest_1m(*, lookback_days: int = 3) -> Dict[str, Any]:
    """Pull recent MNQ 1m from Databento and merge into data/MNQ_1m.csv. Never prints keys."""
    now = datetime.now(timezone.utc)
    merged_path = os.path.join(DATA_DIR, "MNQ_1m.csv")
    known = _file_last_print(merged_path)
    try:
        client = _client(strict=False)
        available = _dataset_available_end(client, "ohlcv-1m")
        end_dt = clamp_query_end(now, available_end=available, lag_minutes=2)
        start_dt = now - timedelta(days=max(1, int(lookback_days)))
        if end_dt <= start_dt:
            start_dt = end_dt - timedelta(days=max(1, int(lookback_days)))
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = _download_1m_latest(client, start, end)
        fresh = _to_ohlcv(raw)
        if fresh is None or getattr(fresh, "empty", True):
            raise RuntimeError("Databento returned no MNQ 1m bars")
        merged = merge_csv(merged_path, fresh)
        merged.to_csv(merged_path, index=False)
        last = merged.iloc[-1]
        last_ts = pd.to_datetime(last["datetime"], utc=True)
        last_px = round(float(last["close"]), 2)
        age_sec = int((now - last_ts.to_pydatetime()).total_seconds())
        meta = {
            "ok": True,
            "error": None,
            "last_price": last_px,
            "last_bar_ts_et": _et_stamp(last_ts),
            "last_bar_ts_utc": last_ts.isoformat(),
            "refreshed_at_et": _et_stamp(now),
            "quote_age_sec": max(0, age_sec),
            "bars_new": int(len(fresh)),
            "source": "databento",
            "path": "data/MNQ_1m.csv",
        }
        write_quote_meta(meta)
        age_note = f"age {meta['quote_age_sec']}s"
        if int(meta["quote_age_sec"] or 0) >= 3600:
            age_note += " — leftover RTH file, overnight uses ticker last"
        if not os.getenv("DAY_HYBRID", "").strip().lower() in ("1", "true", "yes", "on"):
            print(
                f"MNQ 1m refresh ok: {last_px} as of {meta['last_bar_ts_et']} "
                f"({meta['bars_new']} new bars, {age_note})"
            )
        return meta
    except Exception as exc:
        meta = {
            "ok": False,
            "error": f"Databento download failed: {_public_err(exc)}",
            "last_price": None,
            "last_bar_ts_et": None,
            "refreshed_at_et": _et_stamp(now),
            "quote_age_sec": None,
            "source": "databento",
            "path": "data/MNQ_1m.csv",
            **known,
        }
        write_quote_meta(meta)
        err = str(meta.get("error") or "")
        # Caller (PaperCsvFeed) rate-limits 402 spam; keep one-line here only for CLI
        if os.getenv("DATABENTO_REFRESH_QUIET", "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            pass
        elif "402" in err or "delinquent" in err.lower() or "unpaid invoice" in err.lower():
            # Single short line — PaperCsvFeed will suppress further attempts
            print(f"MNQ 1m refresh failed: Databento 402 unpaid invoice")
        else:
            print(f"MNQ 1m refresh failed: {meta['error']}")
        return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-only", action="store_true")
    parser.add_argument("--start", default="2026-03-12")
    parser.add_argument("--end", default="2026-08-18")
    parser.add_argument("--august-1s", action="store_true", help="Also pull ohlcv-1s for August")
    parser.add_argument("--onesec-start", default="", help="Extra 1s download start (YYYY-MM-DD)")
    parser.add_argument("--skip-1m", action="store_true", help="Do not download/merge 1m")
    parser.add_argument(
        "--desk-refresh",
        action="store_true",
        help="Pull last few days of MNQ 1m into data/MNQ_1m.csv for the live desk",
    )
    parser.add_argument("--lookback-days", type=int, default=3)
    args = parser.parse_args()

    if args.desk_refresh:
        meta = refresh_latest_1m(lookback_days=args.lookback_days)
        sys.exit(0 if meta.get("ok") else 1)

    client = _client()
    print("Cost checks (MNQ.c.0 continuous):")
    jobs = [
        ("1m gap/latest", "ohlcv-1m", args.start, args.end),
        ("1m August", "ohlcv-1m", "2026-08-01", args.end),
    ]
    if args.august_1s:
        jobs.append(("1s August", "ohlcv-1s", "2026-08-01", args.end))
    if args.onesec_start:
        jobs.append(("1s extra", "ohlcv-1s", args.onesec_start, args.end))
    costs = {}
    for name, schema, start, end in jobs:
        try:
            c = _cost(client, schema, start, end)
            costs[name] = c
            print(f"  {name} {start}->{end}: ${c:.4f}")
        except Exception as e:
            print(f"  {name}: {e}")
            costs[name] = None

    if args.cost_only:
        return

    if args.skip_1m:
        print("Skipping 1m download.")
    else:
        print("\nDownloading MNQ 1m...")
        raw = download_range(client, "ohlcv-1m", args.start, args.end)
        fresh = _to_ohlcv(raw)
        print(f"  new 1m bars: {len(fresh)}  {fresh['datetime'].min()} -> {fresh['datetime'].max()}")

        merged_path = os.path.join(DATA_DIR, "MNQ_1m.csv")
        merged = merge_csv(merged_path, fresh)
        merged.to_csv(merged_path, index=False)
        resample(merged, "5min").to_csv(os.path.join(DATA_DIR, "MNQ_5m.csv"), index=False)
        print(f"  merged MNQ_1m.csv: {len(merged)}  {merged['datetime'].min()} -> {merged['datetime'].max()}")

        aug = merged[
            (merged["datetime"] >= pd.Timestamp("2026-08-01", tz="UTC"))
            & (merged["datetime"] < pd.Timestamp(args.end, tz="UTC"))
        ].reset_index(drop=True)
        aug_path = os.path.join(DATA_DIR, "MNQ_1m_august_databento.csv")
        aug.to_csv(aug_path, index=False)
        resample(aug, "5min").to_csv(os.path.join(DATA_DIR, "MNQ_5m_august_databento.csv"), index=False)
        print(f"  August 1m slice: {len(aug)} -> {os.path.basename(aug_path)}")

    if args.august_1s and costs.get("1s August") is not None:
        print("\nDownloading MNQ 1s August (build real 30s)...")
        raw1s = download_range(client, "ohlcv-1s", "2026-08-01", args.end)
        s1 = _to_ohlcv(raw1s)
        s1.to_csv(os.path.join(DATA_DIR, "MNQ_1s_august_databento.csv"), index=False)
        s30 = resample(s1, "30s")
        s30.to_csv(os.path.join(DATA_DIR, "MNQ_30s_august_databento.csv"), index=False)
        print(f"  1s: {len(s1)}  30s: {len(s30)}")

    if args.onesec_start and costs.get("1s extra") is not None:
        print(f"\nDownloading MNQ 1s {args.onesec_start} -> {args.end}...")
        raw1s = download_range(client, "ohlcv-1s", args.onesec_start, args.end)
        s1 = _to_ohlcv(raw1s)
        extra_1s = os.path.join(DATA_DIR, "MNQ_1s_databento.csv")
        s1_all = merge_csv(extra_1s, s1)
        s1_all.to_csv(extra_1s, index=False)
        s30 = resample(s1_all, "30s")
        s30_path = os.path.join(DATA_DIR, "MNQ_30s_databento.csv")
        s30.to_csv(s30_path, index=False)
        print(f"  1s total: {len(s1_all)}  30s: {len(s30)} -> MNQ_30s_databento.csv")


if __name__ == "__main__":
    main()
