#!/usr/bin/env python
"""
Phase 4 -- data-driven edge discovery from market behavior, not named
strategies. No "strategy" (breakout/reversion/sweep/etc.) is assumed going
in; this measures the market's own unconditional and conditional
statistical structure and reports everything, positive or null, so any
genuine exploitable structure can be identified before it gets a name.

Uses existing audited 5M OHLCV (data/*_5m.csv), no new data purchased.
Read-only research script -- does not modify any strategy or live file.

Five scans, each strictly causal (a bar's own future range/close is never
used to condition on itself -- signal bar t only uses info through bar t's
close; outcome is measured over bars AFTER t):

  1. Return autocorrelation at lags 1/2/3/5/10/20/50 bars (Ljung-Box-style
     single-lag t-test on the correlation coefficient).
  2. Conditional forward returns after N-sigma bars (|return| > k * rolling
     std) -- continuation vs. mean-reversion signature, multiple horizons.
  3. Volatility-regime persistence -- does elevated ATR now predict elevated
     ATR N bars later, and does realized forward return differ by regime.
  4. Time-of-day conditional mean return, by UTC hour bucket.
  5. Range persistence -- does a narrow-range bar predict next-bar range
     expansion (actionable timing information, not direction).

Usage:
    python scripts/research_market_behavior.py --symbol MES MNQ NQ
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.indicators import atr as atr_ind  # noqa: E402

HORIZONS = [1, 5, 20]          # forward bars measured (5m bars -> 5/25/100 min)
SIGMA_LEVELS = [2.0, 3.0]
NR_LOOKBACK = 20                # "narrow range" percentile window


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{symbol}_5m.csv")
    col = "datetime" if "datetime" in df.columns else "date"
    df["datetime"] = pd.to_datetime(df[col], utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df["ret"] = df["close"].pct_change()
    return df


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from bar t's close to bar (t+horizon)'s close -- purely forward,
    never uses info before/at t to compute anything about t itself beyond
    its own already-known close."""
    return close.shift(-horizon) / close - 1.0


def scan_autocorrelation(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    r = df["ret"].dropna()
    n = len(r)
    for lag in [1, 2, 3, 5, 10, 20, 50]:
        x, y = r.iloc[:-lag].values, r.iloc[lag:].values
        rho = np.corrcoef(x, y)[0, 1]
        # standard error of a correlation coefficient under H0: rho=0
        se = 1.0 / np.sqrt(n - lag - 3) if n - lag > 3 else np.nan
        t_stat = rho / se if se and not np.isnan(se) else np.nan
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
        rows.append({"symbol": symbol, "scan": "autocorrelation", "lag_bars": lag,
                     "n": n - lag, "rho": rho, "t_stat": t_stat, "p_value": p_val})
    return rows


def scan_sigma_moves(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    ret = df["ret"]
    roll_std = ret.rolling(100).std().shift(1)  # std known BEFORE bar t (no lookahead)
    z = ret / roll_std.replace(0, np.nan)
    close = df["close"]

    for sigma in SIGMA_LEVELS:
        for direction, mask in [("up", z >= sigma), ("down", z <= -sigma)]:
            idx = df.index[mask.fillna(False)]
            for h in HORIZONS:
                all_fwd = forward_return(close, h)
                baseline_mean = all_fwd.dropna().mean()  # unconditional drift at this horizon
                fwd = all_fwd.loc[idx].dropna()
                n = len(fwd)
                if n < 2:
                    rows.append({"symbol": symbol, "scan": "sigma_move", "sigma": sigma,
                                 "direction": direction, "horizon_bars": h, "n": n,
                                 "insufficient": True})
                    continue
                # test against the instrument's OWN unconditional mean forward
                # return at this horizon, not zero -- otherwise any regime
                # "looks" significant purely from the underlying's own drift.
                t_stat, p_val = stats.ttest_1samp(fwd.values, baseline_mean)
                rows.append({"symbol": symbol, "scan": "sigma_move", "sigma": sigma,
                             "direction": direction, "horizon_bars": h, "n": n,
                             "mean_fwd_ret_bps": fwd.mean() * 1e4,
                             "baseline_mean_bps": baseline_mean * 1e4,
                             "t_stat": t_stat,
                             "p_value": p_val, "insufficient": n < 30})
    return rows


def scan_vol_regime(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    atr14 = atr_ind(df["high"], df["low"], df["close"], 14)
    atr_pctile = atr14.rolling(500).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False)
    # regime known as of bar t's close -- condition, don't leak
    high_vol = (atr_pctile.shift(0) >= 0.90)
    low_vol = (atr_pctile.shift(0) <= 0.10)
    close = df["close"]

    # (a) persistence: does today's ATR percentile predict ATR percentile N bars later
    for h in [20, 100]:
        future_pctile = atr_pctile.shift(-h)
        both = pd.concat([atr_pctile, future_pctile], axis=1).dropna()
        n = len(both)
        rho = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1] if n > 3 else np.nan
        se = 1.0 / np.sqrt(n - 3) if n > 3 else np.nan
        t_stat = rho / se if se and not np.isnan(se) else np.nan
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
        # NOTE: successive atr_pctile values come from heavily overlapping
        # rolling windows, so consecutive observations are highly
        # autocorrelated -- the iid-based SE (1/sqrt(n-3)) badly overstates
        # significance here. Reported as a descriptive stylized fact (known
        # volatility clustering), NOT compared against the significance bar.
        rows.append({"symbol": symbol, "scan": "vol_persistence", "horizon_bars": h,
                     "n": n, "rho": rho, "t_stat": t_stat, "p_value": p_val,
                     "overlapping_samples_caveat": True})

    # (b) directional edge conditional on regime -- tested against the
    # instrument's own unconditional mean forward return at that horizon,
    # not zero (an equity-index future can drift for reasons that have
    # nothing to do with the vol regime; comparing to 0 would misattribute
    # plain drift to the regime).
    for label, mask in [("high_vol", high_vol), ("low_vol", low_vol)]:
        idx = df.index[mask.fillna(False)]
        for h in HORIZONS:
            all_fwd = forward_return(close, h)
            baseline_mean = all_fwd.dropna().mean()
            fwd = all_fwd.loc[idx].dropna()
            n = len(fwd)
            if n < 2:
                rows.append({"symbol": symbol, "scan": "vol_regime_directional",
                             "regime": label, "horizon_bars": h, "n": n, "insufficient": True})
                continue
            t_stat, p_val = stats.ttest_1samp(fwd.values, baseline_mean)
            rows.append({"symbol": symbol, "scan": "vol_regime_directional", "regime": label,
                         "horizon_bars": h, "n": n, "mean_fwd_ret_bps": fwd.mean() * 1e4,
                         "baseline_mean_bps": baseline_mean * 1e4,
                         "t_stat": t_stat, "p_value": p_val, "insufficient": n < 30})
    return rows


def scan_time_of_day(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    fwd1 = forward_return(df["close"], 1)
    hours = df.index.hour
    overall_mean = fwd1.dropna().mean()
    for h in range(24):
        vals = fwd1[hours == h].dropna()
        n = len(vals)
        if n < 30:
            rows.append({"symbol": symbol, "scan": "time_of_day", "utc_hour": h, "n": n,
                         "insufficient": True})
            continue
        t_stat, p_val = stats.ttest_1samp(vals.values, overall_mean)
        rows.append({"symbol": symbol, "scan": "time_of_day", "utc_hour": h, "n": n,
                     "mean_fwd_ret_bps": vals.mean() * 1e4,
                     "vs_overall_mean_bps": overall_mean * 1e4,
                     "t_stat": t_stat, "p_value": p_val, "insufficient": False})
    return rows


def scan_range_persistence(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    rng = df["high"] - df["low"]
    rng_pctile = rng.rolling(NR_LOOKBACK).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False)
    narrow = rng_pctile <= 0.10   # bottom decile range bar, known at bar t's close
    wide = rng_pctile >= 0.90

    for h in [1, 5]:
        future_rng = (df["high"].shift(-h).rolling(h).max() - df["low"].shift(-h).rolling(h).min()) if h > 1 else rng.shift(-1)
        for label, mask in [("narrow", narrow), ("wide", wide)]:
            idx = df.index[mask.fillna(False)]
            sample = future_rng.loc[idx].dropna()
            baseline = future_rng.dropna()
            n = len(sample)
            if n < 30:
                rows.append({"symbol": symbol, "scan": "range_persistence", "bar_type": label,
                             "horizon_bars": h, "n": n, "insufficient": True})
                continue
            t_stat, p_val = stats.ttest_ind(sample.values, baseline.values, equal_var=False)
            rows.append({"symbol": symbol, "scan": "range_persistence", "bar_type": label,
                         "horizon_bars": h, "n": n,
                         "mean_fwd_range": sample.mean(), "baseline_mean_range": baseline.mean(),
                         "t_stat": t_stat, "p_value": p_val, "insufficient": False})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", nargs="+", default=["MES", "MNQ", "NQ"])
    args = parser.parse_args()

    all_rows = []
    for symbol in args.symbol:
        df = load(symbol)
        print(f"\n{symbol}: {len(df)} 5M bars ({df.index[0]} -> {df.index[-1]})")
        all_rows += scan_autocorrelation(df, symbol)
        all_rows += scan_sigma_moves(df, symbol)
        all_rows += scan_vol_regime(df, symbol)
        all_rows += scan_time_of_day(df, symbol)
        all_rows += scan_range_persistence(df, symbol)

    out = pd.DataFrame(all_rows)
    Path("reports/phase4").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/phase4/market_behavior_scan.csv", index=False)
    print(f"\nWrote {len(out)} rows to reports/phase4/market_behavior_scan.csv")

    # Flag anything that clears a conservative bar: |t| >= 3 (well past the
    # usual 2.0 significance floor, since this is a wide multi-comparison
    # scan across ~5 scan types x 3 symbols x many sub-conditions) AND n>=30.
    sig = out[(out.get("t_stat").abs() >= 3.0) & (out.get("n", 0) >= 30)
              & (~out.get("insufficient", False).fillna(False))
              & (~out.get("overlapping_samples_caveat", False).fillna(False))]
    print(f"\n{'=' * 78}\nCANDIDATES CLEARING |t|>=3.0 (conservative, multi-comparison-aware bar), n>=30\n{'=' * 78}")
    if len(sig) == 0:
        print("  None.")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(sig.sort_values("t_stat", key=lambda s: s.abs(), ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
