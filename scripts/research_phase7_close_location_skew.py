#!/usr/bin/env python
"""
Phase 7 -- single tightly-scoped question carried over from Phase 6:
does close-location-within-range produce a stable, reproducible change in
the SHAPE of the subsequent return distribution, independent of its mean?
Phase 6 already showed the conditional MEAN is null; this only tests shape.

No strategy, entries, exits, stops, or targets anywhere in this file. Does
not touch the live/production framework.

Pre-registered BEFORE running anything (matches Phase 6's locked H2 config,
not re-optimized here):
  loc(t) = (close[t]-low_N[t]) / (high_N[t]-low_N[t]), N=20, causal (uses
  only bar t's own OHLC and its trailing 20-bar window).
  Event groups: TOP = loc >= 0.90 (first bar entering after not being
  there); BOTTOM = loc <= 0.10, same construction. Symmetric, both tested.
  Horizons: 5, 10, 20, 40, 60, 100 bars -- full curve reported, no
  horizon selection.
  Non-overlapping sampling: greedy min-spacing = horizon for the event
  group; a systematic every-Nth-bar sample (N = horizon) of the WHOLE
  series for the baseline, so both groups share the same dependence
  structure and are comparable under iid-style tests.
  Multiple testing: 2 tails x 6 horizons x 3 symbols = 36 primary cells,
  Bonferroni-corrected.
  Discovery = first 80%, Holdout = last 20% (same split point used
  throughout this project). Full 36-cell grid run once on each -- nothing
  is re-selected between discovery and holdout.

Usage:
    python scripts/research_phase7_close_location_skew.py discovery
    python scripts/research_phase7_close_location_skew.py holdout
    python scripts/research_phase7_close_location_skew.py confounds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.indicators import atr as atr_ind  # noqa: E402

SYMBOLS = ["MES", "MNQ", "NQ"]
HORIZONS = [5, 10, 20, 40, 60, 100]
N_WINDOW = 20            # close-location window, matches Phase 6 locked config
DECILE = 0.90            # matches Phase 6 locked config
CONFOUND_HORIZON = 20    # secondary confound checks run at this single horizon only
N_PERM = 2000
N_BOOT = 2000
RNG_SEED = 20260817       # fixed seed, pre-registered, not tuned after seeing results
RTH_START_UTC, RTH_END_UTC = 13.5, 20.0


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{symbol}_5m.csv")
    col = "datetime" if "datetime" in df.columns else "date"
    df["datetime"] = pd.to_datetime(df[col], utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def split_discovery_holdout(df: pd.DataFrame):
    n = len(df)
    split = int(n * 0.8)
    split_date = df.index[split]
    return df[df.index < split_date], df[df.index >= split_date], split_date


def add_close_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hi_n = df["high"].rolling(N_WINDOW).max()
    lo_n = df["low"].rolling(N_WINDOW).min()
    rng = (hi_n - lo_n).replace(0, np.nan)
    df["close_loc"] = (df["close"] - lo_n) / rng
    return df


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def non_overlapping_events(qualify_idx: pd.DatetimeIndex, all_index: pd.DatetimeIndex, min_spacing: int) -> pd.DatetimeIndex:
    pos_map = {ts: i for i, ts in enumerate(all_index)}
    qualify_positions = sorted(pos_map[ts] for ts in qualify_idx if ts in pos_map)
    kept, last_kept = [], -min_spacing - 1
    for p in qualify_positions:
        if p - last_kept >= min_spacing:
            kept.append(p)
            last_kept = p
    return all_index[kept]


def event_bars(df: pd.DataFrame, decile: float) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    top = df["close_loc"] >= decile
    bottom = df["close_loc"] <= (1 - decile)
    top_first = top & ~top.shift(1).fillna(False).astype(bool)
    bottom_first = bottom & ~bottom.shift(1).fillna(False).astype(bool)
    return df.index[top_first.fillna(False)], df.index[bottom_first.fillna(False)]


def systematic_baseline(all_index: pd.DatetimeIndex, spacing: int) -> pd.DatetimeIndex:
    """Every `spacing`-th bar, same non-overlap property as the event
    sample, used as the comparison distribution for KS/permutation tests
    (comparing two dependency-matched samples, not an unmatched full series)."""
    return all_index[::spacing]


def distributional_stats(x: np.ndarray, baseline_p10: float, baseline_p90: float) -> dict:
    n = len(x)
    if n < 2:
        return {"n": n, "insufficient": True}
    large_pos = x > baseline_p90
    large_neg = x < baseline_p10
    return {
        "n": n, "insufficient": n < 30,
        "mean_bps": np.mean(x) * 1e4, "median_bps": np.median(x) * 1e4,
        "std_bps": np.std(x, ddof=1) * 1e4,
        "skew": stats.skew(x) if n >= 8 else np.nan,
        "kurtosis": stats.kurtosis(x) if n >= 8 else np.nan,
        "p5_bps": np.percentile(x, 5) * 1e4, "p10_bps": np.percentile(x, 10) * 1e4,
        "p25_bps": np.percentile(x, 25) * 1e4, "p75_bps": np.percentile(x, 75) * 1e4,
        "p90_bps": np.percentile(x, 90) * 1e4, "p95_bps": np.percentile(x, 95) * 1e4,
        "prob_large_positive": float(large_pos.mean()),
        "prob_large_negative": float(large_neg.mean()),
        "upside_tail_mean_bps": (x[large_pos].mean() * 1e4) if large_pos.any() else np.nan,
        "downside_tail_mean_bps": (x[large_neg].mean() * 1e4) if large_neg.any() else np.nan,
        "mean_abs_bps": np.mean(np.abs(x)) * 1e4,
    }


def permutation_test_skew_diff(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """H0: x and y are drawn from distributions with equal skew. Two-sided
    permutation p-value on skew(x)-skew(y)."""
    if len(x) < 8 or len(y) < 8:
        return np.nan, np.nan
    obs = stats.skew(x) - stats.skew(y)
    pooled = np.concatenate([x, y])
    nx = len(x)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = stats.skew(pooled[:nx]) - stats.skew(pooled[nx:])
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def bootstrap_skew_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 8 or len(y) < 8:
        return np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs[i] = stats.skew(bx) - stats.skew(by)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_cell(df_feat: pd.DataFrame, symbol: str, split_label: str, tail: str, horizon: int, rng: np.random.Generator) -> dict:
    top_ev, bottom_ev = event_bars(df_feat, DECILE)
    events = top_ev if tail == "top" else bottom_ev
    fwd_all = forward_return(df_feat["close"], horizon)

    ev_idx = non_overlapping_events(events, df_feat.index, min_spacing=horizon)
    base_idx = systematic_baseline(df_feat.index, spacing=horizon)

    x = fwd_all.loc[fwd_all.index.intersection(ev_idx)].dropna().values
    y = fwd_all.loc[fwd_all.index.intersection(base_idx)].dropna().values

    if len(x) < 8 or len(y) < 8:
        return {"symbol": symbol, "split": split_label, "tail": tail, "horizon_bars": horizon,
                "n_event": len(x), "n_baseline": len(y), "insufficient": True}

    baseline_p10, baseline_p90 = np.percentile(y, 10), np.percentile(y, 90)
    event_stats = distributional_stats(x, baseline_p10, baseline_p90)
    baseline_stats = distributional_stats(y, baseline_p10, baseline_p90)

    ks_stat, ks_p = stats.ks_2samp(x, y)
    skew_diff_obs, perm_p = permutation_test_skew_diff(x, y, N_PERM, rng)
    ci_lo, ci_hi = bootstrap_skew_diff_ci(x, y, N_BOOT, rng)

    row = {"symbol": symbol, "split": split_label, "tail": tail, "horizon_bars": horizon,
           "insufficient": event_stats["n"] < 30 or baseline_stats["n"] < 30,
           "ks_stat": ks_stat, "ks_p_value": ks_p,
           "skew_diff_obs": skew_diff_obs, "skew_diff_perm_p": perm_p,
           "skew_diff_boot_ci_lo": ci_lo, "skew_diff_boot_ci_hi": ci_hi}
    row.update({f"event_{k}": v for k, v in event_stats.items()})
    row.update({f"baseline_{k}": v for k, v in baseline_stats.items()})
    return row


def full_grid(split_label: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for symbol in SYMBOLS:
        raw = load(symbol)
        disc, hold, split_date = split_discovery_holdout(raw)
        target = disc if split_label == "discovery" else hold
        df_feat = add_close_location(target)
        print(f"  {symbol} {split_label}: {len(df_feat)} bars (split={split_date})")
        for tail in ["top", "bottom"]:
            for h in HORIZONS:
                rows.append(run_cell(df_feat, symbol, split_label, tail, h, rng))
    return pd.DataFrame(rows)


def report_grid(df: pd.DataFrame, label: str):
    n_cells = len(df)
    valid = df[~df["insufficient"].fillna(False)]
    bonferroni_alpha = 0.05 / n_cells
    print(f"\n{'=' * 78}\n{label}: {n_cells} cells, {len(valid)} with sufficient n. "
          f"Bonferroni alpha={bonferroni_alpha:.2e}\n{'=' * 78}")
    cols = ["symbol", "tail", "horizon_bars", "event_n", "baseline_n", "event_skew", "baseline_skew",
            "skew_diff_obs", "skew_diff_perm_p", "skew_diff_boot_ci_lo", "skew_diff_boot_ci_hi",
            "ks_stat", "ks_p_value", "event_mean_bps", "baseline_mean_bps"]
    df2 = df.rename(columns={"event_n": "event_n", "baseline_n": "baseline_n"})
    for c in ["event_n", "baseline_n"]:
        if c not in df2.columns:
            df2[c] = df2.get(f"event_{c}", df2.get(c, np.nan))
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_rows", None):
        present = [c for c in cols if c in df2.columns]
        print(df2[present].to_string(index=False))

    sig = df[(df["skew_diff_perm_p"] < bonferroni_alpha) & (~df["insufficient"].fillna(False))]
    print(f"\nCells surviving Bonferroni correction on skew-difference permutation p-value: {len(sig)} / {n_cells}")
    if len(sig):
        print(sig[["symbol", "tail", "horizon_bars", "skew_diff_obs", "skew_diff_perm_p", "ks_p_value"]].to_string(index=False))


def discovery_report():
    Path("reports/phase7").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nDISCOVERY -- full pre-registered grid (holdout untouched)\n{'=' * 78}")
    df = full_grid("discovery")
    df.to_csv("reports/phase7/discovery_grid.csv", index=False)
    report_grid(df, "DISCOVERY")


def holdout_report():
    Path("reports/phase7").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nHOLDOUT -- identical pre-registered grid, run exactly once\n{'=' * 78}")
    df = full_grid("holdout")
    df.to_csv("reports/phase7/LOCKED_holdout_grid.csv", index=False)
    report_grid(df, "HOLDOUT")


def confound_report():
    """Secondary attribution check at the single CONFOUND_HORIZON, discovery
    only. Not part of the primary 36-cell multiple-testing grid; reported
    separately and explicitly labeled as such."""
    Path("reports/phase7").mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(RNG_SEED + 1)
    h = CONFOUND_HORIZON

    for symbol in SYMBOLS:
        raw = load(symbol)
        disc, hold, split_date = split_discovery_holdout(raw)
        df = add_close_location(disc)
        atr14 = atr_ind(df["high"], df["low"], df["close"], 14)
        df["atr_pctile"] = atr14.rolling(500).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False)
        df["ret20"] = df["close"].pct_change(20)
        df["hour_utc"] = df.index.hour + df.index.minute / 60.0
        df["rth"] = (df["hour_utc"] >= RTH_START_UTC) & (df["hour_utc"] < RTH_END_UTC)
        df["sma200"] = df["close"].rolling(200).mean()
        df["trend_up"] = df["close"] > df["sma200"]
        was_low = df["atr_pctile"].rolling(5).min().shift(1) < 0.30
        was_high = df["atr_pctile"].rolling(5).max().shift(1) > 0.70
        df["near_transition"] = ((was_low & (df["atr_pctile"] > 0.70)) | (was_high & (df["atr_pctile"] < 0.50))).fillna(False)
        print(f"  {symbol}: features built, running confound strata at h={h}")

        top_ev, bottom_ev = event_bars(df, DECILE)
        fwd_all = forward_return(df["close"], h)

        confound_masks = {
            "recent_return_direction": [("prior20_up", df["ret20"] > 0), ("prior20_down", df["ret20"] <= 0)],
            "recent_return_magnitude": [("|ret20|_top_half", df["ret20"].abs() >= df["ret20"].abs().median()),
                                         ("|ret20|_bottom_half", df["ret20"].abs() < df["ret20"].abs().median())],
            "volatility_level": [("atr_pctile>=0.5", df["atr_pctile"] >= 0.5), ("atr_pctile<0.5", df["atr_pctile"] < 0.5)],
            "volatility_transition": [("near_transition", df["near_transition"]), ("not_near_transition", ~df["near_transition"])],
            "rth_overnight": [("RTH", df["rth"]), ("overnight", ~df["rth"])],
            "trend_state": [("above_sma200", df["trend_up"]), ("below_sma200", ~df["trend_up"])],
        }

        for tail, events in [("top", top_ev), ("bottom", bottom_ev)]:
            ev_idx = non_overlapping_events(events, df.index, min_spacing=h)
            for confound_name, variants in confound_masks.items():
                for variant_label, mask in variants:
                    sub_idx = df.index[mask.fillna(False)].intersection(ev_idx)
                    x = fwd_all.loc[fwd_all.index.intersection(sub_idx)].dropna().values
                    if len(x) < 20:
                        rows.append({"symbol": symbol, "tail": tail, "confound": confound_name,
                                      "variant": variant_label, "n": len(x), "insufficient": True})
                        continue
                    rows.append({"symbol": symbol, "tail": tail, "confound": confound_name,
                                  "variant": variant_label, "n": len(x), "insufficient": False,
                                  "skew": stats.skew(x), "mean_bps": np.mean(x) * 1e4})

    out = pd.DataFrame(rows)
    out.to_csv("reports/phase7/confound_strata_discovery.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_rows", None):
        print(f"\n{'=' * 78}\nCONFOUND-STRATIFIED SKEW (discovery only, h={h}, secondary/attribution check)\n{'=' * 78}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    if mode == "discovery":
        discovery_report()
    elif mode == "holdout":
        holdout_report()
    elif mode == "confounds":
        confound_report()
    else:
        print("usage: research_phase7_close_location_skew.py [discovery|holdout|confounds]")
