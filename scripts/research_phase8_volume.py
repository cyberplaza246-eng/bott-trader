#!/usr/bin/env python
"""
Phase 8 -- the single bounded, final experiment authorized after the
information audit. Falsification experiment: does volume carry conditional
information beyond what the price-derived Phase 3-7 hypotheses (all
REJECTED/NULL/INCONCLUSIVE) already covered? No strategy, entries, exits,
stops, or targets anywhere in this file. Does not touch the live/production
framework. Not to be expanded into general volume-feature mining regardless
of intermediate results.

Pre-registered BEFORE running anything:

Hypothesis A -- relative-volume state:
  volume_ratio(t) = volume[t] / rolling_mean(volume, 20)[t]  (causal, bar
  t's own volume against its own trailing 20-bar average).
  vr_pctile(t) = rolling-500-bar percentile rank of volume_ratio(t) (same
  window convention as every prior phase's ATR/close-location percentile).
  Event = first bar entering the top or bottom {70,80,90,95}th-percentile
  band after not having been there. Grid: 4 thresholds x 2 sides (high/low)
  x 6 horizons x 3 symbols = 144 cells.

Hypothesis B -- volume-price divergence:
  "Large move" = |1-bar return| pctile(t) >= 0.90 (single fixed
  definition, not swept). Two variants only: large move + LOW relative
  volume (vr_pctile <= 0.20), large move + HIGH relative volume
  (vr_pctile >= 0.80). Grid: 2 variants x 6 horizons x 3 symbols = 36
  cells. No additional "large move" definitions added.

Total: 180 pre-registered cells. Bonferroni alpha = 0.05/180.

Same methodology as Phases 5-7: baseline-drift-controlled comparison
against a dependence-matched systematic-sample baseline, non-overlapping
event sampling (greedy min-spacing=horizon), permutation test (primary
significance test, on the MEAN difference -- this hypothesis is about
mean/directional information, unlike Phase 7's skew-shape question) and
bootstrap CI as a secondary check, KS test for general shape difference,
discovery/holdout split at the same 80/20 point used throughout, full grid
run once on each, nothing re-selected between them.

Usage:
    python scripts/research_phase8_volume.py discovery
    python scripts/research_phase8_volume.py holdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYMBOLS = ["MES", "MNQ", "NQ"]
HORIZONS = [5, 10, 20, 40, 60, 100]
VR_BASELINE_WINDOW = 20
PCTILE_WINDOW = 500
THRESHOLDS_A = [0.70, 0.80, 0.90, 0.95]
LARGE_MOVE_PCTILE = 0.90
DIVERGENCE_VOL_LOW, DIVERGENCE_VOL_HIGH = 0.20, 0.80
N_PERM = 1000
N_BOOT = 1000
RNG_SEED = 20260817


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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vol_baseline = df["volume"].rolling(VR_BASELINE_WINDOW).mean()
    df["volume_ratio"] = df["volume"] / vol_baseline.replace(0, np.nan)
    df["vr_pctile"] = df["volume_ratio"].rolling(PCTILE_WINDOW).rank(pct=True)
    ret1 = df["close"].pct_change()
    df["abs_ret1"] = ret1.abs()
    df["abs_ret1_pctile"] = df["abs_ret1"].rolling(PCTILE_WINDOW).rank(pct=True)
    return df


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def forward_excursion(df: pd.DataFrame, horizon: int):
    entry = df["close"]
    fwd_high = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    fwd_low = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    return fwd_high / entry - 1.0, fwd_low / entry - 1.0


def non_overlapping_events(qualify_idx: pd.DatetimeIndex, all_index: pd.DatetimeIndex, min_spacing: int) -> pd.DatetimeIndex:
    pos_map = {ts: i for i, ts in enumerate(all_index)}
    qualify_positions = sorted(pos_map[ts] for ts in qualify_idx if ts in pos_map)
    kept, last_kept = [], -min_spacing - 1
    for p in qualify_positions:
        if p - last_kept >= min_spacing:
            kept.append(p)
            last_kept = p
    return all_index[kept]


def systematic_baseline(all_index: pd.DatetimeIndex, spacing: int) -> pd.DatetimeIndex:
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
        "prob_positive": float((x > 0).mean()),
        "mean_abs_bps": np.mean(np.abs(x)) * 1e4,
        "prob_large_positive": float(large_pos.mean()), "prob_large_negative": float(large_neg.mean()),
        "upside_tail_mean_bps": (x[large_pos].mean() * 1e4) if large_pos.any() else np.nan,
        "downside_tail_mean_bps": (x[large_neg].mean() * 1e4) if large_neg.any() else np.nan,
    }


def permutation_test_mean_diff(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])
    nx = len(x)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = np.mean(pooled[:nx]) - np.mean(pooled[nx:])
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def bootstrap_mean_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.mean(bx) - np.mean(by)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def event_bars_hyp_a(df: pd.DataFrame, threshold: float, side: str) -> pd.DatetimeIndex:
    if side == "high":
        cond = df["vr_pctile"] >= threshold
    else:
        cond = df["vr_pctile"] <= (1 - threshold)
    first = cond & ~cond.shift(1).fillna(False).astype(bool)
    return df.index[first.fillna(False)]


def event_bars_hyp_b(df: pd.DataFrame, vol_side: str) -> pd.DatetimeIndex:
    large_move = df["abs_ret1_pctile"] >= LARGE_MOVE_PCTILE
    if vol_side == "low":
        vol_cond = df["vr_pctile"] <= DIVERGENCE_VOL_LOW
    else:
        vol_cond = df["vr_pctile"] >= DIVERGENCE_VOL_HIGH
    cond = large_move & vol_cond
    first = cond & ~cond.shift(1).fillna(False).astype(bool)
    return df.index[first.fillna(False)]


def run_cell(df: pd.DataFrame, symbol: str, split_label: str, hypothesis: str, tag: dict,
             events: pd.DatetimeIndex, horizon: int, rng: np.random.Generator) -> dict:
    fwd_all = forward_return(df["close"], horizon)
    ev_idx = non_overlapping_events(events, df.index, min_spacing=horizon)
    base_idx = systematic_baseline(df.index, spacing=horizon)

    x = fwd_all.loc[fwd_all.index.intersection(ev_idx)].dropna().values
    y = fwd_all.loc[fwd_all.index.intersection(base_idx)].dropna().values

    row = {"symbol": symbol, "split": split_label, "hypothesis": hypothesis, "horizon_bars": horizon, **tag}
    if len(x) < 8 or len(y) < 8:
        row.update({"event_n": len(x), "baseline_n": len(y), "insufficient": True})
        return row

    baseline_p10, baseline_p90 = np.percentile(y, 10), np.percentile(y, 90)
    ev_stats = distributional_stats(x, baseline_p10, baseline_p90)
    base_stats = distributional_stats(y, baseline_p10, baseline_p90)

    ks_stat, ks_p = stats.ks_2samp(x, y)
    mean_diff_obs, perm_p = permutation_test_mean_diff(x, y, N_PERM, rng)
    ci_lo, ci_hi = bootstrap_mean_diff_ci(x, y, N_BOOT, rng)

    up_exc, down_exc = forward_excursion(df, horizon)
    up_x = up_exc.loc[up_exc.index.intersection(ev_idx)].dropna()
    down_x = down_exc.loc[down_exc.index.intersection(ev_idx)].dropna()

    row.update({"insufficient": ev_stats["n"] < 30 or base_stats["n"] < 30,
                "ks_stat": ks_stat, "ks_p_value": ks_p,
                "mean_diff_obs_bps": mean_diff_obs * 1e4, "mean_diff_perm_p": perm_p,
                "mean_diff_boot_ci_lo_bps": ci_lo * 1e4, "mean_diff_boot_ci_hi_bps": ci_hi * 1e4,
                "mean_up_excursion_bps": up_x.mean() * 1e4 if len(up_x) else np.nan,
                "mean_down_excursion_bps": down_x.mean() * 1e4 if len(down_x) else np.nan})
    row.update({f"event_{k}": v for k, v in ev_stats.items()})
    row.update({f"baseline_{k}": v for k, v in base_stats.items()})
    return row


def full_grid(split_label: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for symbol in SYMBOLS:
        raw = load(symbol)
        disc, hold, split_date = split_discovery_holdout(raw)
        target = disc if split_label == "discovery" else hold
        df = add_features(target)
        print(f"  {symbol} {split_label}: {len(df)} bars (split={split_date})")

        for threshold in THRESHOLDS_A:
            for side in ["high", "low"]:
                events = event_bars_hyp_a(df, threshold, side)
                for h in HORIZONS:
                    rows.append(run_cell(df, symbol, split_label, "A_relative_volume",
                                          {"threshold": threshold, "side": side}, events, h, rng))

        for vol_side in ["low", "high"]:
            events = event_bars_hyp_b(df, vol_side)
            for h in HORIZONS:
                rows.append(run_cell(df, symbol, split_label, "B_divergence",
                                      {"vol_side": vol_side}, events, h, rng))
    return pd.DataFrame(rows)


def report_grid(df: pd.DataFrame, label: str):
    n_cells = len(df)
    valid = df[~df["insufficient"].fillna(True)]
    bonferroni_alpha = 0.05 / n_cells
    print(f"\n{'=' * 78}\n{label}: {n_cells} cells, {len(valid)} with sufficient n. "
          f"Bonferroni alpha={bonferroni_alpha:.2e}\n{'=' * 78}")

    cols = ["symbol", "hypothesis", "threshold", "side", "vol_side", "horizon_bars",
            "event_n", "baseline_n", "event_mean_bps", "baseline_mean_bps",
            "mean_diff_obs_bps", "mean_diff_perm_p", "mean_diff_boot_ci_lo_bps",
            "mean_diff_boot_ci_hi_bps", "ks_stat", "ks_p_value",
            "event_prob_positive", "mean_up_excursion_bps", "mean_down_excursion_bps"]
    present = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_rows", None):
        print(df[present].to_string(index=False))

    sig = df[(df["mean_diff_perm_p"] < bonferroni_alpha) & (~df["insufficient"].fillna(True))]
    print(f"\nCells surviving Bonferroni correction on mean-difference permutation p-value: {len(sig)} / {n_cells}")
    if len(sig):
        print(sig[present].to_string(index=False))
    return sig


def discovery_report():
    Path("reports/phase8").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nDISCOVERY -- full pre-registered grid (holdout untouched)\n{'=' * 78}")
    df = full_grid("discovery")
    df.to_csv("reports/phase8/discovery_grid.csv", index=False)
    report_grid(df, "DISCOVERY")


def holdout_report():
    Path("reports/phase8").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nHOLDOUT -- identical pre-registered grid, run exactly once\n{'=' * 78}")
    df = full_grid("holdout")
    df.to_csv("reports/phase8/LOCKED_holdout_grid.csv", index=False)
    report_grid(df, "HOLDOUT")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    if mode == "discovery":
        discovery_report()
    elif mode == "holdout":
        holdout_report()
    else:
        print("usage: research_phase8_volume.py [discovery|holdout]")
