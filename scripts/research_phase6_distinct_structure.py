#!/usr/bin/env python
"""
Phase 6 -- genuinely distinct market-behavior hypotheses, deliberately
excluding anything that duplicates Phase 4/5 (ATR-level persistence,
return autocorrelation, time-of-day, range-persistence-as-vol-clustering)
or the earlier strategy-reconstruction phases (sweep/MSS, breakout, VWAP
filter, ADX gate). Read-only research. No strategy, entries, exits, stops,
or targets anywhere in this file. Does not touch the live/production code.

Three hypotheses, each mathematically defined below, each tested with the
SAME discipline that closed Phase 5: baseline-drift-controlled t-test,
non-overlapping/event-based sampling, discovery/holdout split (80/20,
holdout untouched until locked_holdout_report()), small pre-registered
grids (not hundreds of indicators), MNQ/NQ treated as one correlated
reading, not two confirmations.

H1 -- Directional run persistence (path dependence):
  run_length(t) = length of the current streak of same-sign 1-bar returns
  ending at bar t. Event = the bar where a streak FIRST reaches length k
  (not every bar within it, to avoid recounting one streak many times).
  Question: after k consecutive same-direction bars, does the forward
  h-bar return differ from baseline (continuation) or invert sign
  (reversal)?

H2 -- Close-location-within-range (continuous, not a breakout/sweep event):
  loc(t) = (close[t] - low_N[t]) / (high_N[t] - low_N[t]), where high_N/
  low_N are the N-bar rolling max/min INCLUDING bar t's own high/low (causal
  -- describes where bar t's own close sits in its own recent range, no
  lookahead). Event = the bar where loc(t) FIRST enters the top or bottom
  decile after not having been there. This never requires price to exceed
  the range (unlike breakout) or wick through and recover (unlike sweep)
  -- it is purely a graded closing-location measure.

H3 -- Volatility-regime TRANSITION (not level):
  Using the same atr_pctile(500) as Phase 4/5 ONLY as an input to detect a
  crossing, never as a persistent-state filter. "Release" = atr_pctile
  crosses from <0.30 (within the last 5 bars) to >0.70 now. "Exhaustion" =
  crosses from >0.70 (within the last 5 bars) to <0.50 now. Question: does
  the transition EVENT itself carry directional information beyond the
  already-known fact that volatility will now be elevated/subsiding?

Usage:
    python scripts/research_phase6_distinct_structure.py discovery
    python scripts/research_phase6_distinct_structure.py holdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.instruments.instrument_registry import REGISTRY  # noqa: E402
from src.strategies.indicators import atr as atr_ind  # noqa: E402

SYMBOLS = ["MES", "MNQ", "NQ"]
HORIZONS = [10, 20, 40]  # small, pre-registered grid -- not a wide sweep

# Pre-registered locked configs, fixed BEFORE looking at any result,
# chosen as the natural/central value in each grid, not the best-looking one.
LOCKED = {
    "H1_run_persistence": {"run_length_k": 5, "horizon": 20},
    "H2_close_location": {"decile": 0.90, "horizon": 20},  # symmetric bottom decile = 0.10
    "H3_vol_transition": {"lookback": 5, "horizon": 20},
}
RUN_LENGTH_GRID = [3, 5, 8]
DECILE_GRID = [0.85, 0.90, 0.95]


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


def directional_decomposition(fwd: pd.Series, baseline_fwd: pd.Series) -> dict:
    fwd = fwd.dropna()
    n = len(fwd)
    if n < 2:
        return {"n": n, "insufficient": True}
    pos = fwd[fwd > 0]
    baseline_mean = baseline_fwd.dropna().mean()
    t_stat, p_val = stats.ttest_1samp(fwd.values, baseline_mean)
    skew = stats.skew(fwd.values) if n >= 8 else np.nan
    baseline_p95 = baseline_fwd.dropna().quantile(0.95)
    baseline_p05 = baseline_fwd.dropna().quantile(0.05)
    return {
        "n": n, "insufficient": n < 30,
        "mean_bps": fwd.mean() * 1e4, "baseline_mean_bps": baseline_mean * 1e4,
        "prob_positive": len(pos) / n,
        "mean_abs_bps": fwd.abs().mean() * 1e4,
        "skew": skew,
        "prob_beyond_baseline_p95": float((fwd > baseline_p95).mean()),
        "prob_beyond_baseline_p05": float((fwd < baseline_p05).mean()),
        "t_stat_vs_baseline": t_stat, "p_value_vs_baseline": p_val,
    }


def economic_note(fwd: pd.Series, symbol: str) -> dict:
    spec = REGISTRY[symbol]
    return {
        "round_trip_cost_usd": spec.commission_rt + spec.tick_size * spec.tick_value_usd,
        "mean_abs_fwd_ret_bps": fwd.dropna().abs().mean() * 1e4 if len(fwd.dropna()) else np.nan,
    }


# ───────────────────────────── H1: run persistence ─────────────────────────
def add_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sign = np.sign(df["close"].diff()).fillna(0)
    group = (sign != sign.shift(1)).cumsum()
    df["run_sign"] = sign
    df["run_length"] = sign.groupby(group).cumcount() + 1
    df.loc[sign == 0, "run_length"] = 0
    return df


def h1_event_bars(df: pd.DataFrame, k: int) -> pd.DatetimeIndex:
    just_reached = (df["run_length"] == k) & (df["run_length"].shift(1) == k - 1)
    return df.index[just_reached.fillna(False)]


def run_h1(df: pd.DataFrame, symbol: str, split_label: str, ks, horizons) -> pd.DataFrame:
    rows = []
    for k in ks:
        events = h1_event_bars(df, k)
        for h in horizons:
            fwd_all = forward_return(df["close"], h)
            ev = non_overlapping_events(events, df.index, min_spacing=h)
            fwd = fwd_all.loc[fwd_all.index.intersection(ev)]
            dd = directional_decomposition(fwd, fwd_all)
            dd.update({"symbol": symbol, "split": split_label, "hypothesis": "H1_run_persistence",
                       "run_length_k": k, "horizon_bars": h})
            rows.append(dd)
    return pd.DataFrame(rows)


# ─────────────────────── H2: close-location-within-range ───────────────────
def add_h2_features(df: pd.DataFrame, n_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    hi_n = df["high"].rolling(n_window).max()
    lo_n = df["low"].rolling(n_window).min()
    rng = (hi_n - lo_n).replace(0, np.nan)
    df["close_loc"] = (df["close"] - lo_n) / rng
    return df


def h2_event_bars(df: pd.DataFrame, decile: float) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    top = df["close_loc"] >= decile
    bottom = df["close_loc"] <= (1 - decile)
    top_first = top & ~top.shift(1).fillna(False).astype(bool)
    bottom_first = bottom & ~bottom.shift(1).fillna(False).astype(bool)
    return df.index[top_first.fillna(False)], df.index[bottom_first.fillna(False)]


def run_h2(df: pd.DataFrame, symbol: str, split_label: str, deciles, horizons) -> pd.DataFrame:
    rows = []
    for dec in deciles:
        top_ev, bottom_ev = h2_event_bars(df, dec)
        for h in horizons:
            fwd_all = forward_return(df["close"], h)
            for label, events in [("top_decile", top_ev), ("bottom_decile", bottom_ev)]:
                ev = non_overlapping_events(events, df.index, min_spacing=h)
                fwd = fwd_all.loc[fwd_all.index.intersection(ev)]
                dd = directional_decomposition(fwd, fwd_all)
                dd.update({"symbol": symbol, "split": split_label, "hypothesis": "H2_close_location",
                           "decile": dec, "variant": label, "horizon_bars": h})
                rows.append(dd)
    return pd.DataFrame(rows)


# ───────────────────────── H3: volatility transitions ───────────────────────
def add_h3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr14 = atr_ind(df["high"], df["low"], df["close"], 14)
    df["atr_pctile"] = atr14.rolling(500).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False)
    return df


def h3_event_bars(df: pd.DataFrame, lookback: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    p = df["atr_pctile"]
    was_low = p.rolling(lookback).min().shift(1) < 0.30
    was_high = p.rolling(lookback).max().shift(1) > 0.70
    release = was_low & (p > 0.70)
    exhaustion = was_high & (p < 0.50)
    return df.index[release.fillna(False)], df.index[exhaustion.fillna(False)]


def run_h3(df: pd.DataFrame, symbol: str, split_label: str, lookbacks, horizons) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        release_ev, exhaustion_ev = h3_event_bars(df, lb)
        for h in horizons:
            fwd_all = forward_return(df["close"], h)
            for label, events in [("release", release_ev), ("exhaustion", exhaustion_ev)]:
                ev = non_overlapping_events(events, df.index, min_spacing=h)
                fwd = fwd_all.loc[fwd_all.index.intersection(ev)]
                dd = directional_decomposition(fwd, fwd_all)
                dd.update({"symbol": symbol, "split": split_label, "hypothesis": "H3_vol_transition",
                           "lookback": lb, "variant": label, "horizon_bars": h})
                rows.append(dd)
    return pd.DataFrame(rows)


# ──────────────────────────────── drivers ───────────────────────────────────
def discovery_report():
    Path("reports/phase6").mkdir(parents=True, exist_ok=True)
    h1_all, h2_all, h3_all = [], [], []

    for symbol in SYMBOLS:
        print(f"\n{'=' * 78}\n{symbol} -- DISCOVERY ONLY (holdout untouched)\n{'=' * 78}")
        raw = load(symbol)
        disc, hold, split_date = split_discovery_holdout(raw)
        print(f"  discovery: {len(disc)} bars, holdout: {len(hold)} bars, split={split_date}")

        d1 = add_h1_features(disc)
        h1_all.append(run_h1(d1, symbol, "discovery", RUN_LENGTH_GRID, HORIZONS))

        d2 = add_h2_features(disc)
        h2_all.append(run_h2(d2, symbol, "discovery", DECILE_GRID, HORIZONS))

        d3 = add_h3_features(disc)
        h3_all.append(run_h3(d3, symbol, "discovery", [5], HORIZONS))

    h1_df, h2_df, h3_df = pd.concat(h1_all, ignore_index=True), pd.concat(h2_all, ignore_index=True), pd.concat(h3_all, ignore_index=True)
    h1_df.to_csv("reports/phase6/H1_run_persistence_discovery.csv", index=False)
    h2_df.to_csv("reports/phase6/H2_close_location_discovery.csv", index=False)
    h3_df.to_csv("reports/phase6/H3_vol_transition_discovery.csv", index=False)

    n_comparisons = len(h1_df) + len(h2_df) + len(h3_df)
    bonferroni_alpha = 0.05 / n_comparisons
    from scipy.stats import norm
    bonferroni_t = abs(norm.ppf(bonferroni_alpha / 2))
    print(f"\n{'=' * 78}\nMULTIPLE TESTING: {n_comparisons} discovery-side comparisons run "
          f"(H1={len(h1_df)}, H2={len(h2_df)}, H3={len(h3_df)}).\n"
          f"Bonferroni-corrected alpha={bonferroni_alpha:.2e} -> requires |t| >= {bonferroni_t:.2f} "
          f"to be taken seriously at this stage.\n{'=' * 78}")

    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_rows", None):
        print(f"\n--- H1 run persistence (discovery) ---")
        print(h1_df[["symbol", "run_length_k", "horizon_bars", "n", "mean_bps", "baseline_mean_bps",
                      "prob_positive", "skew", "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))
        print(f"\n--- H2 close-location (discovery) ---")
        print(h2_df[["symbol", "decile", "variant", "horizon_bars", "n", "mean_bps", "baseline_mean_bps",
                      "prob_positive", "skew", "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))
        print(f"\n--- H3 vol transition (discovery) ---")
        print(h3_df[["symbol", "variant", "horizon_bars", "n", "mean_bps", "baseline_mean_bps",
                      "prob_positive", "skew", "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))

    print(f"\n{'=' * 78}\nLOCKED FOR HOLDOUT:\n  H1: run_length_k={LOCKED['H1_run_persistence']['run_length_k']}, "
          f"horizon={LOCKED['H1_run_persistence']['horizon']}\n"
          f"  H2: decile={LOCKED['H2_close_location']['decile']}, horizon={LOCKED['H2_close_location']['horizon']}\n"
          f"  H3: lookback={LOCKED['H3_vol_transition']['lookback']}, horizon={LOCKED['H3_vol_transition']['horizon']}\n"
          f"These were fixed before writing this script, not re-selected from the grids above.\n{'=' * 78}")


def locked_holdout_report():
    Path("reports/phase6").mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in SYMBOLS:
        raw = load(symbol)
        disc, hold, split_date = split_discovery_holdout(raw)
        print(f"\n{symbol}: holdout = {len(hold)} bars from {split_date} onward")

        h = LOCKED["H1_run_persistence"]["horizon"]
        d1 = add_h1_features(hold)
        r1 = run_h1(d1, symbol, "holdout", [LOCKED["H1_run_persistence"]["run_length_k"]], [h])
        r1["econ_mean_abs_fwd_ret_bps"] = economic_note(forward_return(hold["close"], h), symbol)["mean_abs_fwd_ret_bps"]
        rows.append(r1)

        h = LOCKED["H2_close_location"]["horizon"]
        d2 = add_h2_features(hold)
        r2 = run_h2(d2, symbol, "holdout", [LOCKED["H2_close_location"]["decile"]], [h])
        r2["econ_mean_abs_fwd_ret_bps"] = economic_note(forward_return(hold["close"], h), symbol)["mean_abs_fwd_ret_bps"]
        rows.append(r2)

        h = LOCKED["H3_vol_transition"]["horizon"]
        d3 = add_h3_features(hold)
        r3 = run_h3(d3, symbol, "holdout", [LOCKED["H3_vol_transition"]["lookback"]], [h])
        r3["econ_mean_abs_fwd_ret_bps"] = economic_note(forward_return(hold["close"], h), symbol)["mean_abs_fwd_ret_bps"]
        rows.append(r3)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv("reports/phase6/LOCKED_holdout_result.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(f"\n{'=' * 78}\nLOCKED HOLDOUT RESULTS (Phase 6)\n{'=' * 78}")
        cols = [c for c in ["symbol", "hypothesis", "variant", "run_length_k", "decile", "horizon_bars",
                             "n", "mean_bps", "baseline_mean_bps", "prob_positive", "skew",
                             "t_stat_vs_baseline", "p_value_vs_baseline", "econ_mean_abs_fwd_ret_bps"] if c in out.columns]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    if mode == "discovery":
        discovery_report()
    elif mode == "holdout":
        locked_holdout_report()
    else:
        print("usage: research_phase6_distinct_structure.py [discovery|holdout]")
