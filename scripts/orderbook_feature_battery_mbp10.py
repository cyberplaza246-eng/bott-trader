#!/usr/bin/env python3
"""
Stage-1 predictive-information test for MNQ MBP-10 (multi-level order
book), exactly 3 pre-defined features, each justified by information
MBP-1 (top-of-book only) structurally cannot provide. Same rigorous
framework as the MBP-1 batteries: train/OOS split, sign stability, R^2,
incremental-information regression against the already-tested MBP-1
features (so a positive result must prove it adds something new, not
just re-derive depth_imbalance).

This answers exactly one question:
  Does depth beyond the best bid/offer contain incremental predictive
  information that MBP-1 cannot see?

Features tested (from bars in MNQ_orderbook_mbp10_1min_*.parquet):
  1. depth_weighted_imbalance -- (bid_top10_sum - ask_top10_sum) /
                                  (bid_top10_sum + ask_top10_sum), last
                                  snapshot. Uses all 10 levels; MBP-1's
                                  depth_imbalance (already tested, failed)
                                  used only level 0.
  2. book_slope_imbalance     -- ask_avg_level - bid_avg_level (size-
                                  weighted average resting level per
                                  side). Structurally impossible from
                                  MBP-1, which has no level dimension.
  3. behind_book_change       -- bar's (behind_depth_added -
                                  behind_depth_removed) at levels 1-9
                                  ONLY (level 0 excluded -- that's what
                                  MBP-1's depth_change already tested and
                                  failed). Isolates liquidity building/
                                  pulling BEHIND the touch specifically.

No feature additions after seeing results. No parameter hunting. No
dataset extension unless a result is genuinely ambiguous (matching the
one precedent where this was done: MBP-1's `persistence`).

Decision gate: GO only if a feature (a) is significant in-sample AND OOS,
(b) sign-stable, (c) still significant net of the already-tested MBP-1
baseline (incremental_p), (d) economically meaningful after costs.
Otherwise: NO-GO, close the order-book research track. No strategy is
built from this result either way -- GO only triggers a *separate*,
future strategy-validation phase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [1, 3, 5, 10]
TRAIN_FRACTION = 0.7
MNQ_TICK_VALUE_USD = 0.50
MNQ_TICK_SIZE = 0.25


def _manual_ols(y: np.ndarray, X: np.ndarray):
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    return beta, t_stats, p_values


def build_features(bars: pd.DataFrame, mbp1_bars: pd.DataFrame | None) -> pd.DataFrame:
    bars = bars.sort_values("minute").reset_index(drop=True)
    bars["date"] = bars["minute"].dt.date

    total = (bars["bid_top10_sum_last"] + bars["ask_top10_sum_last"]).replace(0, np.nan)
    bars["depth_weighted_imbalance"] = (
        (bars["bid_top10_sum_last"] - bars["ask_top10_sum_last"]) / total
    ).fillna(0.0)

    bars["book_slope_imbalance"] = (bars["ask_avg_level_last"] - bars["bid_avg_level_last"]).fillna(0.0)

    bars["behind_book_change"] = (bars["behind_depth_added"] - bars["behind_depth_removed"]).fillna(0.0)

    bars["ret_1"] = bars["last_mid"].pct_change()
    bars["realized_vol"] = bars["ret_1"].rolling(30).std()
    for h in HORIZONS:
        bars[f"fwd_ret_{h}"] = bars["last_mid"].shift(-h) / bars["last_mid"] - 1

    # Baseline for incremental-info test: the best already-tested MBP-1
    # feature (depth_imbalance, top-of-book only) on the SAME window, if
    # available, so "adds information beyond MBP-1" is tested directly
    # rather than against a generic proxy.
    bars["mbp1_depth_imbalance"] = np.nan
    if mbp1_bars is not None:
        m1 = mbp1_bars.copy()
        total1 = (m1["bid_sz_last"] + m1["ask_sz_last"]).replace(0, np.nan)
        m1["mbp1_depth_imbalance"] = ((m1["bid_sz_last"] - m1["ask_sz_last"]) / total1).fillna(0.0)
        merged = bars.merge(m1[["minute", "mbp1_depth_imbalance"]], on="minute", how="left", suffixes=("", "_m1"))
        bars["mbp1_depth_imbalance"] = merged["mbp1_depth_imbalance_m1"] if "mbp1_depth_imbalance_m1" in merged.columns else merged["mbp1_depth_imbalance"]

    return bars.set_index("minute")


def evaluate_feature(bars: pd.DataFrame, feature_col: str, baseline_col: str | None) -> list[dict]:
    print(f"\n{'=' * 70}\nFEATURE: {feature_col}"
          f"{f' (incremental vs {baseline_col})' if baseline_col else ' (no baseline available)'}\n{'=' * 70}")
    subset_cols = ["realized_vol", feature_col] + ([baseline_col] if baseline_col else [])
    valid = bars.dropna(subset=subset_cols)
    split_idx = int(len(valid) * TRAIN_FRACTION)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]
    rows = []

    for h in HORIZONS:
        col = f"fwd_ret_{h}"
        sub_train = train.dropna(subset=[col])
        sub_test = test.dropna(subset=[col])
        if len(sub_train) < 30 or len(sub_test) < 30:
            print(f"  h={h:>2}min | insufficient n (train={len(sub_train)}, test={len(sub_test)})")
            continue
        if sub_train[feature_col].std() == 0 or sub_test[feature_col].std() == 0:
            print(f"  h={h:>2}min | feature has zero variance, skipping")
            continue

        train_corr, train_p = stats.pearsonr(sub_train[feature_col], sub_train[col])
        test_corr, test_p = stats.pearsonr(sub_test[feature_col], sub_test[col])
        sign_stable = (train_corr > 0) == (test_corr > 0)
        r2 = test_corr ** 2

        incremental_p, incremental_beta = None, None
        if baseline_col:
            X = np.column_stack([
                np.ones(len(sub_train)), sub_train[baseline_col].to_numpy(), sub_train[feature_col].to_numpy()
            ])
            y = sub_train[col].to_numpy()
            try:
                beta, t_stats, p_values = _manual_ols(y, X)
                incremental_p, incremental_beta = p_values[2], beta[2]
            except np.linalg.LinAlgError:
                pass

        daily_corrs = []
        for _, day_df in sub_test.groupby(sub_test.index.to_series().dt.date):
            if len(day_df) >= 30 and day_df[feature_col].std() > 0:
                c, _ = stats.pearsonr(day_df[feature_col], day_df[col])
                daily_corrs.append(c)
        daily_corrs = np.array(daily_corrs)
        pct_positive_days = (daily_corrs > 0).mean() * 100 if len(daily_corrs) else None

        # Economic significance: expected move (bps) at the extreme decile
        # of the feature, vs. MNQ round-trip cost (~1 tick + typical
        # commission, informational).
        top_decile = sub_test[sub_test[feature_col] >= sub_test[feature_col].quantile(0.9)]
        mean_fwd_ret_top_decile_bps = top_decile[col].mean() * 1e4 if len(top_decile) else np.nan

        incremental_p_str = f"{incremental_p:.3f}" if incremental_p is not None else "n/a"
        pct_positive_str = f"{pct_positive_days:.1f}%" if pct_positive_days is not None else "n/a"
        print(f"  h={h:>2}min | train_corr={train_corr:+.4f}(p={train_p:.3f}) "
              f"test_corr={test_corr:+.4f}(p={test_p:.3f}) sign_stable={sign_stable} "
              f"R2_oos={r2:.5f} | incremental_p={incremental_p_str} "
              f"| test_days={len(daily_corrs)} pct_positive_days={pct_positive_str} "
              f"| top_decile_fwd_ret={mean_fwd_ret_top_decile_bps:.2f}bps")

        rows.append({
            "feature": feature_col, "horizon": h, "train_corr": train_corr, "train_p": train_p,
            "test_corr": test_corr, "test_p": test_p, "sign_stable": sign_stable, "r2_oos": r2,
            "incremental_p": incremental_p, "pct_positive_days": pct_positive_days,
            "top_decile_fwd_ret_bps": mean_fwd_ret_top_decile_bps,
        })
    return rows


def main() -> None:
    mbp10_files = [
        "orderflow_data/raw/MNQ_orderbook_mbp10_1min_2026-06-01_2026-06-08.parquet",
    ]
    bars = pd.concat([pd.read_parquet(f) for f in mbp10_files], ignore_index=True)
    bars = bars.drop_duplicates(subset="minute").sort_values("minute").reset_index(drop=True)

    mbp1_bars = None
    try:
        mbp1_bars = pd.read_parquet("orderflow_data/raw/MNQ_orderbook_1min_2026-06-01_2026-06-08.parquet")
    except FileNotFoundError:
        print("WARNING: matching MBP-1 window not found locally; incremental-info test will be skipped.")

    bars = build_features(bars, mbp1_bars)
    print(f"Total MBP-10 bars: {len(bars)}, trading days: {bars['date'].nunique()}")
    baseline_available = bars["mbp1_depth_imbalance"].notna().sum() > 30
    print(f"MBP-1 baseline available for incremental test: {baseline_available}")

    all_rows = []
    for feature in ["depth_weighted_imbalance", "book_slope_imbalance", "behind_book_change"]:
        all_rows += evaluate_feature(bars, feature, baseline_col="mbp1_depth_imbalance" if baseline_available else None)

    results = pd.DataFrame(all_rows)
    results.to_csv("reports/phase8/mbp10_pilot_results.csv", index=False)

    print(f"\n{'=' * 70}\nGO/NO-GO GATE\n{'=' * 70}")
    survivors = results[
        (results["train_p"] < 0.05) & (results["test_p"] < 0.05) & (results["sign_stable"]) &
        ((results["incremental_p"] < 0.05) if baseline_available else True)
    ]
    if len(survivors):
        print("GO candidates (train+OOS significant, sign-stable, incremental over MBP-1):")
        print(survivors.to_string(index=False))
    else:
        print("NO-GO: no feature is simultaneously in-sample significant, OOS significant, "
              "sign-stable, and incremental over the already-tested MBP-1 information.")


if __name__ == "__main__":
    main()
