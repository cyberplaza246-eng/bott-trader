#!/usr/bin/env python3
"""
Stage-1 predictive-information test, round 2, on the SAME already-purchased
MNQ MBP-1 data (2026-06-01 to 2026-06-15) used by orderbook_feature_battery.py.
That round tested 5 features (depth_imbalance, depth_change,
order_book_pressure, persistence, trade_flow_imbalance) -- all failed OOS.

This round tests exactly 2 NEW features, built from raw columns the prior
round never touched (n_quotes, depth_added, depth_removed) -- not a
recombination of the already-failed features:

  1. quote_intensity  -- n_quotes z-scored against its own rolling mean/std.
                          Measures book-update frequency/urgency in the bar,
                          categorically different from a resting-size ratio.
  2. depth_churn_ratio -- depth_added / (depth_added + depth_removed).
                          Measures whether liquidity is net being built or
                          net being pulled over the bar -- a FLOW measure,
                          distinct from depth_imbalance's END-OF-BAR
                          snapshot ratio.

Same decision gate as every prior pilot: fails OOS -> discard, no further
feature invention on this dataset after this round, regardless of result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [1, 3, 5, 10]
TRAIN_FRACTION = 0.7


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


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.sort_values("minute").reset_index(drop=True)
    bars["date"] = bars["minute"].dt.date

    q_roll_mean = bars["n_quotes"].rolling(30).mean()
    q_roll_std = bars["n_quotes"].rolling(30).std().replace(0, np.nan)
    bars["quote_intensity"] = ((bars["n_quotes"] - q_roll_mean) / q_roll_std).fillna(0.0)

    total_churn = (bars["depth_added"] + bars["depth_removed"]).replace(0, np.nan)
    bars["depth_churn_ratio"] = (bars["depth_added"] / total_churn).fillna(0.5) - 0.5  # centered at 0

    # baseline for incremental-info test, same as round 1
    total_trade_vol = (bars["buy_trade_vol"] + bars["sell_trade_vol"]).replace(0, np.nan)
    bars["trade_flow_imbalance"] = (
        (bars["buy_trade_vol"] - bars["sell_trade_vol"]) / total_trade_vol
    ).fillna(0.0)

    bars["ret_1"] = bars["last_mid"].pct_change()
    bars["realized_vol"] = bars["ret_1"].rolling(30).std()
    for h in HORIZONS:
        bars[f"fwd_ret_{h}"] = bars["last_mid"].shift(-h) / bars["last_mid"] - 1

    return bars.set_index("minute")


def evaluate_feature(bars: pd.DataFrame, feature_col: str, baseline_col: str | None) -> None:
    print(f"\n{'=' * 70}\nFEATURE: {feature_col}"
          f"{f' (incremental vs {baseline_col})' if baseline_col else ' (baseline itself)'}\n{'=' * 70}")
    subset_cols = ["realized_vol", feature_col] + ([baseline_col] if baseline_col else [])
    valid = bars.dropna(subset=subset_cols)
    split_idx = int(len(valid) * TRAIN_FRACTION)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]

    for h in HORIZONS:
        col = f"fwd_ret_{h}"
        sub_train = train.dropna(subset=[col])
        sub_test = test.dropna(subset=[col])
        if len(sub_train) < 30 or len(sub_test) < 30:
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
        for _, day_df in sub_test.groupby("date"):
            if len(day_df) >= 30 and day_df[feature_col].std() > 0:
                c, _ = stats.pearsonr(day_df[feature_col], day_df[col])
                daily_corrs.append(c)
        daily_corrs = np.array(daily_corrs)
        pct_positive_days = (daily_corrs > 0).mean() * 100 if len(daily_corrs) else None

        incremental_p_str = f"{incremental_p:.3f}" if incremental_p is not None else "n/a"
        pct_positive_str = f"{pct_positive_days:.1f}%" if pct_positive_days is not None else "n/a"
        print(f"  h={h:>2}min | train_corr={train_corr:+.4f}(p={train_p:.3f}) "
              f"test_corr={test_corr:+.4f}(p={test_p:.3f}) sign_stable={sign_stable} "
              f"R2_oos={r2:.5f} | incremental_p={incremental_p_str} "
              f"| test_days={len(daily_corrs)} pct_positive_days={pct_positive_str}")


def main() -> None:
    files = [
        "orderflow_data/raw/MNQ_orderbook_1min_2026-06-01_2026-06-08.parquet",
        "orderflow_data/raw/MNQ_orderbook_1min_2026-06-08_2026-06-15.parquet",
    ]
    bars = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    bars = bars.drop_duplicates(subset="minute").sort_values("minute").reset_index(drop=True)
    bars = build_features(bars)
    print(f"Total bars: {len(bars)}, trading days: {bars['date'].nunique()}")

    for feature in ["quote_intensity", "depth_churn_ratio"]:
        evaluate_feature(bars, feature, baseline_col="trade_flow_imbalance")


if __name__ == "__main__":
    main()
