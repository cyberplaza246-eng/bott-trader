#!/usr/bin/env python3
"""
Stage-1 predictive-information test for MNQ ORDER-BOOK data (not trade
flow -- that was already tested on NQ and failed). Same rigorous framework
as scripts/orderflow_feature_battery.py: train/OOS split, sign stability,
R^2, incremental-information regression, per-day breakdown.

Features tested (from bars in MNQ_orderbook_1min_*.parquet):
  1. depth_imbalance     -- (bid_sz - ask_sz) / (bid_sz + ask_sz), last
                            book snapshot in each bar.
  2. depth_change        -- bar-over-bar change in bid depth minus change
                            in ask depth (is liquidity building on one
                            side relative to the other?).
  3. order_book_pressure -- depth_imbalance combined with recent (5-bar)
                            price momentum (does book imbalance aligned
                            with the recent trend add anything?).
  4. persistence         -- 5-bar rolling mean of depth_imbalance.
  5. trade_flow_imbalance -- baseline for the incremental-info test: is
                            the order-book information adding anything
                            beyond simple trade-side signed volume
                            (computed from this same MNQ dataset's
                            buy_trade_vol/sell_trade_vol)?

Decision gate: same as the NQ trade-flow test. Fails OOS -> discard.
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

    total_depth = (bars["bid_sz_last"] + bars["ask_sz_last"]).replace(0, np.nan)
    bars["depth_imbalance"] = ((bars["bid_sz_last"] - bars["ask_sz_last"]) / total_depth).fillna(0.0)

    bid_change = bars["bid_sz_last"].diff()
    ask_change = bars["ask_sz_last"].diff()
    bars["depth_change"] = (bid_change - ask_change).fillna(0.0)

    recent_return_sign = np.sign(bars["last_mid"].pct_change(5)).fillna(0.0)
    bars["order_book_pressure"] = bars["depth_imbalance"] * recent_return_sign

    bars["persistence"] = bars["depth_imbalance"].rolling(5).mean().fillna(0.0)

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

    evaluate_feature(bars, "trade_flow_imbalance", baseline_col=None)
    for feature in ["depth_imbalance", "depth_change", "order_book_pressure", "persistence"]:
        evaluate_feature(bars, feature, baseline_col="trade_flow_imbalance")


if __name__ == "__main__":
    main()
