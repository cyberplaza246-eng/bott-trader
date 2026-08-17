#!/usr/bin/env python3
"""
Stage-1 predictive-information test, round 2: does the order-flow DATA
SOURCE contain any incremental predictive information, tested via 4
mechanically distinct features (not just re-parameterizing the same one)?
No new data purchased -- reuses the NQ June 2026 trades pilot.

Features tested, each representing a different microstructure mechanism:
  1. signed_volume_imbalance  -- baseline (already tested; rerun here for
     a consistent comparison table)
  2. block_trade_imbalance    -- imbalance restricted to large ("block-
     like", size >= LARGE_TRADE_SIZE) trades only
  3. flow_acceleration        -- bar-over-bar change in imbalance (is
     pressure building or fading, not just its current level)
  4. flow_persistence         -- rolling mean imbalance over the last 5
     bars (sustained vs one-off pressure)
  5. absorption                -- signed volume per unit of price move
     (high volume + small move = absorption/exhaustion hypothesis)

For each feature x each horizon, reports:
  - train and OOS (test) correlation + p-value
  - sign stability (train vs test same direction?)
  - effect size (R^2)
  - incremental info: partial regression coefficient on the feature AFTER
    controlling for signed_volume_imbalance (does it add anything beyond
    the baseline already tested?)
  - per-day correlation distribution (does the aggregate hide 1-2 unusual
    sessions?)

Decision gate (from the locked research process): fails OOS -> discard.
Survives OOS but R^2 is microscopic -> discard. Survives with a
meaningful effect -> worth testing on another period before spending more.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

BAR_FREQ = "1min"
HORIZONS = [1, 3, 5, 10]
TRAIN_FRACTION = 0.7
LARGE_TRADE_SIZE = 5


def build_bars(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades[trades["side"].isin(["A", "B"])].copy()
    trades["signed_volume"] = np.where(trades["side"] == "A", trades["size"], -trades["size"])
    trades["is_large"] = trades["size"] >= LARGE_TRADE_SIZE
    trades["large_signed_volume"] = np.where(trades["is_large"], trades["signed_volume"], 0)
    trades["large_size"] = np.where(trades["is_large"], trades["size"], 0)
    trades["ts"] = pd.to_datetime(trades["ts_event"], utc=True)
    trades = trades.set_index("ts")

    bars = trades.resample(BAR_FREQ).agg(
        close=("price", "last"),
        volume=("size", "sum"),
        buy_volume=("signed_volume", lambda s: s[s > 0].sum()),
        sell_volume=("signed_volume", lambda s: -s[s < 0].sum()),
        large_buy_volume=("large_signed_volume", lambda s: s[s > 0].sum()),
        large_sell_volume=("large_signed_volume", lambda s: -s[s < 0].sum()),
        large_volume=("large_size", "sum"),
        n_trades=("price", "count"),
    )
    bars = bars.dropna(subset=["close"]).ffill()
    bars["date"] = bars.index.date

    bars["signed_volume_imbalance"] = (
        (bars["buy_volume"] - bars["sell_volume"]) / bars["volume"].replace(0, np.nan)
    ).fillna(0.0)

    bars["block_trade_imbalance"] = (
        (bars["large_buy_volume"] - bars["large_sell_volume"]) / bars["large_volume"].replace(0, np.nan)
    ).fillna(0.0)

    bars["flow_acceleration"] = bars["signed_volume_imbalance"].diff().fillna(0.0)
    bars["flow_persistence"] = bars["signed_volume_imbalance"].rolling(5).mean().fillna(0.0)

    bar_return = bars["close"].pct_change().fillna(0.0)
    net_signed_volume = bars["buy_volume"] - bars["sell_volume"]
    # High when a lot of signed volume moved price very little (potential
    # absorption/exhaustion); denominator uses abs(return)*price so it's
    # scaled in price-points moved rather than raw fractional return.
    bars["absorption"] = net_signed_volume / (bar_return.abs() * bars["close"] + 1)

    return bars


def add_forward_returns_and_vol(bars: pd.DataFrame) -> pd.DataFrame:
    bars["ret_1"] = bars["close"].pct_change()
    bars["realized_vol"] = bars["ret_1"].rolling(30).std()
    for h in HORIZONS:
        fwd_ret = bars["close"].shift(-h) / bars["close"] - 1
        bars[f"fwd_ret_{h}"] = fwd_ret
    return bars


def _manual_ols(y: np.ndarray, X: np.ndarray):
    """OLS with an intercept column already included in X. Returns coefs, t-stats, p-values."""
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

        train_corr, train_p = stats.pearsonr(sub_train[feature_col], sub_train[col])
        test_corr, test_p = stats.pearsonr(sub_test[feature_col], sub_test[col])
        sign_stable = (train_corr > 0) == (test_corr > 0)
        r2 = test_corr ** 2

        # Incremental info: does feature_col add anything beyond baseline_col?
        # Full model: fwd_ret ~ intercept + baseline + feature. Compare feature's
        # own t-stat/p-value in the presence of baseline (on TRAIN, to avoid
        # peeking at test twice for model selection).
        incremental_p, incremental_beta = None, None
        if baseline_col:
            X = np.column_stack([
                np.ones(len(sub_train)), sub_train[baseline_col].to_numpy(), sub_train[feature_col].to_numpy()
            ])
            y = sub_train[col].to_numpy()
            try:
                beta, t_stats, p_values = _manual_ols(y, X)
                incremental_p = p_values[2]
                incremental_beta = beta[2]
            except np.linalg.LinAlgError:
                pass

        # Per-day breakdown: correlation computed separately for each trading
        # day in the TEST period, so one unusual session can't hide inside a
        # single aggregate number.
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
    trades = pd.read_parquet("orderflow_data/raw/NQ_trades_2026-06-01_2026-07-01.parquet")
    bars = build_bars(trades)
    bars = add_forward_returns_and_vol(bars)

    print(f"Total bars: {len(bars)}, trading days: {bars['date'].nunique()}")

    evaluate_feature(bars, "signed_volume_imbalance", baseline_col=None)
    for feature in ["block_trade_imbalance", "flow_acceleration", "flow_persistence", "absorption"]:
        evaluate_feature(bars, feature, baseline_col="signed_volume_imbalance")


if __name__ == "__main__":
    main()
