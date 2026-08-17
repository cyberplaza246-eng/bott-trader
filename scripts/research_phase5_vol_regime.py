#!/usr/bin/env python
"""
Phase 5 -- robustness interrogation of the single Phase 4 candidate
(elevated-ATR-regime forward-return asymmetry). Goal: determine whether
this is a genuine, economically meaningful behavioral effect or a
statistical artifact. Explicitly NOT building a strategy -- no entries,
exits, stops, targets, or optimization anywhere in this file.

Holdout discipline: everything in `discovery_report()` runs ONLY on the
first 80% of each symbol's history (same split point Phase 4 used). The
final 20% is never touched until `locked_holdout_report()`, which applies
ONE pre-registered configuration (90th percentile ATR, 20-bar horizon --
the exact pair flagged in Phase 4, chosen before this script was written,
not selected by looking at which cell in the new threshold/horizon curves
below looks best) exactly once.

Usage:
    python scripts/research_phase5_vol_regime.py discovery
    python scripts/research_phase5_vol_regime.py holdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.instruments.instrument_registry import REGISTRY  # noqa: E402
from src.strategies.indicators import atr as atr_ind, session_vwap  # noqa: E402

SYMBOLS = ["MES", "MNQ", "NQ"]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975]
HORIZONS = [5, 10, 20, 40, 60, 100]
LOCKED_THRESHOLD = 0.90   # pre-registered from Phase 4 -- not re-selected here
LOCKED_HORIZON = 20       # pre-registered from Phase 4 -- not re-selected here
ATR_PCTILE_WINDOW = 500
RTH_START_UTC, RTH_END_UTC = 13.5, 20.0  # approx CME RTH (9:30-16:00 ET), robustness control only


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{symbol}_5m.csv")
    col = "datetime" if "datetime" in df.columns else "date"
    df["datetime"] = pd.to_datetime(df[col], utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr14"] = atr_ind(df["high"], df["low"], df["close"], 14)
    df["atr_pctile"] = df["atr14"].rolling(ATR_PCTILE_WINDOW).apply(
        lambda w: (w.iloc[-1] > w).mean(), raw=False)
    df["rng"] = df["high"] - df["low"]
    df["rng_pctile"] = df["rng"].rolling(20).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False)
    df["ret20"] = df["close"].pct_change(20)          # recent return magnitude/direction, known as of bar t
    df["hour_utc"] = df.index.hour + df.index.minute / 60.0
    df["rth"] = (df["hour_utc"] >= RTH_START_UTC) & (df["hour_utc"] < RTH_END_UTC)
    df["dow"] = df.index.dayofweek
    df["sma200"] = df["close"].rolling(200).mean()
    df["trend_up"] = df["close"] > df["sma200"]
    df["vwap"] = session_vwap(df["high"], df["low"], df["close"], df["volume"])
    df["above_vwap"] = df["close"] > df["vwap"]
    return df


def split_discovery_holdout(df: pd.DataFrame):
    n = len(df)
    split = int(n * 0.8)
    split_date = df.index[split]
    return df[df.index < split_date], df[df.index >= split_date], split_date


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def forward_excursion(df: pd.DataFrame, horizon: int):
    """Max favorable/adverse excursion over the NEXT `horizon` bars after the
    signal bar's close, in both directions (no assumed trade direction)."""
    entry = df["close"]
    fwd_high = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    fwd_low = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    up_excursion = fwd_high / entry - 1.0
    down_excursion = fwd_low / entry - 1.0
    return up_excursion, down_excursion


def directional_decomposition(fwd: pd.Series, baseline_fwd: pd.Series) -> dict:
    fwd = fwd.dropna()
    n = len(fwd)
    if n < 2:
        return {"n": n, "insufficient": True}
    pos = fwd[fwd > 0]
    neg = fwd[fwd <= 0]
    baseline_mean = baseline_fwd.dropna().mean()
    t_stat, p_val = stats.ttest_1samp(fwd.values, baseline_mean)
    return {
        "n": n, "insufficient": n < 30,
        "mean_bps": fwd.mean() * 1e4, "median_bps": fwd.median() * 1e4,
        "baseline_mean_bps": baseline_mean * 1e4,
        "prob_positive": len(pos) / n, "prob_negative": len(neg) / n,
        "mean_positive_bps": (pos.mean() * 1e4) if len(pos) else np.nan,
        "mean_negative_bps": (neg.mean() * 1e4) if len(neg) else np.nan,
        "mean_abs_bps": fwd.abs().mean() * 1e4,
        "p5_bps": fwd.quantile(0.05) * 1e4, "p95_bps": fwd.quantile(0.95) * 1e4,
        "t_stat_vs_baseline": t_stat, "p_value_vs_baseline": p_val,
    }


def non_overlapping_events(qualify_idx: pd.DatetimeIndex, all_index: pd.DatetimeIndex, min_spacing: int) -> pd.DatetimeIndex:
    """Greedy: walk qualifying bars chronologically, keep one, skip forward
    min_spacing bars before the next pick -- guarantees forward-return
    windows used downstream don't overlap, restoring iid-appropriate stats."""
    pos_map = {ts: i for i, ts in enumerate(all_index)}
    qualify_positions = sorted(pos_map[ts] for ts in qualify_idx if ts in pos_map)
    kept = []
    last_kept = -min_spacing - 1
    for p in qualify_positions:
        if p - last_kept >= min_spacing:
            kept.append(p)
            last_kept = p
    return all_index[kept]


def economic_significance(fwd: pd.Series, up_exc: pd.Series, down_exc: pd.Series, symbol: str) -> dict:
    spec = REGISTRY[symbol]
    fwd = fwd.dropna()
    n = len(fwd)
    if n == 0:
        return {"n": 0}
    mean_price = 20000 if symbol in ("MNQ", "NQ") else 6500  # rough scale for bps->$ sanity, informational only
    round_trip_cost_usd = spec.commission_rt + spec.tick_size * spec.tick_value_usd  # 1-tick slippage convention
    # cost expressed in bps of a representative recent price level (informational)
    return {
        "n": n,
        "mean_fwd_ret_bps": fwd.mean() * 1e4,
        "mean_abs_fwd_ret_bps": fwd.abs().mean() * 1e4,
        "mean_up_excursion_bps": up_exc.dropna().mean() * 1e4 if len(up_exc.dropna()) else np.nan,
        "mean_down_excursion_bps": down_exc.dropna().mean() * 1e4 if len(down_exc.dropna()) else np.nan,
        "round_trip_cost_usd_per_contract": round_trip_cost_usd,
        "tick_value_usd": spec.tick_value_usd,
    }


def run_curve(df: pd.DataFrame, symbol: str, thresholds, horizons, label: str) -> pd.DataFrame:
    rows = []
    for h in horizons:
        fwd_all = forward_return(df["close"], h)
        for thr in thresholds:
            mask = df["atr_pctile"] >= thr
            idx = df.index[mask.fillna(False)]
            fwd = fwd_all.loc[fwd_all.index.intersection(idx)]
            dd = directional_decomposition(fwd, fwd_all)
            dd.update({"symbol": symbol, "split": label, "threshold_pctile": thr, "horizon_bars": h})
            rows.append(dd)
    return pd.DataFrame(rows)


def run_nonoverlapping(df: pd.DataFrame, symbol: str, threshold: float, horizon: int, label: str) -> dict:
    fwd_all = forward_return(df["close"], horizon)
    mask = df["atr_pctile"] >= threshold
    qualify_idx = df.index[mask.fillna(False)]
    events = non_overlapping_events(qualify_idx, df.index, min_spacing=horizon)
    fwd = fwd_all.loc[fwd_all.index.intersection(events)]
    dd = directional_decomposition(fwd, fwd_all)
    dd.update({"symbol": symbol, "split": label, "threshold_pctile": threshold,
               "horizon_bars": horizon, "method": "non_overlapping_events"})
    return dd


def run_conditional_controls(df: pd.DataFrame, symbol: str, threshold: float, horizon: int, label: str) -> pd.DataFrame:
    fwd_all = forward_return(df["close"], horizon)
    base_mask = df["atr_pctile"] >= threshold
    rows = []

    controls = {
        "rth": [("RTH", df["rth"]), ("overnight", ~df["rth"])],
        "recent_return_direction": [("prior20_up", df["ret20"] > 0), ("prior20_down", df["ret20"] <= 0)],
        "recent_return_magnitude": [
            ("prior20_|ret|_top_third", df["ret20"].abs() >= df["ret20"].abs().quantile(0.667)),
            ("prior20_|ret|_bottom_third", df["ret20"].abs() <= df["ret20"].abs().quantile(0.333)),
        ],
        "range_expansion": [("range_expanding", df["rng_pctile"] >= 0.667), ("range_contracting", df["rng_pctile"] <= 0.333)],
        "vwap_location": [("above_vwap", df["above_vwap"]), ("below_vwap", ~df["above_vwap"])],
        "day_of_week": [(f"dow_{d}", df["dow"] == d) for d in range(5)],
        "trend_state": [("above_sma200", df["trend_up"]), ("below_sma200", ~df["trend_up"])],
    }

    for control_name, variants in controls.items():
        for variant_label, variant_mask in variants:
            mask = base_mask & variant_mask
            idx = df.index[mask.fillna(False)]
            fwd = fwd_all.loc[fwd_all.index.intersection(idx)]
            dd = directional_decomposition(fwd, fwd_all)
            dd.update({"symbol": symbol, "split": label, "control": control_name,
                       "variant": variant_label, "threshold_pctile": threshold, "horizon_bars": horizon})
            rows.append(dd)
    return pd.DataFrame(rows)


def discovery_report():
    Path("reports/phase5").mkdir(parents=True, exist_ok=True)
    all_curve, all_nonoverlap, all_controls, all_econ = [], [], [], []

    for symbol in SYMBOLS:
        print(f"\n{'=' * 78}\n{symbol} -- DISCOVERY ONLY (holdout untouched)\n{'=' * 78}")
        df = add_features(load(symbol))
        disc, hold, split_date = split_discovery_holdout(df)
        print(f"  discovery: {len(disc)} bars, holdout: {len(hold)} bars, split={split_date}")

        # 1+2+3: directional decomposition across full threshold x horizon curve
        curve = run_curve(disc, symbol, THRESHOLDS, HORIZONS, "discovery")
        all_curve.append(curve)

        # 4: non-overlapping/event-based version at the locked config AND a
        # couple of neighboring cells, to see if the overlap-corrected
        # picture agrees with the naive curve at the point of interest.
        for thr in [0.85, 0.90, 0.95]:
            for h in [10, 20, 40]:
                all_nonoverlap.append(run_nonoverlapping(disc, symbol, thr, h, "discovery"))

        # 5: conditional controls at the locked config
        controls = run_conditional_controls(disc, symbol, LOCKED_THRESHOLD, LOCKED_HORIZON, "discovery")
        all_controls.append(controls)

        # 8: economic significance at the locked config
        fwd_locked = forward_return(disc["close"], LOCKED_HORIZON)
        mask = disc["atr_pctile"] >= LOCKED_THRESHOLD
        idx = disc.index[mask.fillna(False)]
        up_exc, down_exc = forward_excursion(disc, LOCKED_HORIZON)
        econ = economic_significance(fwd_locked.loc[fwd_locked.index.intersection(idx)],
                                      up_exc.loc[up_exc.index.intersection(idx)],
                                      down_exc.loc[down_exc.index.intersection(idx)], symbol)
        econ["symbol"] = symbol
        all_econ.append(econ)

    curve_df = pd.concat(all_curve, ignore_index=True)
    nonoverlap_df = pd.DataFrame(all_nonoverlap)
    controls_df = pd.concat(all_controls, ignore_index=True)
    econ_df = pd.DataFrame(all_econ)

    curve_df.to_csv("reports/phase5/discovery_threshold_horizon_curve.csv", index=False)
    nonoverlap_df.to_csv("reports/phase5/discovery_nonoverlapping.csv", index=False)
    controls_df.to_csv("reports/phase5/discovery_conditional_controls.csv", index=False)
    econ_df.to_csv("reports/phase5/discovery_economic_significance.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_rows", None):
        print(f"\n{'-' * 78}\nFULL THRESHOLD x HORIZON CURVE (t_stat_vs_baseline), discovery only\n{'-' * 78}")
        pivot = curve_df.pivot_table(index=["symbol", "threshold_pctile"], columns="horizon_bars", values="t_stat_vs_baseline")
        print(pivot.round(2).to_string())

        print(f"\n{'-' * 78}\nDIRECTIONAL DECOMPOSITION at locked config (thr={LOCKED_THRESHOLD}, h={LOCKED_HORIZON})\n{'-' * 78}")
        locked = curve_df[(curve_df.threshold_pctile == LOCKED_THRESHOLD) & (curve_df.horizon_bars == LOCKED_HORIZON)]
        print(locked[["symbol", "n", "mean_bps", "median_bps", "baseline_mean_bps", "prob_positive",
                       "mean_positive_bps", "mean_negative_bps", "mean_abs_bps", "p5_bps", "p95_bps",
                       "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))

        print(f"\n{'-' * 78}\nNON-OVERLAPPING (event-based) RESULTS, discovery only\n{'-' * 78}")
        print(nonoverlap_df[["symbol", "threshold_pctile", "horizon_bars", "n", "mean_bps",
                              "baseline_mean_bps", "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))

        print(f"\n{'-' * 78}\nCONDITIONAL CONTROLS at locked config, discovery only\n{'-' * 78}")
        print(controls_df[["symbol", "control", "variant", "n", "mean_bps", "baseline_mean_bps",
                            "t_stat_vs_baseline", "p_value_vs_baseline"]].to_string(index=False))

        print(f"\n{'-' * 78}\nECONOMIC SIGNIFICANCE at locked config, discovery only\n{'-' * 78}")
        print(econ_df.to_string(index=False))

    print(f"\n{'=' * 78}\nLOCKED FOR HOLDOUT: threshold={LOCKED_THRESHOLD} percentile, horizon={LOCKED_HORIZON} bars,\n"
          f"method=non_overlapping_events. This choice was fixed before writing this script\n"
          f"(carried over from the Phase 4 lead) and is NOT re-selected from the curves above.\n{'=' * 78}")


def locked_holdout_report():
    """Run exactly once, using ONLY the pre-registered config."""
    Path("reports/phase5").mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in SYMBOLS:
        df = add_features(load(symbol))
        disc, hold, split_date = split_discovery_holdout(df)
        print(f"\n{symbol}: holdout = {len(hold)} bars from {split_date} onward")

        dd = run_nonoverlapping(hold, symbol, LOCKED_THRESHOLD, LOCKED_HORIZON, "holdout")
        up_exc, down_exc = forward_excursion(hold, LOCKED_HORIZON)
        mask = hold["atr_pctile"] >= LOCKED_THRESHOLD
        events = non_overlapping_events(hold.index[mask.fillna(False)], hold.index, min_spacing=LOCKED_HORIZON)
        econ = economic_significance(
            forward_return(hold["close"], LOCKED_HORIZON).loc[lambda s: s.index.intersection(events)],
            up_exc.loc[up_exc.index.intersection(events)], down_exc.loc[down_exc.index.intersection(events)], symbol)
        dd.update({f"econ_{k}": v for k, v in econ.items() if k != "n"})
        rows.append(dd)

    out = pd.DataFrame(rows)
    out.to_csv("reports/phase5/LOCKED_holdout_result.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(f"\n{'=' * 78}\nLOCKED HOLDOUT RESULT (threshold={LOCKED_THRESHOLD}, horizon={LOCKED_HORIZON}, non-overlapping events)\n{'=' * 78}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    if mode == "discovery":
        discovery_report()
    elif mode == "holdout":
        locked_holdout_report()
    else:
        print("usage: research_phase5_vol_regime.py [discovery|holdout]")
