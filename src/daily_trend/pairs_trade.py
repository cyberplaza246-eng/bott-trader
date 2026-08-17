"""
MNQ/MES statistical arbitrage (relative value, not directional) -- a
genuinely different signal category from everything else tested in this
project: bets on the ratio between two correlated instruments reverting
locally, not on either instrument's own direction.

Ratio = MNQ_close / MES_close. Rolling 20-day mean/std -> Z-score. The
ratio has a long-run drift (Nasdaq structurally outgrew S&P over
2019-2026), so this deliberately does NOT assume global stationarity --
only that LOCAL deviations from the recent rolling mean tend to revert,
which is testable regardless of the drift.

Entry (next-bar-close fill, same convention as the rest of this project's
daily strategies, and for the same reason -- avoids the intrabar-fill
lookahead bug found in enhanced_breakout elsewhere):
  Z <= -z_threshold -> long MNQ, short MES (Nasdaq cheap relative to S&P)
  Z >= +z_threshold -> short MNQ, long MES
Exit: Z crosses back through 0, or max_hold days.

Position sizing: notional-balanced (not 1:1 contracts) -- MNQ and MES
have different point values and price levels, so equal contract counts
would introduce an unintended directional bias. Contracts are sized so
each leg's dollar notional is as close to equal as whole-contract
rounding allows.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.daily_trend.instruments import REGISTRY


def pairs_trade(df_mnq: pd.DataFrame, df_mes: pd.DataFrame, z_threshold: float = 2.5,
                 lookback: int = 20, max_hold: int = 20) -> list[dict]:
    mnq_spec = REGISTRY["MNQ"]
    mes_spec = REGISTRY["MES"]

    df = pd.DataFrame({
        "mnq_close": df_mnq["close"], "mes_close": df_mes["close"],
    }).dropna()

    ratio = df["mnq_close"] / df["mes_close"]
    roll_mean = ratio.rolling(lookback).mean()
    roll_std = ratio.rolling(lookback).std()
    z = (ratio - roll_mean) / roll_std.replace(0, np.nan)

    trades = []
    position: Optional[dict] = None
    n = len(df)

    for i in range(lookback + 5, n):
        mnq_price = df["mnq_close"].iloc[i]
        mes_price = df["mes_close"].iloc[i]
        zi = z.iloc[i]

        if position is not None:
            bars_held = i - position["entry_idx"]
            prev_z = z.iloc[i - 1]
            crossed_zero = (prev_z < 0 <= zi) or (prev_z > 0 >= zi)
            timed_out = bars_held >= max_hold
            if crossed_zero or timed_out:
                exit_type = "Z_CROSS_ZERO" if crossed_zero else "TIMEOUT"
                mnq_pnl = ((mnq_price - position["mnq_entry"]) / mnq_spec.tick_size
                           * mnq_spec.tick_value_usd * position["mnq_contracts"] * position["mnq_dir"])
                mes_pnl = ((mes_price - position["mes_entry"]) / mes_spec.tick_size
                           * mes_spec.tick_value_usd * position["mes_contracts"] * position["mes_dir"])
                commission = (mnq_spec.commission_rt * position["mnq_contracts"]
                              + mes_spec.commission_rt * position["mes_contracts"])
                pnl = mnq_pnl + mes_pnl - commission
                trades.append({
                    "entry_time": position["entry_time"], "exit_type": exit_type,
                    "direction": "long_mnq_short_mes" if position["mnq_dir"] == 1 else "short_mnq_long_mes",
                    "pnl": pnl,
                })
                position = None

        if position is None and i < n - 1 and not pd.isna(zi):
            direction = 0
            if zi <= -z_threshold:
                direction = 1  # long MNQ, short MES
            elif zi >= z_threshold:
                direction = -1  # short MNQ, long MES

            if direction != 0:
                nxt_mnq = df["mnq_close"].iloc[i + 1]
                nxt_mes = df["mes_close"].iloc[i + 1]
                # Notional-balance contract sizing.
                mnq_notional_per_contract = nxt_mnq * mnq_spec.contract_multiplier
                mes_notional_per_contract = nxt_mes * mes_spec.contract_multiplier
                # Fix MNQ at 1 contract, solve MES contracts to balance notional.
                mnq_contracts = 1
                mes_contracts = max(1, round(mnq_notional_per_contract / mes_notional_per_contract))
                position = {
                    "mnq_entry": nxt_mnq, "mes_entry": nxt_mes,
                    "mnq_dir": direction, "mes_dir": -direction,
                    "mnq_contracts": mnq_contracts, "mes_contracts": mes_contracts,
                    "entry_time": df.index[i + 1], "entry_idx": i + 1,
                }

    return trades
