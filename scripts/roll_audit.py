#!/usr/bin/env python3
"""
Roll audit: for every instrument-change in the continuous daily series,
determine whether the price jump is explained by the actual contemporaneous
old-vs-new contract price difference, or looks like an artificial splice.

Method (per the locked research sequence): for each roll, compute the
SAME-DAY spread between the old and new contract using individual-contract
data (old contract's close on roll date vs new contract's close on roll
date, when the old contract still traded that day; otherwise both contracts'
closes on the day before roll).

IMPORTANT: this same-day spread is evaluated on ITS OWN merits (is it small,
as a legitimate calendar-spread/cost-of-carry difference should be), NOT by
comparing it to the raw continuous-series jump. An earlier version of this
script made that comparison and it was wrong: when the old contract still
traded on roll day, "continuous jump minus same-day spread" reduces almost
exactly to the old contract's own 1-day price move -- which is normal daily
volatility, not evidence of anything. Flagging on that basis mostly just
flagged ordinary market moves as "suspicious." The real question is whether
the roll ITSELF introduces a large price gap, independent of whatever the
market did that day.

Never modifies the raw data (Version A). Writes an audit report only.

Usage:
    python scripts/roll_audit.py --symbols ES NQ MES MNQ
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.daily_trend.instruments import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "daily_trend_data" / "raw"
REPORT_DIR = ROOT / "daily_trend_data" / "audit"

SPREAD_FLAG_PCT = 2.0  # flag same-day roll spread if it exceeds this % of price -- generous,
                        # since genuine backwardation/contango can be large in volatile periods


def _outright_pattern(root: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(root)}[FGHJKMNQUVXZ]\d{{1,2}}$")


def audit_symbol(symbol: str) -> dict:
    spec = REGISTRY[symbol]
    root = spec.databento_continuous_symbol.split(".")[0]

    cont_path = RAW_DIR / f"{symbol}_continuous_1d.csv"
    contracts_path = RAW_DIR / f"{symbol}_contracts_1d.csv"
    cont = pd.read_csv(cont_path, parse_dates=["date"])
    contracts = pd.read_csv(contracts_path, parse_dates=["date"])

    pattern = _outright_pattern(root)
    contracts = contracts[contracts["symbol"].str.match(pattern)].copy()

    id_to_symbol = contracts.drop_duplicates("instrument_id").set_index("instrument_id")["symbol"].to_dict()
    close_lookup = contracts.set_index(["instrument_id", "date"])["close"].to_dict()

    cont = cont.sort_values("date").reset_index(drop=True)
    cont["prev_instrument_id"] = cont["instrument_id"].shift(1)
    cont["prev_close"] = cont["close"].shift(1)
    cont["prev_date"] = cont["date"].shift(1)

    roll_mask = (cont["instrument_id"] != cont["prev_instrument_id"]) & cont["prev_instrument_id"].notna()
    rolls = cont[roll_mask]

    events = []
    for _, row in rolls.iterrows():
        old_id, new_id = int(row["prev_instrument_id"]), int(row["instrument_id"])
        old_symbol = id_to_symbol.get(old_id, f"id:{old_id}")
        new_symbol = id_to_symbol.get(new_id, f"id:{new_id}")
        roll_date, prev_date = row["date"], row["prev_date"]
        continuous_old_close, continuous_new_close = row["prev_close"], row["close"]
        abs_diff = continuous_new_close - continuous_old_close
        pct_diff = (abs_diff / continuous_old_close * 100) if continuous_old_close else None

        # Explicit two-component decomposition of the continuous-series jump:
        #   continuous_diff = roll_spread (old vs new, SAME session)
        #                    + market_move (old contract's own return that day)
        # This is an exact identity when the old contract traded on roll day
        # (both terms come from consistent data), and it's the crux of the
        # distinction the audit needs to make: a large continuous jump can be
        # entirely market_move (real event, roll mechanics fine) or partly/
        # mostly roll_spread (the roll itself introduced a large gap).
        roll_spread = None
        market_move = None
        comparison_basis = None
        reference_price = None
        old_close_on_rolldate = close_lookup.get((old_id, roll_date))
        if old_close_on_rolldate is not None:
            roll_spread = continuous_new_close - old_close_on_rolldate
            market_move = old_close_on_rolldate - continuous_old_close
            comparison_basis = "old_vs_new_close_on_roll_date"
            reference_price = old_close_on_rolldate
        else:
            new_close_on_prevdate = close_lookup.get((new_id, prev_date))
            if new_close_on_prevdate is not None:
                roll_spread = new_close_on_prevdate - continuous_old_close
                market_move = None  # can't isolate same-day old-contract move with this basis
                comparison_basis = "old_vs_new_close_on_prev_date"
                reference_price = continuous_old_close

        spread_pct = abs(roll_spread) / reference_price * 100 if roll_spread is not None and reference_price else None
        consistency_check_ok = None
        if roll_spread is not None and market_move is not None:
            consistency_check_ok = abs((roll_spread + market_move) - abs_diff) < 1e-6

        events.append({
            "symbol": symbol, "old_contract": old_symbol, "new_contract": new_symbol,
            "rollover_date": str(roll_date.date()),
            "old_settlement_close": round(float(continuous_old_close), 6),
            "new_contract_price": round(float(continuous_new_close), 6),
            "continuous_jump_abs": round(float(abs_diff), 6),
            "continuous_jump_pct": round(float(pct_diff), 4) if pct_diff is not None else None,
            "roll_spread_same_session": round(float(roll_spread), 6) if roll_spread is not None else None,
            "roll_spread_pct": round(float(spread_pct), 4) if spread_pct is not None else None,
            "market_move_component": round(float(market_move), 6) if market_move is not None else None,
            "decomposition_consistency_check_ok": consistency_check_ok,
            "comparison_basis": comparison_basis,
        })

    # Per-instrument outlier test: is this roll's spread unusual relative to
    # THIS instrument's own history of roll spreads (not a fixed global %).
    valid_spreads = [e["roll_spread_pct"] for e in events if e["roll_spread_pct"] is not None]
    if valid_spreads:
        s = pd.Series(valid_spreads)
        median, mad = s.median(), (s - s.median()).abs().median()
        robust_std = mad * 1.4826 if mad > 0 else s.std()
    else:
        median, robust_std = 0.0, 0.0

    for e in events:
        if e["roll_spread_pct"] is None:
            e["verdict"] = "insufficient_data_to_verify"
            continue
        z = (e["roll_spread_pct"] - median) / robust_std if robust_std else 0.0
        e["roll_spread_zscore_vs_own_history"] = round(float(z), 2)
        is_outlier = z > 5 or e["roll_spread_pct"] > SPREAD_FLAG_PCT
        e["verdict"] = "LARGE_ROLL_SPREAD_NEEDS_REVIEW" if is_outlier else "explained_by_roll"

    n_total = len(events)
    n_explained = sum(1 for e in events if e["verdict"] == "explained_by_roll")
    n_suspect = sum(1 for e in events if e["verdict"] == "LARGE_ROLL_SPREAD_NEEDS_REVIEW")
    n_insufficient = sum(1 for e in events if e["verdict"] == "insufficient_data_to_verify")

    return {
        "symbol": symbol, "total_rolls": n_total,
        "explained_by_roll": n_explained, "large_roll_spread_needs_review": n_suspect,
        "insufficient_data_to_verify": n_insufficient,
        "instrument_roll_spread_median_pct": round(float(median), 4),
        "instrument_roll_spread_robust_std_pct": round(float(robust_std), 4),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["ES", "NQ", "MES", "MNQ"])
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overall_summary = []

    for symbol in args.symbols:
        result = audit_symbol(symbol)
        summary = {k: v for k, v in result.items() if k != "events"}
        overall_summary.append(summary)
        print(json.dumps(summary, indent=2))

        suspects = [e for e in result["events"] if e["verdict"] == "LARGE_ROLL_SPREAD_NEEDS_REVIEW"]
        if suspects:
            print(f"  Large roll spreads for {symbol} (needs review):")
            for e in sorted(suspects, key=lambda x: abs(x["roll_spread_pct"] or 0), reverse=True)[:10]:
                print(f"    {e['rollover_date']}: {e['old_contract']}->{e['new_contract']} "
                      f"roll_spread={e['roll_spread_same_session']} ({e['roll_spread_pct']}%, "
                      f"z={e.get('roll_spread_zscore_vs_own_history')}) "
                      f"market_move={e['market_move_component']} "
                      f"[continuous jump was {e['continuous_jump_abs']} ({e['continuous_jump_pct']}%)]")

        out_path = REPORT_DIR / f"{symbol}_roll_audit.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Wrote {out_path}\n")

    (REPORT_DIR / "summary.json").write_text(json.dumps(overall_summary, indent=2))


if __name__ == "__main__":
    main()
