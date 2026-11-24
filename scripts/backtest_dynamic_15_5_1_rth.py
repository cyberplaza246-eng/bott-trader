#!/usr/bin/env python3
"""
Backtest: dynamic_15_5_1_rth vs locked asymmetric (same data, same costs).

Dynamic spec (user-approved):
  - 15m strict structure+EMA10/20+VWAP (completed bars)
  - 5m EMA10/20 confirm (completed)
  - 1m trigger Pattern A (quiet) and/or B (pullback)
  - 1m EMA8/21 agrees with side
  - RTH 9:35-15:50 continuous scan — NO time-direction lock
  - One position; reversal only on opposite full trigger (close then enter)
  - Stop: 1m ATR14 x1.5 clip [6,25]; TP: 1R long / 2R short
  - Noon+ optional sep/ATR >= 0.45 quality filter (not directional)

Compare to locked: morning long recipe + afternoon short recipe (prior PASS).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_level_strategy_search as lvl
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt
from src.strategy.mnq_15m_ema_eod import (
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    ONE_MIN_MAX_BODY,
    SEP_MIN,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    window_requires_quality,
)

ET = resp.ET
WARMUP = 80
PT, COMM = official.PT, official.COMM
MORNING_END = 12 * 60

LOCKED_SHORT = {
    "stack": "15_5",
    "loop": "window_loop",
    "entry": "quiet",
    "win": "w3_pm",
    "stop": "atr1_1.5_6_25",
    "tp": "2R",
    "side": "short",
    "sep": False,
    "cooldown": 0,
    "max_day": 4,
    "max_lots": 1,
    "morning": False,
    "vwap": "with_side",
    "sep_short_min": None,
}


@dataclass
class SimStats:
    signals_stack: int = 0
    signals_pattern: int = 0
    filtered_sep: int = 0
    filtered_max_day: int = 0
    filtered_same_bar: int = 0
    filtered_entry_block: int = 0
    entries: int = 0
    reversals: int = 0
    exits_sl: int = 0
    exits_tp: int = 0
    exits_eod: int = 0
    exits_reversal: int = 0


def prepare_arrays(df: pd.DataFrame) -> Dict[str, Any]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    tmp = work.set_index("datetime")
    if tmp.index.tz is None:
        tmp.index = tmp.index.tz_localize("UTC")
    side15, _, side5, struct, ema15, vwap_side = lvl.precompute_bias_arrays(tmp)
    t1 = base.trend_1m_completed(work)
    atr1 = resp.atr1_on_1m(work).to_numpy(dtype=float)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    return {
        "dt": dt,
        "side15": side15,
        "side5": side5,
        "t1": t1,
        "struct": struct,
        "ema15": ema15,
        "vwap_side": vwap_side,
        "atr1": atr1,
        "sep": sep,
        "op": work["open"].to_numpy(dtype=float),
        "hi": work["high"].to_numpy(dtype=float),
        "lo": work["low"].to_numpy(dtype=float),
        "cl": work["close"].to_numpy(dtype=float),
        "mins": (et.hour * 60 + et.minute).to_numpy(),
        "hours": et.hour.to_numpy(),
        "dow": et.weekday.to_numpy(),
        "dates": np.array(et.date),
    }


def stack_side(i: int, a: Dict) -> int:
    s15 = int(a["side15"][i])
    s5 = int(a["side5"][i])
    if s15 == 0 or s15 != s5:
        return 0
    return s15


def pattern_a(side: int, i: int, a: Dict) -> bool:
    if int(a["t1"][i]) != side:
        return False
    return quiet_1m_ok(side, a["op"][i], a["cl"][i], max_body=ONE_MIN_MAX_BODY)


def pattern_b(side: int, i: int, a: Dict) -> bool:
    if not pattern_a(side, i, a):
        return False
    if i < 2:
        return False
    prev = float(a["cl"][i - 1]) - float(a["cl"][i - 2])
    if side > 0:
        return prev <= 0
    return prev >= 0


def trigger(side: int, i: int, a: Dict, *, pattern: str) -> bool:
    if pattern == "A":
        return pattern_a(side, i, a)
    if pattern == "AB":
        return pattern_a(side, i, a) or pattern_b(side, i, a)
    raise ValueError(pattern)


def tp_for_side(side: int, entry: float, stop_dist: float) -> float:
    r = 2.0 if side < 0 else 1.0
    dist = stop_dist * r
    return entry + dist if side > 0 else entry - dist


def simulate_dynamic(
    a: Dict[str, Any],
    *,
    pattern: str = "A",
    sep_noon: bool = True,
    morning_only: bool = False,
    block_long_minutes: Optional[Tuple[int, int]] = None,
    max_day: int = MAX_TRADES_DAY,
    capture_meta: bool = False,
) -> Tuple[List[scalp_bt.ScalpTrade], SimStats, Optional[List[Dict[str, Any]]]]:
    stats = SimStats()
    trades: List[scalp_bt.ScalpTrade] = []
    meta: Optional[List[Dict[str, Any]]] = [] if capture_meta else None
    pos: Optional[Dict] = None
    counts: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])

    for i in range(WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]

        if pos is not None:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(a["cl"][i])
            elif pos["dir"] == 1:
                if a["lo"][i] <= pos["sl"]:
                    reason, exit_px = "SL", pos["sl"]
                elif a["hi"][i] >= pos["tp"]:
                    reason, exit_px = "TP", pos["tp"]
            else:
                if a["hi"][i] >= pos["sl"]:
                    reason, exit_px = "SL", pos["sl"]
                elif a["lo"][i] <= pos["tp"]:
                    reason, exit_px = "TP", pos["tp"]
            if reason:
                tr = gapped._close_position(pos, ts, float(exit_px), reason)
                tr.version = f"dynamic_{pattern}"
                trades.append(tr)
                if reason == "SL":
                    stats.exits_sl += 1
                elif reason == "TP":
                    stats.exits_tp += 1
                else:
                    stats.exits_eod += 1
                pos = None

        if a["dow"][i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if morning_only and m >= MORNING_END:
            continue

        side = stack_side(i, a)
        if side == 0:
            continue
        stats.signals_stack += 1

        if sep_noon and m >= MORNING_END:
            sep = float(a["sep"][i])
            if sep != sep or sep < SEP_MIN:
                stats.filtered_sep += 1
                continue

        if not trigger(side, i, a, pattern=pattern):
            continue
        stats.signals_pattern += 1

        if block_long_minutes is not None and side == 1:
            start, end = block_long_minutes
            if start <= m < end:
                stats.filtered_entry_block += 1
                continue

        if counts.get(day, 0) >= max_day:
            stats.filtered_max_day += 1
            continue

        atr = float(a["atr1"][i])
        if atr != atr or atr <= 0:
            continue
        use = sl_pts_from_atr(atr, mult=1.5, sl_min=6.0, sl_max=25.0)
        entry = float(a["cl"][i])
        sl = entry - use if side == 1 else entry + use
        tp = tp_for_side(side, entry, use)

        if pos is not None:
            if pos["dir"] == side:
                stats.filtered_same_bar += 1
                continue
            # Reversal: close existing at market (close), then open opposite
            exit_px = float(a["cl"][i])
            tr = gapped._close_position(pos, ts, exit_px, "REVERSAL")
            tr.version = f"dynamic_{pattern}"
            trades.append(tr)
            stats.exits_reversal += 1
            stats.reversals += 1
            pos = None

        pos = {
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_pts": use,
        }
        if capture_meta and meta is not None:
            trigger_type = "B" if pattern_b(side, i, a) else "A"
            struct = int(a["struct"][i])
            ema = int(a["ema15"][i])
            vwap = int(a["vwap_side"][i])
            meta.append({
                "bar_i": i,
                "side": side,
                "entry_minute": m,
                "hour": int(a["hours"][i]),
                "pattern": trigger_type,
                "struct": struct,
                "struct_label": "HH_HL" if struct > 0 else ("LH_LL" if struct < 0 else "mixed"),
                "ema15": ema,
                "ema_aligned": side == ema or ema == 0,
                "vwap_side": vwap,
                "vwap_aligned": side == vwap or vwap == 0,
                "sep": float(a["sep"][i]),
                "sep_high": float(a["sep"][i]) >= SEP_MIN if a["sep"][i] == a["sep"][i] else False,
                "atr1": atr,
            })
        counts[day] = counts.get(day, 0) + 1
        stats.entries += 1

    if pos is not None:
        i = n - 1
        tr = gapped._close_position(pos, pd.Timestamp(dt[i]), float(a["cl"][i]), "EOD")
        tr.version = f"dynamic_{pattern}"
        trades.append(tr)
        stats.exits_eod += 1

    return trades, stats, meta


def r_multiple(t: scalp_bt.ScalpTrade) -> float:
    entry = float(t.entry_price)
    exit_px = float(t.exit_price)
    sl = float(t.sl)
    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        return 0.0
    pts = (exit_px - entry) if t.direction == "long" else (entry - exit_px)
    return pts / stop_dist


def max_consecutive_losses(trades: List[scalp_bt.ScalpTrade]) -> int:
    best = cur = 0
    for t in trades:
        if t.pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def avg_win_loss(trades: List[scalp_bt.ScalpTrade]) -> Tuple[float, float]:
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    aw = float(np.mean(wins)) if wins else 0.0
    al = float(np.mean(losses)) if losses else 0.0
    return round(aw, 2), round(al, 2)


def bucket_trades(trades: List[scalp_bt.ScalpTrade], key_fn) -> Dict[str, List]:
    out: Dict[str, List] = {}
    for t in trades:
        k = key_fn(t)
        out.setdefault(str(k), []).append(t)
    return out


def trade_metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    if not trades:
        z = resp.group_metrics([])
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_consec_losses": 0,
        }
    s = resp.group_metrics(trades)
    aw, al = avg_win_loss(trades)
    return {
        "trades": s["trades"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "net_pnl": round(s["total_pnl"], 2),
        "expectancy": round(s["expectancy"], 2),
        "max_drawdown": round(s["max_drawdown"], 2),
        "avg_winner": aw,
        "avg_loser": al,
        "max_consec_losses": max_consecutive_losses(trades),
    }


def full_report(
    label: str,
    trades: List[scalp_bt.ScalpTrade],
    stats: Optional[SimStats] = None,
) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    overall = trade_metrics(trades)
    is_m = trade_metrics(ins)
    oos_m = trade_metrics(oos)

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    morning = []
    afternoon = []
    by_hour: Dict[int, List] = {}
    r_vals = [round(r_multiple(t), 2) for t in trades]
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        et = ts.tz_convert(ET)
        h = et.hour
        by_hour.setdefault(h, []).append(t)
        m = h * 60 + et.minute
        (morning if m < MORNING_END else afternoon).append(t)

    by_hour_m = {f"{h:02d}": trade_metrics(v) for h, v in sorted(by_hour.items())}

    pass_ok = (
        oos_m["trades"] >= 30
        and oos_m["profit_factor"] >= 1.15
        and oos_m["net_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["net_pnl"] > 0
    )

    r_buckets = {
        "lt_0": sum(1 for r in r_vals if r < 0),
        "0_0.5": sum(1 for r in r_vals if 0 <= r < 0.5),
        "0.5_1": sum(1 for r in r_vals if 0.5 <= r < 1.0),
        "1_2": sum(1 for r in r_vals if 1.0 <= r < 2.0),
        "ge_2": sum(1 for r in r_vals if r >= 2.0),
    }

    rep = {
        "label": label,
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "long": trade_metrics(longs),
        "short": trade_metrics(shorts),
        "morning": trade_metrics(morning),
        "afternoon": trade_metrics(afternoon),
        "by_hour": by_hour_m,
        "r_multiples": {
            "mean": round(float(np.mean(r_vals)), 2) if r_vals else 0.0,
            "median": round(float(np.median(r_vals)), 2) if r_vals else 0.0,
            "buckets": r_buckets,
        },
        "pass": pass_ok,
    }
    if stats is not None:
        rep["sim_stats"] = {
            "signals_stack": stats.signals_stack,
            "signals_pattern": stats.signals_pattern,
            "filtered_sep": stats.filtered_sep,
            "filtered_entry_block": stats.filtered_entry_block,
            "filtered_max_day": stats.filtered_max_day,
            "entries": stats.entries,
            "reversals": stats.reversals,
            "exits_sl": stats.exits_sl,
            "exits_tp": stats.exits_tp,
            "exits_eod": stats.exits_eod,
            "exits_reversal": stats.exits_reversal,
        }
    return rep


def print_report(r: Dict[str, Any]) -> None:
    o, oo = r["overall"], r["oos"]
    flag = "PASS" if r["pass"] else "FAIL"
    print(f"\n{'=' * 90}")
    print(f"{r['label']}  [{flag}]")
    print(f"{'=' * 90}")
    print(
        f"  ALL  n={o['trades']:4d} WR={o['win_rate']:5.1f}% PF={o['profit_factor']:5.2f} "
        f"Net=${o['net_pnl']:8.1f} Exp=${o['expectancy']:6.2f} DD=${o['max_drawdown']:7.1f} "
        f"maxLL={o['max_consec_losses']}"
    )
    print(
        f"  OOS  n={oo['trades']:4d} WR={oo['win_rate']:5.1f}% PF={oo['profit_factor']:5.2f} "
        f"Net=${oo['net_pnl']:8.1f} Exp=${oo['expectancy']:6.2f} DD=${oo['max_drawdown']:7.1f}"
    )
    print(
        f"  Long PF={r['long']['profit_factor']:.2f} ${r['long']['net_pnl']:.0f} n={r['long']['trades']} | "
        f"Short PF={r['short']['profit_factor']:.2f} ${r['short']['net_pnl']:.0f} n={r['short']['trades']}"
    )
    print(
        f"  Morning PF={r['morning']['profit_factor']:.2f} ${r['morning']['net_pnl']:.0f} | "
        f"Afternoon PF={r['afternoon']['profit_factor']:.2f} ${r['afternoon']['net_pnl']:.0f}"
    )
    if "sim_stats" in r:
        st = r["sim_stats"]
        print(
            f"  Signals stack={st['signals_stack']} pattern={st['signals_pattern']} "
            f"sep_filtered={st['filtered_sep']} reversals={st['reversals']}"
        )
    print(f"  R-mult mean={r['r_multiples']['mean']} median={r['r_multiples']['median']} buckets={r['r_multiples']['buckets']}")


def simulate_locked(df: pd.DataFrame) -> List[scalp_bt.ScalpTrade]:
    ha = hunt.prepare(df)
    lt = hunt.simulate_ext(ha, hunt.LONG_WINNER)
    st = hunt.simulate_ext(ha, LOCKED_SHORT)
    return lt + st


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "dynamic_15_5_1_rth_backtest.json")

    print("DYNAMIC vs LOCKED — same MNQ_1m.csv, $2/pt + $1.24 RT, IS/OOS split Jun 2026")
    df = official.load_1m(csv)
    a = prepare_arrays(df)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    tr_a, st_a, _ = simulate_dynamic(a, pattern="A", sep_noon=True)
    tr_ab, st_ab, _ = simulate_dynamic(a, pattern="AB", sep_noon=True)
    tr_locked = simulate_locked(df)

    rep_a = full_report("DYNAMIC Pattern A (quiet 1m only)", tr_a, st_a)
    rep_ab = full_report("DYNAMIC Pattern A+B (quiet + pullback)", tr_ab, st_ab)
    rep_locked = full_report("LOCKED asymmetric (AM long / PM short)", tr_locked)

    for r in (rep_a, rep_ab, rep_locked):
        print_report(r)

    print(f"\n{'=' * 90}")
    print("HEAD-TO-HEAD (OOS)")
    print(f"{'=' * 90}")
    for r in (rep_a, rep_ab, rep_locked):
        oo = r["oos"]
        print(
            f"  {r['label'][:45]:<45} PF={oo['profit_factor']:5.2f} "
            f"${oo['net_pnl']:7.0f} n={oo['trades']:3d} L/S={r['long']['trades']}/{r['short']['trades']}"
        )

    payload = {
        "spec": "dynamic_15_5_1_rth",
        "csv": csv,
        "costs": f"${PT}/pt + ${COMM} RT",
        "oos_split": "2026-06-01",
        "execution": {
            "stop": "1m ATR x1.5 clip [6,25]",
            "tp": "1R long / 2R short",
            "session": "9:35-15:50 ET",
            "one_position": True,
            "reversal": "close on opposite full trigger then enter",
            "no_time_direction_lock": True,
        },
        "dynamic_pattern_A": rep_a,
        "dynamic_pattern_AB": rep_ab,
        "locked_asymmetric": rep_locked,
        "verdict": {
            "dynamic_A_pass": rep_a["pass"],
            "dynamic_AB_pass": rep_ab["pass"],
            "locked_pass": rep_locked["pass"],
            "best_oos_pf": max(rep_a["oos"]["profit_factor"], rep_ab["oos"]["profit_factor"], rep_locked["oos"]["profit_factor"]),
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
