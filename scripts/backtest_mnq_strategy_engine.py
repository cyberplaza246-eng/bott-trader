#!/usr/bin/env python3
"""
Strategy-engine backtest ONLY (no MT5 EA / TradeManager).

Tests the MNQ brain described for the TrendCatcher skeleton replacement:

  15m direction  →  5m direction  →  VWAP  →  EMA10/20  →  1m entry trigger

No chart-object signals. No level interaction (PDH/PDL). No execution plumbing.
Shared exits: 1m ATR×1.5 clip 6-25, 1R TP, flatten 15:50, 6 windows, 2-lot,
$max/day, $2/pt + $1.24 RT. IS/OOS split Jun 2026.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_level_strategy_search as lvl
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    TP_R,
    _as_utc_frame,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    window_index,
    window_label,
    window_requires_quality,
)
from src.strategy.session_levels import EMA_FAST_CHART, EMA_SLOW_CHART

ET = lvl.ET
WARMUP = lvl.WARMUP
PT, COMM = official.PT, official.COMM
MORNING_ONLY_WINS = frozenset({0, 1, 2})

# Strategy-engine variants (logic only)
VARIANTS = {
    "SE1_strict": {
        "label": "15m structure+EMA10/20+VWAP strict + 5m EMA10/20 + quiet 1m",
        "side15": "strict",
        "require_vwap": True,
        "morning_only": False,
    },
    "SE2_soft": {
        "label": "15m structure+EMA10/20 soft + VWAP side filter + 5m + quiet 1m",
        "side15": "soft",
        "require_vwap": True,
        "morning_only": False,
    },
    "SE3_ema_vwap": {
        "label": "15m EMA10/20 + 5m EMA10/20 + VWAP side + quiet 1m (no structure)",
        "side15": "ema",
        "require_vwap": True,
        "morning_only": False,
    },
    "SE4_ema_only": {
        "label": "15m EMA10/20 + 5m EMA10/20 + quiet 1m (no VWAP, no structure)",
        "side15": "ema",
        "require_vwap": False,
        "morning_only": False,
    },
    "SE5_strict_am": {
        "label": "SE1 strict, morning windows only (9:35/10:15/11:00)",
        "side15": "strict",
        "require_vwap": True,
        "morning_only": True,
    },
    "SE6_ema_vwap_am": {
        "label": "SE3 ema+vwap, morning windows only",
        "side15": "ema",
        "require_vwap": True,
        "morning_only": True,
    },
}


def simulate_engine(
    df: pd.DataFrame,
    *,
    side15_mode: str,
    require_vwap: bool,
    morning_only: bool,
    version: str,
) -> List[lvl.TradeRec]:
    tmp = _as_utc_frame(df)
    strict, soft, side5, _struct, ema15, vwap_side = lvl.precompute_bias_arrays(tmp)
    atr1 = lvl.atr1_on_1m(tmp).to_numpy(dtype=float)
    sep = sep15_on_1m(tmp).to_numpy(dtype=float)
    op = tmp["open"].to_numpy(dtype=float)
    hi = tmp["high"].to_numpy(dtype=float)
    lo = tmp["low"].to_numpy(dtype=float)
    cl = tmp["close"].to_numpy(dtype=float)
    et = tmp.index.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date
    dt = tmp.index

    if side15_mode == "strict":
        side15 = strict
    elif side15_mode == "soft":
        side15 = soft
    elif side15_mode == "ema":
        side15 = ema15
    else:
        raise ValueError(side15_mode)

    recs: List[lvl.TradeRec] = []
    opens: List[Dict] = []
    fired: Set[int] = set()
    counts: Dict = {}

    for i in range(WARMUP, len(tmp)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                tr = gapped._close_position(p, ts, float(exit_px), reason)
                tr.version = version
                recs.append(
                    lvl.TradeRec(tr, p["win_name"], p["sl_pts"], version)
                )
                if p.get("win") is not None:
                    fired.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue

        win = window_index(m)
        if win is None or win in fired:
            continue
        if morning_only and win not in MORNING_ONLY_WINS:
            continue

        side = int(side15[i])
        if side == 0 or side != int(side5[i]):
            continue
        if require_vwap and int(vwap_side[i]) != side:
            continue
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        # noon+ quality: sep gate (same safety idea as live stack)
        conf = side != 0 and float(sep[i]) >= SEP_MIN and not (
            isinstance(sep[i], float) and np.isnan(sep[i])
        )
        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        a = float(atr1[i])
        if np.isnan(a) or a <= 0:
            continue
        use = sl_pts_from_atr(a, mult=1.5, sl_min=6.0, sl_max=25.0)
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp = entry + use * TP_R if side == 1 else entry - use * TP_R
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_pts": use,
            "win": win,
            "win_name": window_label(win),
        })
        fired.add(win)
        counts[day] = counts.get(day, 0) + 1

    return recs


def dump(label: str, trades: List) -> Dict[str, Any]:
    m = lvl.group_metrics(trades)
    print(
        f"  {label:<18} n={m['trades']:4d} WR={m['win_rate']:5.1f}% "
        f"PF={m['profit_factor']:5.2f} Net=${m['total_pnl']:8.1f} "
        f"Exp=${m['expectancy']:7.2f} DD=${m['max_drawdown']:7.1f}"
    )
    return m


def report(vid: str, label: str, recs: List[lvl.TradeRec]) -> Dict[str, Any]:
    trades = [r.trade for r in recs]
    ins, oos = edge.split_is_oos(trades)
    print(f"\n{'=' * 96}")
    print(f"{vid}: {label}")
    print("=" * 96)
    overall = dump("ALL", trades)
    is_m = dump("IS", ins)
    oos_m = dump("OOS", oos)
    by_side = {
        s: dump(s, [t for t in trades if t.direction == s])
        for s in ("long", "short")
    }
    ok = (
        oos_m["trades"] >= 30
        and oos_m["profit_factor"] >= 1.15
        and oos_m["total_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["total_pnl"] > 0
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return {
        "id": vid,
        "label": label,
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_side": by_side,
        "pass": ok,
        "n_trades": len(trades),
    }


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "mnq_strategy_engine_backtest.json")

    print("MNQ STRATEGY ENGINE BACKTEST (logic only - no EA)")
    print("Stack: 15m -> 5m -> VWAP -> EMA10/20 -> quiet 1m entry")
    print(f"EMA periods: {EMA_FAST_CHART}/{EMA_SLOW_CHART} | stop 1m ATR x1.5 [6,25] | TP 1R")
    print(f"Costs ${PT}/pt + ${COMM} RT | IS/OOS Jun 2026 | file={csv}")
    print("NOTE: Not implementing TradeManager / MT5 - edge test only.\n")

    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    results: List[Dict[str, Any]] = []
    for vid, cfg in VARIANTS.items():
        recs = simulate_engine(
            df,
            side15_mode=cfg["side15"],
            require_vwap=cfg["require_vwap"],
            morning_only=cfg["morning_only"],
            version=vid,
        )
        results.append(report(vid, cfg["label"], recs))

    ranked = sorted(
        results,
        key=lambda r: (
            r["oos"]["expectancy"],
            r["oos"]["profit_factor"],
            -r["oos"]["max_drawdown"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 96)
    print("RANKED BY OOS EXPECTANCY")
    print("=" * 96)
    for i, r in enumerate(ranked, 1):
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"  #{i} [{flag}] {r['id']:<18} "
            f"OOS n={r['oos']['trades']:3d} Exp=${r['oos']['expectancy']:7.2f} "
            f"PF={r['oos']['profit_factor']:5.2f} ${r['oos']['total_pnl']:8.1f} | "
            f"ALL PF={r['overall']['profit_factor']:.2f} ${r['overall']['total_pnl']:.0f}"
        )

    best = ranked[0]
    print("\nCONCLUSION")
    print(
        f"  Best OOS: {best['id']} — Exp ${best['oos']['expectancy']:.2f}/tr, "
        f"PF {best['oos']['profit_factor']:.2f}, n={best['oos']['trades']}"
    )
    if best["pass"]:
        print("  Strategy engine shows a tradable edge under PASS gates.")
    else:
        print(
            "  No variant PASSes. Keep EA plumbing as a future shell only — "
            "do not wire live execution to this brain yet."
        )  # noqa: E501 — intentional conclusion text

    # Prior level-mode reference (includes level interaction — different from SE*)
    prior_path = os.path.join("data", "level_strategy_search.json")
    prior_note = None
    if os.path.isfile(prior_path):
        with open(prior_path, encoding="utf-8") as f:
            prior = json.load(f)
        prior_note = prior.get("recommendation", {}).get("text")
        print(f"\nPrior level-mode search note:\n  {prior_note}")

    payload = {
        "purpose": "strategy_engine_edge_test",
        "not_implemented": [
            "MT5 EA",
            "TradeManager.mqh port",
            "TrendCatcherExtractor chart objects",
            "break-even / trail / partials",
        ],
        "stack": "15m → 5m → VWAP → EMA10/20 → quiet 1m",
        "csv": csv,
        "range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
        "variants": results,
        "ranked": [r["id"] for r in ranked],
        "best": best["id"],
        "prior_level_mode_note": prior_note,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
