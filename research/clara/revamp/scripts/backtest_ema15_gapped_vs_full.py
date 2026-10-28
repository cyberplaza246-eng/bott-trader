#!/usr/bin/env python3
"""
Compare gapped 6-window ema15_eod vs full RTH continuous on real MNQ 1m bars.

Variant A — six locked windows (current live recipe).
Variant B — any bar 9:35–15:50 ET when filters pass, 5m entry cooldown.

Both use: daily+15m+60m, quiet 1m body≤4, sep/ATR noon+, ATR×2 stop 20–60,
1R TP, flatten 15:50, MAX_TRADES_DAY=20, window/slot re-entry after close, 2-lot cap.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    QUALITY_AFTER_MINUTE,
    SEP_MIN,
    TP_R,
    atr15_on_1m,
    daily_trend_on_1m,
    high_confidence,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_index,
    window_requires_quality,
)

ET = pytz.timezone("US/Eastern")
PT, COMM = official.PT, official.COMM
WARMUP = 80
FULL_COOLDOWN_MIN = 5


def _close_position(p, ts, exit_px, reason) -> scalp_bt.ScalpTrade:
    pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
    return scalp_bt.ScalpTrade(
        entry_time=p["ts"],
        direction="long" if p["dir"] == 1 else "short",
        entry_price=p["entry"],
        sl=p["sl"],
        tp=p["tp"],
        exit_time=ts,
        exit_price=exit_px,
        pnl=pts * PT - COMM,
        exit_reason=reason,
        version=p.get("variant", ""),
        hold_seconds=(ts - p["ts"]).total_seconds(),
    )


def simulate(
    df: pd.DataFrame,
    *,
    variant: str,
    cooldown_min: int = 0,
) -> List[scalp_bt.ScalpTrade]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    op = work["open"].to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    trades: List[scalp_bt.ScalpTrade] = []
    opens: List[Dict] = []
    fired_windows: Set[int] = set()
    counts: Dict = {}
    last_entry: Optional[Tuple] = None  # (day, et_minute)

    for i in range(WARMUP, len(work)):
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
                trades.append(_close_position(p, ts, float(exit_px), reason))
                if variant == "gapped" and p.get("win") is not None:
                    fired_windows.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue

        win: Optional[int] = None
        if variant == "gapped":
            win = window_index(m)
            if win is None or win in fired_windows:
                continue
        else:
            if cooldown_min > 0 and last_entry is not None:
                ld, lm = last_entry
                if ld == day and m - lm < cooldown_min:
                    continue

        side = int(t15[i])
        if side == 0 or side != int(td[i]) or side != int(t60[i]):
            continue
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        conf = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        if variant == "gapped":
            if window_requires_quality(win) and not conf:
                continue
        elif m >= QUALITY_AFTER_MINUTE and not conf:
            continue

        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp_dist = use * TP_R
        tp = entry + tp_dist if side == 1 else entry - tp_dist
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": win,
            "variant": variant,
        })
        if variant == "gapped" and win is not None:
            fired_windows.add(win)
        else:
            last_entry = (day, m)
        counts[day] = counts.get(day, 0) + 1

    return trades


def fills_per_day(trades: List[scalp_bt.ScalpTrade]) -> Tuple[float, float]:
    days: Dict = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        d = ts.tz_convert(ET).date()
        days[d] = days.get(d, 0) + 1
    if not days:
        return 0.0, 0.0
    vals = list(days.values())
    return sum(vals) / len(vals), float(max(vals))


def report(label: str, trades: List[scalp_bt.ScalpTrade]) -> Dict:
    ins, oos = edge.split_is_oos(trades)
    s_all = edge.summarize(trades)
    s_is = edge.summarize(ins)
    s_oos = edge.summarize(oos)
    avg_fpd, max_fpd = fills_per_day(trades)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n=== {label} ===")
    print(
        f"  ALL  n={s_all['trades']:4d}  WR={s_all['win_rate']:5.1f}%  PF={s_all['profit_factor']:5.2f}  "
        f"${s_all['total_pnl']:9.1f}  DD=${s_all['max_drawdown']:7.1f}  avg/day={avg_fpd:.2f}  exits={reasons}"
    )
    print(
        f"  IS   n={s_is['trades']:4d}  WR={s_is['win_rate']:5.1f}%  PF={s_is['profit_factor']:5.2f}  "
        f"${s_is['total_pnl']:9.1f}  DD=${s_is['max_drawdown']:7.1f}"
    )
    print(
        f"  OOS  n={s_oos['trades']:4d}  WR={s_oos['win_rate']:5.1f}%  PF={s_oos['profit_factor']:5.2f}  "
        f"${s_oos['total_pnl']:9.1f}  DD=${s_oos['max_drawdown']:7.1f}"
    )
    return {
        "label": label,
        "trades": s_all["trades"],
        "win_rate": s_all["win_rate"],
        "pf": s_all["profit_factor"],
        "pnl": s_all["total_pnl"],
        "max_dd": s_all["max_drawdown"],
        "avg_fpd": avg_fpd,
        "is_trades": s_is["trades"],
        "is_pf": s_is["profit_factor"],
        "is_pnl": s_is["total_pnl"],
        "is_dd": s_is["max_drawdown"],
        "oos_trades": s_oos["trades"],
        "oos_pf": s_oos["profit_factor"],
        "oos_pnl": s_oos["total_pnl"],
        "oos_dd": s_oos["max_drawdown"],
    }


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("ema15_eod: GAPPED WINDOWS vs FULL RTH")
    print(f"file={csv}")
    print(
        "Rules: daily+15m+60m, quiet 1m body<=4pt, sep/ATR>=0.45 noon+, ATRx2 SL 20-60, "
        f"1R TP, flatten 15:50, cap={MAX_TRADES_DAY}/day, re-entry after close, max {DEFAULT_MAX_OPEN_LOTS} lots"
    )
    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    gapped = simulate(df, variant="gapped")
    full = simulate(df, variant="full", cooldown_min=FULL_COOLDOWN_MIN)

    rows = [
        report("A - Gapped windows (6 slots)", gapped),
        report(f"B - Full RTH ({FULL_COOLDOWN_MIN}m cooldown)", full),
    ]

    print("\n" + "=" * 95)
    print("COMPARISON TABLE")
    print("=" * 95)
    hdr = (
        f"{'Variant':<32} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'MaxDD $':>9} "
        f"{'F/day':>6} | {'IS PF':>6} {'IS $':>9} | {'OOS PF':>6} {'OOS $':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:<32} {r['trades']:6d} {r['win_rate']:6.1f} {r['pf']:6.2f} "
            f"{r['pnl']:10.1f} {r['max_dd']:9.1f} {r['avg_fpd']:6.2f} | "
            f"{r['is_pf']:6.2f} {r['is_pnl']:9.1f} | {r['oos_pf']:6.2f} {r['oos_pnl']:9.1f}"
        )

    winner = max(rows, key=lambda x: (x["oos_pnl"], x["pnl"]))
    loser = min(rows, key=lambda x: (x["oos_pnl"], x["pnl"]))
    print("\nRECOMMENDATION")
    if winner["pnl"] > loser["pnl"] and winner["oos_pnl"] > loser["oos_pnl"]:
        print(f"  Winner: {winner['label']}")
        print(
            f"  Beats on ALL (${winner['pnl']:.0f} vs ${loser['pnl']:.0f}) "
            f"and OOS (${winner['oos_pnl']:.0f} vs ${loser['oos_pnl']:.0f})."
        )
    elif winner["oos_pnl"] > loser["oos_pnl"]:
        print(f"  OOS winner: {winner['label']} (${winner['oos_pnl']:.0f} vs ${loser['oos_pnl']:.0f})")
        print(f"  ALL-period winner: {max(rows, key=lambda x: x['pnl'])['label']}")
    else:
        print("  No clear winner - review table above.")
    print(f"  Full RTH uses {FULL_COOLDOWN_MIN}-minute cooldown between entries to reduce bar spam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
