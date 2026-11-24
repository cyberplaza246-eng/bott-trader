#!/usr/bin/env python3
"""
Short-only hunt: profitable SHORT SETUP across full RTH (all 6 windows).

Fixed: short only, 15/5/1 bearish stack, quiet 1m (or pullback variant), w6_all windows.
Tests setup-quality filters — NOT time-direction locks.

For promising recipes, splits OOS by morning / midday / afternoon to detect hidden clock bias.

PASS: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_level_strategy_search as lvl
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_shorts as hunt

ET = resp.ET
MORNING_END = base.MORNING_END
MIDDAY_END = 14 * 60


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    a = hunt.prepare(df)
    work = df.copy()
    work["datetime"] = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    tmp = work.set_index("datetime")
    if tmp.index.tz is None:
        tmp.index = tmp.index.tz_localize("UTC")
    _, _, _, struct, _, _ = lvl.precompute_bias_arrays(tmp)
    a["struct15"] = struct
    return a


def structure_ok(mode: str, side: int, i: int, a: Dict) -> bool:
    if mode == "none":
        return True
    s = int(a["struct15"][i])
    if mode == "lh_ll":
        return side < 0 and s < 0
    if mode == "hh_hl":
        return side > 0 and s > 0
    raise ValueError(mode)


def sep_ok(cfg: Dict, side: int, i: int, a: Dict, win: Optional[int]) -> bool:
    mode = cfg.get("sep_mode", "none")
    if mode == "none":
        return True
    sep = float(a["sep"][i])
    if sep != sep:
        return False
    smin = float(cfg.get("sep_min", base.SEP_MIN))
    if mode == "always":
        return sep >= smin
    if mode == "noon_windows":
        from src.strategy.mnq_15m_ema_eod import window_requires_quality

        wcheck = win if win is not None else base.window_index(int(a["mins"][i]))
        if wcheck is not None and window_requires_quality(wcheck):
            return sep >= smin
        return True
    raise ValueError(mode)


def simulate_short(a: Dict[str, Any], cfg: Dict[str, Any]) -> List:
    winset = hunt.WINSETS.get(cfg["win"])
    stop_cfg = base.STOPS[cfg["stop"]]
    tp_cfg = base.TPS[cfg["tp"]]
    tp_kind, tp_val = tp_cfg
    loop = cfg["loop"]
    vwap_mode = cfg.get("vwap", "none")
    struct_mode = cfg.get("structure", "none")
    max_day = cfg["max_day"]
    max_lots = cfg["max_lots"]

    trades: List = []
    opens: List[Dict] = []
    fired: Set[int] = set()
    counts: Dict = {}
    dt = a["dt"]
    n = len(a["cl"])

    for i in range(hunt.WARMUP, n):
        ts = pd.Timestamp(dt[i])
        m = int(a["mins"][i])
        day = a["dates"][i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= base.FLAT_MINUTE:
                reason, exit_px = "EOD", float(a["cl"][i])
            elif p["dir"] == 1:
                if a["lo"][i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif a["hi"][i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if a["hi"][i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif a["lo"][i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                trades.append(gapped._close_position(p, ts, float(exit_px), reason))
                if loop == "window_loop" and p.get("win") is not None:
                    fired.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if a["dow"][i] >= 5:
            continue
        if m < base.ENTRY_MINUTE or m >= base.FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= max_day:
            continue

        win: Optional[int] = None
        if loop == "window_loop":
            win = base.window_index(m)
            if win is None or win not in (winset or set()):
                continue
            if win in fired:
                continue
        else:
            raise ValueError(loop)

        side = base.stack_side(cfg["stack"], i, a)
        if side != -1:
            continue
        if not base.entry_ok(cfg["entry"], side, i, a):
            continue
        if not hunt.vwap_ok(side, i, a, vwap_mode):
            continue
        if not structure_ok(struct_mode, side, i, a):
            continue
        if not sep_ok(cfg, side, i, a, win):
            continue
        if len(opens) >= max_lots:
            continue

        use = base.stop_pts(stop_cfg, float(a["atr1"][i]), float(a["atr5"][i]))
        if use != use or use <= 0:
            continue
        entry_px = float(a["cl"][i])
        sl = entry_px + use
        if tp_kind == "r":
            tp_px = entry_px - use * tp_val
        else:
            tp_px = entry_px - tp_val

        opens.append({
            "ts": ts, "dir": side, "entry": entry_px, "sl": sl, "tp": tp_px,
            "sl_pts": use, "win": win,
        })
        fired.add(int(win))
        counts[day] = counts.get(day, 0) + 1

    return trades


def session_bucket(trade) -> str:
    ts = pd.Timestamp(trade.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert(ET)
    m = et.hour * 60 + et.minute
    if m < MORNING_END:
        return "morning"
    if m < MIDDAY_END:
        return "midday"
    return "afternoon"


def session_splits(trades: List) -> Dict[str, Dict]:
    oos = edge.split_is_oos(trades)[1]
    out: Dict[str, Dict] = {}
    for label in ("morning", "midday", "afternoon"):
        subset = [t for t in oos if session_bucket(t) == label]
        out[label] = resp.group_metrics(subset)
    return out


def time_dependency_flag(splits: Dict[str, Dict], oos_net: float) -> Optional[str]:
    """Flag if >70% of OOS $ comes from one session (diagnostic only)."""
    if abs(oos_net) < 50:
        return None
    nets = {k: splits[k]["total_pnl"] for k in splits}
    dominant = max(nets, key=lambda k: nets[k])
    if oos_net > 0 and nets[dominant] / oos_net > 0.7:
        return f"OOS profit concentrated in {dominant} ({nets[dominant]:.0f}/{oos_net:.0f})"
    if oos_net < 0 and abs(nets[dominant]) / abs(oos_net) > 0.7:
        return f"OOS loss concentrated in {dominant} ({nets[dominant]:.0f}/{oos_net:.0f})"
    return None


def cfg_label(c: Dict) -> str:
    return (
        f"entry={c['entry']}|tp={c['tp']}|vwap={c['vwap']}|struct={c['structure']}|"
        f"sep={c['sep_mode']}"
    )


def build_grid() -> List[Dict]:
    cfg_base = {
        "stack": "15_5_1",
        "loop": "window_loop",
        "win": "w6_all",
        "stop": "atr1_1.5_6_25",
        "side": "short",
        "max_day": 4,
        "max_lots": 1,
        "sep_min": base.SEP_MIN,
    }
    rows: List[Dict] = []
    for entry in ("quiet", "1m_pullback"):
        for tp in ("1R", "2R"):
            for vwap in ("none", "with_side"):
                for structure in ("none", "lh_ll"):
                    for sep_mode in ("none", "always", "noon_windows"):
                        rows.append({
                            **cfg_base,
                            "entry": entry,
                            "tp": tp,
                            "vwap": vwap,
                            "structure": structure,
                            "sep_mode": sep_mode,
                        })
    # dedupe
    seen, out = set(), []
    for r in rows:
        k = tuple(sorted((k, str(v)) for k, v in r.items()))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def score(cfg: Dict, trades: List) -> Dict:
    ins, oos = edge.split_is_oos(trades)
    is_m, oos_m = resp.group_metrics(ins), resp.group_metrics(oos)
    splits = session_splits(trades)
    flag = time_dependency_flag(splits, oos_m["total_pnl"])
    passed = hunt.is_pass(is_m, oos_m)
    return {
        "label": cfg_label(cfg),
        "config": cfg,
        "is": is_m,
        "oos": oos_m,
        "overall": resp.group_metrics(trades),
        "session_oos": splits,
        "pass": passed,
        "time_dependency": flag,
        "interesting": (
            not passed
            and oos_m["trades"] >= 20
            and oos_m["profit_factor"] >= 1.05
            and oos_m["total_pnl"] > 0
            and is_m["total_pnl"] > 0
        ),
    }


def print_session_table(row: Dict) -> None:
    print(f"  {row['label']}")
    oos = row["oos"]
    print(
        f"    OOS n={oos['trades']} PF={oos['profit_factor']:.2f} ${oos['total_pnl']:.0f} "
        f"WR={oos['win_rate']:.1f}%  [{'PASS' if row['pass'] else 'FAIL'}]"
    )
    for period in ("morning", "midday", "afternoon"):
        s = row["session_oos"][period]
        sign = "+" if s["total_pnl"] > 0 else "−"
        print(
            f"    {period:10s} n={s['trades']:3d} PF={s['profit_factor']:.2f} "
            f"${s['total_pnl']:7.0f} ({sign})"
        )
    if row.get("time_dependency"):
        print(f"    ⚠ {row['time_dependency']}")


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "short_rth_all_windows_hunt.json")

    print("SHORT-ONLY HUNT — all 6 RTH windows, setup filters (no time-direction lock)")
    print("Stack 15/5/1 bearish | costs ${}/pt + ${} RT | PASS gates standard\n".format(
        official.PT, official.COMM
    ))

    t0 = time.time()
    df = official.load_1m(csv)
    a = prepare(df)
    grid = build_grid()
    print(f"bars={len(df):,} | candidates={len(grid)}")

    results: List[Dict] = []
    pass_rows: List[Dict] = []
    interesting: List[Dict] = []

    for i, cfg in enumerate(grid, 1):
        trades = simulate_short(a, cfg)
        row = score(cfg, trades)
        results.append(row)
        if row["pass"]:
            pass_rows.append(row)
        if row["interesting"]:
            interesting.append(row)
        if i % 24 == 0:
            print(f"  ... {i}/{len(grid)} tested pass={len(pass_rows)} interesting={len(interesting)}")

    elapsed = time.time() - t0
    print(f"\nDone {elapsed:.0f}s | PASS={len(pass_rows)} | interesting={len(interesting)}")

    if pass_rows:
        print("\n" + "=" * 90)
        print("PASS (full gates)")
        print("=" * 90)
        for row in sorted(pass_rows, key=lambda r: (-r["oos"]["profit_factor"], -r["oos"]["total_pnl"])):
            print_session_table(row)

    if interesting:
        print("\n" + "=" * 90)
        print("INTERESTING (OOS PF>=1.05, both periods $>0, n>=20 — not full PASS)")
        print("=" * 90)
        for row in sorted(interesting, key=lambda r: (-r["oos"]["profit_factor"], -r["oos"]["total_pnl"]))[:10]:
            print_session_table(row)

    if not pass_rows and not interesting:
        best = max(results, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print("\nNo PASS or interesting. Best OOS:")
        print_session_table(best)

    payload = {
        "spec": "short_only_w6_all_setup_hunt",
        "tested": len(grid),
        "elapsed_s": round(elapsed, 1),
        "pass": pass_rows,
        "interesting": interesting,
        "best_fail": max(results, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"])),
        "all_results_summary": sorted(
            [
                {
                    "label": r["label"],
                    "pass": r["pass"],
                    "interesting": r["interesting"],
                    "oos_pf": r["oos"]["profit_factor"],
                    "oos_net": r["oos"]["total_pnl"],
                    "oos_n": r["oos"]["trades"],
                    "session_oos": r["session_oos"],
                    "time_dependency": r["time_dependency"],
                }
                for r in results
            ],
            key=lambda x: (-x["oos_pf"], -x["oos_net"]),
        )[:30],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0 if pass_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
