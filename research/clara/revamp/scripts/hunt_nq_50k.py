"""Hunt an NQ setup that fits a 50k prop account ($2,000 end-of-day trailing max loss).

1 NQ = $20/pt, $5 round trip, 1 tick slippage on entries and stops. One position at a time.
Every entry is rebuilt with a hard point stop: stop = min(structural stop, CAP pts),
target = R x stop, no $ lock (the MNQ $ lock is meaningless at NQ size).
Day loss limit $500, overnight $300.

Score per run
  - plain stats (net, PF, trade-sequence drawdown) per period
  - eval sim: start a 50k eval every 5 trading days; pass at +$3,000, fail when end-of-day
    balance <= trailing floor (peak EOD - $2,000, floor stops at +$100). 90 trading day timeout.
Selection uses IS = Nov24-Oct25 only; OOS = Nov25-Sep26 is reported, never used to pick.

  python scripts/hunt_nq_50k.py            stage 1: every building block alone
  python scripts/hunt_nq_50k.py combos     stage 2: combos of the best IS blocks
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.backtest_revamp_live_stack as bt  # noqa: E402

CACHE = os.path.join(ROOT, "data", "_nq50k_signals.pkl")
IS_END = pd.Timestamp("2025-11-01", tz="UTC")
CAPS = (10.0, 15.0, 25.0)
RS = (1.0, 1.5, 2.0, 3.0)
EVAL_TARGET, EVAL_DD, EVAL_LOCK, EVAL_DAYS, EVAL_STEP = 3000.0, 2000.0, 100.0, 90, 5
_D: dict = {}


def build_cache() -> None:
    m1, m5 = bt.load_bars(os.path.join(ROOT, "data", "MNQ_1m.csv"))
    b = bt.featured(m5)
    blocks = {}
    ens = bt.ensemble_signals(m5)
    for sl in ("breakout", "trend", "mean_reversion"):
        blocks[f"ens_{sl}"] = ens[ens.sleeve == sl]
    blocks.update(bt.addon_rows(b))
    blocks.update(bt.candidate_rows(b))
    close = m5["close"]
    for k, v in blocks.items():
        v = v.copy()
        v["ref"] = close.reindex(v.index).values
        blocks[k] = v[np.isfinite(v.ref)]
    with open(CACHE, "wb") as f:
        pickle.dump({"m1": m1, "idx": m5.index, "blocks": blocks}, f)


def _load():
    if not _D:
        with open(CACHE, "rb") as f:
            _D.update(pickle.load(f))
    return _D


def rebuild(sig: pd.DataFrame, cap: float, r: float) -> pd.DataFrame:
    s = sig.copy()
    dist = np.minimum((s.ref - s.sl).abs(), cap)
    dist = np.maximum((dist / bt.TICK).round() * bt.TICK, 4 * bt.TICK)
    s["sl"] = s.ref - s.dir * dist
    s["tp"] = s.ref + s.dir * r * dist
    s["lock"] = False
    s["sleeve"] = s["sleeve"].where(s.sleeve != "addon", "trend")
    return s


def eval_sim(daily: pd.Series, days: pd.DatetimeIndex) -> dict:
    pnl = daily.reindex(days, fill_value=0.0).to_numpy()
    out = {"pass": 0, "fail": 0, "timeout": 0, "days_to_pass": []}
    for s in range(0, len(days), EVAL_STEP):
        bal = peak = 0.0
        res = "timeout"
        for k in range(s, min(s + EVAL_DAYS, len(days))):
            bal += pnl[k]
            if bal >= EVAL_TARGET:
                res = "pass"
                out["days_to_pass"].append(k - s + 1)
                break
            peak = max(peak, bal)
            if bal <= min(peak - EVAL_DD, EVAL_LOCK):
                res = "fail"
                break
        if s + EVAL_DAYS > len(days) and res == "timeout":
            continue
        out[res] += 1
    n = out["pass"] + out["fail"] + out["timeout"]
    out["n"] = n
    out["pass_pct"] = round(100 * out["pass"] / n, 1) if n else 0.0
    out["fail_pct"] = round(100 * out["fail"] / n, 1) if n else 0.0
    out["med_days"] = int(np.median(out.pop("days_to_pass"))) if out["pass"] else None
    return out


def run(cfg) -> dict:
    name, parts, cap, r, day_only = cfg
    bt.PV, bt.COMM_RT, bt.MAX_RISK_USD = 20.0, 5.0, 1e9
    bt.DAY_LOSS, bt.OVN_LOSS, bt.DAILY_TARGET = 500.0, 300.0, 0.0
    d = _load()
    sig = pd.concat([rebuild(d["blocks"][p], cap, r) for p in parts])
    rules = bt.Rules(overnight_sleeves="all", rth_main_only=day_only, max_open=1)
    if day_only:
        sig = sig[[bt.session_of(t + pd.Timedelta(minutes=5))[0] == "day" for t in sig.index]]
    tr = bt.in_window(bt.simulate(d["m1"], d["idx"], sig, rules))
    days = pd.DatetimeIndex(sorted({t.normalize().tz_localize(None)
                                    for t in d["idx"].tz_convert(bt.ET) if t.weekday() < 5}))
    days = days[(days >= "2024-11-01") & (days < "2026-10-01")]
    daily = pd.Series([t.pnl for t in tr], index=[t.exit_time.tz_convert(bt.ET).normalize().tz_localize(None)
                                                  for t in tr], dtype=float).groupby(level=0).sum()
    is_tr = [t for t in tr if t.entry_time < IS_END]
    oos_tr = [t for t in tr if t.entry_time >= IS_END]
    cut = IS_END.tz_localize(None)
    return {"name": name, "parts": list(parts), "cap": cap, "r": r, "day_only": day_only,
            "total": bt.stats(tr), "is": bt.stats(is_tr), "oos": bt.stats(oos_tr),
            "periods": {k: v["net"] for k, v in bt.by_period(tr).items()},
            "worst_day": round(float(daily.min()), 1) if len(daily) else 0.0,
            "eval_is": eval_sim(daily, days[days < cut]), "eval_oos": eval_sim(daily, days[days >= cut])}


def show(res: list, top: int) -> None:
    print(f"{'name':<34} {'cap':>4} {'R':>4} {'day':>3} | {'IS n':>5} {'IS net':>7} {'PF':>4} {'DD':>6} "
          f"{'pass%':>5} {'fail%':>5} | {'OOS n':>5} {'OOS net':>7} {'PF':>4} {'DD':>6} {'pass%':>5} {'fail%':>5}")
    for x in res[:top]:
        i, o, ei, eo = x["is"], x["oos"], x["eval_is"], x["eval_oos"]
        print(f"{x['name']:<34} {x['cap']:4.0f} {x['r']:4.1f} {'Y' if x['day_only'] else 'N':>3} | "
              f"{i['n']:5d} {i['net']:7.0f} {i['pf']:4.2f} {i['dd']:6.0f} {ei['pass_pct']:5.1f} {ei['fail_pct']:5.1f} | "
              f"{o['n']:5d} {o['net']:7.0f} {o['pf']:4.2f} {o['dd']:6.0f} {eo['pass_pct']:5.1f} {eo['fail_pct']:5.1f}")


def is_score(x: dict) -> float:
    return x["eval_is"]["pass_pct"] - x["eval_is"]["fail_pct"]


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "blocks"
    if not os.path.exists(CACHE):
        build_cache()
    blocks = list(_load()["blocks"])
    if stage == "blocks":
        grid = []
        for b in blocks:
            sess = (True, False) if b.startswith("ens_") else (False,)
            for cap, r, day in itertools.product(CAPS, RS, sess):
                grid.append((b, (b,), cap, r, day))
        out = os.path.join(ROOT, "data", "nq50k_blocks.json")
    else:
        prev = json.load(open(os.path.join(ROOT, "data", "nq50k_blocks.json")))
        good = [x for x in prev if x["is"]["pf"] >= 1.15 and x["is"]["n"] >= 15 and x["is"]["net"] > 0]
        best = {}
        for x in sorted(good, key=is_score, reverse=True):
            best.setdefault(x["name"], x)
        pick = list(best.values())[:6]
        print("combo blocks:", [(x["name"], x["cap"], x["r"], x["day_only"]) for x in pick])
        grid = []
        for k in range(2, len(pick) + 1):
            for combo in itertools.combinations(pick, k):
                for cap, r in ((15.0, 1.5), (15.0, 2.0), (25.0, 2.0), (10.0, 2.0)):
                    day = all(x["day_only"] for x in combo if x["name"].startswith("ens_"))
                    grid.append(("+".join(x["name"] for x in combo), tuple(x["name"] for x in combo), cap, r, day))
        out = os.path.join(ROOT, "data", "nq50k_combos.json")
    print(f"{len(grid)} runs")
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run, grid, chunksize=2))
    res.sort(key=is_score, reverse=True)
    show(res, 40)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
