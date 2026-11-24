"""Stress test REVAMP_OVERNIGHT_SLEEVES choices against worse overnight fills.

Uses scripts/backtest_revamp_live_stack.py. Writes data/revamp_overnight_sleeves_stress.json.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

import scripts.backtest_revamp_live_stack as bt  # noqa: E402

VARIANTS = [(s, k) for s in ("trend", "all", "trend,breakout", "trend,mean_reversion") for k in (0, 1, 2)]
_CACHE: dict = {}


def _inputs():
    if not _CACHE:
        m1, m5 = bt.load_bars(os.path.join(ROOT, "data", "MNQ_1m.csv"))
        b = bt.featured(m5)
        sig = pd.concat([bt.ensemble_signals(m5)] + list(bt.addon_rows(b).values()))
        _CACHE.update(m1=m1, idx=m5.index, sig=sig)
    return _CACHE


def run(variant):
    sleeves, extra = variant
    d = _inputs()
    tr = bt.in_window(bt.simulate(d["m1"], d["idx"], d["sig"],
                                  bt.Rules(overnight_sleeves=sleeves, ovn_extra_slip_ticks=extra)))
    return {"sleeves": sleeves, "extra_slip_ticks": extra, "total": bt.stats(tr), "periods": bt.by_period(tr),
            "by_session": bt.by_key(tr, lambda t: t.session)}


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(run, VARIANTS))
    base = {r["extra_slip_ticks"]: r for r in res if r["sleeves"] == "trend"}
    for r in res:
        b = base[r["extra_slip_ticks"]]
        d = {k: round(r["periods"][k]["net"] - b["periods"][k]["net"]) for k in r["periods"]}
        r["delta_vs_trend_same_slip"] = d
        print(f"{r['sleeves']:22s} +{r['extra_slip_ticks']}tick  {bt.fmt(r['total'])}  "
              f"ovn {bt.fmt(r['by_session'].get('overnight', bt.stats([])))}  "
              f"vs trend-only: {sum(d.values()):+,} ({sum(v > 0 for v in d.values())}/4 periods)")
    out = os.path.join(ROOT, "data", "revamp_overnight_sleeves_stress.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
