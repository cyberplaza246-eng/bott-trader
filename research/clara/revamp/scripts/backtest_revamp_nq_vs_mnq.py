"""Live revamp stack (overnight all sleeves) on MNQ vs NQ.

NQ and MNQ trade the same price; NQ is $20/pt vs $2/pt. The bot's $ rules (profit lock,
per-trade risk cap, session limits, daily target) are in dollars, so on NQ they trigger at
1/10th the price move unless scaled. Variants:
  MNQ            : $2/pt, $3 round trip
  NQ_same_usd    : $20/pt, $5 round trip, $ rules unchanged (what the bat would do with --symbols NQ)
  NQ_scaled_usd  : $20/pt, $5 round trip, every $ rule x10 (same behaviour as 10 MNQ)
Writes data/revamp_nq_vs_mnq.json.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

import scripts.backtest_revamp_live_stack as bt  # noqa: E402

USD_KEYS = ("LOCK_ACTIVATE", "LOCK_KEEP", "LOCK_TRAIL_START", "MAX_RISK_USD", "DAY_LOSS", "OVN_LOSS", "DAILY_TARGET")
BASE = {k: getattr(bt, k) for k in USD_KEYS + ("PV", "COMM_RT")}

VARIANTS = {
    "MNQ": {"PV": 2.0, "COMM_RT": 3.0, "usd_scale": 1.0},
    "NQ_same_usd": {"PV": 20.0, "COMM_RT": 5.0, "usd_scale": 1.0},
    "NQ_scaled_usd": {"PV": 20.0, "COMM_RT": 5.0, "usd_scale": 10.0},
}


def main() -> None:
    m1, m5 = bt.load_bars(os.path.join(ROOT, "data", "MNQ_1m.csv"))
    b = bt.featured(m5)
    sig = pd.concat([bt.ensemble_signals(m5)] + list(bt.addon_rows(b).values()))
    out = {}
    for name, v in VARIANTS.items():
        bt.PV, bt.COMM_RT = v["PV"], v["COMM_RT"]
        for k in USD_KEYS:
            setattr(bt, k, BASE[k] * v["usd_scale"])
        tr = bt.in_window(bt.simulate(m1, m5.index, sig, bt.Rules(overnight_sleeves="all")))
        tot, per = bt.stats(tr), bt.by_period(tr)
        worst_day = min(pd.Series([t.pnl for t in tr], index=[t.exit_time.tz_convert(bt.ET).date() for t in tr])
                        .groupby(level=0).sum())
        out[name] = {"total": tot, "periods": per, "worst_day": round(float(worst_day), 1),
                     "by_tag": bt.by_key(tr, lambda t: t.tag)}
        print(f"\n=== {name}  TOTAL {bt.fmt(tot)}  worst day ${worst_day:,.0f}")
        for k, s in per.items():
            print(f"  {k:15s} {bt.fmt(s)}")
    for k, val in BASE.items():
        setattr(bt, k, val)
    path = os.path.join(ROOT, "data", "revamp_nq_vs_mnq.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
