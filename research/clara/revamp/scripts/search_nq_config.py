"""Search NQ settings for the live revamp stack that keep drawdown survivable.

1 NQ = $20/pt, $5 round trip. Profit lock stays the tested MNQ rule in points (x10 in $).
Grid:
  max stop risk ($ per NQ)   300 / 500 / 800       (15 / 25 / 40 pts)
  max open trades            1 / 3
  sessions                   day-only (no overnight main, no 6 PM add-on) / all (overnight all sleeves)
  session loss limits ($)    day 1000 + ovn 600  /  day 2000 + ovn 1200
Writes data/nq_config_search.json, sorted by net / max drawdown.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

import scripts.backtest_revamp_live_stack as bt  # noqa: E402

GRID = list(itertools.product((300.0, 500.0, 800.0), (1, 3), ("day_only", "all"), ((1000.0, 600.0), (2000.0, 1200.0))))
_CACHE: dict = {}


def _inputs():
    if not _CACHE:
        m1, m5 = bt.load_bars(os.path.join(ROOT, "data", "MNQ_1m.csv"))
        b = bt.featured(m5)
        sig = pd.concat([bt.ensemble_signals(m5)] + list(bt.addon_rows(b).values()))
        _CACHE.update(m1=m1, idx=m5.index, sig=sig)
    return _CACHE


def run(cfg):
    risk, max_open, sess, (day_loss, ovn_loss) = cfg
    bt.PV, bt.COMM_RT = 20.0, 5.0
    bt.LOCK_ACTIVATE, bt.LOCK_KEEP, bt.LOCK_TRAIL_START = 1000.0, 200.0, 1500.0
    bt.MAX_RISK_USD, bt.DAY_LOSS, bt.OVN_LOSS, bt.DAILY_TARGET = risk, day_loss, ovn_loss, 0.0
    if sess == "day_only":
        rules = bt.Rules(rth_main_only=True, disabled_tags=("addon_e1",), max_open=max_open)
    else:
        rules = bt.Rules(overnight_sleeves="all", max_open=max_open)
    d = _inputs()
    tr = bt.in_window(bt.simulate(d["m1"], d["idx"], d["sig"], rules))
    tot = bt.stats(tr)
    per = bt.by_period(tr)
    daily = pd.Series([t.pnl for t in tr], index=[t.exit_time.tz_convert(bt.ET).date() for t in tr]).groupby(level=0).sum()
    return {"max_risk_usd": risk, "max_open": max_open, "sessions": sess, "day_loss": day_loss, "ovn_loss": ovn_loss,
            "total": tot, "periods": {k: v["net"] for k, v in per.items()},
            "periods_positive": sum(v["net"] > 0 for v in per.values()),
            "worst_day": round(float(daily.min()), 1) if len(daily) else 0.0,
            "losing_days_pct": round(100 * float((daily < 0).mean()), 1) if len(daily) else 0.0,
            "net_per_dd": round(tot["net"] / tot["dd"], 2) if tot["dd"] else 0.0,
            "by_tag": bt.by_key(tr, lambda t: t.tag)}


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(run, GRID))
    res.sort(key=lambda r: r["net_per_dd"], reverse=True)
    print(f"{'risk':>5} {'open':>4} {'sess':>8} {'dayL':>5} {'ovnL':>5} | {'n':>5} {'net':>9} {'PF':>5} {'DD':>8} "
          f"{'worstDay':>9} {'net/DD':>6} per+  periods")
    for r in res:
        t = r["total"]
        print(f"{r['max_risk_usd']:5.0f} {r['max_open']:4d} {r['sessions']:>8} {r['day_loss']:5.0f} {r['ovn_loss']:5.0f} | "
              f"{t['n']:5d} {t['net']:9.0f} {t['pf']:5.2f} {t['dd']:8.0f} {r['worst_day']:9.0f} {r['net_per_dd']:6.2f} "
              f"{r['periods_positive']}/4  {[round(v) for v in r['periods'].values()]}")
    path = os.path.join(ROOT, "data", "nq_config_search.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
