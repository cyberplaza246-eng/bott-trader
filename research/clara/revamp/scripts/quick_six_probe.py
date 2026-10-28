#!/usr/bin/env python3
"""Quick probes for 6 trades/day configs."""
import os, sys
from collections import Counter
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth

df = official.load_1m("data/MNQ_1m.csv")
ha = hunt.prepare(df)
a = combo.prepare(df)
end = df["datetime"].iloc[-1].tz_convert("America/New_York").date()
JUNE = pd.Timestamp("2026-06-01", tz="UTC")


def entry_ts(x):
    ts = pd.Timestamp(x.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def report(name, trades):
    t = [x for x in trades if entry_ts(x) >= JUNE]
    days = Counter(entry_ts(x).tz_convert("America/New_York").date() for x in t)
    cal = 0
    d = date(2026, 6, 1)
    while d <= end:
        if d.weekday() < 5:
            cal += 1
        d += timedelta(days=1)
    avg = len(t) / cal if cal else 0
    m = resp.group_metrics(t)
    ge6 = sum(1 for c in days.values() if c >= 6)
    print(
        f"{name}: n={len(t)} avg/wd={avg:.2f} max={max(days.values()) if days else 0} "
        f"days>=6={ge6}/{cal} PF={m['profit_factor']:.2f} net=${m['total_pnl']:.0f}"
    )


lc = dict(hunt.LONG_WINNER)
lc.update(win="w6_all", entry="quiet", max_day=12, max_lots=2)
sc = dict(combo.SHORT_RECIPE)
sc.update(win="w6_all", entry="quiet", max_day=12, max_lots=2)
sc.pop("structure", None)
report("stack w6 quiet x2 md12 no struct", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc["entry"] = "1m_pullback"
sc["entry"] = "1m_pullback"
sc["structure"] = "lh_ll"
report("stack w6 pullback x2 md12 lh_ll", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(max_day=99, max_lots=2, entry="quiet")
sc.update(max_day=99, max_lots=2, entry="quiet")
sc.pop("structure", None)
report("stack w6 quiet x2 md99", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(max_lots=1)
sc.update(max_lots=1)
report("stack w6 quiet x1 md99", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

combo.LONG_RECIPE = dict(combo.LONG_RECIPE_W6)
combo.LONG_RECIPE["max_day"] = None
combo.LONG_RECIPE["entry"] = "quiet"
combo.SHORT_RECIPE = dict(combo.SHORT_RECIPE)
combo.SHORT_RECIPE["max_day"] = None
combo.SHORT_RECIPE["entry"] = "quiet"
combo.SHORT_RECIPE.pop("structure", None)
report("combo w6 quiet unlimited no struct", combo.simulate_combined(a))

combo.LONG_RECIPE["entry"] = "quiet"
combo.SHORT_RECIPE["entry"] = "1m_pullback"
combo.SHORT_RECIPE["structure"] = "lh_ll"
report("combo w6 long quiet short pullback", combo.simulate_combined(a))

# More probes toward 6/day + edge
lc = dict(hunt.LONG_WINNER)
lc.update(win="w6_all", entry="quiet", max_day=12, max_lots=3)
sc = dict(combo.SHORT_RECIPE)
sc.update(win="w6_all", entry="1m_pullback", max_day=12, max_lots=3, structure="lh_ll")
report("stack quiet long x3 + pullback short x3 md12", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(win="w4_no_noon", entry="1m_pullback", max_lots=2)
sc.update(entry="1m_pullback", max_lots=2)
report("stack w4 long pullback + w6 short pullback x2 md12", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(win="w6_all", entry="quiet2", max_lots=2)
sc.update(entry="quiet2")
sc.pop("structure", None)
report("stack w6 quiet2 x2 md12", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(entry="quiet", max_lots=2)
sc.update(entry="1m_pullback", structure="lh_ll")
report("stack long quiet + short pullback w6 x2 md12", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(max_day=99, max_lots=2)
sc.update(max_day=99)
report("stack long quiet + short pullback w6 x2 md99", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(entry="1m_pullback", max_lots=2)
sc.update(entry="1m_pullback", structure="lh_ll")
report("stack w6 pullback both x2 md99", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(max_day=12)
sc.update(max_day=12)
report("stack w6 pullback both x2 md12", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(win="w4_no_noon", max_lots=2)
report("stack w4 long pullback + w6 short pullback x2 md12 (v2 windows)", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc.update(win="w6_all", max_lots=2, max_day=12)
sc.update(win="w6_all")
report("stack w6 pullback both x2 md12 w6", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

# asymmetric: profitable long recipe + aggressive short quiet
lc = dict(hunt.LONG_WINNER)
lc.update(win="w4_no_noon", entry="1m_pullback", max_day=99, max_lots=1)
sc = dict(combo.SHORT_RECIPE)
sc.update(win="w6_all", entry="quiet", max_day=12, max_lots=2)
sc.pop("structure", None)
report("stack v4 long pullback + short quiet x2", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc = dict(hunt.LONG_WINNER)
lc.update(win="w4_no_noon", entry="1m_pullback", max_day=99, max_lots=1)
sc = dict(combo.SHORT_RECIPE)
sc.update(win="w6_all", entry="quiet", max_day=12, max_lots=3)
sc.pop("structure", None)
report("v4 long + short quiet x3", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

sc["max_lots"] = 4
report("v4 long + short quiet x4", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))

lc["max_lots"] = 2
report("v4 long x2 + short quiet x3", hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc))
