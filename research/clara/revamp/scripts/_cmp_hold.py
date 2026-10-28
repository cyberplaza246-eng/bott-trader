import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.backtest_ema15_official as o
import scripts.test_ema15_tp_modes as t
import scripts.hunt_runners_exits as h
import scripts.find_real_1m_edge as e
from scripts.hunt_runners_exits import in_range, metrics, JUNE

df = o.load_1m("data/MNQ_1m.csv")
ent = t.collect_official_entries(df)
a = t.simulate_same_entries(df, ent, "hold")
b = h.simulate_exit(df, ent, {"mode": "hold"})
print("official_hold_oos", e.summarize(e.split_is_oos(a)[1]))
print("ours_hold_oos", e.summarize(e.split_is_oos(b)[1]))
print("official_junsep", metrics(in_range(a, JUNE, None)))
print("ours_junsep", metrics(in_range(b, JUNE, None)))
print("len", len(a), len(b))
da = {(x.entry_time, x.direction): x.pnl for x in a}
db = {(x.entry_time, x.direction): x.pnl for x in b}
diffs = [(k, da[k], db[k]) for k in da if abs(da[k] - db.get(k, 0)) > 0.01]
print("diff_count", len(diffs))
print("sample", diffs[:8])
