"""Live MNQ revamp stack (overnight all sleeves) through the same 50k eval sim as hunt_nq_50k.py,
at 1/2/3/5 contracts (P&L and commission scale with size)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

import scripts.backtest_revamp_live_stack as bt  # noqa: E402
import scripts.hunt_nq_50k as h  # noqa: E402


def main() -> None:
    d = h._load()
    m1, idx = d["m1"], d["idx"]
    m5 = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    sig = pd.concat([bt.ensemble_signals(m5)] + list(bt.addon_rows(bt.featured(m5)).values()))
    tr = bt.in_window(bt.simulate(m1, idx, sig, bt.Rules(overnight_sleeves="all")))
    days = pd.DatetimeIndex(sorted({t.normalize().tz_localize(None) for t in idx.tz_convert(bt.ET) if t.weekday() < 5}))
    days = days[(days >= "2024-11-01") & (days < "2026-10-01")]
    cut = h.IS_END.tz_localize(None)
    daily1 = pd.Series([t.pnl for t in tr], index=[t.exit_time.tz_convert(bt.ET).normalize().tz_localize(None)
                                                   for t in tr], dtype=float).groupby(level=0).sum()
    print(f"1 MNQ: {bt.fmt(bt.stats(tr))}")
    for k in (1, 2, 3, 5):
        daily = daily1 * k
        eq = daily.reindex(days, fill_value=0.0).cumsum()
        eod_dd = float((eq.cummax() - eq).max())
        a, b = h.eval_sim(daily, days[days < cut]), h.eval_sim(daily, days[days >= cut])
        print(f"{k} MNQ  net {daily.sum():8.0f}  EOD DD {eod_dd:6.0f}  worst day {daily.min():6.0f} | "
              f"eval Nov24-Oct25 pass {a['pass_pct']:4.1f}% fail {a['fail_pct']:4.1f}% med {a['med_days']}d | "
              f"Nov25-Sep26 pass {b['pass_pct']:4.1f}% fail {b['fail_pct']:4.1f}% med {b['med_days']}d")


if __name__ == "__main__":
    main()
