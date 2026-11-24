# Professional-priority check, 2026-11-07

No family qualified. The cell chosen on 2025 had to be positive in 2026 H1, positive from July 1 to September 21, positive in at least 4 of 6 walk-forward folds, and still positive in both 2026 windows at 3 ticks of slippage. None were.

Order-flow absorption was not tested. The MNQ files, including the Databento 1-second bars, are open/high/low/close/volume only. There is no bid, ask, or aggressor side, so delta and absorption cannot be computed.

| Family | Chosen on 2025 | 2025 | 2026 H1 | Jul–Sep 21 | Walk-forward | Last 6 weeks |
|---|---|---:|---:|---:|---:|---:|
| 15-min ORB | 2R, no filter | +8,844 | −828 | −2,872 | +856 (3/6) | −1,349 |
| Initial balance | close outside, 1-range target | −1,266 | +839 | −73 | +160 (3/6) | +1,180 |
| Failed auction | prior-day high/low | +693 | −157 | −22 | +3,546 (3/6) | −820 |
| VWAP trend hold | the one rule | +1,600 | −629 | −793 | −74 (3/6) | +479 |
| Overnight inventory | fade the overnight move | +2,025 | −3,390 | −870 | −7,229 (1/6) | −1,095 |

The 15-minute breakout is the same pattern as the earlier 192-variant test. It made money in 2025 (profit factor 1.34) and lost it in 2026. Adding the 60-minute trend, skipping a large opening range, and stopping new entries at 11:00 did not repair it: that version was about flat in 2025 (−$550 at 2R) and lost in both 2026 windows.

The $150 risk cap, 2-trade limit, and −$300 day stop were applied after the fact. On the 2R breakout they turned the 2025 result from +$8,844 into −$124, because the stop is the other side of the opening range and that is usually more than 75 points ($150) on MNQ.

One cell the 2025 rule did not pick, the overnight-high/low failed auction, was positive in both 2026 windows (+$2,483 then +$2,107) and then lost $1,679 in the six-week holdout. It lost money in 2025. That is one window, not an edge.

Reproduce with `venv\Scripts\python.exe research\pro_priority_2026-11-07\run.py`.
