# New-edge search, 2026-11-07

No candidate. Seven pre-registered strategies (16 parameter settings) were tested on MNQ at 1 contract. None passed the locked rules: positive validate, positive test, at least 4 of 6 walk-forward folds positive, still positive at 3 ticks of slippage, and no $2,000 mark-to-market drawdown on the combined in-sample window.

Parameters were chosen on 2025 only. Validate is 2026 H1, test is 2026-07-01 to 2026-09-21, holdout is 2026-09-22 to 2026-11-06. Costs are $4 round trip plus 1 tick per side ($2/point).

Selected-on-train results, net dollars, 1 tick:

| Strategy | Train | Validate | Test | Walk-forward | Holdout | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Opening-range breakout, 15 min, 2R | +1,617 | −4,018 | +628 | −4,891 (1/6) | −1,100 | 6,152 |
| VWAP reversion, k=1.5 | +1,010 | −2,269 | +679 | −1,569 (2/6) | +28 | 3,735 |
| Pullback, 1.5R | −4,170 | +1,105 | −189 | −3,564 (2/6) | −916 | 5,737 |
| Morning momentum, m=0.25 | +3,416 | +708 | −1,311 | −281 (2/6) | −1,496 | 2,990 |
| VWAP + range regime, k=1.5 | +692 | −397 | +451 | −520 (2/6) | −500 | 1,510 |
| Pullback + trend regime, 1.5R | −3,316 | −608 | +1,056 | −2,607 (1/6) | −588 | 4,793 |
| Momentum + trend regime, m=0.25 | +1,431 | +741 | −1,005 | −1,546 (1/6) | −854 | 2,361 |

The only setting that never breached $2,000 was VWAP reversion gated to range days. It still lost money in validate, walk-forward, and holdout (+$747 over 21 months, t-stat 0.44).

One setting the train rule did not select, pullback at 2R, was positive on both validate (+$1,502) and test (+$435). It lost $4,732 in 2025, lost on the holdout (−$758), and its in-sample drawdown was $5,708. It is not an edge.

Every cell, breach, and walk-forward fold is in `results/`. Reproduce with `venv\Scripts\python.exe research\new_edge_2026-11-07\run.py`. `sanity_checks.py` rechecked fills (next-bar open, stop before target), the 60-point stop cap, and the trade limits; those checks passed on 2026-11-07.

No live bot, `src`, launcher, or broker file was changed by this study.
