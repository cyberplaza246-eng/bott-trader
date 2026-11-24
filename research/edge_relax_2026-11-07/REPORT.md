# Edge search with the $2,000 cap removed, 2026-11-07

No new candidate. Five pre-registered families were scored on 2025 (train), 2026 H1 (validate), July–Sep 21 2026 (test), and a walk-forward. The Sep 22–Nov 6 holdout was reported and not used to pick. Costs are $4 round trip plus 1 tick per side, 1 MNQ. A long-only idea also had to beat simply being long during the cash session (09:30–15:55). None did.

| Family | Picked on 2025 | 2025 | 2026 H1 | Jul–Sep 21 | Walk-forward | Holdout | Drawdown, 2026 H1+summer |
|---|---|---:|---:|---:|---:|---:|---:|
| Overnight long, stop chosen from 80/150/300 | 80 pt stop | +11,397 | −82 | +2,600 | +15,230 (5/6) | −220 | 5,751 |
| Overnight, up-day or down-day filter | up-day, 150 pt | +9,281 | −1,451 | +2,913 | +7,556 (5/6) | −1,576 | 4,448 |
| Long from 18:00, 150 pt stop | that one cell | +466 | −2,074 | +3,940 | +3,597 (4/6) | −3,321 | 6,117 |
| 10/20-day momentum, long and short | 20-day | +8,337 | +2,373 | −3,385 | +3,146 (4/6) | +677 | 7,773 |
| Same momentum, long only | 20-day | +3,084 | +2,316 | −1,877 | +1,834 (4/6) | −775 | 5,709 |

The overnight long is the bull market, not a separate edge. With the 80-point stop it made +$11.4k in 2025 (t-stat 2.5, profit factor 1.54, drawdown $2.6k), then the first quarter of 2026 was −$4.6k and the first half of 2026 was flat. Over 2026 H1 plus the summer it made +$2.5k, less than the cash-session long control (+$3.0k) over the same dates. The 300-point stop was positive in both 2026 windows (+$2.2k then +$4.0k) and then lost $3.6k in the six-week holdout. The stop width that looks best changes with the window.

Daily momentum made money in 2025 and in 2026 H1, then lost it in the summer test. Over that out-of-sample stretch it did worse than being long every cash session.

The setup that still has the strongest evidence is the one already tested, with this cap removed: the live revamp with the breakout target at 3R. Through 2026-09-21 that was +$29.3k, profit factor 1.22, max drawdown $4.4k, 5 of 6 walk-forward folds positive. The unseen Sep 22–Nov 6 window was about flat (+$39). That drawdown is the "little drawdown" this book actually has. Nothing in this pass beat it out of sample.

Reproduce with `venv\Scripts\python.exe research\edge_relax_2026-11-07\run.py`.
