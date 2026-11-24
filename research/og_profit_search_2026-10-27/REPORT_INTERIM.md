# OG / revamp profit search - INTERIM (2026-10-27 11:55 PM ET)
Backtest only. Costs: $4 RT/contract + 1 tick slip each side (+1 more tick on stops). Max-DD cap lifted per owner.
Data MNQ_1m 2024-09-22..2026-09-21 (revamp window Nov24-Sep26). Train = through 2025, test = 2026.

## Revamp stack (different strategy than the OG sweep bot) - results_p1.jsonl, results_descent.jsonl
baseline: n2518 net +25,686 PF1.19 DD4,629 | train +15,096 (DD4,311) | 2026 +10,590 (DD3,510)
no_revamp_exits (REVAMP_EXITS=0): +29,026 PF1.25 DD4,368 | train +16,058 | 2026 +12,968
bo_r=3.0 (breakout target 3R): +29,313 PF1.22 DD4,449 | train +17,361 | 2026 +11,952
drop daytime trend: +27,245 DD4,673 | train +14,969 | 2026 +12,276 (DD3,028)
drop overnight breakout 00-09 ET: +21,822 DD3,729 PF1.23 | train +15,500 (DD2,753) | 2026 +6,323  (train-only win)
day-only (all day sleeves): +17,375 PF1.36 DD3,066 | train +12,386 (DD1,842) | 2026 +4,988 (DD1,960)
wide lock (150/30/250): +33,008 DD8,507 | train +19,779 | 2026 +13,229
wide lock + bo_r 3.0: +39,475 PF1.26 DD7,284 | train +25,897 (DD5,865) | 2026 +13,578 (DD6,787); periods P1..P4 +17.1k +8.9k -0.0k +13.5k
lock off / tight: worse. max_open 1/2: much worse (+9.7k / +19.1k). 2-tick slippage baseline: +23,816 (still profitable).
Descent (rv_descent.py) still running; see out/descent_*.log.
## OG sweep bot (fallback ON/OFF)
Fallback-ON signal generation (fb_worker2, 10 procs) ~25% done newest-first; og_specs1.json (43 variants) ready, run with: python og_run.py og_specs1.json results_og.jsonl once out/fb_n*.prog show full coverage.
Partial-coverage smoke test only (NOT valid): fb ON on covered bars net -5.3k DD 8.2k. Earlier sweeps-only replay: -2,110 / 118 trades / PF0.70.
