# OG bot profit search - FINAL (2026-10-28 ~3:10 AM ET) - backtest only, nothing deployed, live bot untouched

## Verdict: NO OG configuration is profitable after costs. The revamp is still the only profitable candidate.
Data: MNQ 1m 2024-09-23..2026-09-21, full fallback-ON signal coverage (10 workers FINISHED, 70,409 signal bars). Costs: $4 RT/contract + 1 tick slip each side (+1 tick on stops). Train = through 2025, test = 2026.

| Config | Trades | Net | PF | Max DD | Train | 2026 | Green months |
|---|---|---|---|---|---|---|---|
| OG sweeps-only (fallback OFF, as live) | 211 | -2,780 | 0.79 | 4,484 | -1,914 | -867 | 12/25 |
| OG fallback ON | 12,596 | -41,755 | 0.92 | 42,310 | -25,747 | -16,008 | 4/25 |
| fallback only | 12,563 | -42,926 | 0.92 | 43,480 | -27,004 | -15,922 | 4/25 |
| best OG, sweeps + SL x1.25 | 210 | -586 | 0.95 | 2,848 | +182 | -768 | 13/25 |
| best fb-ON, lock off | 11,339 | -16,443 | 0.97 | 22,754 | -19,654 | +3,211 | 10/25 |
| sweeps + SL x1.25 + max 1 pos (only net-positive) | 149 | +376 | 1.04 | 1,708 | +875 | -499 | 13/25 |
| fb ON with ZERO costs (diagnostic) | 12,596 | +19,111 | 1.04 | 9,260 | +12,215 | +6,896 | 14/25 |
| fb ON 2-tick slip | 12,596 | -52,049 | 0.91 | 52,578 | | | |
| sweeps-only 2-tick slip | 211 | -3,046 | 0.77 | 4,700 | | | |

All 43 variants: out/results_og.jsonl (+ 4 stress: out/results_og_stress.jsonl), table out/og_table.txt.
- Earlier partial replay said sweeps-only -2,110 / 118 trades; with full coverage it is -2,780 / 211 trades.
- Fallback ON has a tiny gross edge (~+$1.5/trade before costs) that ~$5/trade of costs wipes out. Not fixable by lock, TP, SL, session, direction, regime, size, or position caps.
- The single +$376 variant is 149 trades, PF 1.04, and red in 2026: noise, not an edge.

## Compare: revamp candidates (validation/REPORT_VALIDATION.md)
- A: revamp + breakout 3R (24/7): +29,313, PF 1.22, DD 4,449, 2,473 trades; train +17.4k / 2026 +12.0k; positive in all stress tests.
- D2: day-only + 3R: +14,880, PF 1.30, DD 2,872, 863 trades (lower drawdown).
- Base live revamp (2R): +25,686, PF 1.19, DD 4,629.
Recommendation: retire the OG sweep logic as a profit source; if anything goes forward, it is revamp A (or D2 for lower DD), paper first, owner OK required.

Caveats: same as REPORT_VALIDATION (params tuned on same sample, TP fills at limit, stop-first intrabar, data ends 2026-09-21).
