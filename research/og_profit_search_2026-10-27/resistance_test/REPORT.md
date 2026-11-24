# REPORT - resistance/support take-profit vs fixed 3R for the revamp BREAKOUT sleeve (backtest only)

Nothing live was touched: no bot, launcher or order. All outputs are in this folder. Workers ran at below-normal priority, 3 processes.

## VERDICT: NO. No resistance-based target beats the fixed 3R target robustly. Keep 3R.
- Best total result: day-levels (+$235, +0.8%) and swing-50 (+$116, +0.4%). Both are inside noise. Each is better in one window and worse in the other (day: -$112 train / +$347 2026; swing-50: +$587 train / -$471 2026). Day-levels is worse in walk-forward OOS (-$923, 4/6 folds better but the one bad fold is big) and a paired day-resample says the gain is a coin flip (P(gain) 0.54-0.56, 5th-95th pct about -$1.6k..+$2.1k and -$3.3k..+$4.0k). That is noise, not an edge.
- Round-number targets (50/100 pts) clearly hurt: -$10.2k (50) and -$5.1k (100). They cap the winners at a nearby level on almost every trade and lose in train and 2026. The 50-pt version loses in all 6 walk-forward folds.
- Swing-100 loses $1.3k. Swing-20 never fires, because the breakout is a 20-bar Donchian break, so there is no 20-bar swing high above a long breakout by construction.
- Half off at the level and the rest to 3R: every one of the 6 partial variants is at or below baseline (-$121 to -$4.0k). The partials slightly lower drawdown at best (day: DD 4,398 vs 4,449, round50: 4,301), but that costs net profit.
- Why: resistance only matters rarely here. A breakout by definition has just cleared the 20-bar high, so a prior swing high within 1R-3R above entry exists on only 4% (swing-50) to 13% (swing-100) of breakout trades, and day/session levels on about 37%. When a level does exist, the price often runs through it (breakouts are momentum trades), and trimming the target gives up more in the runners than it saves in reversals. Round numbers are always nearby, so they simply cut the average winner.

## Method (same engine and costs as candidate A)
- Baseline = candidate A: revamp stack + REVAMP_BREAKOUT_TARGET_R=3.0, 24/7, lock 100/20/150/0.5, 1 MNQ, max 3 open, $4 round trip + 1 tick slippage each side, Nov 2024 - Sep 21 2026 (data MNQ_1m cleaned). The harness reproduces A exactly (2,473 trades, +$29,313.3, PF 1.22, DD $4,449), so the comparison is like for like. The bot's own code and files were not changed; the target override lives only in this folder's rt_engine.py.
- Only breakout-sleeve entries (tag ens_breakout; 3,233 raw signals) get a new target; trend, mean-reversion and add-ons are identical. All other rules (stops, lock, cooldowns, loss limits) unchanged.
- Rule: R = |signal close - stop|. Long target = nearest level above entry that lies between +1R and +3R, set 1 tick inside the level (never below +1R). If no level is in that band the target stays 3R. Shorts mirrored with support. A level closer than 1R is ignored.
- Levels use only bars closed before entry: (2) 5m fractal pivots (2 bars each side, confirmed 2 bars later) within the last 20/50/100 bars; (3) prior-day RTH high/low, prior full-session high/low, current overnight high/low, current session running high/low; (4) multiples of 50 and of 100 MNQ points; (5) half off at the same level, remainder to 3R (same lock/stop for the remainder).
- Walk-forward: no parameter is fitted (fixed rules), so the same six 3-month OOS test windows as the validation report are just sliced by entry date. Paired bootstrap = resample days 2,000 times on the daily P&L difference vs baseline.

## Table (net of costs; delta columns vs fixed 3R; green months counted by entry month)
| variant | trades | net $ | PF | win% | maxDD | green mo | train (to 2025) | 2026 | d vs 3R (all) | d train | d 2026 | WF OOS d (folds better) | BO-sleeve trades w/ level | BO sleeve net (d) | paired day-boot p5..p95 d / P(gain) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2473 | +29,313 | 1.22 | 61.0 | 4,449 | 19/24 | +17,361 | +11,952 | +0 | +0 | +0 | +0 (0/6) | 0/1823 | +22,838 (+0) | - |
| swing20 | 2473 | +29,313 | 1.22 | 61.0 | 4,449 | 19/24 | +17,361 | +11,952 | +0 | +0 | +0 | +0 (0/6) | 0/1823 | +22,838 (+0) | +0..+0 / 0.0 |
| swing50 | 2475 | +29,429 | 1.22 | 61.0 | 4,449 | 19/24 | +17,948 | +11,481 | +116 | +587 | -471 | +52 (3/6) | 78/1825 | +22,978 (+140) | -1644..+2093 / 0.542 |
| swing100 | 2474 | +27,994 | 1.21 | 60.9 | 4,449 | 18/24 | +16,761 | +11,233 | -1,319 | -600 | -719 | -1,895 (3/6) | 236/1824 | +21,512 (-1,326) | -4471..+1904 / 0.256 |
| day | 2486 | +29,548 | 1.22 | 61.1 | 4,472 | 18/24 | +17,249 | +12,299 | +235 | -112 | +347 | -923 (4/6) | 679/1832 | +23,288 (+450) | -3272..+3988 / 0.559 |
| round50 | 2548 | +19,068 | 1.14 | 60.6 | 4,241 | 19/24 | +10,883 | +8,185 | -10,245 | -6,478 | -3,767 | -5,131 (0/6) | 1883/1887 | +13,667 (-9,171) | -19576..-1654 / 0.025 |
| round100 | 2532 | +24,227 | 1.18 | 60.7 | 4,468 | 19/24 | +15,302 | +8,925 | -5,086 | -2,059 | -3,027 | -2,276 (2/6) | 1778/1875 | +17,710 (-5,128) | -13488..+2842 / 0.155 |
| part_swing20 | 2473 | +29,313 | 1.22 | 61.0 | 4,449 | 19/24 | +17,361 | +11,952 | +0 | +0 | +0 | +0 (0/6) | 0/1823 | +22,838 (+0) | +0..+0 / 0.0 |
| part_swing50 | 2473 | +29,192 | 1.22 | 61.0 | 4,449 | 19/24 | +17,166 | +12,026 | -121 | -195 | +74 | -113 (2/6) | 78/1823 | +22,716 (-122) | -693..+438 / 0.402 |
| part_swing100 | 2473 | +28,760 | 1.22 | 61.0 | 4,449 | 19/24 | +16,584 | +12,175 | -553 | -777 | +223 | -655 (3/6) | 233/1823 | +22,284 (-554) | -1724..+439 / 0.229 |
| part_day | 2473 | +29,084 | 1.22 | 61.0 | 4,398 | 19/24 | +17,064 | +12,020 | -229 | -297 | +68 | -452 (4/6) | 675/1823 | +22,608 (-230) | -1457..+1030 / 0.414 |
| part_round50 | 2475 | +25,266 | 1.19 | 61.1 | 4,301 | 19/24 | +14,031 | +11,234 | -4,047 | -3,330 | -718 | -1,453 (2/6) | 1821/1825 | +18,790 (-4,048) | -7065..-1096 / 0.008 |
| part_round100 | 2473 | +26,744 | 1.2 | 61.0 | 4,340 | 19/24 | +15,443 | +11,300 | -2,569 | -1,918 | -652 | -521 (3/6) | 1731/1823 | +20,268 (-2,570) | -5251..+93 / 0.056 |

Walk-forward folds (OOS test windows, entry date): 2025-05-01..2025-08-01, 2025-08-01..2025-11-01, 2025-11-01..2026-02-01, 2026-02-01..2026-05-01, 2026-05-01..2026-08-01, 2026-08-01..2026-10-01

| variant | fold1 | fold2 | fold3 | fold4 | fold5 | fold6 |
|---|---|---|---|---|---|---|
| base | +264 | +7,410 | +3,282 | +5,699 | +5,352 | -1,221 |
| swing20 | +264 | +7,410 | +3,282 | +5,699 | +5,352 | -1,221 |
| swing50 | -188 | +7,971 | +3,687 | +5,938 | +4,651 | -1,221 |
| swing100 | -1,650 | +8,231 | +3,169 | +5,752 | +4,536 | -1,147 |
| day | -1,482 | +7,980 | +3,595 | +5,363 | +5,447 | -1,040 |
| round50 | +175 | +5,469 | +1,929 | +4,469 | +5,170 | -1,557 |
| round100 | -708 | +7,446 | +3,707 | +4,485 | +5,193 | -1,613 |
| part_swing20 | +264 | +7,410 | +3,282 | +5,699 | +5,352 | -1,221 |
| part_swing50 | +110 | +7,307 | +3,348 | +5,826 | +5,303 | -1,221 |
| part_swing100 | -324 | +7,438 | +2,948 | +5,999 | +5,246 | -1,176 |
| part_day | -366 | +7,542 | +3,318 | +5,642 | +5,407 | -1,209 |
| part_round50 | +46 | +6,849 | +2,683 | +5,517 | +5,382 | -1,144 |
| part_round100 | -152 | +7,526 | +3,326 | +5,645 | +5,133 | -1,213 |

## Caveats
1. Partial variants model two legs of 0.5 contract (commission $2 each leg, same per-contract cost). Real MNQ has no half contract: it needs 2 MNQ (double exposure and the lock/risk caps would behave differently) so partials are only an approximation.
2. Changing exits reorders which later signals are allowed (cooldowns, loss limits, max-3, in-profit rule), so non-breakout trades also shift slightly; the table shows total and breakout-sleeve net.
3. TP fills are modelled at the limit price (as in the validation); a level target placed 1 tick inside the level is more likely to fill than a 3R target in real markets, which would make level targets look a bit better than they should in a backtest, and they still did not win.
4. Level definitions are one reasonable choice each (fractal pivots, 1R floor, 1-tick buffer). Other definitions could differ; I did not tune them, to avoid curve-fitting. Results are from the same data and period as the original 3R selection (2026 is not unseen).
5. Data ends 2026-09-21. The last walk-forward fold has only 3 weeks and is negative for every variant including baseline.
6. Baseline green months here is 19/24 (grouped by entry month); the validation report shows 18/24 (grouped by exit month). Same trades.

## Files
- rt_engine.py (variants, simulator copy with partial legs, levels), rt_analyze.py (tables, walk-forward, bootstrap)
- results_runs.jsonl (summary per variant), analysis.json, REPORT_TABLE.md, trades_<variant>.csv (per-position trades)