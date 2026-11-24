# REPORT_VALIDATION (backtest only)

Cost model: $4 round trip per contract + 1 tick slippage each side (stops slip 1 tick; overnight same), 1 MNQ, next-1m-open + 1 tick fills, stop-first on same-bar touches, max 3 open, session loss limits. Data MNQ_1m cleaned of roll spikes; trades Nov-2024 .. 2026-09-21. Candidates: **A** = revamp stack + REVAMP_BREAKOUT_TARGET_R=3.0; **D** = A with overnight sleeves dropped (day-only; ONL + 6PM add-ons kept); **D2** = D without 6PM add-on; base = current revamp live (2.0R).

## 1. Headline (all data)

| run | net | maxDD | PF | trades | win | avg/trade |
|---|---|---|---|---|---|---|
| A | +29,313 | 4,449 | 1.22 | 2473 | 61.0% | 11.9 |
| D | +15,337 | 3,380 | 1.28 | 1033 | 62.2% | 14.8 |
| D2 | +14,880 | 2,872 | 1.3 | 863 | 66.5% | 17.2 |
| base | +25,686 | 4,629 | 1.19 | 2518 | 60.7% | 10.2 |

## 2. Yearly / monthly / drawdown shape

| run | 2024 | 2025 | 2026 | green months | worst month | worst day | worst week | longest DD (days) | still in DD (days) | green weeks |
|---|---|---|---|---|---|---|---|---|---|---|
| A | +4,517 | +13,075 | +11,721 | 18/24 | -2,794 | -1,024 | -1,783 | 85 | 51 | 61/99 |
| D | +1,595 | +8,679 | +5,062 | 18/23 | -944 | -620 | -872 | 118 | 45 | 54/98 |
| D2 | +1,623 | +9,050 | +4,206 | 17/23 | -1,081 | -620 | -878 | 117 | 39 | 53/98 |
| base | +5,529 | +9,798 | +10,359 | 18/24 | -2,570 | -1,024 | -1,783 | 190 | 51 | 61/99 |

Monthly PnL (ET, by exit):

| month | A | D | D2 | base |
|---|---|---|---|---|
| 2024-10 | +123 | +0 | +0 | +123 |
| 2024-11 | +1,280 | +944 | +880 | +2,114 |
| 2024-12 | +3,114 | +651 | +743 | +3,292 |
| 2025-01 | +3,713 | +848 | +860 | +2,944 |
| 2025-02 | -2,794 | +914 | +947 | -2,570 |
| 2025-03 | +1,125 | +454 | +420 | +643 |
| 2025-04 | +1,966 | +1,896 | +2,028 | +1,702 |
| 2025-05 | -1,026 | +295 | -153 | -884 |
| 2025-06 | +542 | -944 | -1,081 | +512 |
| 2025-07 | +619 | +1,199 | +1,397 | -94 |
| 2025-08 | +2,246 | +1,487 | +1,620 | +1,784 |
| 2025-09 | +3,575 | +2,083 | +2,023 | +3,186 |
| 2025-10 | +1,717 | +543 | +930 | +1,828 |
| 2025-11 | +2,461 | +162 | +155 | +2,267 |
| 2025-12 | -1,071 | -257 | -95 | -1,520 |
| 2026-01 | +1,892 | -774 | -463 | +954 |
| 2026-02 | -637 | +1,744 | +1,613 | -476 |
| 2026-03 | +2,439 | -367 | -632 | +2,354 |
| 2026-04 | +3,897 | +1,516 | +1,392 | +2,201 |
| 2026-05 | +500 | +357 | +391 | +1,162 |
| 2026-06 | +1,864 | +788 | +56 | +1,333 |
| 2026-07 | +2,988 | +1,981 | +2,126 | +3,812 |
| 2026-08 | -1,207 | +24 | +79 | -1,164 |
| 2026-09 | -15 | -206 | -355 | +184 |

## 3. Walk-forward (6-month train / 3-month out-of-sample test, rolled every 3 months)

Fixed params = A, D, base run once over full history and sliced by entry date (signals have no fitted parameters, so the only look-ahead is the choice of params). Re-select = on each train window pick the best of the grid (D, A, base, D2, g_A_notrend, g_A_norevamp, g_A_no0009, g_A_tight2, g_D_notrend, g_A_wide, g_base_notrend, g_bo25) by net - 0.75*maxDD, then score it on the next 3 months.

| train window | test window | A OOS net (DD) | D OOS net (DD) | base OOS net (DD) | re-selected pick (train net) | pick OOS net (DD) |
|---|---|---|---|---|---|---|
| 2024-11-01..2025-05-01 | 2025-05-01..2025-08-01 | +264 (2,276) | +550 (1,517) | -338 (2,319) | g_A_wide (+17,097) | -891 (3,620) |
| 2025-02-01..2025-08-01 | 2025-08-01..2025-11-01 | +7,410 (1,006) | +4,113 (854) | +6,670 (1,167) | g_D_notrend (+3,925) | +3,671 (854) |
| 2025-05-01..2025-11-01 | 2025-11-01..2026-02-01 | +3,282 (3,706) | -869 (3,380) | +1,701 (4,629) | g_A_no0009 (+7,708) | +2,503 (3,500) |
| 2025-08-01..2026-02-01 | 2026-02-01..2026-05-01 | +5,699 (3,619) | +2,893 (1,740) | +4,078 (3,164) | A (+10,693) | +5,699 (3,619) |
| 2025-11-01..2026-05-01 | 2026-05-01..2026-08-01 | +5,352 (2,145) | +3,125 (1,446) | +6,307 (2,044) | g_A_norevamp (+12,813) | +1,252 (2,123) |
| 2026-02-01..2026-08-01 | 2026-08-01..2026-10-01 | -1,221 (3,572) | -182 (1,134) | -980 (3,449) | g_A_norevamp (+13,030) | -1,224 (3,395) |

Total OOS net: A: +20,786, D: +9,630, base: +17,438, reselect: +11,010  | OOS folds positive: A 5/6, D 4/6, base 4/6, reselect 4/6

## 4. Stress tests

| spec | net | maxDD | PF | trades | train net | 2026 net |
|---|---|---|---|---|---|---|
| A (reference) | +29,313 | 4,449 | 1.22 | 2473 | | |
| s_A_slip2 | +27,612.8 | 4,604.6 | 1.21 | 2467 | +15,977.7 | +11,635.1 |
| s_A_slip3 | +25,477.1 | 4,761.1 | 1.19 | 2458 | +14,332.5 | +11,144.5 |
| s_A_tpthru | +29,466.3 | 4,448.6 | 1.22 | 2472 | +17,529.7 | +11,936.6 |
| s_A_tpthru_slip2 | +27,758.3 | 4,604.6 | 1.21 | 2467 | +16,138.2 | +11,620.1 |
| s_A_ovnslip2 | +27,117.1 | 4,666.6 | 1.2 | 2463 | +16,120.0 | +10,997.0 |
| s_A_bo24 | +25,623.6 | 4,205.0 | 1.19 | 2494 | +14,623.9 | +10,999.7 |
| s_A_bo36 | +29,821.9 | 4,349.7 | 1.23 | 2468 | +17,824.0 | +11,997.9 |
| s_A_trend12 | +30,091.1 | 4,119.3 | 1.23 | 2473 | +18,314.2 | +11,776.9 |
| s_A_trend18 | +29,326.7 | 4,595.9 | 1.22 | 2491 | +16,895.0 | +12,431.7 |
| s_A_lock_m20 | +25,001.3 | 5,003.3 | 1.2 | 2655 | +12,792.7 | +12,208.6 |
| s_A_lock_p20 | +27,348.4 | 6,292.0 | 1.19 | 2365 | +18,931.5 | +8,416.9 |
| s_A_comm6 | +24,296.8 | 4,830.6 | 1.18 | 2472 | +14,387.2 | +9,909.6 |
| s_A_tpthru4 | +28,315.0 | 4,448.6 | 1.22 | 2471 | +16,597.4 | +11,717.6 |
| s_A_tpthru2 | +29,262.8 | 4,448.6 | 1.22 | 2472 | +17,529.7 | +11,733.1 |
| s_A_tpthru2_slip2 | +27,554.3 | 4,604.6 | 1.21 | 2467 | +16,138.2 | +11,416.1 |
| x_A_early | +31,665.8 | 4,448.6 | 1.23 | 2595 | +19,713.7 | +11,952.1 |
| D (reference) | +15,337 | 3,380 | 1.28 | 1033 | | |
| s_D_slip2 | +14,462.1 | 3,472.8 | 1.26 | 1029 | +9,287.1 | +5,175.0 |
| s_D_slip3 | +13,469.6 | 3,568.3 | 1.24 | 1028 | +8,340.1 | +5,129.5 |
| s_D_tpthru | +15,321.1 | 3,379.8 | 1.28 | 1032 | +10,379.6 | +4,941.5 |
| s_D_tpthru_slip2 | +14,447.1 | 3,472.8 | 1.26 | 1028 | +9,287.1 | +5,160.0 |
| s_D_ovnslip2 | +15,031.6 | 3,416.8 | 1.27 | 1033 | +10,188.6 | +4,843.0 |
| s_D_bo24 | +16,021.8 | 3,514.4 | 1.29 | 1038 | +10,331.8 | +5,690.0 |
| s_D_bo36 | +13,894.5 | 3,379.8 | 1.25 | 1035 | +9,185.2 | +4,709.3 |
| s_D_trend12 | +15,958.4 | 3,409.3 | 1.29 | 1035 | +10,956.2 | +5,002.2 |
| s_D_trend18 | +15,166.7 | 3,267.8 | 1.27 | 1034 | +10,269.1 | +4,897.7 |
| s_D_lock_m20 | +13,176.8 | 2,064.6 | 1.26 | 1071 | +8,357.4 | +4,819.4 |
| s_D_lock_p20 | +13,250.9 | 3,952.3 | 1.21 | 1007 | +7,460.9 | +5,790.0 |
| s_D_comm6 | +13,270.6 | 3,603.8 | 1.24 | 1033 | +9,077.6 | +4,193.0 |
| s_D_tpthru2 | +15,321.1 | 3,379.8 | 1.28 | 1032 | +10,379.6 | +4,941.5 |
| s_D_tpthru4 | +15,078.1 | 3,379.8 | 1.27 | 1032 | +10,136.6 | +4,941.5 |
| s_D_tpthru2_slip2 | +14,447.1 | 3,472.8 | 1.26 | 1028 | +9,287.1 | +5,160.0 |
| x_D_early | +16,978.9 | 3,379.8 | 1.29 | 1102 | +12,021.9 | +4,957.0 |

slip2/slip3 = 2x/3x slippage; x_*_early = same params but starting 2024-09-25 (all data); tpthru2/tpthru4 = TP needs 2/4 ticks through; tpthru = take-profit fills only if price trades 1 tick THROUGH the limit; ovnslip2 = +2 ticks extra on overnight fills; bo24/bo36 = breakout target 3.0R -/+20%; trend12/18 = trend target 1.5R -/+20%; lock_m20/p20 = $ profit-lock thresholds -/+20%; comm6 = $6 round trip. The 4.0 ATR trail is applied inside the strategy engine and was not re-run at +/-20% (see caveats).

## 5. Bootstrap (500 runs) and account sizing

| run | method | net p5 | net p50 | net p95 | maxDD p50 | maxDD p95 | maxDD worst | P(net<0) |
|---|---|---|---|---|---|---|---|---|
| A | random 10% of trades removed | +22,401 | +26,588 | +30,305 | 4,208 | 5,042 | 5,829 | 0.0% |
| A | resample trading days with replacement | +13,632 | +29,646 | +45,892 | 4,881 | 8,052 | 13,823 | 0.0% |
| D | random 10% of trades removed | +11,099 | +13,866 | +16,403 | 3,018 | 3,742 | 4,459 | 0.0% |
| D | resample trading days with replacement | +5,776 | +15,051 | +24,856 | 3,134 | 5,366 | 8,125 | 0.6% |
| D2 | random 10% of trades removed | +10,779 | +13,401 | +15,993 | 2,621 | 3,282 | 3,758 | 0.0% |
| D2 | resample trading days with replacement | +4,860 | +14,718 | +24,926 | 3,064 | 5,371 | 9,325 | 0.2% |

Account sizing for 1 MNQ (cushion = 1.25 x the larger of observed max DD and bootstrap p95 DD, rounded up to $500):

| run | observed maxDD | worst bootstrap p95 DD | suggested min account | equity at bootstrap net p5 (from that start) |
|---|---|---|---|---|
| A | 4,449 | 8,052 | $10,500 | $24,132 (p5 net +13,632) |
| D | 3,380 | 5,366 | $7,000 | $12,776 (p5 net +5,776) |
| D2 | 2,872 | 5,371 | $7,000 | $11,860 (p5 net +4,860) |