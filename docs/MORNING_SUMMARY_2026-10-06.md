# bott-trader overnight research — MORNING SUMMARY
_generated 2026-10-06 00:55 EDT · 249 variants evaluated · paper/backtest only, not live_

Protocol: MNQ 5m, sprint-after engine (next-open fills, 1-tick slip both sides, gap-aware stops, flatten 16:00 ET, cooldown 3, daily halt 2%, max 10 ct, $50k @ 0.5% risk). Locked holdout ≥ 2025-09-12. WF = 180d train / 60d test, fixed params. **Selection used only in-sample + pre-holdout WF folds; holdout is confirmation.**

**Baseline (current ensemble):** holdout $1,802 (122 tr, DD -1,582) · WF OOS $9,940 (t=1.762, 7/10 green) · pre-holdout WF $6,923 · full $8,847

## Winners (beat baseline on holdout AND full WF AND pre-holdout WF; ≥30 holdout trades; holdout DD ≤ 1.5× baseline)

| # | id | holdout $ | Δ vs base | tr | PF | DD | WF OOS $ | WF t | green | pre-WF $ | stress holdout/WF | MES WF $ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 1 | `combo_stack7` | 1,896 | 94 | 93 | 1.16 | -1,945 | 23,806 | 2.892 | 10/10 | 20,097 | 1,636 / 22,661 | 1,784 |
| 2 | `combo_pair__trail_mult4.0__tgt_breakout_r2.0` | 4,745 | 2,943 | 109 | 1.44 | -1,377 | 18,453 | 2.806 | 8/10 | 13,809 | n/a | n/a |
| 3 | `combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far` | 5,927 | 4,125 | 108 | 1.54 | -1,481 | 23,290 | 2.804 | 9/10 | 17,716 | n/a | n/a |
| 4 | `combo_pair__tgt_breakout_r2.0__cooldown_48` | 3,933 | 2,132 | 97 | 1.44 | -1,412 | 16,041 | 2.721 | 10/10 | 11,994 | n/a | n/a |
| 5 | `combo_tri__trail_mult4.0__tgt_mr_bb_far__entry_bo_don10` | 2,510 | 708 | 137 | 1.19 | -1,684 | 23,260 | 2.708 | 9/10 | 19,093 | 2,022 / 21,969 | 3,686 |
| 6 | `combo_pair__tgt_breakout_r2.0__tgt_mr_bb_far` | 6,216 | 4,414 | 108 | 1.61 | -1,270 | 20,660 | 2.642 | 9/10 | 14,807 | n/a | n/a |
| 7 | `combo_tri__trail_mult4.0__tgt_breakout_r2.0__entry_bo_don10` | 2,283 | 481 | 133 | 1.17 | -1,720 | 18,058 | 2.579 | 9/10 | 15,080 | n/a | n/a |
| 8 | `combo_pair__trail_mult4.0__cooldown_48` | 2,293 | 491 | 97 | 1.25 | -1,284 | 15,621 | 2.57 | 9/10 | 12,326 | n/a | n/a |
| 9 | `tgt_breakout_r2.0` | 5,105 | 3,303 | 110 | 1.51 | -1,270 | 15,981 | 2.549 | 9/10 | 11,112 | n/a | n/a |
| 10 | `combo_pair__trail_mult4.0__tgt_mr_bb_far` | 3,713 | 1,911 | 113 | 1.34 | -1,243 | 18,708 | 2.347 | 10/10 | 14,760 | n/a | n/a |
| 11 | `combo_tri__tgt_breakout_r2.0__tgt_mr_bb_far__entry_bo_don10` | 2,899 | 1,097 | 133 | 1.21 | -1,370 | 18,789 | 2.306 | 9/10 | 15,156 | n/a | n/a |
| 12 | `combo_pair__tgt_mr_bb_far__cooldown_48` | 2,080 | 278 | 102 | 1.28 | -1,484 | 15,788 | 2.24 | 9/10 | 12,027 | n/a | n/a |
| 13 | `tgt_breakout_r1.5` | 3,840 | 2,038 | 113 | 1.37 | -1,270 | 13,556 | 2.228 | 9/10 | 9,221 | n/a | n/a |
| 14 | `tgt_breakout_r4.0` | 4,259 | 2,457 | 108 | 1.42 | -1,270 | 14,122 | 2.216 | 10/10 | 9,584 | n/a | n/a |
| 15 | `trail_mult3.0` | 2,664 | 862 | 116 | 1.27 | -1,336 | 13,741 | 2.204 | 7/10 | 9,984 | n/a | n/a |
| 16 | `trail_mult4.0` | 2,240 | 438 | 112 | 1.21 | -1,693 | 14,154 | 2.198 | 8/10 | 11,152 | n/a | n/a |
| 17 | `tgt_breakout_r3.0` | 4,494 | 2,692 | 108 | 1.45 | -1,270 | 13,949 | 2.196 | 9/10 | 9,393 | n/a | n/a |
| 18 | `trail_mult2.5` | 3,417 | 1,615 | 120 | 1.38 | -1,165 | 13,204 | 2.156 | 8/10 | 8,893 | n/a | n/a |
| 19 | `cooldown_24` | 2,470 | 668 | 112 | 1.33 | -1,220 | 11,830 | 2.128 | 9/10 | 8,135 | n/a | n/a |
| 20 | `combo_pair__tgt_breakout_r2.0__entry_bo_don10` | 2,201 | 399 | 134 | 1.17 | -1,626 | 14,161 | 2.126 | 7/10 | 11,849 | n/a | n/a |
| 21 | `trail_atr_period7` | 3,334 | 1,532 | 122 | 1.38 | -1,121 | 11,932 | 2.051 | 8/10 | 7,763 | n/a | n/a |
| 22 | `cooldown_12` | 2,151 | 349 | 116 | 1.27 | -1,566 | 11,506 | 2.05 | 8/10 | 8,169 | n/a | n/a |
| 23 | `route_mr_not_trend` | 2,617 | 815 | 138 | 1.27 | -1,652 | 12,261 | 2.009 | 8/10 | 8,239 | n/a | n/a |
| 24 | `tgt_mr_bb_far` | 2,628 | 826 | 122 | 1.29 | -1,815 | 14,628 | 1.991 | 10/10 | 10,446 | n/a | n/a |
| 25 | `stop_trend_swing12` | 2,784 | 982 | 109 | 1.38 | -1,127 | 10,766 | 1.988 | 8/10 | 7,043 | n/a | n/a |

Descriptions:
- `combo_stack7` — stack top-7: entry_mr_rsi35_65 + trail_mult4.0 + tgt_breakout_r2.0 + tgt_mr_bb_far + entry_bo_don10 + cooldown_48 + time_morning_only
- `combo_pair__trail_mult4.0__tgt_breakout_r2.0` — trail_mult4.0 + tgt_breakout_r2.0
- `combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far` — trail_mult4.0 + tgt_breakout_r2.0 + tgt_mr_bb_far
- `combo_pair__tgt_breakout_r2.0__cooldown_48` — tgt_breakout_r2.0 + cooldown_48
- `combo_tri__trail_mult4.0__tgt_mr_bb_far__entry_bo_don10` — trail_mult4.0 + tgt_mr_bb_far + entry_bo_don10
- `combo_pair__tgt_breakout_r2.0__tgt_mr_bb_far` — tgt_breakout_r2.0 + tgt_mr_bb_far
- `combo_tri__trail_mult4.0__tgt_breakout_r2.0__entry_bo_don10` — trail_mult4.0 + tgt_breakout_r2.0 + entry_bo_don10
- `combo_pair__trail_mult4.0__cooldown_48` — trail_mult4.0 + cooldown_48
- `tgt_breakout_r2.0` — breakout fixed target=r:2.0 (no trail)
- `combo_pair__trail_mult4.0__tgt_mr_bb_far` — trail_mult4.0 + tgt_mr_bb_far
- `combo_tri__tgt_breakout_r2.0__tgt_mr_bb_far__entry_bo_don10` — tgt_breakout_r2.0 + tgt_mr_bb_far + entry_bo_don10
- `combo_pair__tgt_mr_bb_far__cooldown_48` — tgt_mr_bb_far + cooldown_48
- `tgt_breakout_r1.5` — breakout fixed target=r:1.5 (no trail)
- `tgt_breakout_r4.0` — breakout fixed target=r:4.0 (no trail)
- `trail_mult3.0` — ATR trail multiple 3.0
- `trail_mult4.0` — ATR trail multiple 4.0
- `tgt_breakout_r3.0` — breakout fixed target=r:3.0 (no trail)
- `trail_mult2.5` — ATR trail multiple 2.5
- `cooldown_24` — cooldown 24 bars
- `combo_pair__tgt_breakout_r2.0__entry_bo_don10` — tgt_breakout_r2.0 + entry_bo_don10
- `trail_atr_period7` — trail ATR period 7
- `cooldown_12` — cooldown 12 bars
- `route_mr_not_trend` — MR allowed whenever ADX<trend_min
- `tgt_mr_bb_far` — MR target=bb_far
- `stop_trend_swing12` — trend stop=swing:12

## Best per research dimension (by full WF OOS $)

| group | best id | holdout $ | WF OOS $ | WF t | pre-WF $ |
|---|---|---:|---:|---:|---:|
| 00_baseline | `baseline_ensemble` | 1,802 | 9,940 | 1.762 | 6,923 |
| 01_routing | `route_mr_not_trend` | 2,617 | 12,261 | 2.009 | 8,239 |
| 02_entry | `entry_mr_rsi35_65` | -1,910 | 14,109 | 2.082 | 13,778 |
| 03_stop | `stop_breakout_atr1.0` | 8,385 | 14,052 | 1.27 | 8,137 |
| 04_target | `tgt_breakout_r2.0` | 5,105 | 15,981 | 2.549 | 11,112 |
| 05_mechanics | `cooldown_24` | 2,470 | 11,830 | 2.128 | 8,135 |
| 06_time | `time_morning_only` | 1,348 | 10,434 | 2.024 | 8,321 |
| 07_new_system | `new_vwap_band_mr_range_only` | 3,463 | 9,919 | 1.922 | 4,556 |
| 08_hybrid | `hyb_add_vwap_band_mr_range` | 2,882 | 15,448 | 2.187 | 10,557 |
| 09_risk | `risk_1.0pct` | 2,450 | 12,983 | 1.475 | 7,969 |
| 10_combo | `combo_new_volume_shock_solo__top3` | 6,044 | 33,176 | 3.096 | 26,486 |
| 11_stress | `stress_combo_new_volume_shock_solo__top3` | 5,800 | 32,038 | 3.001 | 25,678 |
| 12_mes | `mes_combo_tri__trail_mult4.0__tgt_mr_bb_far__entry_bo_don10` | -157 | 3,686 | 0.492 | 1,819 |

## Statistically significant on WF (t ≥ 2): 65

- `combo_new_volume_shock_solo__top3` t=3.096 WF $33,176 holdout $6,044
- `stress_combo_new_volume_shock_solo__top3` t=3.001 WF $32,038 holdout $5,800
- `combo_stack6` t=3.06 WF $28,034 holdout $-48
- `combo_hyb_add_vwap_band_mr_range__top5` t=2.795 WF $27,934 holdout $1,736
- `stress_combo_stack6` t=2.958 WF $27,035 holdout $-50
- `combo_hyb_add_vwap_band_mr_range__top3` t=3.196 WF $26,726 holdout $3,971
- `stress_combo_hyb_add_vwap_band_mr_range__top5` t=2.573 WF $25,151 holdout $905
- `combo_stack4` t=2.604 WF $24,831 holdout $1,323
- `stress_combo_hyb_add_vwap_band_mr_range__top3` t=2.908 WF $24,019 holdout $3,384
- `combo_stack7` t=2.892 WF $23,806 holdout $1,896
- `combo_stack5` t=2.441 WF $23,469 holdout $-919
- `combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far` t=2.804 WF $23,290 holdout $5,927
- `stress_combo_stack4` t=2.456 WF $23,262 holdout $504
- `combo_tri__trail_mult4.0__tgt_mr_bb_far__entry_bo_don10` t=2.708 WF $23,260 holdout $2,510
- `combo_stack3` t=3.007 WF $23,236 holdout $2,248

## Caveats

- 249 variants tested → multiple-comparison risk; treat single-dimension wins under ~$500 as noise.
- WF windows overlap the holdout period for the 'all' WF figure; pre-holdout WF is the clean selection number.
- Risk-% variants (09_risk) only scale P&L/DD and are excluded from winners.
- Nothing here was deployed. Live bott-trader on Paragan untouched; no commits.

Full table: `/workspace/bott-trader-research/artifacts/overnight_2026-10-06/leaderboard.csv` · per-variant JSON: `/workspace/bott-trader-research/artifacts/overnight_2026-10-06/stage*/results/`