# REPORT_VALIDATION - revamp + 3R breakout target (A) and day-only (D2) - backtest only, nothing deployed

## VERDICT: PROFITABLE in backtest (with caveats below) - A: +$29.3k / 23 months on 1 MNQ, max DD $4.4k, PF 1.22, 5 of 6 walk-forward test folds positive, still +$25.5k at 3x slippage.
Not proof of future profit: parameters were tuned on this same 23-month sample, and the last ~2 months and the final walk-forward fold are negative.

| | A: revamp + breakout 3R (24/7) | D2: day-only + 3R (no 6PM add-on) | D: day-only + 3R (keeps 6PM add-on) | base: live revamp (2R) |
|---|---|---|---|---|
| Net (Nov24-Sep26) | +29,313 | +14,880 | +15,337 | +25,686 |
| Max DD | 4,449 | 2,872 | 3,380 | 4,629 |
| PF / win % / trades / avg | 1.22 / 61% / 2,473 / $11.9 | 1.30 / 66.5% / 863 / $17.2 | 1.28 / 62% / 1,033 / $14.8 | 1.19 / 61% / 2,518 / $10.2 |
| Train (to 2025) / 2026 | +17.4k / +12.0k | see json | +10.4k / +4.9k | +15.1k / +10.6k |
| Green months | 18/24 | 17/23 | 18/23 | 18/24 |
| Worst day / week / month | -1,024 / -1,783 / -2,794 | -620 / -878 / -1,081 | -620 / -872 / -944 | -1,024 / -1,783 / -2,570 |
| Longest DD (days) / in DD now | 85 / 51 | 117 / 39 | 118 / 45 | 190 / 51 |
| Walk-forward OOS total (6 folds) | +20.8k (5/6 pos.) | n/a | +9.6k (4/6) | +17.4k (4/6) |
| Stress: 3x slip / TP 4 ticks through / +2 tick overnight slip | +25.5k / +28.3k / +27.1k | +14.9k-ish (D: 13.5k / 15.1k / 15.0k) | 13.5k / 15.1k / 15.0k | - |
| Bootstrap 10% trades removed: net p5 / DD p95 | +22.4k / 5.0k | +10.8k / 3.3k | +11.1k / 3.7k | - |
| Bootstrap resampled days: net p5 / DD p95 | +13.6k / 8.1k | +4.9k / 5.4k | +5.8k / 5.4k | - |
| Suggested min account (1 MNQ) | $10,500 | $7,000 | $7,000 | - |

(The D2 column's stress cells were only run for D; D2 is within about $0.5k of D everywhere and has the smaller observed DD.)

## What the numbers say, honestly
- Positive in every stress run: 2x/3x slippage, TP fills requiring 1/2/4 ticks through the limit, +2 ticks overnight slip, $6 round-trip commission, breakout target +/-20%, trend target +/-20%, profit-lock +/-20%, extra data from 2024-09-25 (+$31.7k, same DD). Bootstrap with 10% of trades removed: 0% of 500 runs lost money. Resampling whole days: 0% losing for A, 0.2-0.6% for day-only.
- 3R vs 2R is a modest improvement (+$3.6k, DD -$0.2k), not a different strategy. The base revamp is also profitable (+$25.7k, 4 of 6 OOS folds); do not expect 3R alone to transform results. Walk-forward fixed A beat base OOS (+20.8k vs +17.4k), but the margin is small relative to noise.
- Parameter RE-selection per train window did worse (+$11.0k OOS) than just fixing A (+$20.8k): picking "the best" grid variant each quarter chases noise. Use fixed params; do not keep re-tuning.
- Drawdown is real: A max DD $4.4k observed, p95 of resampled-days $8.1k (worst case $13.8k). Worst day -$1.0k, worst week -$1.8k, worst month -$2.8k (Feb 2025). 6 of 24 months were red. Longest time under water 85 days (base 190 days). Currently ~51 days under water (peak around late July 2026): Aug 2026 -$1.2k, Sep ~$0, and the last walk-forward fold (Aug-Sep 2026) was -$1.2k.
- Profit lock is the most sensitive knob: lock thresholds +20% cut 2026 profit to $8.4k and raised DD to $6.3k; -20% raised trades and DD to $5.0k. Keep the lock exactly as is.
- Day-only (D/D2) is the lower-risk choice: DD $2.9-3.4k, PF 1.28-1.30, worst day -$620, but about half the profit and 2026 is only +$4.2-5.1k with 4 of 6 OOS folds positive. Day-only had 2 red months in 2026 (Jan, Mar).
- Overnight is where over half the trade count and ~$14k of A's profit come from, but at about $4-5/trade it is the most cost-sensitive part: +2 ticks overnight slip costs A only $2.2k, so it survives.

## Caveats
1. Parameters (3R, which sleeves, lock) were chosen on this same data; 2026 is not truly unseen.
2. TP fills are modeled at the limit (generous). Re-run with 1/2/4 ticks required through the limit: still +$29.5k / +29.3k / +28.3k, so the effect is small, but real queue position and partial fills are not modeled.
3. Intra-bar order assumed stop-first when both touched; slippage is a fixed tick count, not event-driven (news/roll gaps can be far worse; contract-roll spikes were cleaned from the data).
4. The engine's 4.0 ATR trail was not perturbed +/-20% (it lives inside the strategy engine); only target, lock, slippage, commission and fill assumptions were stressed.
5. MNQ_1m.csv ends 2026-09-21; the last ~5 weeks (and any live results) are untested here. Walk-forward uses one continuous run sliced by dates, because the signals have no fitted parameters.
6. Trade results are heavily clustered (Sep 2025, Apr 2026, Jul 2026 each +$2-4k); 6 of 24 months red; the profit factor is 1.2, so a few months of untypical conditions can erase months of gains.
7. If the account is a prop-firm account with a trailing max-drawdown below roughly $4.5k (A) / $3.4k (day-only), the observed drawdown alone would have breached it; check the account's rules before going live.
8. Walk-forward fold windows: 6-month train / 3-month test rolled quarterly (6 folds, May 2025 to Sep 2026). Tests are not independent of the earlier tuning.

## Exact steps to run it (NOT done - nothing deployed, live files untouched)
Breakout target is an ENV VAR, no code change: `REVAMP_BREAKOUT_TARGET_R` (read in src\research_ensemble\revamp_exits.py; 2.0 today in research\clara\revamp\start_mnq_live_revamp.bat).
- Candidate A (24/7 revamp + 3R): copy start_mnq_live_revamp.bat to a new file and change ONE line `set "REVAMP_BREAKOUT_TARGET_R=2.0"` to `3.0`. Everything else stays (REVAMP_EXITS=1, trail 4.0, MR bb_far, lock 100/20/150/0.5, 1 contract, MAX_CONCURRENT_TRADES=3, overnight on). Ready-made text: validation\PROPOSED_start_mnq_live_revamp_bo3.bat.txt (rename to .bat in the same folder to use).
- Candidate D2 (day-only + 3R): same plus `OVERNIGHT_TRADING=false`, `SESSION_MODE=rth`, `REVAMP_RTH_ONLY=1`, `ADDON_E1_6PM=0` (keep ADDON_ONL_RECLAIM=1). Ready-made: validation\PROPOSED_start_mnq_live_revamp_dayonly_bo3.bat.txt. Verify live that the 6 PM add-on and overnight entries are really off (the REVAMP_OVERNIGHT_SLEEVES line in the .bat is not referenced in src; REVAMP_RTH_ONLY/OVERNIGHT_TRADING/SESSION_MODE are the gates that matter). For D (keep 6PM long), leave ADDON_E1_6PM at 1 and SESSION_MODE extended; that needs a check that overnight sleeves stay off.
- Before going live: stop the old bot only with the owner's OK, back up first, paper/demo it for at least 1-2 weeks and compare fills with this backtest (target: slippage near 1 tick, no negative PF in the first 100 trades), start with 1 MNQ, and size the account per the table (A about $10.5k, day-only about $7k).

