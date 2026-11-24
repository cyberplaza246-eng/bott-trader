# Strategy research & improvement tests — 2026-10-28 (backtest only)
Baseline = revamp 5m ADX ensemble, 3R, 1 MNQ, max 3 open, $4 RT + 1 tick slip/side, Nov 2024–Sep 21 2026: **2,473 trades, +$29,313, PF 1.22, DD $4,449, 18/24 months green** (train <2026 +$17.4k, 2026 +$12.0k).
Nothing live was touched. No orders. No purchases. Social/YouTube/Reddit = ideas only; every verdict below rests on backtests.

## 1. Bottom line
1. **No entry filter / rule / sizing tweak tested (25 specs + 4 extras) robustly beats the 3R baseline.** Everything that "won" did so in one window only (train *or* 2026) → noise. The baseline is already near a local optimum; most filters just delete profitable trades.
2. **One real candidate add-on: RTH gap-fade** (separate 1-contract sleeve, 09:30 ET): fade the 09:30 open vs prior 15:59 close when |gap| ≥ expanding 75th-pct of past |gaps|; hold 90 min, time exit. Standalone +$7.6k, PF 1.43, n=155, DD $2.9k, positive in train (+$3.7k) AND 2026 (+$3.9k) AND 6/6 walk-forward folds; survives 3-tick slippage (+$7.3k). Daily-P&L correlation with the baseline −0.05. Base+gap: +$36.9k, PF 1.25, DD $3.9k (< baseline $4.4k), WF-OOS +$27.6k vs +$20.8k.
   **Caveat: t-stat only ~1.4–1.7 (not statistically significant), 2 years of data, params picked off a 15-cell grid (but results plateau across hold 60–180 min and q 0.5–0.75; an unfiltered gap-fade LOSES, so the size filter matters).** Paper-trade first, do not size up.
3. Billionaire / "secret formula" claims: **no verified trading system exists for Elon Musk or Donald Trump** — nothing public to test. Documented rules of legendary traders were translated into mechanical tests (§4); none improved the 3R stack.

## 2. Ranked public strategies (credibility = how independently verified)
| # | Strategy | Source | Credibility / what holds up |
|---|---|---|---|
|1| Trend-following / time-series momentum, diversified futures (AQR "Century of Evidence", Moskowitz-Ooi-Pedersen) | https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf | Highest (peer-reviewed, 1880–2016, net Sharpe ~0.76). Slow, multi-asset — not an MNQ intraday edge. |
|2| Rob Carver pysystemtrade (open source, live since 2014) | https://github.com/robcarver17/pysystemtrade | High (real P&L published, Sharpe ~0.58, worst year −16%). Multi-week holds. |
|3| Vol-targeting (Harvey et al., Man AHL) | https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf | Peer-reviewed; helps equities Sharpe & tails; on our intraday stack not shown to help (§4). |
|4| Intraday momentum last-30-min (Gao et al. JFE 2018; Baltussen et al. JFE 2021) | https://www.sciencedirect.com/science/article/abs/pii/S0304405X21001598 | Peer-reviewed but post-2013 decay (Rosa 2022; 2026 replication finds none on Nasdaq). **Our MNQ test: loses (−$4.2k).** |
|5| Zarattini et al. ORB / Noise-area (SSRN) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172 , https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416622 | Medium-low: authors sell education; independent replication (github giovannibrusco/zarattini-2023-orb-qqq) shows net≈0 after ~2¢ slippage, 76% from 2022. Prior work already ported (only noise-band was positive). |
|6| Pre-FOMC drift (Lucca-Moench, NY Fed) | https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf | Peer-reviewed, but gone after 2015 (PMC7525326). Our test: +$431 / 16 events = noise. |
|7| Kevin Davey / Andrea Unger (futures, WCTC wins) | linkedin / worldcuptradingchampionship | Self-reported or contest results; course sellers. Process ideas (walk-forward, incubation) only. |
|8| Collective2 "NQ Positional" | https://collective2.com/details/128871252 | Low: labeled hypothetical, 37% DD, 3.4-day holds. |
|9| Reddit/GitHub MNQ/NQ ORB & filter threads (r/algotrading, r/Daytrading, r/FuturesTradingNQ; repos dws-data/nas-orb-backtester, charlesbx/futures-backtester, Stolzel/systematic-mnq-trader) | see links in file list | Lowest: self-reported; several honest repos show ORB edge ≈0 after costs. Useful only as filter ideas (calendar guard, Friday filter, ATR-ratio filter, VWAP slope, loss caps) — all tested below. |

## 3. Backtest table (entry-date folds; WF-OOS = sum of 6 quarterly folds May-2025→Sep-2026; boot_p = P(paired day-bootstrap Δnet>0))
Verdict rule: ROBUST needs Δ>0 in train, 2026, WF-OOS, ≥4/6 folds and boot_p≥0.8. None met it.
| spec | n | net | PF | maxDD | green | train | 2026 | Δtrain | Δ2026 | WF-OOS | ΔWF | folds beat | boot_p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**base**|2473|29313|1.22|4449|18/24|17361|11952|–|–|20786|–|–|–|–|
|vol_lo07 (drop ATR<0.7×med)|2130|28818|1.24|4171|18/23|17534|11284|+173|−668|18982|−1804|2/6|0.45|noise|
|vol_hi15 (drop ATR>1.5×med)|2127|24539|1.23|3884|16/24|10329|14210|−7033|+2258|19354|−1432|2/6|0.18|noise (2026-only)|
|vol_hi20|2324|26077|1.21|5435|18/24|12443|13634|−4918|+1682|20453|−333|3/6|0.18|noise|
|htf576 (trend-align EMA)|2152|28038|1.25|4187|16/23|16768|11270|−593|−682|18045|−2741|2/6|0.41|worse|
|htf2880 (200-day-style proxy)|1987|21032|1.20|4956|18/23|8706|12325|−8655|+373|12906|−7881|2/6|0.11|worse|
|vwap_bt / vwap_bo|2376/2388|28251/28558|1.22/1.23|3836|19/24|18430/18461|9821/10097|+1069/+1100|−2131/−1855|17918/18168|−2868/−2618|2/6|0.34/0.37|train-only → noise (lower DD only)|
|rthopen (side of 09:30 open)|2449|28123|1.22|4263|19/24|16534|11589|−827|−363|19481|−1305|2/6|0.25|worse|
|news_all (blackout −30/+60m NFP/CPI/FOMC)|2461|29177|1.22|4449|19/24|17174|12002|−187|+50|21049|+263|3/6|0.44|zero effect|
|news_fomc|2461|28940|1.22|4449|18/24|17197|11743|−165|−209|20913|+126|1/6|0.27|zero effect|
|nofri|2157|24350|1.21|3977|17/24|11910|12441|−5451|+488|19576|−1210|2/6|0.10|noise|
|nomon|2131|30382|1.28|4364|20/24|17649|12733|+288|+781|20241|−545|3/6|0.59|noise (WF negative, day-of-week data-mined)|
|consec2 (stop after 2 losses)|2438|28176|1.22|4295|17/24|15923|12253|−1438|+301|19377|−1409|1/6|0.22|noise|
|losscd30|2371|25246|1.20|4755|17/24|15097|10149|−2264|−1803|17834|−2952|2/6|0.01|worse|
|dl300 / dl750 (daily loss cap)|2394/2535|27235/29857|1.21/1.22|4436/4598|19/24|15435/17851|11800/12006|−1926/+490|−152/+54|18232/19917|−2554/−869|2/6|0.16/0.57|worse / noise|
|notarget (drop $2k daily target)|2475|29437|1.22|4449|18/24|17485|11952|+124|0|20786|0|0/6|0.63|nothing|
|risk300 / risk200 (skip wide stops)|2233/1924|21155/15016|1.18/1.16|5448/5458|20/24|9804/7918|11351/7098|−7557/−9443|−601/−4854|16492/10769|−4294/−10017|1/6|0.03|worse (wide-risk trades carry the profit)|
|stop125 / stop080 (scale stop)|2260/2737|17599/23452|1.13/1.17|6642/5694|17/23|9603/15773|7996/7679|−7758/−1588|−3956/−4273|13936/12885|−6850/−7901|2/6|0.02/0.15|worse|
|open1 / open2|1805/2273|12266/22084|1.12/1.18|3326/4612|16/24|6558/13918|5708/8167|−10803/−3444|−6245/−3786|7957/15926|−12829/−4860|1/6|0.00|worse (concurrency is part of the edge)|
|open4 / open5 (pyramid more)|2552/2586|30752/31553|1.23|4426/4621|17/23|19737/20189|11014/11363|+2376/+2828|−938/−589|19789/19735|−997/−1051|2/6|0.75/0.81|train-only → noise|
|drop3_5 (ET hours 3,5 — chosen on TRAIN only)|2373|32654|1.27|3828|20/24|23174|9480|+5813|**−2472**|20513|−273|2/6|0.87|**overfit: fails 2026**|
|drop3_5_6|2277|30983|1.26|3188|20/24|21925|9058|+4563|−2894|20522|−264|3/6|0.71|overfit|
**Standalone event/time add-ons (1 MNQ, same costs; independent of the ensemble):**
| add-on | n | net | PF | train | 2026 | verdict |
|---|---|---|---|---|---|---|
|Last-30-min momentum (Gao/Baltussen)|461|−4237|0.80|−3994|−244|fails|
|Last-30-min fade|461|−373|0.98|+1094|−1466|fails|
|Pre-FOMC long drift|16|+431|1.28|+661|−230|noise (16 events)|
|Post-event spike continuation NFP+CPI (+5m, hold 30m)|40|+221|1.14|−102|+324|noise|
|… fade NFP+CPI|40|−621|0.70|−138|−484|fails|
|FOMC spike fade (n=16)|16|+744|2.27|+70|+674|noise (tiny sample, 2026-only)|
|FOMC spike continuation|16|−904|0.38|−170|−734|fails|
|Gap continuation (|gap|≥median)|243|−8914|0.73|−3386|−5528|fails (mirror of fade)|
|**Gap fade, q50, H60**|243|+6484|1.26|+2036|+4448|ok, weaker|
|**Gap fade, q75, H90 (candidate)**|155|+7580|1.43|+3704|+3877|train+2026+6/6 folds|
|Gap fade q75/H90 @3-tick slip|155|+7270|1.41|+3546|+3725|survives|
Gap-fade grid (q×hold) and quarterly detail: `gapfade_robust.json`. Unfiltered gap fade (q=0): −$2.7k to +$1.1k → the |gap| filter is essential.

## 4. Legendary-trader / "billionaire" rules → mechanical tests
Note: these are widely quoted (Market Wizards, Way of the Turtle) but individual claims are unverifiable and the 24-month MNQ sample is not a test of a multi-decade rule.
| Rule (who) | Test | Result |
|---|---|---|
|"Don't be long below the 200-day MA" (P. T. Jones)|trend-align filter, EMA576 / EMA2880 proxy (only 2y of data; true 200d SMA not testable)|worse (−$1.3k / −$8.3k)|
|Cut losses fast (Livermore, Jones, Turtles 2N stop)|stop ×0.8, ×1.25, stop-after-2-losses, cooldown, daily loss cap|stop0.8 −$5.9k; stop1.25 −$11.7k; others noise|
|Pyramid winners (Livermore, Turtles add every 0.5N)|max_open 4/5 (same-direction adds only when prior in profit)|+$1.4k/+$2.2k train-only; 2026 negative → noise|
|Concentrate on conviction / size to volatility (Druckenmiller, Turtle N)|skip high-ATR / low-ATR regimes; skip wide-risk trades|worse or noise; wide-risk trades carry the edge. True vol-sized qty NOT tested post-hoc (sizing study not completed).|
|Trade only after news? / avoid news (generic)|blackout around NFP/CPI/FOMC|zero effect|
|Event-shock continuation vs fade|NFP/CPI/FOMC +5m spike|no consistent edge; samples 16–40|
|Gap fade|see above|only credible new thing|
Soros/Dalio/Simons: no published, testable intraday rules; Renaissance info is public-marketing level. Skipped honestly. Elon Musk / Trump: no verified system; nothing to test.

## 5. Recommendation
- **Keep the baseline 3R stack; do not add any of the tested filters.** (Prior chosen trim — cut 2–8 AM overnight breakouts — is a separate, already-validated tweak that this research does not contradict.)
- **Paper-trade the gap-fade sleeve** (q75 of expanding |gap|, 09:30 ET entry, 90-min time exit, 1 MNQ, ideally with an unlevered max-loss stop ≥1.5×|gap| — stop variants didn't change P&L). Ignore anything else.
- Ignore: last-30-min momentum, pre-FOMC drift, news blackout, day-of-week drops, hour drops, ATR band filters, trend filters.

## 6. Caveats
- 2 years of data (Nov 2024–Sep 2026); 155–243 gap trades; t≈1.4–1.7 → **not statistically proven**. Param plateau is the only supporting argument.
- Gap-fade entry fills at the 09:30 1-min bar open + slippage; real open prints can be worse (3–4 tick stress still positive).
- Gap sleeve combined with baseline by adding P&L (independent contracts; overlap/margin not modeled).
- Hour/time-of-day selection used train only; failure on 2026 shows how fragile such cuts are.
- Event dates: BLS/Fed calendars (public); Nov–Dec 2024 NFP/CPI dates entered from memory of BLS releases; Oct 2025 releases cancelled by shutdown.
- REA (morluto/rea) was installed and launched against a Collective2 page via headless Chrome/CDP, but argument errors and low value → time-boxed; no REA-derived rules are used. Web research was the working path.
- Post-hoc vol-sized quantity study and a combined multi-filter "candidate" were not run because no single filter passed.

## 7. What prior work already concluded (not redone)
- Revamp alone: +$25.2k, PF 1.19, DD ≈$4.8k, 17/23 months. Daytime breakout is the best sleeve (+$14.1k); overnight breakout weak; daytime trend −$0.9k.
- Clara Core replica fused with revamp: worse in every clean test (early win = UTC/ET look-ahead bug). Pinnacle Quant NQ log audit done.
- 4 OSS ports: ORB5 −$1.0k, NQ breakout −$13.6k, VWAP trend −$39.3k, noise-momentum +$3.1k (DD $5.2k).
- OG regime fix worse; OG bot unprofitable in any version.
- Chosen trim (cut overnight breakouts 2–8 AM): +$25.0k, DD $4.4k, 20/23 months. "Double" idea (+3 MNQ daytime breakout) +$53.2k, DD $7.4k, paper-test only.
- Resistance-target test: nothing beat fixed 3R; round numbers hurt.

## 8. Files (Paragan) `C:\Users\shawa\Projects\bott-trader\research\og_profit_search_2026-10-27\strategy_research_2026-10-28\`
REPORT.md, REPORT_TABLE.md/.csv (25+4 specs), sr_engine.py (spec runner), sr_analyze.py, sr_addons.py + addon_*.csv + addons_summary.json, sr_gap.py + gapfade_robust.json + gapfade_q50_H60_trades.csv, comb.py (base+gap combos), trades_*.csv, results_runs.jsonl, extra_specs.json.
Sources (ideas only): r/algorithmictrading 1rfyyak (ORB doesn't work), r/algotrading 1rvfy12 & 1s9wrmb (regime filters), r/FuturesTradingNQ 1umuf8o (ORB audit), r/Daytrading 1rrn609 & 1ubjp3f; GitHub dws-data/nas-orb-backtester, charlesbx/futures-backtester, Stolzel/systematic-mnq-trader, giovannibrusco/zarattini-2023-orb-qqq, jadenmachado2009/nq-algo-strategy-lab.


---
# Addendum 2026-10-28 (second pass): public-content scouting + value-area / POC / SMA200-500 tests

Backtest only. Same engine/costs as above ($4 RT + 1 tick/side, 1 MNQ, next-bar-open fills, no lookahead). Baseline = revamp 3R: n 2,473, +$29,313, PF 1.22, DD $4,449 (train +17,361 / 2026 +11,952 / WF-OOS +20,786).
Scripts: sr_pub.py, sr_va.py, sr_size.py, sr_engine.py (+sma200/sma500 filters), sr_analyze.py. Outputs: pub_results.json, va_results.json, size_results.json, REPORT_TABLE.md, pub_*.csv, va_*.csv.

## 1. Scouting (public/free only, no logins)
- YouTube/TradingView/EliteTrader/Substack/blogs: used web search + public pages. Most "bot builder" content gives no mechanical rules or shows only cherry-picked equity curves. Credible, rule-bearing items found: (a) prior-day gap fade / gap fill on index futures, (b) Overnight Dip Buy + Morning Trend Breakout (vendor duediligence.fyi write-up; their own paper shows these lost before 2021 = regime dependent), (c) Asian-range breakout, (d) overnight-drop z-score rebound, (e) value-area / "80% rule" (Steidlmayer/Dalton, widely published), (f) SMA200/500 + prior-day POC bounce (from the streamer screenshot).
- Facebook/Instagram: login-walled, nothing usable. Ignored.
- Hype discounted: anything without a stated rule set, trade count, and costs.

## 2. Public-idea sleeves (standalone add-on, 1 MNQ)
| sleeve | n | net | PF | DD | train | 2026 | WF folds+ | boot_p | 3-tick slip net | base+sleeve net / DD |
|---|---|---|---|---|---|---|---|---|---|---|
| gap fade q75 H90 (no confirm) | 137 | +7,780 | 1.50 | 2,612 | +3,414 | +4,365 | 6/6 | 0.97 | 7,506 | 37,093 / 4,004 (PF 1.25, WF 27,378) |
| gap fade + 5m confirm | 76 | +2,252 | 1.25 | - | -76 | +2,328 | - | - | - | worse (one-window) |
| gap fill to prior close (CE) 0.25% | 305 | +5,494 | 1.13 | 3,895 | +3,356 | +2,138 | 4/6 | 0.82 | 5,086 | weaker |
| Overnight Dip Buy (filtered) | 25 | +1,399 | 2.52 | 540 | +502 | +897 | 5/6 | 0.94 | 1,394 | 30,712 / 4,099 (tiny n) |
| Dip Buy (no filter) | 80 | +1,596 | 1.51 | 862 | +207 | +1,389 | 4/6 | 0.88 | - | 30,909 / 4,073 |
| Morning Trend Breakout | 63 | +2,218 | 1.63 | 838 | +658 | +1,560 | 4/6 | 0.94 | 2,122 | 31,532 / 4,676 |
| Morning breakout w/o trend filter | 98 | +2,265 | 1.32 | - | -216 | +2,482 | - | - | - | one-window = noise |
| Asian-range breakout | 136 | -517 | 0.86 | - | - | - | - | - | - | FAIL |
| overnight-drop z<=-2 / -1.5 | 8 / 13 | +386 / -472 | - | - | - | - | - | - | - | n too small / fail |

Verdict: only the gap fade is a credible add (positive in both windows, 6/6 folds, survives 3-tick slip, boot_p 0.97, but t~1.9 over ~2 yrs / 137 trades => paper-trade first). Dip Buy and Morning Breakout are positive in both windows but tiny and underpowered (optional, not recommended). Rest = ignore.

## 3. SMA200 / SMA500 (5m) as trend filter on the baseline (sr_engine)
| variant | n | net | PF | DD | train | 2026 | WF-OOS | folds beat | boot_p |
|---|---|---|---|---|---|---|---|---|---|
| base | 2473 | 29,313 | 1.22 | 4,449 | 17,361 | 11,952 | 20,786 | - | - |
| SMA200 trend-sleeve filter | 2260 | 23,500 | 1.19 | 5,100 | 18,042 | 5,458 | 11,624 | 2/6 | 0.05 |
| SMA200 breakout-sleeve filter | 2285 | 23,709 | 1.19 | 5,177 | 17,799 | 5,910 | 12,594 | 2/6 | 0.06 |
| SMA500 trend-sleeve filter | 2146 | 29,317 | 1.26 | 4,331 | 16,294 | 13,023 | 18,872 | 2/6 | 0.50 |
| SMA500 breakout-sleeve filter | 2192 | 28,623 | 1.25 | 4,145 | 16,685 | 11,938 | 17,832 | 2/6 | 0.45 |
SMA200 filter helps train slightly, wrecks 2026 (-$6k): reject. SMA500 filter ~neutral net, tiny PF/DD gain, hurts train and WF: not worth adding (noise).

## 4. Value area / dPOC / SMA bounce, 1m and 5m (sr_va.py)
Prior-day RTH volume profile (volume spread across each 1m bar's range, 1-pt bins, 70% VA from POC). Long: low touches level, bar closes up above level (reversal candle), price was above level 3 bars earlier; short mirror. Stop beyond signal bar, target 1.5R or 3R, timed exit, entry next-bar open + 1 tick. 1m = 1m entries with 5m SMA context; 5m = 5m bars. Standalone (not combined) results, 24 runs:
- Every level/timeframe/R combination LOSES: 1m net -$5.7k to -$40k (PF 0.63-0.80, t -3.5 to -16, 0/6 folds in 23 of 24 cases... SMA500 3R 1m 1/6, VA 5m 3R 1/6); 5m net -$2.4k to -$26.8k (PF 0.72-0.92, 0/6 folds in most, boot_p 0.00). Best of the lot: 5m SMA500 3R -$2,404 (PF 0.92), still negative both windows (train -1,872, 2026 -532).
- Adding any of these to the baseline cuts net (e.g. +1m dPOC 1.5R: 18,148 vs 29,313; +ANY level 1m: negative total).
- Read: bounce-at-level has ~zero gross edge on MNQ; with ~$5 all-in cost per trade and 1,000-7,000 trades the cost drag is the loss. The streamer's chart is one visually clean example (survivorship); the mechanical rule does not reproduce an edge. "ANY level" is the worst (more trades, same zero edge).
- Trial count: ~44 runs across this grid + filters -> even a lone positive would need discounting; here there is no positive to discount.

80% rule (open outside prior VA, N consecutive 30-min closes back inside, target opposite VA edge or POC):
| variant | n | net | PF | train | 2026 (n) | folds+ | boot_p | t |
|---|---|---|---|---|---|---|---|---|
| 1 close -> opposite VA edge | 107 | +1,214 | 1.17 | +1,141 | +74 (27) | 4/6 | 0.74 | 0.66 |
| 1 close -> POC | 73 | +683 | 1.20 | +1,179 | -496 (15) | 3/6 | 0.71 | 0.54 |
| 2 closes -> opposite | 77 | -1,084 | 0.79 | -1,206 | +122 | 3/6 | 0.21 | -0.82 |
| 2 closes -> POC | 51 | -318 | 0.89 | -158 | -160 | 4/6 | 0.39 | -0.30 |
Verdict: noise (best variant earns its money in train, ~0 in 2026, t 0.66). Ignore.

## 5. Gap fade x volatility-conditioned sizing (approximate, post-hoc; sr_size.py)
Vol ratio = trailing 20-day mean NY-session range / expanding median.
| sizing | net | PF | DD | train | 2026 | boot_p | base+sleeve net / DD |
|---|---|---|---|---|---|---|---|
| flat 1 MNQ | 7,780 | 1.50 | 2,612 | 3,414 | 4,365 | 0.97 | 37,093 / 4,004 |
| 2 MNQ when vol >= median | 12,898 | 1.43 | 5,223 | 6,030 | 6,867 | 0.95 | 42,211 / 4,905 |
| 2 MNQ when vol < median | 10,441 | 1.62 | 2,612 | 4,213 | 6,228 | 0.99 | 39,754 / 4,004 |
| flat 2 MNQ (reference) | 15,559 | 1.50 | 5,223 | 6,829 | 8,730 | 0.97 | 44,872 / 4,905 |
High-vol upsizing does NOT improve risk-adjusted return (PF falls, DD doubles = just leverage). Low-vol upsizing looks better (PF 1.62, DD unchanged) but this is one of two variants picked after seeing results, on 137 trades: weak evidence, do not adopt without more data. Gap fade at flat size is the recommendation.

## 6. Updated recommendation
- Baseline revamp: KEEP unchanged. Nothing from this pass (SMA200/500 filters, value-area/POC/SMA bounces on 1m or 5m, 80% rule, vendor sleeves) beats it with robustness.
- ADD (paper-trade first, then 1 MNQ): gap-fade sleeve q75, H90, no confirmation. Combined: +$7.8k, PF 1.25, DD 4,004 (lower than baseline), both windows, 6/6 folds. Caveats: t~1.9, 137 trades, ~2 years of data, threshold from an expanding quantile; paper results should confirm before real size.
- OPTIONAL/not recommended: Overnight Dip Buy, Morning Breakout (tiny n, regime dependent).
- IGNORE: SMA200 filter, SMA500 filter, dPOC/VA/SMA bounce (1m and 5m, any R), 80% rule, 5m-confirmed gap fade, CE gap fill, Asian breakout, overnight-drop, vol-based sizing, FB/IG ideas.
- Caveats: bounce rules were defined by me from a screenshot description (reversal-candle definition, stop, R multiples, 70% VA via uniform volume spread from 1m bars), so a different discretionary execution could differ; but results are uniformly negative across 24 variants, so a hidden edge is unlikely at these costs. All results are in-sample on ~2 years; sizing results are post-hoc.


---
# Addendum 2026-10-28 (third pass): making the gap-fade sleeve fire more often

Backtest only; same engine/costs as above ($4 RT + 1 tick/side, 1 MNQ, entry at event open +1 tick, time exit at open of bar a+H -1 tick; no lookahead: the gap/move is known at the event open). Baseline = revamp 3R (n 2,473, +$29,313, PF 1.22, DD $4,449; 109 trades/mo). Reference sleeve = gap fade q75, H90 (re-run here: n 134, +$7,594, PF 1.50, DD 2,612, 5.9 trades/mo; 3 of the earlier 137 trades drop because their 90-min bar run was not contiguous).
Scripts: sr_gap2.py (103 variants), sr_gap2b.py (delay stress, marginal-trade test, portfolios, mirror), sr_gap2c.py (weekend continuation), mk_gap2_table.py. Outputs: gap2_results.json, GAP2_TABLE.md (all 103 rows), gap2_stage2.json, gap2_stage3.json, gap2_trades.pkl.
Validation per variant: train(<2026) vs 2026, 6 WF folds, paired day-bootstrap p (sleeve P&L > 0), slip 3 and 4 ticks, daily-P&L correlation with the baseline. Flag ROBUST = train>0 and 2026>0 and >=4/6 folds and boot_p>=0.9 and slip4 net>0; one-window/noise = positive total but only one of train/2026 positive.

**Trials counted: 103 pre-declared fade variants** (P1 NY-open gap thresholds/holds 32; P2 other boundaries 27 [Globex reopen, London, Asian open, post-lunch, 15:30]; P3 intraday extension fades 24; P4 confirmation/partial-fill 11; P5 weekend 9) **+ 12 distinct post-hoc "mirror" (continuation) variants** chosen after seeing losses + 14 entry-delay stress runs. Outcome of the 103: 16 ROBUST (12 in P1, 1 in P2, 3 in P4), 25 both-windows-positive-but-weak, 9 one-window/noise, 53 fail. With ~100 correlated trials an isolated ROBUST flag is not evidence on its own; only clusters that move together (the NY-open family) are.

## 1. NY-open gap (09:30 vs 15:59 close): lower thresholds and holds
| variant | n | /mo | net | PF | DD | train | 2026 | folds+ | boot_p | slip4 | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q75 H90 (ref) | 134 | 5.9 | 7,594 | 1.50 | 2,612 | 3,414 | 4,180 | 6/6 | 0.97 | 7,192 | ROBUST |
| q65 H90 | 179 | 7.9 | 7,430 | 1.35 | 3,282 | 2,698 | 4,732 | 5/6 | 0.95 | 6,894 | ROBUST |
| q60 H90 | 201 | 8.9 | 6,424 | 1.26 | 3,420 | 1,710 | 4,714 | 5/6 | 0.90 | 5,821 | ROBUST (train thin) |
| q50 H90 | 238 | 10.5 | 7,568 | 1.27 | 3,658 | 3,460 | 4,108 | 4/6 | 0.93 | 6,854 | ROBUST |
| q75 H180 | 132 | 5.8 | 9,628 | 1.52 | 3,141 | 5,884 | 3,744 | 5/6 | 0.96 | 9,232 | ROBUST |
| q50 H180 | 233 | 10.3 | 10,106 | 1.32 | 4,160 | 6,298 | 3,808 | 5/6 | 0.94 | 9,408 | ROBUST |
| abs >=40 pts H180 | 337 | 14.9 | 11,181 | 1.24 | 4,147 | 7,112 | 4,069 | 5/6 | 0.94 | 10,170 | ROBUST |
| abs >=40 pts H90 | 346 | 15.3 | 6,636 | 1.16 | 4,418 | 3,010 | 3,626 | 5/6 | 0.85 | 5,598 | weak |
| abs 10/20/30 pts H90 | 375-420 | 17-19 | +286..+2,598 | 1.01-1.06 | 5.5-6.4k | | 2026 -567..+1,522 | 3-4/6 | 0.5-0.66 | | weak/noise |
| abs 10/20/30 pts H180 | 365-410 | 16-18 | 7.5-8.3k | 1.14-1.16 | 4.5-5k | 4.5-6.3k | 1.6-2.9k | 4-5/6 | 0.84-0.86 | | weak |
| 0.12 ATR H180 | 339 | 15.0 | 9,652 | 1.21 | 4,132 | 4,348 | 5,303 | 4/6 | 0.89 | 8,634 | weak (boot .89) |
| 0.04-0.08 ATR H90/H180 | 376-413 | 17-18 | 1.9-9.3k | 1.04-1.18 | 4.4-6.4k | | | 3-5/6 | 0.6-0.88 | | weak |
(ATR = trailing 20-day mean NY range, median ~300 pts, so 0.12 ATR ~ 36 pts. |gap| quantiles: q50 ~108 pts, q75 ~220 pts, q90 ~357 pts.)

**Key finding: lowering the threshold adds trades but no profit.** Marginal trades vs the q75 H90 sleeve: q65 adds 45 trades = -$164 (PF 0.97); q50 adds 104 trades = -$26 (PF 1.00, 3/6 folds); q50 H180 adds 101 trades = +$478 (PF 1.04). Frequency rises 5.9 -> 10.5/mo but PF falls 1.50 -> 1.27 and portfolio DD rises (4,004 -> 4,751). Longer hold (H180) lifts net on the SAME core trades (q75: 7,594 -> 9,628) but the gain is all train (+2.5k); 2026 is slightly lower (4,180 -> 3,744), so use H90-H120 as the core. The abs>=40pt/H180 sleeve earns +$11.2k at 15/mo, but its 205 extra trades beyond the core earn only +$1,552 (PF 1.06, 3/6 folds, boot_p 0.63): the extra frequency is ~breakeven filler. Small gaps (<30-40 pts) have no fade edge.
Entry-delay stress (enter 1 / 3 min after the open, same exit): q75H90 6,406 / 5,932; q65H90 6,514 / 5,771; q50H90 6,928 / 6,218; q50H180 9,462 / 8,620; abs40H180 10,964 / 9,907; 0.12ATR H180 9,332 / 7,802 -> the NY-open family is robust to execution delay.

## 2. Other session boundaries
| variant | n | /mo | net | PF | train | 2026 | folds+ | boot_p | flag |
|---|---|---|---|---|---|---|---|---|---|
| Globex reopen 18:00 weekday vs 16:59 close (q0/q50/q75, H60-180) | 84-329 | 3.7-14.6 | -1.8k..-5.9k | 0.49-0.86 | neg | neg/~0 | 1-4/6 | <=0.25 | FAIL (9/9) |
| London 03:00 vs prior 3h or 9h move (q50/q75, H60/120) | 129-239 | 5.7-10.6 | -1.5k..-3.6k | 0.73-0.84 | mostly neg | neg | 1-3/6 | <=0.21 | FAIL (8/8) |
| Asian open 20:00 vs prior 2h (q50/q75, H60/120) | 129-219 | 5.7-9.7 | +0.4k..+2.4k | 1.07-1.28 | -810..-149 | +1.2k..+2.6k | 3/6 | 0.6-0.86 | one-window (2026 only) = noise (4/4) |
| Post-lunch 13:00 vs prior 60/90 min (q75, H30/60) | 91-113 | 4-5 | -2.7k..-3.4k | 0.54-0.68 | neg | mixed | 1-4/6 | <=0.22 | FAIL (4/4) |
| 15:30 vs prior 30 min (q75, H25) | 104 | 4.6 | +688 | 1.12 | 622 | 66 | 4/6 | 0.66 | weak |
| **15:30 vs prior 60 min (q75, H25)** | 99 | 4.4 | **+2,584** | **1.63** | 886 | 1,698 | 4/6 | 0.96 | ROBUST by rule, see below |
15:30 fade: DD only $808 and slip4 +$2,287, but it is isolated (its 30-min sibling is weak, post-lunch fails), and a 1-minute entry delay cuts it to +$1,389 with train only +$98 (3-min delay +1,940). Paper-trade candidate, not an add. With the q75 H90 sleeve: net 39,491, PF 1.26, DD 3,646, WF-OOS 28,599, corr 0.02.

## 3. Intraday extension fades (>= N-ATR move over 30/60 min, or distance from RTH VWAP / session open; first trigger per day 10:00-14:30, time exit 45/90 min): 24 variants
22 of 24 fail (negative total, 1-3/6 folds, PF 0.67-0.97). The other two are weak: VWAP 0.2 ATR H90 +$2.2k (train -$1.3k, 2026 only) and 60-min 0.25 ATR H90 +$2.2k (2026 +$124). Intraday extension fading has no edge on MNQ at these costs; MNQ extends after big intraday moves more often than it snaps back (consistent with the baseline's trend sleeve).

## 4. Confirmation / partial-fill filters on the NY gap
| variant | n | /mo | net | PF | DD | train | 2026 | folds+ | boot_p | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| q50 + first-1m bar closes toward fill, H90 | 123 | 5.4 | 5,218 | 1.40 | 2,554 | 1,387 | 3,832 | 6/6 | 0.92 | ROBUST (fewer trades than q50) |
| q60 / q75 + 1m confirm | 106 / 76 | 4.7 / 3.4 | 4,271 / 2,370 | 1.36 / 1.26 | | 379 / 244 | | 6/6 | 0.89 / 0.79 | weak |
| q50, >=15% filled by 09:35 / 09:45 | 80 / 75 | 3.5 / 3.3 | 2,148 / 174 | 1.29 / 1.03 | | | 460 / -1,544 | 3/6 | | weak / one-window |
| q50, >=30% filled by 09:35 | 32 | 1.4 | 3,331 | 3.10 | 540 | 1,616 | 1,716 | 5/6 | 0.99 | tiny n (not usable) |
| q75, >=30% filled by 09:35 | 10 | 0.4 | 1,243 | 3.37 | | | | 4/6 | | n=10 |
Confirmation and partial-fill reduce frequency (they are filters, not frequency raisers) and do not beat the unfiltered q75 H90 on PF/DD in a robust way.

## 5. Weekend gap (Sunday 18:00 vs Friday 16:59), separate
Fade loses in every cut: q0/q50/q75 x H60/120/180: n 30-86 (1.3-3.8/mo), PF 0.31-0.45, net -$2.7k..-$6.2k, 0-2/6 folds, boot_p <=0.05 (9/9 FAIL). Weekend gaps tend to continue, not fill.
Post-hoc mirror (continuation, chosen AFTER seeing the fade lose): q0 H120 n 86, +5,160, PF 2.00, DD 716, train +2,560 / 2026 +2,600, 5/6 folds, boot_p 0.99, slip4 +4,902. Do NOT trust: the top 3 trades = $2,452 (47% of profit) and a 1-minute entry delay collapses it (H60 -1,230, H120 +455, H180 +1,644; q50 H60/H120 -1,514/-290). The edge sits in the first minutes after the reopen and is likely not capturable with a market order. Other mirrors (Globex weekday q50 H60 +2,040 PF 1.36 but 2026 only +490, 3/6 folds; London +320 ~0; post-lunch +2,268 train-only, 2026 -66) are noise.

## 6. Combined portfolios (baseline + sleeves, each sleeve 1 MNQ, additive; sleeves can overlap baseline positions so live exposure is higher than the baseline's 3-max)
| portfolio | trades | sleeve /mo | net | PF | DD | train | 2026 | WF-OOS | green mo | sleeve corr w/ base |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2,473 | - | 29,313 | 1.22 | 4,449 | 17,361 | 11,952 | 20,786 | 18/24 | - |
| + gap q75 H90 (ref) | 2,607 | 5.9 | 36,907 | 1.25 | 4,004 | 20,776 | 16,132 | 27,192 | 18/24 | 0.00 |
| + gap q65 H90 | 2,652 | 7.9 | 36,744 | 1.24 | 4,794 | 20,060 | 16,684 | 26,339 | 20/24 | -0.01 |
| + gap q50 H90 | 2,711 | 10.5 | 36,881 | 1.23 | 4,751 | 20,821 | 16,060 | 25,824 | 20/24 | -0.05 |
| + gap q50 H180 | 2,706 | 10.3 | 39,420 | 1.24 | 3,858 | 23,660 | 15,760 | 28,482 | 20/24 | -0.05 |
| + gap q75 H180 | 2,605 | 5.8 | 38,942 | 1.26 | 4,806 | 23,245 | 15,697 | 28,713 | 20/24 | -0.01 |
| + gap abs40 H180 | 2,810 | 14.9 | 40,494 | 1.23 | 4,111 | 24,473 | 16,021 | 30,223 | 19/24 | -0.09 |
| + gap 0.12 ATR H180 | 2,812 | 15.0 | 38,965 | 1.22 | 4,387 | 21,710 | 17,255 | 28,514 | 17/24 | -0.09 |
| + gap q50 1m-confirm H90 | 2,596 | 5.4 | 34,532 | 1.24 | 4,066 | 18,748 | 15,784 | 26,102 | 20/24 | -0.04 |
| + gap q75 H90 + 15:30 fade | 2,706 | 10.3 | 39,491 | 1.26 | 3,646 | 21,662 | 17,830 | 28,599 | 20/24 | 0.02 |
| + gap abs40 H180 + 15:30 fade | 2,909 | 19.3 | 43,078 | 1.24 | 4,201 | 25,359 | 17,719 | 31,629 | 19/24 | -0.08 |
| + gap q75 H90 + 15:30 + Asian (Asian = one-window, completeness only) | 2,835 | 16.0 | 41,501 | 1.26 | 4,024 | 21,513 | 19,988 | 29,855 | 19/24 | 0.02 |
All sleeves are essentially uncorrelated with the baseline (daily P&L corr -0.09..+0.02). Every portfolio beats the baseline in train, 2026 and WF-OOS, but the lift over the q75 H90 reference is modest.

## 7. Verdict
- **Frequency AND robustness together: only the NY-open gap family.** q65/q50 at H90 and abs>=40pt (or ~0.12 ATR) at H180 raise frequency to 8-15 trades/mo, pass the robustness flags, survive 3-4 tick slippage and 1-3 min entry delay, and are positive in train and 2026. But the extra trades are ~zero-edge filler (marginal PF 1.00-1.06, boot_p 0.5-0.63): net barely moves, PF falls 1.50 -> 1.24-1.35, and DD rises at q65/q50 H90.
- **Best frequency-raising choices if wanted:** gap >= 40 points, H180 (14.9/mo, +$11.2k, PF 1.24, portfolio net 40.5k, DD 4,111, WF-OOS 30.2k) or q50 H180 (10.3/mo, portfolio DD 3,858, WF-OOS 28.5k). The q75-q65, H90-H120 core remains the dependable engine; treat extra trades as optional.
- **Paper-trade candidate only:** 15:30 fade vs prior 60 min (+2.6k, PF 1.63, DD 808; fragile to 1-min delay, isolated among ~100 trials).
- **Ignore:** Globex weekday reopen fade, London fade, Asian-open fade (2026-only), post-lunch fade, all intraday ATR/VWAP/open extension fades, weekend gap fade, thresholds under 40 pts, partial-fill and confirmation filters (they cut frequency), weekend continuation (post-hoc, outlier-driven, vanishes with 1-min delay).
- **Caveats:** ~23 months of data; the q75 sleeve itself has t ~1.9; 103 pre-declared + 12 post-hoc variants (expect several false positives); expanding-quantile thresholds skip the first 20 events; fills are market-at-open with 1-4 tick slippage (opening prints on 09:30/18:00 can be worse); portfolio numbers are additive sums of independent 1-MNQ sleeves with no combined position cap; H180 gains are train-heavy.


---
## Addendum 4 (2026-10-28): NY-open gap fade built LIVE-CAPABLE + big-candle / exhaustion test

### A. Gap-fade live sleeve (q75, H90, no confirmation, 1 MNQ) - built, OFF by default
- New files only (no live file edited; originals backed up + hash-verified in `backup_pre_gapfade_20261028\`): `src/core/gapfade_sleeve.py`, `run_revamp_gapfade.py` (shim launcher; stock script untouched), `scripts/seed_gapfade_state.py`, `scripts/gapfade_offline_test.py`, `START_MNQ_LIVE_REVAMP_GAPFADE.bat`.
- Switch `GAPFADE_ENABLED` (default 0 => shim == stock bot). Rule: gap = 09:30 ET live price - 15:59 ET prior close; trade if |gap| >= expanding q75 of past |gap| (first 20 skipped); fade it; 1 MNQ; bracket market order with protective stop (default 300 pts = $600) and far TP; time exit 11:00 ET (H90).
- Integration: uses the bot's own place_order/process_exit (costs, ledger, session/loss limits), position cap, and an opposite-direction/netting guard (Rithmic nets per symbol). Roll-day fake gaps skipped via stored contract code. Time exit flattens (alone) or closes 1 contract with a market order then cancels only its own legs (partial close; untested on a real broker).
- **Stop test (backtest, 134 trades Nov24-Sep26, 1-tick slip):** no stop +$7,594 PF1.50 6/6 folds; stop 60/80/100/125/150/200 pts -> +$1,050/-$762/+$1,162/+$2,444/+$2,186/+$3,238 (fragile, 4/6 folds, boot_p<0.8); 250 -> +$4,771 5/6; **300 -> ~94% of no-stop net, 6/6 folds** (default). In-cap alternative `GAPFADE_STOP_PTS=240` ($480) ~ +$4.8-5.0k, 5/6 folds. 300 pts needs `GAPFADE_RISK_CAP_USD=650` (lifts the $500 per-trade cap for this sleeve only). Wide stop = catastrophe protection, not an edge.
- **Netting policy cost:** with stop300 + cap/opposite skips, 85 of 134 trades are taken; gap P&L +$6,786, baseline trades blocked while a gap position is open cost -$469 => net effect ~ +$7.3k. Baseline positions open opposite at 09:30 had net -$32 historically, so the skip policy costs ~0.
- **Offline safe-mode test** (`gapfade_offline_test.py`, run from an empty sandbox dir, no broker, state/ledger files redirected): 33/33 checks pass - entry sizing/stop/TP, skips (session limit, cap, opposite, roll, outside window), time exit alone / with others, market-close failure (stop stays), broker reject, restart-state warning; plus a 480-session replay through the real `LiveRithmicTrader` in paper mode: 155 trades, 8 stops, 147 time exits, net +$7,898, worst -$602, trade dates/sides identical to an independent numpy implementation of the rule.
- Caveats: positions are not re-tagged after a mid-day restart (boot check warns); partial-close path is untested against a real broker, so supervise the first live day; first live gap needs the 15:59 close recorded (or the 15:55 5m bar in the live candle window); q75 sleeve t ~1.9.

### B. Big-candle / exhaustion fade + continuation (BACKTEST ONLY)
Method: 1m and 5m signals on bar close; entry at open of next 1m bar + 1 tick slip (delay 0), $4 RT; one position at a time per variant; signal contiguity checked; ATR(14)/volume avg(20) from prior bars only. Families: range >= N x ATR (N=2,3,4); volume spike (2x,3x avg, range>=1 ATR); long-wick rejection (wick>=60% of range, range>=1.5/2 ATR); climax bar (N=2,3 ATR, vol>=2x) + opposite reversal bar; news-time candle 08:30/10:00/14:00 ET (all, or range>=1.5 ATR). Each tested as fade and as continuation, with exits T5/T15/T30/T60 min, fixed-R (stop 1 ATR, target 1R or 2R, 60-min cap). **Variants counted: 30 signal definitions x 2 modes x 6 exits = 360** (all pre-declared; plus 7 stress reruns on the 5 screen-passers). Script `sr_bigc.py` (+`sr_bigc2.py`, `sr_bigc3.py`), results `bigc_all.csv`, `bigc_stress.json`.

- **Fades lose, almost without exception.** Mean net by family (fade/continuation): range -$25.0k / -$20.8k; volume -$51.2k / -$49.4k; wick -$10.6k / -$11.6k; climax -$9.6k / -$8.9k; news -$2.4k / -$0.7k. Positive fade variants: 4 of 180 (best +$1.5k, t 0.7, boot_p 0.76), none robust. Several fades are significantly NEGATIVE (e.g. 5m N=4 range fade T60: -$14.2k, t -2.7; 10:00 news fade T60: -$7.3k, t -2.6) - i.e. big candles tend to continue, not revert. 1m versions lose to costs outright (thousands of trades, $5 cost each).
- **Continuation: 5 variants pass the screen (train>0, 2026>0, >=4/6 folds, boot_p>=0.9)**, all 5m: (1) range>=4 ATR cont T60: n 420, +$9,976, PF 1.34, t 1.93, folds 4/6 (last two -$1.7k/-$0.2k), slip4 +$8.7k, delay3 +$9.2k, corr 0.17, but ex top-10 days only +$0.9k, 15/24 months green; T30 +$6.6k same shape. (2) 10:00 ET news bar (range>=1.5 ATR) cont T60: n 188, +$5,462, PF 1.46, t 1.95, folds 5/6 (last -$1.3k), slip4 +$4.9k, delay3 +$3.9k, corr 0.05, DD $2.4k, but ex top-10 days only +$0.5k; T30 +$3.3k, T15 +$2.3k. Neighbours decay (N=3 +$3.2k t 0.5, N=2 +$1.8k), unfiltered 10:00 news fails 2026.
- **Verdict: no fade edge; no new sleeve to trade.** With 360 variants and a one-sided boot_p>=0.9 bar, ~10% would pass by luck; 5 correlated survivors (2 distinct ideas, t<2, profit concentrated in ~10 days, latest folds negative) is not distinguishable from chance. Portfolio effect if added to baseline: range cont T60 net 39.3k but DD 5.2k (vs 4.4k); 10:00 news cont T60 net 34.8k, DD 4.9k. Not recommended for live; the 10:00 5m news-continuation is the only one worth a paper watch. The gap fade remains the only intraday-reversion edge found.

---
## Addendum 5 (2026-10-28): 6 PM ET Globex-reopen deep dive + strict double-test of the NY gap fade (BACKTEST ONLY)
Scripts: `sr_globex.py`, `sr_globex2.py`, `sr_globex3.py`, `sr_dbl.py`, `sr_dbl2.py`; outputs `globex_all.csv`, `globex_robust.csv`. Same costs/validation as before ($4 RT + 1 tick/side, train<2026 vs 2026, 6 folds, day-bootstrap, slip 4, delay, corr with baseline).

### A. Globex 18:00 ET reopen (gap = 18:00 open - 16:59 prior close; weekday = Mon-Thu, 402 events; Sunday = 102 events)
**172 variants:** 144 gap trades (2 sets x thresholds q0/q50/q75/q90/abs20/abs40 pts x holds 15/30/60/90/180/360 min x fade/continuation) + 24 opening-range breakouts (range 15/30 min; breakout vs reversion; hold 60/120/240) + 4 gap-fill reversion (target prior close, 1x-gap stop, 180 min cap). Roll-week events excluded in an "ex-roll" rerun (same results).
- **Fade loses everywhere:** 0 of 72 weekday fades and 0 of 72 Sunday fades are positive (best -$1.1k / -$0.7k; mean -$2.5k / -$4.1k). Gap-fill reversion: 4/4 lose (-$6.6k to -$9.0k). Opening-range reversion/breakout: 10 of 24 positive, mean -$0.3k (Sunday) / -$1.4k (weekday); only one passes the screen (Sunday R30 H60 breakout, +$846, 58 trades - negligible).
- **Continuation is where the signal is (35 variants pass the screen, all continuation):**
  - **Weekday:** 34/36 continuations positive but small (best +$2.7k, abs>=20pt H90, n 64, PF 1.9, DD $0.8k); the profit is 72-87% in 3 trades and ex-top-5 is ~0 (q90 H90: -$164). Noise.
  - **Sunday:** 36/36 continuations positive: n 42-85 (2.2-4.1 trades/mo), net +$3.4k to +$6.0k, PF 2.0-5.8, DD $0.2-1.1k, 5-6/6 folds, boot_p 0.93-1.00, slip4 holds, corr with baseline 0.00-0.09. Example abs>=40pt H360: n 42, +$5,898, PF 4.05, DD $570, 6/6 folds, worst trade -$444, top-3 trades = 34% of profit, ex-top-5 +$2.8k. Sunday gaps average 72 pts (weekday 13.6); the gap continues ~20 pts in its own direction over the next 30 min (corr gap vs first-minute move 0.48; weekday 0.19).
  - **But it lives in the opening print:** a 1-minute entry delay turns H15-H90 holds to ~0 or negative (e.g. abs40 H90 +$4.4k -> -$0.5k). Only long holds survive: abs40 H360 +$5.9k -> +$2.1k (1 min), +$1.8k (3 min), +$1.7k (3 min + 4-tick slip); q50 H360 +$5.3k -> +$1.9k / +$1.6k / +$1.4k. Realistic expectation ~ +$1.5-2k over 22 months (~2 trades/mo, ~$80/mo), 42 trades.
- **Verdict:** nothing to trade live. Sunday-gap continuation (hold 6h, delayed entry) is the only survivor: real-looking, tiny, 42 trades, chosen after seeing the fade lose (post-hoc direction), 172 variants tried - paper-watch only. Weekday reopen: no edge either way.

### B. Strict double-test of the NY-open gap fade (freeze on first 12 months, score on the rest)
Window A = 2024-11-01..2025-10-31 (12 mo); window B = 2025-11-01..2026-09-21 (~10.7 mo). Grid on A only: q {.5,.6,.65,.7,.75,.8,.85,.9} x H {30,60,90,120,180,240} = 48 cells; frozen = best A net.
- **Frozen-on-A pick = q50/H240** (A: 112 trades, +$7,042, PF 1.53). **Scored on B it FAILS: 130 trades (12.2/mo), +$677, PF 1.03, DD $6,046, boot_p 0.56**; slip4 +$287, delay3 -$149, delay3+slip4 -$539; with stop 300 +$97 (worst trade -$604, worst week -$1,316, losing streak 5). A strict walk-forward selection does NOT hand you q75/H90.
- **The family replicates though:** all 48 cells positive in B (48/48) and 47/48 in A. But the cell ranking does not carry over (Spearman A-vs-B net across cells = 0.02). Mean net by q in B: q50 +$3.0k, q65 +$3.3k, q75 +$5.0k, q80 +$5.3k, q90 +$3.5k; in A: q50 +$4.1k, q75 +$2.2k, q90 +$0.6k. So q and H are not identifiable; the edge is "fade big NY-open gaps", not "q75/H90".
- **The chosen q75/H90 (picked using all data) scored on B only (selection-contaminated):** 80 trades (7.5/mo), +$5,206, PF 1.58, DD $2,612, t 1.62, boot_p 0.95; with stop 300: +$4,510, PF 1.46, DD $2,612. In A it was weaker: 57 trades, +$2,574, PF 1.40, t 0.96. Stress on B (stop 300): slip4 +$4,277; delay 1 min +$4,459; delay 3 min +$3,132; delay3+slip4 +$2,900; ex-roll +$4,516.
- **Worst-day dollars (1 MNQ, q75/H90, B):** worst trade/day -$636 (no stop) / -$604 (stop 300); worst week -$1,058 / -$1,076; max DD $2,612; longest losing streak 7. Corr with baseline -0.03 (A +0.05). Baseline alone in B +$13,112 (DD $3,706) -> baseline + gap fade +$18,317 (DD $3,553) / +$17,621 with stop 300 (DD $3,553).
- **Lookahead checks:** (1) independent point-in-time rebuild (threshold from earlier events only, using only the 09:30 open and 15:59 close) gives the identical 137 trades (dates and sides); (2) placebo with a random side on the same days and holds: mean -$752 (sd $4,158), real +$7,780 beats it in 98% of draws (p 0.02; one-sided); (3) volume peaks at the 09:30 label (10,350 vs 2,323 at 09:29), so labels are bar-start and the 09:30 open is the first print; (4) delay and 4-tick stress above cover opening-print fill risk.
- **Verdict:** keep the gap fade, but with honest expectations: the effect survives out of sample across every parameter cell, yet the optimal-looking q75/H90 and its $5.2k are partly selection luck. Plan on roughly +$3-5k per ~11 months at 1 MNQ (q65-q85, H60-H180 neighbourhood) at ~5-8 trades/mo, t ~1.4-1.9, and a $600-ish worst day. Do not treat q75/H90 as individually validated; a q70-q80 / H90-H120 neighbourhood is equally good.

---
## Addendum 6 (2026-10-28): standalone ORB sleeve, ES/NQ cross-market filter, gap-size scaling (BACKTEST ONLY)
Same validation as before (train <2026 vs 2026, 6 quarterly folds, window A = 2024-11..2025-10 vs B = 2025-11..2026-09 frozen-forward, slip to 4 ticks, 1-3 min entry delay, day-bootstrap, corr and paired bootstrap vs baseline). Scripts `sr_x.py`, `sr_x1.py` (ORB), `sr_x2.py` (cross-market), `sr_x3.py` (scaling); outputs `orb_all.csv`, `orb_robust.csv`, `x2_all.csv`, `x2_features.csv`, `scale_all.csv`. **Variants tried: 192 (ORB) + 32 (cross-market: 7 filters x 4 parameter cells + 4 unfiltered) + 32 (scaling: 8 schemes x 4 cells) = 256.**

### 1. Standalone opening-range-breakout sleeve - IGNORE
Rules fixed up front: sessions 09:30 NY, 08:30, 18:00 Globex; opening range 5/15/30/60 min; breakout = first 1m close beyond the range (entry next open, window 120 min); fade-of-failed-break = after a break, close back inside the range within 10 bars, trade against the break; stop = opposite side of range (fade: beyond the break extreme) or 1 x ATR(14, 5m); exits 1R/2R/3R (session backstop 15:55; Globex 02:00) or 120-min time exit; one trade per session per day; 1 MNQ; $4 RT + 1 tick/side.
- **Mean net of all 192 = -$2.3k; only 33 are positive; 0 have t>=2; mean by group (breakout/fade): NY -$2.3k/-$0.6k, 08:30 -$5.1k/-$1.4k, Globex -$2.9k/-$1.3k.** Breakouts lose broadly (as the papers said, ~zero edge before costs, negative after).
- **Frozen-forward:** best-on-A = NY 15-min breakout, opposite-side stop, 3R: A +$8,639 -> **B -$6,220**. Top-5 on A all lose in B (-$2.7k to -$6.7k); rank correlation A-vs-B across the 192 = 0.01. Classic one-window win.
- **Only 1 variant passes the screen** (train>0, 2026>0, >=4/6 folds, boot_p>=0.9): NY 30-min fade-of-failed-break, 1 ATR stop, 2R: n 307, +$3,722, PF 1.20, t 1.4, but train +$152 / window A -$54 vs 2026 +$3,571 (all profit in one window), slip4 +$1.9k, delay 3 min -$159. Post-hoc, noise. The other NY fade-of-failure variants (30/60 min, ATR stop) are positive only in 2026 for the same reason.
- Corr with baseline -0.05; adding that one to the baseline: net $33.0k (vs $29.3k) but DD $5.4k (vs $4.4k). **Do not build.**

### 2. ES / NQ cross-market filter on the NY-open gap fade - IGNORE (optional shadow log)
- **Data:** ES/NQ 1m is NOT on the PC (MES 1m covers only 2025-12-29..2026-03-11; NQ 1m only 2026-05-20..06-10). Used free public yfinance hourly ES=F and NQ=F (2024-06..2026-10, no login). Aligned to the same dates: 458 of 462 gap days; yf NQ=F pre-open move vs local MNQ correlation 0.993 (median diff 0.2 bp; 5 days >40 bp = roll/stale). Features are known before 09:30: move from 16:00 prior close to 09:00 ET (hourly bars cannot give 09:30), ES move, and divergence = MNQ move minus ES move (pct). Raw ES=F can have roll jumps; caveat.
- **Tested on q75/H90 (300-pt stop) and three neighbouring cells (q70/H120, q80/H120, q65/H90):** ES agrees with the gap direction, ES disagrees, big divergence (|NQ-ES| >= median 16.7 bp), small divergence, NQ-led divergence, NQ-lagged divergence, ES move >= half the NQ move.
- **ES direction carries no information:** ES agrees with the NQ gap on 132 of 137 q75 days (disagrees 4, NQ-lagged 5) - too few to test; "agrees" +/-0.
- **Best idea: big NQ-ES divergence (idiosyncratic NQ gap) kept, small divergence (market-wide gap) dropped.** q75/H90: 137 -> 118 trades (5.2/mo vs 6.1; B 6.4/mo), net +$7,136 -> +$7,994, PF 1.44 -> 1.59, DD $2,612 -> $2,150, window A +$2,626 -> +$3,084, B +$4,510 -> +$4,911, 6/6 folds, slip4 +$7,650, delay3 +$5,834. The 19 dropped trades lost -$859 (small-divergence trades overall: n 18, -$668, PF 0.73, 0/6 folds). Same direction in all 4 cells (net +$0.4k..+$1.9k vs unfiltered in each cell) and >=4 trades/mo kept (4.2-6.6). **But paired day-bootstrap that the filter beats unfiltered: p 0.65-0.88 (q75/H90 0.76 all / 0.70 on B) - not significant; chosen after seeing 7 filters (post-hoc), and the fade it improves is itself only t~1.4-1.9.** "ES move >= half NQ move": no improvement (p 0.3-0.5).
- **Verdict:** not worth gating live trades on. Optional: log the 09:00 NQ-ES divergence in shadow and re-test after ~6 more months of live trades.

### 3. Gap-size scaling of the gap fade - IGNORE as a signal; 2 MNQ only if risk limits are raised
- **Is gap size predictive?** No: Spearman(|gap|, trade P&L) = 0.03 (p 0.77) for q75 trades; 0.04 / 0.07 (p 0.5 / 0.3) for the wider q50 set; per-trade mean by size rank bucket q75-85 +$25, q85-95 +$80, q95+ +$60 (non-monotone. So scaling gains come from more size, not information.
- **Note on the "60 points" idea:** with the gap measured 15:59 -> 09:30, every q75 gap is >=166 pts , so a 60/100-pt trigger would just be flat 2 MNQ; I used 400/500 pts instead.
- **q75/H90, 300-pt stop, 137 trades (flat 1 MNQ: +$7,136, PF 1.44, DD $2,612, worst trade/day -$604, A +$2,626, B +$4,510, 6/6 folds):**
  - flat 2 MNQ: +$14,271, PF 1.44, DD $5,223, worst -$1,209; B +$9,019 (just doubling).
  - 2x if gap >= q90 (58 of 137 trades): +$10,720, PF 1.47, DD $3,428, worst -$1,209, A +$3,652, B +$7,068 (+$2,558 vs flat 1 in B), 5/6 folds, slip4 +$10,151, delay3 +$7,021, delay3+slip4 +$6,454; paired bootstrap vs flat 1: p 0.89 (B) / 0.88 (all).
  - buckets 1-2-3 by q85/q95 (max 3 MNQ): +$15,648, PF 1.53, DD $4,413, worst -$1,814, A +$5,790, B +$9,858, 6/6 folds, slip4 +$14,916, delay3+slip4 +$9,746; p 0.90 (B) / 0.94 (all).
  - buckets 1-2-3 by points (300/450): +$13,516, PF 1.50, DD $3,732, worst -$1,814; p 0.88 / 0.90.
  - 2x if >=400 / >=500 pts: +$8,936 / +$8,834, p 0.67-0.78 (no gain). Anti-scheme (2x on SMALLER gaps): +$10,687 but DD $5,084, p 0.78 B - it also "wins" because it adds size, which shows the test mostly measures leverage.
  - Neighbouring cells (q70/H120, q80/H120, q65/H90): the q90 scheme also beats flat 1 in net (e.g. q70/H120 +$7,584 vs +$3,966, q80/H120 +$10,826 vs +$7,208, q65/H90 +$10,305 vs +$6,721) - see scale_all.csv for p-values.
- **Risk vs the bot's loss limits:** stop 300 pts = $600 per MNQ; 2 MNQ = $1,200 (+ costs -> -$1,209 observed worst), 3 MNQ = $1,800. Bot limits: per-trade risk cap $500 (sleeve lifted to $650 -> a 2-MNQ entry would be trimmed/skipped unless `GAPFADE_RISK_CAP_USD` >= ~1,300), day-session loss limit -$500, overnight -$300, daily profit target +$2,000 (limits only block NEW entries after they are hit; they do not cap an open trade). One 2-MNQ stop-out = 2.4x the day-session limit (3 MNQ = 3.6x); the bot is also configured MAX_CONTRACTS=1. The sleeve currently supports a fixed contract count only (no sizing logic).
- **Does it beat flat 1 MNQ out of sample?** In dollars yes (B +$7.1-9.9k vs +$4.5k) but return-per-drawdown only improves modestly (net/DD 2.7 flat -> 3.1 q90-scheme, 3.5 buckets) and the p-values are 0.88-0.94 with a leverage confound; it is not an independent edge. **Recommendation: stay flat 1 MNQ. If Shawayne wants more size, run flat 2 MNQ deliberately (or the q90 2x scheme) only after raising the per-trade cap and loss limits to match (~$1,200-1,300 per trade), and expect worst day -$1.2k.**

---
## Addendum 7 (2026-10-28): do other markets give a "tell" for MNQ? (BACKTEST / RESEARCH ONLY)
Question from Shawayne: besides ES, does any symbol lead MNQ or move inversely in a way that predicts MNQ's next 30-120 min, the gap fade (q75/H90) or the 3R baseline? Scripts `fetch_multi.py`, `sr_x7_data.py` (features), `sr_x7a.py` (Q1), `sr_x7b.py` (Q2), `sr_x7c.py`/`sr_x7d.py`/`sr_x7e.py`/`sr_x7f.py` (Q3); outputs `x7_ovn.csv`, `x7_daily.csv`, `x7a_all.csv`, `x7b_hourly.csv`, `x7b_5m.csv`, `x7c_gapfade.csv`, `x7c_baseline.csv`, `x7d_baseline_bigday.csv`. No live file touched (bot PID 35724 still running).

### Data, alignment, limits
- **Free sources only:** yfinance (hourly 730 d, 5-minute last ~60 d, daily). 48 symbols: ES, YM, RTY futures; ^VIX, ^VIX9D, ^VIX3M, VIXY, UVXY; DX-Y.NYB, UUP, EURUSD, USDJPY; ^TNX, ^FVX, ^IRX, ZN/ZB/ZT; GC, CL; BTC, ETH; IWM, ^RUT, RSP, SPY, QQQ, ^SPX, ^NDX, SOXX, SMH, XLK, NVDA/AAPL/MSFT/AMZN/GOOGL/META/TSLA; ^N225, ^GDAXI, ^FTSE, ^HSI, ^STOXX50E; HYG, LQD, TLT. Spreads RSP/SPY, IWM/SPY, SMH/QQQ computed.
- **Time zones:** yfinance stamps are UTC; converted to ET with DST. Overnight move = log change from the 16:00 ET prior-session close (last bar starting <=15:59 ET) to 09:00 ET (last bar starting <=08:00 ET), both known before 09:30; MNQ target and gap come from the local 1m file (corr of yf NQ=F with local MNQ 0.97 on this basis; 465 days). Asia/Europe are covered in that window (N225 and HSI sessions are fully inside it; DAX/FTSE/STOXX only until 09:00).
- **Resolution:** hourly only for the overnight features and the 2-year lead/lag; the 5-minute lead/lag exists only for 2026-08-19..10-28 (about 50 days; yfinance 60-day cap), using NQ=F as the MNQ proxy because local MNQ 1m ends 2026-09-21. Stock/ETF premarket hourly bars are thin (RSP etc. have gaps); continuous futures can have roll jumps.
- **Cash indices with no overnight data:** ^SPX, ^NDX, ^RUT, ^VIX3M, ^VIX9D, ^TNX, ^FVX, ^IRX are 97% zero overnight (cash hours only) so they were NOT used for the overnight test; they are used only as prior-session daily features. 24h yield proxies = ZN/ZB/ZT futures.
- **Not available free / skipped:** VX1/VX2 futures (proxy: VIX9D/VIX/VIX3M term ratios and VIXY/UVXY); CBOE put/call (free archive stops at 2019-10-04, useless for 2024-26); intraday breadth (proxies RSP/SPY, IWM/SPY, HYG, SMH/QQQ); FRED yields are end-of-day lagged and duplicate ^TNX/ZN so were not added.
- **Sanity:** overnight moves correlate with the MNQ overnight gap as expected: QQQ 0.96, ES 0.93, SPY 0.91, SMH 0.88, RTY 0.83, NVDA 0.74, BTC 0.50, VIX/VIXY/UVXY about -0.82/-0.83 (the "moves opposite" market), DXY 0.03, TNX/ZN ~0, GC 0.16, CL -0.21. These are the gap restated, not information beyond it.

### Variants tried (total about 1,254) and the multiple-testing discount
Q1 474 (39 symbols x raw/residual x all-days/strong-days x 3 horizons, minus MNQ's own gap); Q2 45 hourly + 210 five-minute; Q3 gap fade 412 evaluated filters, baseline 56 skip-filters + 54 big-move-day splits + 3 composites. At a 5% level about 60 would "pass" by luck; all conclusions below are calibrated against **permutation nulls** (shuffling days/sessions so the symbol is unrelated to MNQ, 100-1,000 shuffles) rather than a flat discount.

### Q1. Does the overnight move predict MNQ's next 30/60/120 min from the 09:30 open?
Rule: trade MNQ at the 09:30 open (1 tick/side + $4), direction = sign of the overnight move (raw) or of its residual after removing the MNQ gap (expanding beta), held 30/60/120 min; direction (momentum vs inverse) frozen on window A (2024-11..2025-10), scored on B (2025-11..2026-09); pass = B net>0, t>=1.28, slip-4 and 3-min delay net>0, both halves of B>0.
- **20 of 474 variants pass; the permutation null passes 31 on average (95th percentile 63).** Real max B net +$15.4k vs null median-max +$11.9k (95th pct +$17.2k). **Nothing beats chance.**
- Best: MSFT residual (momentum, 120 min): n 210, B +$15,431, t 2.73, A +$2,658, halves +$9.6k/+$5.9k, slip4 +$14.8k, delay3 +$15.8k, corr 0.12; AAPL residual +$14.7k (A +$1.4k), TSLA/AMZN/SPY residual +$8.8-9.9k. But NVDA residual loses (mean -$11.2k), so there is no coherent "mega-cap" cluster, and none of this exceeds the null. Asia/Europe raw (FTSE -0.13, N225 -0.12 correlation with the next 2 h, i.e. same direction as fading the gap) are within the best-of-117 range expected from chance and are largely the gap itself. VIX/VIXY/UVXY overnight moves (the inverse market) add nothing beyond the gap: residual versions lose.
- **Verdict:** no overnight tell for the next 30-120 min.

### Q2. Do intraday moves lead MNQ by 5-60 minutes?
- **Hourly, 2 years (45 symbols, bar-exact MNQ windows from local 1m, RTH 10:00-15:30):** partial lead correlation (X's hour -> MNQ next hour, controlling for MNQ's own hour) is 0.00-0.11; only TLT 0.07, LQD 0.10, HYG 0.11 reach |t|>=2 (bond/credit ETFs; likely NAV/timing artefacts) and they fail the frozen test (HYG's direction flips between windows, B net -$13k). **0 of 45 tradable variants pass (null passes 0.4 on average)**; costs of $5 per trade exceed any gross edge.
- **5-minute, last ~50 days only (42 symbols x lags 5/10/15/30/60 min = 210 tests):** 16 tests have |t|>=2 versus ~10.5 expected by chance, none >=3; best are CL (-0.05, lag 10 min), GC/HYG/LQD (+0.05, lag 10), IWM/RTY (about -0.05 at 30, +0.05 at 15). Best gross edge about $2 per MNQ trade vs $5 cost. ES itself: partial correlation +0.04 at 10 min (t 1.8), -0.03 at 30 min; gross +$1.4 per trade. Window is one regime of 50 days, so even these are unconfirmed.
- **Verdict:** no symbol leads MNQ enough to trade; MNQ and ES/QQQ/SPY price in the same minute.

### Q3a. Filter / direction check on the q75/H90 NY gap fade (137 trades, +$7,136, A +$2,626, B +$4,510)
Filters per symbol: keep trades when its overnight move (raw and gap-residual) or prior-session move/level is large vs small, or moves with / against the trade direction; VIX level and term structure (VIX/VIX3M, VIX9D/VIX) high/low; prior-session DXY, yields, gold, oil, BTC, RSP/SPY etc.
- **412 variants (kept >=40 trades), 0 pass** (both windows improve and t>=1.64). Permutation null: 0.1 pass on average (max 6); real max t 0.61 vs null 95th percentile 1.71, i.e. **the real data is weaker than random.**
- Largest improvements, none significant: AMZN residual with the trade (87 trades = 3.8 a month, +$8,472, paired p 0.73), VIX moving with the trade direction (122 trades, +$7,670, p 0.68), CL calm prior day (52 trades). The earlier ES/NQ divergence idea (Addendum 6, p 0.76) remains the best of all and is also not significant.
- **Verdict:** nothing filters or confirms the gap fade. Do not add a cross-market gate.

### Q3b. Filters on the 3R baseline (2,473 trades, +$29,313, A +$16,078, B +$13,112)
Features known before entry: last completed US session's move/level per symbol (VIX level, term, DXY, 2y/5y/10y yields, gold, oil, BTC, ...).
- **Skip-filters (56 variants, kept >=45% of trades): none improves net; all are negative (-$3k to -$7k)**. Skipping big-move days hurts: the baseline earns its money in volatile regimes. VIX-high-only (-$7.3k) and VIX-low-only also lose.
- **Opposite idea (boost on big-move days; informational only, live is 1 contract):** mean P&L/trade after a big prior-session move (|move| >= past 80th pct) vs the rest: MSFT $32 vs $7 (t 2.8), UVXY $35 vs $8 (2.8), HYG $37 vs $9 (2.6), SMH/QQQ $29 vs $8 (2.5), BTC $28 vs $8 (2.3), SOXX/SMH/TSLA/ETH about $25-30 vs $8 (t 2.2-2.3), same sign in A and B. 10 of 54 features have t>=2. Permutation null: max t has median 2.20 (95th pct 3.45), so 2.84 is **P=0.20**; count of t>=2 is 10 vs null mean 2.3 (P=0.044). A single composite "number of risk assets with a big move" score: top third vs rest t 0.78 (P=0.27 vs null); the ">= median+1" cut gave t 2.32 (B: $24 vs -$2 per trade) but was one of three cuts tried, so post-hoc. The effect is a broad volatility-regime signal (the baseline likes volatile days), not a specific symbol.
- **Verdict:** no skip rule; a volatility-regime boost is plausible but unproven (and conflicts with MAX_CONTRACTS=1).

### Overall recommendation
No symbol gives a usable standalone tell for MNQ. Ignore: ES/QQQ/SPY (same information as MNQ), VIX/VIXY/UVXY (inverse but just the gap), DXY/EURUSD/USDJPY, yields (TNX/ZN/ZB/ZT), gold, oil, BTC/ETH, Asia/Europe, RSP/SPY, IWM, put/call (no free data), mega-cap residuals and credit ETFs as signals. Only pre-registered shadow logging is justified (below).
Shadow-log daily at ~09:25 ET (read-only, no trading): overnight moves (16:00->09:00 ET) of NQ, ES, VIX, UVXY, DXY, ZN, GC, CL, BTC, N225, FTSE, HYG, MSFT, AAPL, AMZN, TSLA, SMH; prior-session VIX level and VIX/VIX3M; the number of risk assets with a big prior-session move. Re-test in ~6 months with only three frozen hypotheses: (1) MSFT/AAPL residual momentum over 120 min; (2) the Addendum 6 NQ-ES divergence filter on the gap fade; (3) the baseline "big-move regime" score >= median+1.
---
## Addendum 8 (2026-10-28): is the baseline worse in the afternoon / on Fed days? (BACKTEST ONLY)
Question from Shawayne (a trader's claim: "no trades after 1:15 PM; big moves 2:30-4:30 PM ET on FOMC days"). Scripts `sr_x8_desc.py` (buckets), `sr_x8.py` (full re-simulations; patches a copy of the engine in memory, repo files untouched), `sr_x8_an.py` (metrics + paired bootstrap); outputs `x8_buckets.csv`, `x8_rules.csv`, `x8_trades_*.csv`. Backup of this report: `REPORT.md.bak_pre_add8`. Live bot (PID 35724) not touched, no orders.

### FOMC dates used (16 in sample, all verified on federalreserve.gov/monetarypolicy/fomccalendars.htm, statement 2:00 PM ET on day 2, press conference 2:30 PM ET)
2024-11-07, 2024-12-18; 2025-01-29, 03-19, 05-07, 06-18, 07-30, 09-17, 10-29, 12-10; 2026-01-28, 03-18, 04-29, 06-17, 07-29, 09-16. (2025-08-22 was a notation vote, not a meeting; excluded.) None unverifiable. Out of the data (it ends 2026-09-21): 2026-10-28 (today) and 2026-12-09. "FOMC day" = the ET calendar date of the trade entry. **Small sample: 16 FOMC days (10 train, 6 in 2026), 85 trades.**

### Part 1: baseline trades by entry time (ET) and day type (baseline: 2,473 trades, +$29,313, PF 1.22, DD $4,449)
| Entry bucket (all dates) | n | net | PF | $/trade | DD |
|---|---|---|---|---|---|
| overnight / pre-open 18:00-09:30 | 1,663 | +$13,333 | 1.2 | $8.0 | n/a |
| RTH 09:30-13:15 | 586 | +$12,333 | 1.36 | $21.0 | |
| 13:15-14:00 | 26 | +$646 | 1.49 | $24.9 | |
| 14:00-16:00 | 142 | +$3,118 | ~1.4 | $22.0 | |
| 16:00-18:00 | 56 | -$117 | ~0.97 | -$2.1 | |
Non-FOMC days (2,388 trades): 13:15-14:00 +$646 (PF 1.49, 26 trades); 14:00-16:00 +$2,642 (PF 1.36, 129); 16:00-18:00 -$200 (PF 0.93, 55). Train (2024-11..2025-12) vs 2026 on non-FOMC days: 13:15-14:00 +$1,216 (16 trades) vs -$569 (10 trades); 14:00-16:00 +$1,829 (75, PF 1.46) vs +$813 (54, PF 1.24); 16:00-18:00 -$528 (38) vs +$329 (17). All 13:15-16:00 entries together: 168 trades, +$3,764, PF 1.41, $22.4/trade; train $31.8/trade (101), 2026 $8.3/trade (67, PF 1.13).
- **The afternoon is not where the baseline loses.** 13:15-18:00 entries (224 trades): $16.3/trade vs $11.4 for the rest; difference 95% CI -$18 to +$28 (P(afternoon worse) 0.34) - no evidence of a problem. 14:00-16:00: $22.0 vs $11.2, P(worse) 0.24. Only the 16:00-18:00 sliver is weak (56 trades, -$117, not significant).
- Caveat: the afternoon edge did shrink in 2026 ($31.8 -> $8.3 per trade over 168 trades, 13:15-16:00). It is still positive (+$554) and cutting it still cost money (see Part 2). Worth watching, not acting on.
- **FOMC days (85 trades on 16 dates): +$5,034, PF 2.72, $59/trade vs $10/trade on other days.** Train 59 trades +$2,411 (PF 2.11); 2026 26 trades +$2,623 (PF 4.46). 13 of 16 FOMC dates were net positive (worst -$431). Most of it is overnight/pre-open trades dated that calendar day (54 trades, +$4,327, PF 3.85: breakouts before the 14:00 statement and after 18:00), RTH 09:30-13:15 17 trades +$148 (PF 1.16, 2026: -$218), **14:00-16:00 on FOMC days 13 trades +$476 (PF 2.0; train +$165, 2026 +$311)**. Bootstrap FOMC avg $59 vs $10: diff CI +$11 to +$92, P(worse) 0.006 - but 16 days, one of many cuts and driven by overnight trades, so I treat it as "not worse", not as an edge to size on. Trades that were open across 14:00 on FOMC days: 6 trades, net -$69 (train +$146, 2026 -$215; one -$328 and one -$295 stop-out) - 6 trades prove nothing.

### Part 2: rules re-simulated through the full engine (concurrency, daily loss/target, cooldown re-run). 3R base reproduced: 2,473 trades +$29,313.
Skipped entries are dropped at the signal; "flat" closes open trades at the 1m close minus 1 tick slippage. Train = entry before 2026-01-01; 2026 = from 2026-01-01. delta = rule minus baseline net. "P(better)" = paired day-bootstrap probability (10,000 resamples of per-day P&L deltas, by exit date) that the rule beats baseline; folds = 6 equal-time folds (rule beats baseline in x/6).
| Rule | trades | net | PF | DD | delta all | delta train | delta 2026 | P(better) all / 2026 | folds |
|---|---|---|---|---|---|---|---|---|---|
| **Base** | 2,473 | +$29,313 | 1.22 | $4,449 | | | | | |
| R1 ALL days: no entries 13:15-18:00 (the trader's rule) | 2,364 | +$22,975 | 1.18 | $4,385 | **-$6,339** | -$3,896 | -$2,442 | 0.045 / 0.11 | 2/6 |
| R1b ALL days: no entries 13:15-16:00 | 2,381 | +$23,532 | 1.19 | $4,597 | -$5,781 | -$4,601 | -$1,180 | 0.018 / 0.25 | 2/6 |
| **R2 FOMC days only: no entries 13:15-16:00** | 2,460 | +$28,838 | 1.22 | $4,449 | **-$476** | -$165 | -$311 | 0.22 / 0.00 | 1/6 |
| **R3 FOMC days: no entries 13:45-16:00 + flat at 13:45** | 2,459 | +$28,721 | 1.22 | $4,310 | **-$592** | -$339 | -$254 | 0.23 / 0.28 | 1/6 |
Sensitivity variants (all worse than base, same direction): ALL days no entries 14:00-18:00 -$6,677 (2026 -$3,219); 15:00-18:00 -$4,296 (2026 -$2,079); 12:00-18:00 -$6,317 (2026 -$2,927); FOMC no entries 13:15-18:00 -$600; FOMC no entries 14:00-16:00 -$476; FOMC no entries 13:45-16:00 w/o flatten -$476; FOMC flatten at 13:45 only (entries allowed) -$23 (train -$80, 2026 +$58, DD $4,310 vs $4,449, 2 more trades taken afterwards (2,475 vs 2,473) - a wash); FOMC no entries 13:45-16:00 + flat at 13:55 -$600; skipping FOMC days entirely -$3,357 (2026 -$1,232).
- **Variants run: 13 re-simulated rules + 26 descriptive bucket splits (5 buckets x FOMC/non x all/train/2026). None of the 13 improves the baseline in net, in either window, or in the fold count.** One rule shows a nominal +$58 in 2026 (flatten-only) but -$80 in train and -$23 overall, so it fails the both-windows test. Nothing passes, so no slippage/entry-delay stress or live-code change was needed, and there is no post-hoc "winner" to discount.
- **Drawdown:** the only DD gains are FOMC flatten rules ($4,310 vs $4,449, -$139), which appear to come from closing a few open trades early (e.g. the 2025-01-29 -$328 stop-out, entered 13:05, stopped 14:41); they cost $23-$592 of net. Not a real risk reduction.
- **FOMC-day sample is small:** 16 days / 85 trades (rules R2/R3 change 13-14 trades in total). Even so, the direction is consistent: these trades made money in both windows (13:15-16:00 entries on FOMC days: train +$165, 2026 +$311), so there is nothing to remove. Cannot prove the rules are harmless with 13 trades; only that nothing here supports them.

### Verdict
- **Keep the 3R baseline as is.** The claim "afternoon is bad / stop after 1:15 PM" is not supported: the baseline earns $22/trade from 13:15-16:00 (+$3,764 over 168 trades), and banning those entries on all days costs about $5.8-6.3k over the 2 years (2026 alone -$1.2k to -$2.4k), significant at about 95% that the rule is worse. FOMC days were the baseline's best days (+$5,034, $59/trade, 13/16 positive), and a Fed-day pause or flatten gives up $0.5-0.6k with no drawdown benefit worth the loss.
- The one soft spot to monitor: 2026 afternoon entries average $8/trade vs $32 in train (still positive), and 16:00-18:00 entries are about flat (-$117 over 56 trades). Neither is tested as something to cut.
- Nothing about the Warsh-era first Fed meeting (and today's FOMC, 2026-10-28) can be tested yet; today's data is outside the backtest.

---
## Addendum 9 (2026-10-28): "capitalize on the midday selloff" (BACKTEST / ANALYSIS ONLY)
Question from Shawayne: news around 12-1 PM ET today changed the chart's trajectory (dip and selloff) - how do we capitalize? Scripts `x9_today.py`, `x9_today2.py` (today's bars), `x9_test.py` (backtest); outputs `x9_rules.csv`, `x9_events.json`, `x9.out`, `x9_nq_1m_today.csv` (Yahoo NQ=F 1m). Backup of this report: `REPORT.md.bak_pre_add9`. Live bot (PID 35724) not touched, no orders, nothing bought.

### Part 1: what actually happened today (Yahoo NQ=F 1m; MNQ tracks it within a few points)
- **No news-driven selloff at noon-1 PM.** 12:00-12:59 ET NQ ranged only 29,827-29,921 (open 29,849, close 29,885). The largest midday move was a **~109-point dip, 29,920 (12:54) to 29,811 (13:28)**, a normal-size wiggle, and no news item I found explains it. I cannot verify any cause for it and will not invent one.
- **What the news really was:** (a) Fed held at 3.75-4.00% with two hawkish dissents and Warsh's first press conference (2:00 / 2:30 PM): NQ whipsawed 29,786 to 30,009 between 14:16 and 14:43, but AP/Barron's say stocks "shrugged"; S&P +0.3%, Nasdaq +0.27% at the close. (b) Earlier AI news was bullish (SK Hynix record profit, Oracle +11% on a 2030 outlook). (c) **The real selloff came AFTER the 4 PM cash close**: NQ spiked to 30,178 at 16:01 then fell to 29,674 by 16:50 (**about -504 points, roughly $1,000 per MNQ**), as Alphabet fell about 9% after a revenue/op-income miss and negative free cash flow, with ServiceNow -15% and Fortinet -25%, while Microsoft rose about 2% (CNBC after-hours movers, IBD, Newsquawk). That is the move the bot's 4:40 PM log shows (29,740). So the "selloff" happened roughly 3 hours later than he remembers, and it was an earnings gap, not midday news.
- Sources: CNBC (cnbc.com/2026/10/28/stocks-making-the-biggest-moves-after-hours-googl-msft-cmg-now.html), AP News (Fed/stocks recap), Economic Times Fed live blog, IBD after-hours piece.
- Bot note: at 16:41 the bot printed a trend SELL signal and skipped it ("SHORT setup is against open trade 1 (LONG)") because the paper long from the morning was still open. That is a design rule (only same-direction adds), not a bug.

### Part 2: objective backtest of "midday selloff" rules (2-year MNQ 1m; 466 trading days; train = before 2026-01-01, 2026 window = from 2026-01-01)
**Rule grid, fixed before any result was seen:** at a 1-minute close between 11:30 and 14:00 ET, if price is X points below the highest high of the trailing W minutes, take the first such event per day. X in {80, 120, 180} pts, W in {60, 90} min (6 event definitions; 299/321, 209/244, 106/142 event days). Enter at the next 1m open with 1 tick slippage; exit at the open H minutes later (H in {30, 60, 120}, capped at 15:55) with 1 tick slippage; $4 round trip; 1 contract; no stop. Two directions: **buy-the-dip (fade)** and **continuation short**. **36 variants total; every one is reported (no dropped variants).**
**Significance:** each variant's average is compared with a placebo of the same number of random entries (any day, any minute 11:30-14:00, same side and hold, 4,000 draws); Bonferroni x36 for the family. Pass requires positive average in train AND 2026.

| Result across the 36 variants | count |
|---|---|
| Positive average after costs (all data) | 8 of 36 (28 lose) |
| Positive in BOTH train and 2026 | 3 of 36 |
| Raw placebo p < 0.05 | 0 of 36 (about 1.8 expected by chance) |
| Raw p < 0.10 | 1 of 36 (about 3.6 expected) |
| Bonferroni-adjusted p < 0.05 | 0 |

- **Continuation short (chase the selloff): loses in every threshold.** all 18 variants lose money overall (2 of them are positive in train only by pennies); the worst, X=180/W=60, H=60 is -$30/trade (-$3,179, 106 trades, both windows negative). Shorting after a midday drop is the wrong side on this data.
- **Buy-the-dip: nothing reliable.** Small thresholds (80, 120) average about -$8 to +$6/trade, essentially the cost of trading ($5 per trade). The large-drop buckets look better but are post-hoc: **X=180/W=60 fade, H=60: +$20/trade, +$2,119, 106 trades, PF 1.22, train +$15.1 (59), 2026 +$26.1 (47), raw p 0.08, Bonferroni p 1.0.** Its sibling H=120 is +$41.5/trade in train but -$4.5 in 2026, and X=180/W=90 H=120 is +$21.2 in train and -$32.5 in 2026, which are exactly the flips a noise result shows. The two others positive in both windows (X=120/W=60/H=120 +$6.5, X=180/W=60/H=30 +$6.4) are $6/trade, below the cost noise. **All three are post-hoc picks from a 36-variant grid: treat as noise, not an edge.**
- **Slippage stress was not run** because no variant passed the both-windows + significance screen.

### Part 3: did the 3R baseline already capture these days?
- Baseline average day P&L is $65 on X=80/W=60 event days vs $39 on other days (299 vs 167 days), $66 vs $43 for X=120/W=90, but **$41 vs $56 on X=180/W=60 days** (106 days; baseline is a little worse than usual on very big drop days). Mixed and not a systematic effect.
- Baseline trades entered within 4 hours after the trigger on event days (X=80/W=60): 133 trades, +$3,324, PF 1.52 (train +$2,876, 2026 +$448); X=180/W=60: 40 trades, +$1,520, PF 2.13 (train +$991, 2026 +$529). So **the baseline already trades through these days and is positive after the trigger**, with the usual 2026 softening (X=120/W=60: 2026 -$457). This is an overlap description, not a new rule.
- If the dip-buy fade were added on top it would trade on the same days as the baseline (which is long/short by its own signals) and add concurrency risk without proven edge.

### Part 4: what today's data says (anecdote, outside the 2-year data, one day)
Only the X=80 rules fired (13:18 ET; the 109-point dip). Buying it: H=30 +$3.5, H=60 -$49, H=120 +$38 per MNQ; shorting it: -$13.5, +$39, -$48. A coin flip, as the backtest says. None of the larger thresholds fired today. The biggest drop of the day (-504 pts) came after 16:00, outside both this test's window and the regular session.

### Verdict
- **Nothing to add.** Neither "buy the midday dip" nor "short the midday selloff" holds up on both windows after costs and a 36-variant discount. Shorting after a drop loses in every setting. The only both-window-positive large-drop dip buys are post-hoc and not significant.
- Today's big move was an after-hours earnings gap (Alphabet miss), not midday news; the bot already trades the 16:00-18:00 session (baseline there is about flat: -$117 over 56 trades, Addendum 8) and a rule built on one earnings night would be a guess.
- Possible follow-up only if wanted: shadow-log midday drops >=180 pts (60 min) and the follow-through, then re-test in ~6 months (about 1 event per 4-5 days at that threshold, so the sample grows fast). No live change.

## Addendum 10 (2026-10-28): is there a proven edge at night, and should the bot trade it exactly like the day? (BACKTEST ONLY)

Question (Shawayne): "i want the bot to sell if the setups appears same like it would in the day.. backtest and find a proven edge in the nights." Concern: the 6PM add-on goes long at 18:00 even in a down-trend. Nothing here touches the live bot; new files only (`sr_x10_lib.py`, `x10_variants.py`, `x10_run.py`, `x10_part1.py`, `x10_an*.py`, `x10_*.csv`, 62 `x10_trades_*.csv`).

Method: the 3R baseline re-run through a patched copy of `run_spec3` (exactly reproduces 2,473 trades, +$29,313, PF 1.22, max DD $4,449; train +$17,361, 2026 +$11,952, 19/24 months green). Sessions by signal-bar end (ET): day 09:30-16:00, evening 16:00-18:00, night 18:00-02:00, overnight 02:00-09:30. 59 night variants (+3 post-hoc combos, kept out of the family) were each re-simulated through the full engine (so concurrency/re-routing is included), then scored with: train(<2026) vs 2026, window A (2024-11..2025-10) vs B (2025-11..2026-09), 6 equal-time folds, paired day-bootstrap (10,000, per-exit-day P&L deltas) P(rule better), slippage 3 and 4 ticks per side, frozen-forward selection, expanding walk-forward over the family, and a random-removal null (same number of night trades removed at random) plus Sidak discount. Strict screen = delta>0 in train AND 2026, >=4/6 folds better, bootstrap >=0.90. Backtest does not model live re-entry.

### Part 1: baseline by session / direction / sleeve (standalone, `x10_part1.csv`)
| Slice | Trades | Net | PF | Avg | Train net | 2026 net | Months green |
|---|---|---|---|---|---|---|---|
| ALL | 2,473 | +29,313 | 1.22 | 11.9 | +17,361 | +11,952 | 19/24 |
| Day 09:30-16:00 | 754 | +16,097 | 1.38 | 21.3 | +10,790 | +5,307 | 18/23 |
| Evening 16:00-18:00 | 56 | -117 | 0.96 | -2.1 | -445 | +329 | 8/20 |
| Night 18:00-02:00 | 833 | +12,220 | 1.30 | 14.7 | +8,505 | +3,715 | 17/24 |
| Overnight 02:00-09:30 | 830 | +1,113 | 1.02 | 1.3 | -1,488 | +2,601 | 15/23 |
| 18:00-09:30 all | 1,663 | +13,333 | 1.15 | 8.0 | +7,017 | +6,316 | 18/24 |

By direction (night = both already traded): night long 485 / +5,064 (PF 1.24), night short 348 / +7,156 (PF 1.38); overnight long 414 / +2,124 (train -2,052, 2026 +4,176), overnight short 416 / -1,011 (train +563, 2026 -1,575). The overnight long/short signs flip between windows = noise.
By sleeve at night (18-02): breakout 579 / +9,100 (short 297 / +7,918, PF 1.47, train +6,090, 2026 +1,828; long 282 / +1,182, PF 1.08, 2026 -818); mean-rev 73 / +1,366 (long 39 / +1,459, PF 2.46); trend 38 / -209 (shorts 17 / -669, PF 0.35); 6PM long 143 / +1,963. Overnight (02-09:30): breakout 713 / +1,608 (PF 1.04), mean-rev 69 / -746, trend 48 / +250.
Evening: breakout shorts 18 / -1,286 (PF 0.36); the rest tiny.
So the only night window with a standing edge is 18:00-02:00, driven by breakouts, strongest on the short side. 02:00-09:30 is statistically zero (t=0.26) and window-dependent. The day session carries more than half the P&L.

### Part 2: the 6PM long (addon_e1) and a mirror 6PM short
- Standalone the 6PM long is 143 trades, +$1,963, PF 1.39 (train +$188 only, 2026 +$1,776) = helped in 2026 only.
- Removed from the full baseline (`e1_off`, re-routing included): -$449 (train +$1,329, 2026 -$1,778; folds better 3/6; P(better) 0.41; slip3 -$239, slip4 -$536). Net: neutral. Turning it off is harmless (about -$20/month) but is not proven better or worse.
- Mirror 6PM short (RSI<=40.5 and down >=1.48 ATR; 975 raw signals): added on top -$3,438 (2026 -$3,352); swapped for the long -$3,580. **Symmetry does not hold; do not add a 6PM short.** (The evening 16-18 shorts also lose.)

### Part 3: re-check of "trim overnight breakouts 2-8 AM is worth ~+$25k, 20/23 months green"
That +$25k figure was a TOTAL net of a different (earlier-engine) run, not a gain. On the 3R baseline: removing 2-8 AM breakouts = **-$4,073** (train +$34, 2026 -$4,107, 2/6 folds, P(better) 0.19); trimming all 2-8 AM trades -$2,377 (2026 -$3,436); 2-9:30 breakouts -$4,009; 0-9 breakouts -$9,503; 3-6 AM breakouts +$755 (train +$2,776, 2026 -$2,021, P 0.59, slip4 +$897) = one window only. **Claim not supported; do not trim.**

### Part 4: night filters (59 variants; full table `x10_results.csv`, with null columns `x10_results2.csv`)
Headline deltas vs baseline (all / train / 2026 / folds better / P(better)):
- align_htf2880 (night breakout/trend only with the ~10-day trend): +4,529 / +2,140 / +2,389 / 3 / 0.81. Slip3 +4,090, slip4 +3,631. Positive in A and B (+3,898 / +631). Fails strict screen (3/6 folds, P<0.90; fold deltas -31, +4,602, -500, -4,010, +3,260, +1,208: one fold drives it).
- off_18_0930_trend_short (no trend-sleeve shorts 18:00-09:30; ~42 trades): +1,985 / +1,115 / +870 / 5 / 0.99. Slip3 +2,131, slip4 +2,254. **The only strict-screen pass.** But trades removed are few (standalone +$872), found after reading the cell table, and the random-removal null gives emp p = 0.11; expected chance passes of the screen among 59 variants = 1.21, observed 1.
- off_h03 (+1,949) 2026 -2,569; off_evening_short +1,913 (2026 -354); off_h05 +1,516 (2026 +254, P 0.86); off_overnight_short +1,549 (train -1,022, 2026 +2,572; slip3 +1,847, slip4 +1,237) -> all one-window.
- Everything that turns whole night sleeves/directions off loses: off_night_all -10,097, off_18_0930_all -14,314, off_night_short -3,434, off_night_long -4,691, off_night_breakout -7,518, off_18_0930_long -11,712.
- Best-looking minimum random-removal p = 0.010 (off_h03, which fails 2026); Sidak over 59 variants = 0.447 (not significant).

### Part 5: forward / walk-forward / discount
- Frozen forward A->B: best-on-A align_htf2880 +3,898 -> +631 (holds, small); 27 variants positive in A, only 5 positive in B. Train->2026: best off_h03 +4,518 -> -2,569; 29 positive in train, only 3 positive in 2026. Spearman first vs later delta 0.05-0.18 (p 0.18-0.73): no persistence.
- Expanding 6-fold walk-forward (pick best on earlier folds, score next): total out-of-sample -$6,444 vs baseline over folds 2-6. Cell-pruning walk-forward (drop cells with negative avg on earlier folds): -$2,445 / -$301 / -$1,119 (three versions). Selecting night cuts from history lost money out of sample.
- Slip3/Slip4 baselines: +$25,477 / +$22,954 (still positive; every conclusion above unchanged).

### Post-hoc flags (not in the 59)
combo_A (6PM long off + no night trend shorts) +1,238 (2026 -1,206); combo_B (A + htf2880) +3,526 (2026 +1,510, 4/6 folds, P 0.73, B window -803); combo_E1htf (6PM long off + night with 10-day trend only) +2,856 (train +2,027, 2026 +829, 4/6 folds, P 0.69; slip3 +2,286, slip4 +1,493; DD 3,972 vs 4,449). All were assembled after seeing the tables = hindsight; not proven.

### Verdict
- **No night rule is proven.** Keep the baseline unchanged: it already takes breakout/trend/mean-rev setups both long and short at night (REVAMP_OVERNIGHT_SLEEVES=all). Night 18:00-02:00 is where the proven edge lives (+$12,220, PF 1.30, positive in train and 2026, driven by breakouts, shorts best). 02:00-09:30 is ~zero (+$1,113, PF 1.02) and not proven either way.
- Do not trim 2-8 AM breakouts, do not ban hours, do not add a 6PM short, do not switch whole night sleeves/directions off (all lose big).
- 6PM long: neutral (-$449 if off). OK to turn off to match Shawayne's wish: `set "ADDON_E1_6PM=0"` in the launcher (default 1), plus restart; NOT applied.
- Forward paper-watch only: align_htf2880 for night trades, and no night trend shorts. Re-test after ~6 more months.


## Addendum 11 (2026-10-28): holdout re-test of the two night filters (off_18_0930_trend_short, align_htf2880) (BACKTEST ONLY)

Scope: Shawayne asked for a paper watch on the two Addendum-10 filters; before building it he asked for a strictest-possible holdout re-test with 4-tick slippage, folds and trade counts. Nothing live touched; new files only: `x11_run.py`, `x11_an.py`, `x11_extra.py`, `x11_trades_*.csv` (16), `x11_windows.csv`, `x11_results.txt`. REPORT.md backed up to `REPORT.md.bak_pre_add11`. No paper-watch script was built.

**Holdout honesty (read first).** There is NO truly untouched data. The bar file ends 2026-09-21 and every month from 2024-11 to 2026-09-21 was in the sample used to look at and pick both filters (they came out of the 59-variant cell table). No newer MNQ bars exist on the box (the 1m file was last built 2026-09-22; free Yahoo intraday does not go back far enough). The only never-used slice is 2024-09-30..2024-10-31 (about 5 weeks, 105 trades), which was warm-up before the Nov-2024 start. So the "holdout" below = (a) the last 3-4 months as a pseudo-holdout (already seen, so NOT independent), (b) frozen-forward splits, (c) the unused Oct-2024 month, (d) 6 equal-time folds and calendar quarters. Method: baseline 3R re-simulated (reproduces 2,473 trades, +$29,313 at 1 tick; +$22,954 on 2,456 trades at 4-tick slippage), each filter re-simulated through the full engine (concurrency/re-routing included).

### Results (delta vs baseline, same engine, same period)
| Window | F1 trend-short off: delta 1-tick / 4-tick | F1 trades (base->filt) | F2 align_htf2880: delta 1-tick / 4-tick | F2 trades (base->filt) |
|---|---|---|---|---|
| Full Nov24-Sep26 | +1,985 / +2,254 (P better 0.99) | 2,473->2,441 (49 removed, 17 re-routed in) | +4,529 / +3,631 (P 0.81 / 0.75) | 2,473->2,149 (578 removed, 254 in) |
| Train (<2026) | +1,115 / +1,364 | 1,487->1,468 | +2,140 / +1,591 | 1,487->1,314 |
| 2026 YTD | +870 / +890 | 986->973 | +2,389 / +2,040 | 986->835 |
| Last 4m (May22-Sep21) | +714 / +730 (P 0.995) | 421->415 | +1,728 / +1,745 (P 0.88) | 421->352 |
| Last 3m (Jun22-Sep21) | +634 / +646 (P 0.98) | 325->321 (5 removed, 1 in) | +1,788 / +2,113 (P 0.91-0.95) | 325->274 |
| Unused Oct-2024 | 0 / 0 (rule never fired) | 106->106 | -608 / -736 | 106->91 |

Six equal-time folds (delta, 4-tick; 1-tick in brackets): 
- F1: +392 [+383], -346 [-352], +459 [+235], +943 [+929], +77 [+76], +730 [+714] -> 5/6 folds better, never worse than -$352.
- F2: -405 [-31], +4,383 [+4,602], -310 [-500], -4,545 [-4,010], +2,875 [+3,260], +1,632 [+1,208] -> 3/6 better; swings of +/-$4k per fold.
Calendar quarters (4-tick): F1 positive in 6 of 7 (Q2-25 -205, Q4-24 -40). F2 positive in 5 of 7, but Q4-25 is -2,724 (baseline +2,298 -> -426) and Q4-24 -415.
Months with delta>0 / <0 (4-tick): F1 13 / 7 (4 flat); F2 12 / 12.
Frozen-forward (delta before cutoff | after cutoff, 4-tick): F1 cutoff 2025-07 +45 | +2,208; 2025-10 +504 | +1,750; 2026-01 +1,364 | +890; 2026-04 +1,644 | +610 (positive after every cutoff). F2 cutoff 2025-07 +3,895 | -265; 2025-10 +4,315 | -684; 2026-01 +1,591 | +2,040; 2026-04 +1,719 | +1,911 (a rule picked in mid/late 2025 would have lost afterwards).

### Where the gain comes from (important)
- F1: the 49 removed trades net -$956 (4-tick) and the 18 trades that took their place net +$1,297. So about 58% of the +$2,254 is "different trades filled the freed slots", not removal of losers. Last 3m: 5 trades removed (-$576), 1 added (+$70). The whole 3-month "holdout win" is 5 trades.
- F2: the 578 removed trades net **+$1,544** (they were net winners, 338 wins) and the 264 trades that replaced them net +$5,175. The entire +$3,631 comes from the re-routing/path effect of the freed slots, not from filtering out bad trades. That is the signature of noise/path dependence, which also explains the +/-$4k swings by fold and the loss on the unused month.

### Verdict
- **align_htf2880 (F2): NOT proven. Do not use and do not paper-watch as a candidate edge.** Fails the fold test (3/6), P(better) 0.75-0.81 full sample, loses in the one unused month, loses in Q4-25 by $2.7k, would have lost money out of sample if picked in mid-2025, and its gain comes from replacement trades rather than removed losers. The last-3-month +$1.8-2.1k looks good but is not independent and flips sign by period.
- **off_18_0930_trend_short (F1): holds up on every lens we can run, but is small and not independently confirmed.** 5/6 folds, 6/7 quarters, positive after every frozen cutoff, positive at 4-tick (+$2,254, PF 1.17->1.19, max DD $4,977->$4,728), last 3m +$646. Size: about $100/month on 1 MNQ, 49 trades in 22 months (~2/month); more than half the gain is replacement-trade luck; found after reading the cell table (59 variants tried, ~1.2 chance passes expected; random-removal null p = 0.11 from Addendum 10); no fresh data exists to confirm it. Honest label: **plausible, low-risk, tiny; unproven**. It does not meet "profitable and proven" on independent data. If applied it needs a code change (not an env switch today) and I would NOT change live for ~$100/month on an unconfirmed result. Better: collect real forward trades, then re-test; the only way to get a real holdout is new bars after 2026-09-21 (refresh the bar file, then re-run `x11_run.py`/`x11_an.py`, after raising the engine end date past 2026-10-01 and rebuilding revamp_base.pkl) and compare only trades after 2026-10-28.
- Both filters together add +$4,347 at 4-tick, but that is F2 driven and carries F2's problems.

