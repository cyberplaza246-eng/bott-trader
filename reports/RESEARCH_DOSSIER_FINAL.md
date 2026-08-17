# Final Research Dossier: MES/MNQ/NQ Trading-Edge Recovery & Discovery Program

**Status: DISCOVERY TRACK CLOSED**

---

## 1. Executive Conclusion

**The recovered live trading system and its available 5M OHLCV + Volume information set did not demonstrate a validated exploitable edge in the tested MES/MNQ/NQ data.**

This is not a statement that "no strategy was found" through insufficient effort. Across eight phases, every candidate behavioral effect that emerged from the recovered live system's own logic, from a bottom-up empirical scan of the market's statistical behavior, and from the one remaining information axis (volume) the live system consumed, was subjected to a disciplined, standardized process:

1. Reconstruction with an explicit no-lookahead audit before any test was run.
2. Discovery-side analysis only, with the final 20% of history held untouched.
3. Non-overlapping / event-based sampling wherever forward windows or persistent states could otherwise inflate apparent significance.
4. A dependence-matched baseline (never a comparison against zero, to avoid conflating instrument drift with a conditional effect).
5. Multiple-testing correction (Bonferroni) applied to every grid, not just a headline cell.
6. A single, pre-registered, locked configuration run exactly once against the untouched holdout.
7. MES and the MNQ/NQ pair reported and interpreted separately (MNQ and NQ are correlated readings of the same underlying index, not independent confirmations).

Several candidates looked real under naive analysis. Every one of them either failed to clear multiple-testing correction, failed to replicate in the untouched holdout, or turned out to be explainable by a phenomenon already rejected earlier in the program (chiefly: ordinary volatility clustering). None cleared all of discovery significance + holdout replication + economic magnitude + genuine novelty simultaneously.

---

## 2. Original Live-System Reconstruction

Source: `legacy/root_scripts/start_live_rithmic.py`, `legacy/src/core/ensemble_trader.py`, `legacy/src/ai/liquidity_sweep.py` (git tip `afd3623`, the version live at the end of the recovered trading window).

### Actual production signal path
`start_live_rithmic.py` → `EnsembleTrader.get_trading_signal()` → `LiquiditySweepAnalyzer.get_signal()` (sweep + MSS, the hard gate) → confirmation layers (EMA crossover, technical indicators, IntelligentTrader ML) → final signal.

**`enhanced_breakout` was NOT the live system's real decision logic.** It was only the exception-path fallback (`_check_simple_breakout()`), invoked when the ensemble was unavailable or threw an error — despite every saved trading log hardcoding `"strategy": "enhanced_breakout"` regardless of which path actually fired.

### Data resolution actually used in production
Both `start_live_rithmic.py`'s main loop (`get_candles(symbol, timeframe_minutes=5)`) and `ensemble_trader.py`'s internal re-fetch (`self.broker.get_candles(pair, 5, num_candles=250)`) requested **5-minute bars only**.

**Critical finding**: `LiquiditySweepAnalyzer.get_signal(self, df_1m, pair, df_5m=None, ...)` — the first parameter is literally named `df_1m`, designed for genuine 1-minute liquidity-sweep detection per the module's own docstring ("1M Liquidity Sweep of swing-based levels + 5M invalidation gate"). In production, this parameter was fed 5-minute data. `ensemble_trader.py` contains a `_looks_like_five_minute_data()` fallback check, confirming the codebase's own authors were aware the caller frame was 5-minute cadence, not 1-minute. **The live bot's designed dual-timeframe architecture was silently collapsed to single-timeframe (5-minute) in actual production.** This was discovered during the Phase 8 information audit, not assumed.

### Volume usage
`volume_ratio = volume / rolling_mean(volume, 10)`, used only inside the displacement-candle confirmation check. The confirmation threshold (`VOLUME_CONFIRMATION = 1.0`) made this filter effectively a no-op in practice.

### VWAP status
Not present in `liquidity_sweep.py` at all. Added later, in a separate commit (`c7e3137`), as a bolt-on pre-trade filter inside `ensemble_trader.py`/`backtest_engine.py`, computed as a cumulative-since-inception series (not session-reset — a methodological flaw noted at the time it was found). Never isolation-tested in this program; explicitly excluded from every phase's core hypothesis per the isolation discipline established in Phase 3.

### Absence of order-flow / order-book data
Confirmed by direct source search: zero references to order book, market depth, bid/ask, or level-2 data anywhere in `start_live_rithmic.py`, `ensemble_trader.py`, or `liquidity_sweep.py`. The order-flow/order-book pilot studies referenced in the broader research log were a **separate, later research exploration**, never part of the recovered live bot's actual signal path.

### Absence of higher-timeframe inputs
No 15-minute, 1-hour, or daily context referenced anywhere in the live decision path.

### Execution behavior
Market-style bracket orders (stop-loss + take-profit) via the broker connector; no limit orders, no smart/staged execution. Cooldown = 5 bars (25 minutes on the 5-minute timeframe), applied globally across symbols, not per-symbol. No time-of-day or session gate (`OPTIMAL_SESSIONS` was defined in config but never referenced anywhere else in the file — dead configuration).

### Intended architecture vs. what production actually executed
| | Intended (per docstrings/code structure) | Actually executed in production |
|---|---|---|
| Sweep detection timeframe | 1-minute | 5-minute (mislabeled) |
| Structure/regime timeframe | 5-minute | 5-minute (correct) |
| Primary strategy label in logs | Accurate | Hardcoded `enhanced_breakout` regardless of actual path |
| Volume filter | Meaningful confirmation gate | Effectively disabled (threshold=1.0) |
| Session/time filter | Config present | Never enforced |

---

## 3. Complete Hypothesis Ledger

| # | Hypothesis | Data/Input | Phase | Methodology | Discovery Result | Holdout Result | Multiple-Testing | Economic Significance | Final Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Sweep + MSS (combined, faithful reconstruction of live logic) | Real 1M+5M OHLCV | 3 | Non-lookahead reconstruction, fixed 1.5R bracket, next-bar execution | MES t=-1.31 (p=0.19), MNQ t=0.24 (p=0.81), NQ t=0.40 (p=0.69) | Same script includes its own holdout split; sign inconsistent, none significant | N/A (3 cells) | Negligible | **NULL / NO EDGE** |
| 2 | Sweep-only | Real 1M OHLCV | 3 | Same as above, sweep event without MSS confirmation | MES t=-5.40 (p<0.0001, significant negative); pooled t=-1.59 (p=0.11) | — | N/A | Loses to costs | **REJECT (negative)** |
| 3 | MSS-only (break-of-structure) | Real 1M OHLCV | 3 | Displacement-candle break of structure, no sweep required | Pooled t=-1.99 (p=0.046, negative); MES t=-7.79 (p<0.0001) | — | N/A | Loses to costs | **REJECT (negative)** |
| 4 | Named strategy suite (14 strategies: trend, mean-reversion, breakout, VWAP-cross, ensemble, ORB, ORB-failure, VWAP-pullback ×2, regime-bot, vol-expansion-momentum, extreme-displacement-reversion, volume-shock-continuation) | 5M OHLCV | Pre-1 | Walk-forward search, OOS aggregation | Zero validated edge across all 14 | — | Walk-forward | — | **NULL** |
| 5 | Daily-bar legends (Dennis breakout, Simons z-score fade, Raschke turtle soup, PTJ 200-SMA filter — 16-variant grid) | Daily OHLCV, NQ/ES | Pre-1 | Parameter-plateau grid across lookbacks | Significant NEGATIVE, replicated on both NQ and ES | — | Plateau-tested | Real negative edge (against the strategy) | **REJECT (negative, high confidence)** |
| 6 | MNQ/MES pairs / statistical arbitrage | Daily OHLCV | Pre-1 | Notional-balanced relative-value | Null | — | — | — | **NULL** |
| 7 | Overnight gap reversion | 5M OHLCV, RTH open | Pre-1 | Real 10:30 ET time-stop, ATR catastrophe stop | Null | — | — | — | **NULL** |
| 8 | Return autocorrelation (lags 1-50) | 5M OHLCV | 4 | Lag correlation, t-test | Statistically significant (t up to -10.8) but ρ<0.03 | — | N/A | **Economically negligible** — smaller than round-trip cost | **NULL (not exploitable)** |
| 9 | Time-of-day (clock-hour, UTC) | 5M OHLCV | 4 | Per-hour mean vs. overall mean | Nothing cleared the bar | — | — | — | **NULL** |
| 10 | Volatility persistence (ATR/range level, autocorrelation of state) | 5M OHLCV | 4 | Rolling-window percentile correlation | Real but driven by heavily overlapping windows (invalid iid assumption); descriptive stylized fact, not directional | — | — | Not directional | **NULL (known stylized fact, not an edge)** |
| 11 | ATR percentile → forward directional return | 5M OHLCV | 4/5 | Full threshold×horizon curve, then non-overlapping event correction | Naive t=3.06-3.21; **collapsed to t=0.51-0.99 under non-overlapping correction** | t=0.30/-0.02/-0.05 (all null) | Bonferroni-aware | None (P(positive) only 53-55%, near coin-flip; magnitude not direction) | **REJECT** |
| 12 | Sigma-move behavior (single-bar N-sigma move → forward return) | 5M OHLCV | 4 | Conditional forward return after 2σ/3σ bars | Never cleared significance bar even before overlap correction | — | — | — | **NULL (closed without needing holdout)** |
| 13 | Directional run persistence (H1) | 5M OHLCV | 6 | Run-length event, non-overlapping sampling | Max \|t\|=2.50, 0/27 clear Bonferroni (bar 3.48); no MES/MNQ-NQ cross-consistency | t=-0.40/0.23/0.10 (null) | Bonferroni (99 cells) | — | **REJECT** |
| 14 | Close-location mean effect (H2) | 5M OHLCV | 6 | Continuous close-location-in-range, non-overlapping sampling | Max \|t\|=1.83, 0/54 clear Bonferroni | t=0.08-0.47 (null) | Bonferroni (99 cells, shared) | — | **REJECT** |
| 15 | Volatility-regime transitions (H3) | 5M OHLCV | 6 | Release/exhaustion crossing events | n=10-42 per cell, mostly below n=30 floor, sign flips across horizons | n=8-14 (still insufficient) | N/A (insufficient) | Cannot be assessed | **INCONCLUSIVE — insufficient event count, not a rejection** |
| 16 | Close-location distributional shape (skew) | 5M OHLCV | 7 | Permutation test on skew difference vs. dependence-matched baseline, KS test, bootstrap CI | 0/36 cells clear Bonferroni on the primary (skew permutation) test; one secondary KS signal at h=5/bottom-tail (p=1e-6 to 1e-8) | 0/36 skew cells; the KS signal **collapsed** (p=0.014-0.019, 4-5 orders of magnitude weaker) | Bonferroni (36 cells) | Not reached | **REJECT** |
| 17 | Relative volume state (Hyp A) | 5M OHLCV + Volume | 8 | Percentile-threshold event, permutation test on mean difference, KS test | 0/144 cells clear Bonferroni on primary test; 0/144 even nominal p<0.05; 9 cells KS-significant | 0/144 primary; **all 9 KS survivors failed to replicate** (p rose to 0.017-0.35) | Bonferroni (180 cells, shared with #18) | None | **REJECT** |
| 18 | Volume-price divergence (Hyp B) | 5M OHLCV + Volume | 8 | Large-move × volume-level event (fixed definitions), same tests as #17 | 0/36 primary; 24 KS-significant (both "low-vol divergence" and "high-vol confirmation" variants) | 0/36 primary; only 5 KS survivors, **all "high-volume confirmation" only** — the "low-volume divergence" variant (the more novel concept) fully failed to replicate; surviving cells show zero mean signal | Bonferroni (shared) | None; surviving fragment explainable by already-rejected volatility clustering (#10/#11) | **REJECT** |

---

## 4. Methodological Lessons (Permanent Research Standards)

These were discovered the hard way, mid-program, and should govern any future research on this codebase or dataset:

1. **Overlapping forward windows inflate significance, sometimes catastrophically.** Rolling-window features (ATR percentile, range percentile) evaluated at every bar produce forward-return samples that overlap almost entirely with their neighbors. Standard iid-based standard errors (e.g., `1/sqrt(n-3)`) are invalid here and can report t-statistics in the hundreds (observed: t=307) from an effect that is genuine noise once corrected. **Fix used throughout Phases 5-8: greedy non-overlapping event sampling with minimum spacing equal to the forward horizon.**

2. **Persistent-state variables (e.g., "ATR is currently elevated") must not be tested as if each qualifying bar were an independent observation.** A single volatility episode can span dozens of consecutive bars; counting each as a separate "event" is the same bug as #1 wearing a different hat. The correct unit of observation is the *episode*, not the *bar*.

3. **Large sample sizes make economically meaningless correlations statistically "significant."** Return autocorrelation of ρ=0.03 produced t=-10.8 purely from n=150,000+ — nowhere near large enough in magnitude to survive real trading costs. **Statistical significance and economic significance are separate questions and both must be checked.**

4. **Multiple-testing correction must be applied honestly and to the whole grid, not just the interesting cell.** Every phase from 4 onward pre-registered a full threshold × horizon × symbol grid and applied Bonferroni correction to all of it, not just the cell that happened to look best. This caught several would-be false positives (Phase 6, Phase 7, Phase 8) that looked real under a naive single-cell view.

5. **Correlated instruments are not independent confirmations.** MNQ and NQ are the same underlying index at different contract sizes. Every phase explicitly separated MES (genuinely independent) from the MNQ/NQ pair (one correlated reading, reported twice) rather than treating 3-symbol agreement as 3 independent confirmations.

6. **Discovery/holdout separation must be genuinely locked before looking.** In every phase from 4 onward, the analytical configuration (thresholds, horizons, methodology) was fixed and written into code *before* the holdout was touched, and the holdout was run exactly once. This caught real discovery-side "signals" that evaporated on contact with untouched data (Phase 7's KS signal fell by 4-5 orders of magnitude; most of Phase 8's discovery-side KS survivors vanished entirely).

7. **Baseline drift must be controlled, not assumed to be zero.** Testing a conditional forward return against zero (rather than the instrument's own unconditional mean return at that horizon) can misattribute ordinary market drift to whatever condition is being tested. This bug was caught and fixed mid-Phase-4, before any conclusion was drawn from the flawed version.

8. **A distribution-shape difference (KS-significant) is not the same as a directional difference (mean-significant).** Phases 7 and 8 both produced cases where general distributional tests flagged something while the specific, economically relevant test (mean/skew difference via permutation) found nothing — and in both cases, the KS-only signal failed to survive holdout. Always test the specific quantity that would actually be exploitable, not just "is the distribution different in some way."

---

## 5. What Remains Untested

### A. Information available to the original live system, now tested
- 5-minute OHLCV price structure (the system's actual operating resolution) — extensively tested across Phases 4-8.
- Genuine 1-minute granularity for sweep/MSS detection — tested in Phase 3 with **more fidelity than the live system itself ever had** (production fed 5-minute data into the `df_1m` parameter; our reconstruction used real 1-minute data).
- Volume (level and divergence-with-price) — tested in Phase 8.
- Return autocorrelation, time-of-day, session structure, directional runs, close-location (mean and shape), volatility level/persistence/transitions — all tested.

### B. Information NOT available to the original live system
The following were never part of what the recovered live bot actually consumed, and were correctly excluded from this program rather than invented as new hypothesis territory:
- Genuine order-flow / order-book / market-depth / bid-ask data.
- Higher-timeframe context (15m/1h/daily) beyond the (degraded, 5-minute-only) structure the live system used.
- Prior-day reference levels (conceptually ICT-adjacent, but never actually coded into the live analyzer).
- Market internals (breadth, TICK, VIX, etc.).
- Tick-level/microstructure data.

**Do not conflate these two categories.** Category B represents genuinely different information sources that could form the basis of a *new* research program with a materially different dataset — not a continuation of the current one.

---

## 6. Final Status

```
DISCOVERY TRACK:              CLOSED
VALIDATED EDGE:                NONE
PRODUCTION STRATEGY:           NONE
FURTHER OHLCV+VOLUME MINING:   PROHIBITED
```

No production or live trading code was modified at any point in this research program. No strategy was constructed. All work is confined to `scripts/research_*.py` (read-only research scripts) and `reports/` (result artifacts).

---

## 7. Future Research Boundary

Any future attempt to continue this research **must introduce a materially different information source or materially different dataset.** It must NOT simply be:
- another indicator,
- another threshold,
- another combination of existing OHLCV variables,
- another strategy label,
- another parameter sweep.

Any of the above would be reopening a research space that has already been extensively tested across 8 phases, 18 distinct hypotheses, discovery/holdout discipline, dependence-aware sampling, and multiple-testing correction, with a consistent and repeated null result. The only legitimate continuation path is Category B above: a genuinely new information source (e.g., real order-flow/order-book data) paired with a new, equally disciplined discovery/holdout research program — not a variation on what has already been closed here.

---

*Artifacts referenced in this dossier: `reports/phase4/`, `reports/phase5/`, `reports/phase6/`, `reports/phase7/`, `reports/phase8/`, `scripts/research_sweep_mss.py`, `scripts/research_market_behavior.py`, `scripts/research_phase5_vol_regime.py`, `scripts/research_phase6_distinct_structure.py`, `scripts/research_phase7_close_location_skew.py`, `scripts/research_phase8_volume.py`, `reports/RESEARCH_LOG.md` (pre-Phase-1 program).*

---

## Addendum: MNQ order-book round 2 (post-closure follow-up)

After this dossier's closure, a second round of Stage-1 testing was run on the *same already-purchased* MNQ MBP-1 data (2026-06-01 to 2026-06-15) — no new data purchased. Two features were tested, both built from raw columns the original 5-feature round never used (`n_quotes`, `depth_added`, `depth_removed`), to avoid re-testing the same information under a new name:

- **quote_intensity** (book-update frequency, z-scored): never significant in-sample or OOS at any of 4 horizons, R² ≈ 0.
- **depth_churn_ratio** (liquidity build vs. pull, a flow measure distinct from `depth_imbalance`'s snapshot ratio): not significant in-sample at any horizon; one OOS cell nominally significant (p=0.046) but with a sign flip from train — consistent with chance given 8 cells tested, not a real signal.

**Both REJECT.** Script: `scripts/orderbook_feature_battery_v2.py`. This brings total order-book features tested on this window to 7 (5 original + 2 here); none survived. No further feature invention is planned on this dataset. Per the dossier's future-research-boundary section, any further order-book research requires new data (a different depth level, symbol, or time window), not another feature on this same window.

---

## Addendum 2: MBP-10 pilot — aborted (operational failure, not a research result)

An MNQ MBP-10 pilot (same 2026-06-01/06-08 window, quoted at $36.59) was authorized and attempted. The streaming download (`scripts/download_orderbook_mbp10_pilot.py`) exceeded Databento's own 5GB streaming-request threshold and stalled. It was killed based on an incorrect diagnosis (Python-level CPU/network idleness does not reflect native-level streaming I/O still in progress) before completion. **No data file was produced or saved.**

Actual cost incurred: ~$53 (per account billing, "last 12 hours" usage), substantially exceeding the $36.59 quote. A follow-up check found `metadata.get_cost()` unreliable at this data volume — the real billable size for the identical request was confirmed at 78.58 GB, and the observed real-world rate from the failed attempt implied a true cost closer to $55-60 to complete via the correct (batch) API, not the original quote.

**Decision: stop.** Given the cost overrun and quote-reliability failure, the MBP-10 pilot was not retried.

**Important distinction for any future work**: this is an operational/cost failure, not a scientific result. Unlike every hypothesis in the ledger above, **MBP-10's information content relative to MBP-1 was never actually tested** — no data was obtained, no features were computed, no GO/NO-GO verdict was reached. This should not be recorded or treated as a "rejected" hypothesis; it is an open question that was never reached. Should MBP-10 be revisited in the future, use the batch-download API (`client.batch.submit_job`/`get_job_details`/`download`) from the start, verify `metadata.get_billable_size()` against the actual observed per-GB billing rate before committing, and start with a much smaller window (1-2 days) to bound any single mistake.

Total order-book/order-flow spend across the whole project is now approximately $166 (~$113 original program total + ~$53 from this aborted attempt), against the original research-log figure of ~$113. No further order-book spend is planned.
