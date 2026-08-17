# Research log: MNQ / MES / NQ / ES strategy search

## STATUS: PROJECT CLOSED, fully resolved (2026-08-17)

## Conclusion (final)

**36 hypotheses tested across six research branches -- intraday strategies,
order-flow/order-book information, two recovered legacy "profitable"
strategies, a 16-variant daily grid modeled on four named traders'
philosophies, cross-index pairs statistical arbitrage, and overnight gap
reversion. Zero produced a validated, replicated edge. One branch
(daily index breakout-following) produced something stronger than "no
edge": a statistically robust, parameter-plateau-stable, cross-instrument
NEGATIVE result -- real evidence the specific strategy family loses money,
not just an absence of evidence that it wins.**

**14 strategy hypotheses failed to demonstrate robust replicated edge on
MNQ/MES/NQ/ES (5-minute and daily timeframes). Nine order-flow/order-book
features across two pilots (NQ trade-flow, MNQ order-book) failed to
demonstrate OOS predictive information. One feature (`persistence`, MNQ
order-book) looked ambiguous on a 1-week sample; extending to 2 weeks
resolved it to a clean fail (p-values collapsed, sign flipped) --
no unresolved threads remain.**

These are distinct claims, kept deliberately separate: the 14 are
hypotheses about tradable strategies; the order-flow/order-book work is a
predictive-information test (Stage 1 of Information -> Signal -> Strategy
-> Profitability -> Live) and covers a specific, narrow slice (2 symbols,
short time windows, a handful of feature definitions). None of this
supports the broader claim that no exploitable information exists in
these markets — only that these specific, tested approaches, on this
data, didn't clear a real bar. The branch was closed by choice, at a
point of genuine ambiguity, not because every avenue was exhausted.

`absorption` (net signed volume per unit of price move) is worth
remembering as a case study: p<0.001 and significant even after controlling
for the baseline imbalance feature, in-sample — and it fell apart
completely out-of-sample (p=0.26-0.97, sign flipped at 2 of 4 horizons).
The clearest example in this project of why the OOS gate exists.

This is evidence against these specific hypotheses on this specific
data/timeframe/instrument set — **not** evidence that no exploitable
information exists in these markets. That broader claim was never tested
and this research doesn't support it either way.

## What was tested

### Intraday (5-minute bars, MNQ/MES/NQ, 2024-01-01 to 2026-03-11)
Walk-forward tested (180-day train / 60-day test, re-optimized per fold),
significance bar: t-stat >= 2.0 on >= 30 pooled OOS trades, cost-adjusted,
positive total P&L.

| Strategy | Best result (any symbol) | Significant? |
|---|---|---|
| trend (EMA cross) | NQ: +$612, t=0.31 | No |
| mean_reversion (Bollinger+RSI+ADX) | NQ: -$1,179, t=-0.12 | No |
| breakout (Donchian) | NQ: -$723, t=-0.97 | No |
| ensemble (ADX regime router) | NQ: -$520, t=-0.62 | No |
| vwap_cross (EMA10/20 + VWAP + confirm delay) | NQ: -$6,948, t=-1.39 | No |
| orb (opening range breakout) | MNQ: -$489, t=-1.14 | No |
| orb_failure (fade failed breakout) | NQ: -$723, t=-0.97 | No |
| vwap_pullback_trend (daily trend + VWAP pullback) | MNQ: +$4,056, t=0.92 (optimized); +$760, t=0.13 (unoptimized default params) | No |
| vol_expansion_momentum | NQ: -$182, t=-0.04 | No |
| extreme_displacement_reversion | NQ: +$31,026, t=0.45 (9.6% win rate -- likely a tiny-stop position-sizing artifact, not a real edge) | No |
| volume_shock_continuation | MNQ: +$11,383, t=0.71 | No |
| regime_bot (regime-score argmax router) | worst performer of all: MES -$59,928, t=-8.07 | No |
| adaptive engine (expectancy-conditioned regime router) | MNQ: -$6,843 full-history, -$2,022 final 90 days | No |

Selection-bias audit on vwap_pullback_trend's walk-forward optimizer: not
pure noise-harvesting (beats naive-average-parameter-choice by ~3x, picks
consistent winning params across folds), but the underlying signal is weak.

### Daily breakout (NQ/ES primary, MES/MNQ implementation-validation, 2010-2026 where available)
Deliberately simple: N-day Donchian breakout (corrected non-lookahead
formulation), ATR stop/trail, lookbacks 40/60/80/100/120, no optimization,
same rule on every instrument.

- NQ: positive and smooth across all 5 lookbacks (+$9.1K to +$10.6K) --
  but only 7-12 trades over 16 years, far below any reasonable sample-size
  bar.
- **ES (the required independent-replication instrument): negative on 4 of
  5 lookbacks.** Failed to replicate.
- MES (49-63 trades, much larger sample): negative across all 5 lookbacks.
- MNQ: mostly negative, marginal positive only at longest lookbacks.

**Verdict: NQ's result did not survive independent replication on ES and
was very likely a small-sample artifact. Hypothesis retired without
running the remaining robustness battery (cost sensitivity, time-stability,
exposure), since it already failed the primary replication gate.**

### Order-flow predictive-information test (Stage 1 only -- not a strategy)
NQ, June 2026, tick-level trades (Databento `trades` schema, includes
exchange-tagged aggressor side). 1-minute bars, forward returns at
1/3/5/10-minute horizons, 70/30 chronological train/test split, per-day
correlation breakdown, incremental-information regression against a
baseline feature.

Five mechanically distinct features tested: signed-volume imbalance
(baseline), block-trade imbalance (size >= 5, ~99th pctile), flow
acceleration (bar-over-bar change in imbalance), flow persistence
(5-bar rolling mean imbalance), absorption (signed volume per unit of
price move -- an exhaustion proxy).

**All 5 failed the OOS gate at all 4 horizons** (best OOS R^2 = 0.0004,
i.e. <0.05% of forward-return variance explained). `absorption` is the
standout case: p<0.001 in-sample including after controlling for the
baseline feature, collapsed OOS (p=0.26-0.97, sign flipped at 2 of 4
horizons) -- textbook in-sample-only artifact.

**Scope of this result: one symbol, one month, five specific feature
definitions, 1-minute bars.** Does not test order-book depth, sweep
detection, finer time resolution, or other symbols/periods. Per the
research gate: none of the 5 survived Stage 1, so none were escalated to
Stage 2 (signal) or Stage 3 (profitability), and no further order-flow
data was purchased beyond the initial ~$10.58 pilot.

### MNQ order-book predictive-information test (Stage 1 only)
MNQ, 2026-06-01 to 2026-06-07 (1 week, 7 trading days), MBP-1 order-book
data (~$20.93, streamed via `DBNStore.replay()` and aggregated directly
into 1-minute bars to avoid materializing the full 156.1M-record raw
stream, which OOM-crashed on the first attempt). Same framework: 70/30
chronological split, sign stability, R^2, incremental info vs. a
trade-flow-imbalance baseline (computed from the same dataset), per-day
correlation breakdown.

5 features tested: depth_imbalance, depth_change (bid vs ask depth
build-up), order_book_pressure (imbalance x recent momentum), persistence
(5-bar rolling imbalance), trade_flow_imbalance (baseline).

**4 of 5 failed cleanly** (no OOS significance, mostly sign-unstable).
**`persistence` was ambiguous on 1 week:** OOS-significant at 3 of 4
horizons (p=0.026-0.039), sign-stable, mean-reversion direction -- but
train-set correlation was NOT significant for the same feature (backwards
from the usual overfitting pattern), and the OOS test split was only 2-3
trading days -- too thin to trust.

**Follow-up: extended to 2 weeks (2026-06-01 to 2026-06-15, 13 trading
days, ~$47.55 total order-book spend), ambiguity resolved to NO.** With
4 OOS test days instead of 3, `persistence`'s p-values went from
0.026-0.039 to 0.364-0.839 and the sign flipped at 3 of 4 horizons --
the signature of noise that briefly looked real on too small a sample,
not genuine signal that needed more power to detect. **All 5 order-book
features now fail cleanly, no ambiguity remaining.**

### Recovered legacy strategies (git archaeology)
The project's git history contained an earlier, lost implementation
(orphaned from `main` after an apparent history rewrite, recovered via
`git log --all`, tags, and a stash entry). Two strategies with explicit
"profitable" claims in commit messages/paper-trade logs were found and
re-verified against this project's real data and engine, not taken on
faith.

**`enhanced_breakout`** (EMA50 trend filter + 10-bar breakout, 5m MES):
original paper-trade log showed 444 trades, PF=1.50, +$4,623 over ~2.5
months. Re-run on the full 2-year dataset: t=8.4-13.9 (MES/MNQ/NQ), far
beyond anything else in this project -- until a look-ahead bug was found
and fixed (the trend filter used the *same bar's own closing price* while
treating entry as an earlier intrabar stop-order fill). Corrected version:
**MES t=-2.91 (significantly negative), MNQ t=-0.07 (flat), NQ t=1.78
(not significant).** The entire apparent edge was the bug.

**`clean_scalper`** (6-confirmation ICT/liquidity-sweep system, 1m/5m,
`LiquiditySweepAnalyzer` 4-layer pipeline): commit claimed "MES PF=1.11,
MNQ PF=2.76, NQ PF=2.41" with no saved verification data behind it.
Re-run across 8 independent quarters (MNQ/NQ, 2024-2025): **1 of 8
profitable (NQ Q1 2024, PF=1.09); total across all 8: -$26,464, 601
trades.** The claimed PF=2.76 never appeared on any independently tested
period.

### Daily strategies modeled on named traders' philosophies
Four building blocks, each grounded in a specific real trading approach
(Paul Tudor Jones' 200-SMA regime gate, Richard Dennis' Donchian breakout,
Jim Simons-style Z-score mean reversion, Linda Raschke's "Turtle Soup"
failed-breakout fade), tested individually before any combination (per
this project's established rule: combining untested components has never
helped). Entry uses the bar's own intraday high/low against a
shift(1)-only channel (no lookahead); fill is the *next* bar's close
(deliberately conservative, avoiding the intrabar-fill-vs-same-bar-filter
mismatch that caused the `enhanced_breakout` bug above).

**Parameter-plateau grid: entry lookback in {20, 40, 60, 80}, with/without
the PTJ regime filter, on NQ and ES (2010-2026):**

| System | NQ (all 4 lookbacks) | ES (all 4 lookbacks) |
|---|---|---|
| Pure Dennis breakout | t = -2.10 to -2.70, all significant | t = -1.94 to -3.49, 3 of 4 significant |
| Dennis + PTJ filter | t = -1.58 to -2.22 | t = -2.39 to -2.69, all significant |
| Pure Raschke (Turtle Soup) | t = -0.73 to +0.19, no signal | t = -0.80 to +0.62, no signal |
| Raschke + PTJ | t = -0.49 to +1.90, doesn't replicate; n collapses to 4-21 at lb>=60 | mostly n too few |

**This is the strongest finding in the project, and it's a negative one:**
daily breakout-following on these instruments is not merely untested, it's
significantly disproven across a full parameter plateau (stable sign,
mostly significant at every lookback) and replicated across two
instruments. The PTJ 200-SMA filter reduces losses versus pure breakout
but never flips the result positive. Raschke's fade is inconclusive
(no stable sign or significance anywhere), not a lead.

Scope: this is specific to the tested formulation (N-day Donchian entry,
fixed 10-day exit, next-close fill, these two instruments, 2010-2026) --
not a claim about breakout trading in general or about other markets/
timeframes.

### Post-closure follow-ups (4 more hypotheses, session of 2026-08-17 continued)
Reopened briefly for four additional, genuinely distinct ideas rather than
re-tweaking anything already tested.

**`vwap_pullback_trend_v2`** (`src/strategies/vwap_pullback_trend_v2.py`):
v1 + audit-validated params (pullback_lookback_bars=8, min_body_atr_mult=0.5,
not the untested v1 defaults) + a new ADX>=20 selectivity filter, grounded
in the one consistent cross-project pattern (selectivity correlated with
less damage everywhere else). MNQ (primary): t=1.04 (up slightly from v1's
0.92, still not significant). NQ: over-filtered to 3 trades, inconclusive.
**MES: t=-2.15, significantly negative** -- the filter failed its own
replication test. Reasoned, evidence-grounded tweaks still aren't a
substitute for a real edge.

**Four-legend daily grid** (`src/daily_trend/legends_strategies.py`):
PTJ 200-SMA regime gate x Dennis breakout x Simons Z-score fade x Raschke
Turtle Soup, parameter-plateau tested (lookback 20/40/60/80, with/without
PTJ filter) on NQ and ES, 2010-2026. **This is the strongest and most
useful finding in the whole project: Pure Dennis and Dennis+PTJ are both
significantly NEGATIVE across nearly every lookback on both instruments**
(t as low as -3.49, ES pure Dennis). The PTJ filter reduces losses but
never flips the sign positive -- daily breakout-following on these
instruments is disproven, not just unproven, across a genuine parameter
plateau and cross-instrument replication. Raschke (pure or PTJ-gated):
no stable sign or significance anywhere -- inconclusive, not a lead.

**MNQ/MES pairs statistical arbitrage** (`src/daily_trend/pairs_trade.py`):
relative-value bet on the rolling Z-score of the MNQ/MES price ratio
reverting (2019-2026 daily data, both launched 2019-05-05). Grid over
lookback (10/20/30/40) x z-threshold (1.5/2.0/2.5): win rates consistently
55-87% (the known mean-reversion signature -- many small wins, occasional
large loss, not evidence of edge by itself), but total P&L scatters around
zero with t-stats of 0.01-0.78 wherever sample size is large enough to
trust (n>=60). The one eye-catching cell (lookback=10, z=2.5, t=2.05) has
only 8 trades over 7 years -- a small-sample fluke, not a result. **Null,
not disproven** -- genuinely no signal either direction.

**Overnight gap reversion** (`src/strategies/gap_reversion.py`, a
self-contained backtest function rather than routed through the generic
engine, because it needed a real 10:30 ET time-stop the generic engine
doesn't support): fade the RTH opening gap vs. prior session's close when
it exceeds a volatility threshold, target the gap fill, time-stop at
10:30 ET. Note: the originally proposed rule had no protective stop-loss
(unlimited single-trade downside) -- an ATR-based stop was added as a
necessary risk-management fix, not part of the original spec. Tested on
MNQ/MES/NQ (2024-2026) across a threshold sweep (0.3-1.0x ATR): all
negative but not significant, t-stats confined to [-0.72, +0.44] across
every symbol and threshold -- clean null result, no cherry-picked
threshold shows anything.

## What was NOT tested (i.e. the actual open questions)

- Order-flow/order-book features on other symbols (NQ order-book, MES),
  other periods, other bar sizes/time resolutions.
- MBP-10 (multi-level depth) or trade-size/sweep-specific order-book
  features.
- Any strategy/profitability test using order-flow or order-book
  information (never reached -- nothing cleared Stage 1 with confidence).
- Raschke-style mean-reversion at other lookbacks/instruments/timeframes
  (inconclusive, not disproven -- the one branch left genuinely open).
- Anything outside MNQ/MES/NQ/ES: other instruments, other asset classes,
  fundamentally different information sources (news, cross-market signals).
- Discretionary trading using this project's infrastructure (regime state,
  VWAP, session structure) as live decision support rather than a fully
  automated rule set -- never tested because it isn't backtestable the
  same way; not ruled out by anything found here.

## Research framework built (reusable regardless of hypothesis)

- `src/backtest/engine.py` -- generic bar-by-bar simulator, no-lookahead,
  realistic commission/slippage, ATR trailing stop, breakeven-at-R.
- `src/backtest/walk_forward.py` -- rolling re-optimization + OOS
  aggregation + t-stat significance testing.
- `scripts/audit_selection_bias.py` -- tests whether a parameter optimizer
  is finding real structure vs. harvesting noise (compares selected-combo
  OOS performance against naive-average-combo and best-possible-in-hindsight
  baselines).
- `src/daily_trend/` + `scripts/roll_audit.py` -- continuous-contract data
  pipeline with an explicit roll-artifact-vs-real-market-move audit
  (same-session contract-to-contract spread comparison, per-instrument
  outlier detection, not a fixed global threshold).
- `src/regime/` -- regime-score engine and two selection approaches tried
  (argmax and expectancy-conditioned); both retired, but the underlying
  regime-scoring machinery could still inform feature engineering later.

- `scripts/orderflow_feature_battery.py` / `scripts/orderbook_feature_battery.py`
  -- Stage-1 predictive-information test harness: per-feature train/OOS
  correlation, sign stability, R^2, incremental-information regression vs.
  a baseline feature, per-day correlation breakdown. Reusable for any new
  feature/symbol/period.
- `scripts/download_orderbook_pilot.py` -- memory-safe MBP-1 streaming
  aggregation via `DBNStore.replay()`, straight into 1-minute bar features.
  Necessary fix after the first attempt (materializing the full tick
  stream as a DataFrame) OOM-crashed on 156M rows; keep this pattern for
  any future order-book pull rather than repeating the naive `to_df()`
  approach.
- `src/daily_trend/legends_strategies.py` -- parameterized Dennis/Simons/
  Raschke daily strategies with a toggleable PTJ regime filter, built for
  clean single-variable comparison grids (mode x filter x lookback) rather
  than one-off scripts.
- `src/daily_trend/pairs_trade.py` -- notional-balanced two-leg relative-
  value backtester (MNQ/MES), reusable for any correlated-instrument pair.
- `src/strategies/gap_reversion.py` -- self-contained session-based
  backtest with a real intraday time-stop (not routed through the generic
  engine, which doesn't support time-based exits); reusable pattern for
  any future strategy needing an exact clock-time exit rule.

## Status: PROJECT CLOSED, fully resolved (2026-08-17)

**36 hypotheses tested end to end** across six branches:
- 14 intraday strategies (5-minute bars, MNQ/MES/NQ)
- 9 order-flow/order-book predictive-information features (2 pilots, 1
  ambiguous result followed up and resolved to fail, not left open)
- 2 recovered "profitable" legacy strategies, both re-verified against
  real data and found to be either a look-ahead artifact
  (`enhanced_breakout`) or simply unprofitable on independent data
  (`clean_scalper`, 1 of 8 quarters positive)
- 16-variant daily grid modeling four named traders' philosophies
  (Dennis, PTJ, Simons, Raschke) across a full lookback plateau on two
  instruments
- MNQ/MES pairs statistical arbitrage (relative-value, not directional)
- Overnight gap reversion (session-based, real time-stop)

**Zero produced a validated, replicated edge.** The daily-breakout branch
produced something more useful than a null result: statistically
significant NEGATIVE expectancy, stable across the entire tested
parameter region and replicated on both NQ and ES -- real evidence against
that specific strategy family, not merely an absence of evidence for it.

Total spend across the entire research program: **~$113** (2 years of
intraday bars, 16 years of daily bars across 12 markets, roll-quality
audits, order-flow pilot, 2-week order-book pilot). No further data
purchases pending.

Closed with the open threads explicitly named above (Raschke variants,
other instruments/timeframes, MBP-10, discretionary use of the
infrastructure) rather than implied to be exhausted. If any of those get
picked up later, they're genuinely new bets, not re-tests of anything
already tried and rejected here.

---

## Addendum: LiquiditySweepAnalyzer sweep + MSS (isolated reconstruction)

Recovered from `legacy/src/ai/liquidity_sweep.py` -- this was the actual
primary live entry signal generator (not `enhanced_breakout`, which was
only its exception-path fallback). Reconstructed standalone as
`scripts/research_sweep_mss.py`: swing-pivot detection, liquidity sweep of
a swing level, and market-structure-shift displacement confirmation,
isolated from the ML confidence layer, dynamic SL/TP, and every other
filter (VWAP/ADX/EMA/RSI-filter/PTJ) it was bundled with live. Tested on
existing 1M/5M OHLCV (MES/MNQ/NQ, 2024-01-01 to 2026-03-11, no new data
purchased), fixed 1.5R bracket exit, next-bar-close execution, real costs,
zero parameter tuning.

Pooled across all three symbols: sweep-only n=48,186 (t=-1.59, p=0.11,
negative point estimate, consistent negative sign across all 3 symbols
individually -- MES t=-5.40 p<0.0001, MNQ t=-1.89 p=0.06); MSS-only
n=44,784 (t=-1.99, p=0.046, negative); the actual reconstructed live
sequence (sweep required, then MSS) n=5,033 (t=0.35, p=0.73, essentially
flat, PF=1.02, sign inconsistent across symbols and unstable in the
untouched holdout tail).

**Verdict: NULL / NO EDGE.** No component tested positive and significant.
Not rescued with parameter tuning per the standing rule. Hypothesis #3
(from the Phase 2 forensic recovery) is closed.
