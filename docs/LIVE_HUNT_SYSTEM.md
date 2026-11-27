# Live hunt system — TopstepX Combine (MNQ)

The live ruleset for the discretionary hunt run on Carib's TopstepX Combine
account, as of **2026-11-26 ~10:00 PM ET**. This document is the current
source of truth for the live system; it supersedes the gate and sizing
details in the earlier study files where they differ.

Design origin: FIVEMIN_CALLS ("5m calls, 15m direction"), judged on 2026
data out of sample: 178 trades, win rate 52.2%, PF 1.437, +$27.45/trade,
net +$4,886. The 100-point cap is the strategy: without it the same shape
collapses.

## Direction gate (closed 15m bars)

- Bias requires: EMA20 vs EMA50 aligned in the bias direction, DI
  confirming (+DI above -DI for longs, reversed for shorts), and the close
  on the correct side of the EMA200 series.
- **H1**: no entries within 1x ATR(15m) of EMA200.
- **H2: REMOVED 2026-11-26** (Carib's order). The fresh MACD-flip +
  RSI-50-cross veto no longer blocks entries. Reason: the 2026-11-25 study
  found its vetoed trades were net winners (OOS +$673 over 10 trades, 60%
  win rate), and on 2026-11-26 it blocked the re-entry into the session's
  strongest long trend after trade #12 exited. MACD and RSI are still read
  at every gate for context; they veto nothing.
- Loose gate is live (no ADX minimum for a bias). **HIGH CONFIDENCE** =
  15m ADX at least 25 + H1 clear at the governing gate. High-confidence
  trades get NO protect line; ordinary trades do.

## Cap

- Flat 100 points: closed 5m ATR(14) must be at most 50.0, judged at the
  arm and again at the call. Wider caps (125/150/200, tiered) were tested
  2026-11-25 and rejected: they destroy value out of sample.

## Entry sequence (closed 5m bars)

- Arm on the first 5m close against the bias; the arm refreshes on each
  further against close. The first 5m close in the bias direction after
  the arm is the CALL.
- One hunt position at a time; fresh sequence after any exit, flatten, or
  bias re-formation.
- Entry at the call: LONG via JOIN BID (buy limit at the bid), SHORT via
  JOIN ASK (sell limit at the ask). The side is named and the working
  order verified against the live bid/ask before the close.
- **Bracket is PRE-SET at the arm** (process fix, 2026-11-26 evening):
  at the arm, set the DOM quantity to the ladder size and set the bracket
  dollars from the arm bar's ATR; at the call the only steps are a quick
  ATR re-check and the join click. Editor flow on TopstepX: fill each box,
  press Tab, click SAVE CHANGES, verify the toolbar (closing with X
  discards the values; whole dollars only).
- **Keep test (refined 2026-11-26)**: block only a fill WORSE than the
  call close by more than 6 points (above the close for a long, below for
  a short). A fill at the call close, better than it, or within 6 points
  worse is taken. (The original either-direction wording lapsed the
  2026-11-26 9:50 PM high-confidence call when the fill would have been
  ~15.75 points better than the close.)

## Bracket and exits

- Stop = 2x ATR(14) 5m; target = 1.5R. Per contract: Risk $ = 4x ATR,
  Profit $ = 6x ATR (MNQ $2/point). Bracket dollars scale linearly with
  quantity.
- **The take profit is NEVER removed** (Carib, 2026-11-26). The only
  management allowed is trailing the stop UP, and only on Carib's word or
  a plan he approved. A trade ends at its TP, its trailed stop, or the
  session hard end.
- **Protect line** (ordinary trades only): when a fresh run reaches the
  band at 73-89% of the entry-to-target distance and stalls or bounces, a
  fixed line is set at a bounce level banking roughly half or more of the
  target value; any trade through the line closes the position by hand at
  once. Never armed retroactively.
- TopstepX Position Brackets aggregates: a second fill in the same
  contract grows the existing bracket orders at the existing levels.

## Sizing (ladder, Carib 2026-11-26)

- Base size is **2 MNQ** on every hunt entry (set 2026-11-26, replacing
  the 1-contract default and the old high-confidence-only 2-contract rule).
- Ladder: +1 contract per $1,000 of balance — 3 at $102,500, 4 at
  $103,500, 5 at $104,500, and so on (10 MNQ = 1 NQ in size, reached in
  steps the cushion can carry instead of one $20/point jump).
- DOM quantity is set to the ladder size at every arm and verified; after
  a fill, Positions must show that quantity with matching Risk/To Make.

## Hunt stop and run rule

- The hunt ends only at net realized hunt P&L of **-$800** (raised from
  -$400 on 2026-11-26 to match the 2-contract size), Carib's stop word,
  or market close. It restarts when the market opens. A dead bias only
  halts entries; the gate is verified at every 15m close until the close.
- Holiday/session hard ends override: e.g. Fri 2026-11-27, entries stop
  12:30 PM ET, any open hunt trade flattened by hand by 12:55 PM ET
  (Combine flat-by 1:00 PM ET).
- Placement discipline: a chart share or a bare "go" is never an order.
  Trades are placed only when a setup exists under this gate, on the
  operator's judgment, without waiting for a per-trade go. Anything
  off-system goes to Carib first.
- Accounting separation: Carib's own trades are his ledger alone — never
  in the hunt record, never toward the -$800 hunt stop. While he holds a
  position, same-direction hunt setups are placed without asking;
  opposite-direction setups are flagged to him first.

## Record

Through 2026-11-26 9:17 PM ET: **13 hunt trades, 4 losers, +$1,404.08 net
realized**; balance $101,968.52; day RP&L +$941.90. Trade #13 (first under
2-contract sizing with H2 removed): long 2 @ 26,451.75, target filled
26,559.00, +$426.56 net.
