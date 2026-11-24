# Add-on setups: frozen rules (2026-10-10)

Two backtested LONG setups run inside BOTH live bots (start_mnq_live_revamp.bat = REVAMP and
START_ENSEMBLE_VISIBLE.bat = original Ensemble). Both bots run start_live_rithmic.py. The code is in
src/core/addon_setups.py, wired into start_live_rithmic.py. Tests are in tests/test_addons_20261010.py,
with a fixture of real bars and the backtest's own signals. Research files on the box:
out/ideas/IDEA_LIST_RESULTS.md (#6) and out/ideas/PATTERN_MINING_RESULTS.md (E1).

Toggles (both default ON, no .bat change needed): ADDON_ONL_RECLAIM=1, ADDON_E1_6PM=1. Set either to 0 to turn it off.

## Bars and indicators (same formulas as the backtest, pattern_mining/build.py)
- 5-minute MNQ bars, keyed by bar START time in ET. Live Rithmic bars are stamped by bar END, so the code subtracts 5 min. Only finished bars are used.
- Trading day = ET time + 6h (a trading day starts at 6:00 PM ET the evening before).
- ATR(5m) = EWM(alpha 1/14) of true range. RSI(5m) = EWM(alpha 1/14) Wilder RSI on closes. Both run continuously through the 5-6 PM halt.
- Live loads ADDON_CANDLE_COUNT=400 bars (its own Rithmic request, bypassing the 30 s candle cache). That reaches back past 6 PM ET on every weekday morning. A reading needs at least 150 bars.

## 1) ONL reclaim LONG (ADDON_ONL_RECLAIM)
- **Overnight low** = lowest low of this trading day's Globex bars, from the 6:00 PM ET bar up to the 9:25 bar (the 4-6 PM bars are not included). If the loaded bars don't reach back to 6:00 PM ET, the overnight low counts as unknown and no trade is taken.
- **Reclaim candle** = a day-session 5m bar whose START is 9:30-11:25 ET (first 2 hours), with low < overnight low AND close > overnight low (both strict).
- **First per day only:** only the day's first reclaim candle counts. If the bot isn't flat at that bar, there's no ONL trade that day.
- **Only when flat:** the bot (and the broker account) has no open trade.

## 2) E1 6 PM momentum LONG (ADDON_E1_6PM)
- Signal bar = a 5m bar whose START is 6:00-6:55 PM ET (Sun-Thu evenings), with
  RSI(14) >= 59.5 AND (close - close 12 bars earlier) / ATR >= 1.48.
  "12 bars earlier" counts bars, not clock time, so for the 6:00 PM bar it is the 4:00 PM bar's close.
  These are the frozen mined thresholds (candidates.json: rsi >= 59.5, ret12 >= 1.48). The shorthand "1.5 ATR / RSI 60" is the rounded version.
- It can fire again on a later bar in the window whenever the bot is flat (the backtest took one position at a time).
- Not blocked by the bad-hour/Friday filter. 6 PM ET = 22 UTC (EDT) / 23 UTC (EST), and the 7:00 PM bar close in winter = 0 UTC; the add-on doesn't consult bad_hours_utc [0,4,21] at all.

## Trade (both setups = backtest "label A")
- Entry: market BUY right after the signal bar closes (backtest: next bar open + 1 tick slippage).
- Size: 1 MNQ, fixed (confidence sizing is bypassed).
- Stop: entry_ref - max(1.25 x ATR(signal bar), 3.0 pts). entry_ref = signal bar close, rounded to tick.
- Target: entry_ref + 2.0 x stop distance (= 2.5 ATR).
- Time limit: 24 x 5m bars = 2h after the signal bar closes. If neither SL nor TP has filled, the position is closed at market (flatten), but only when the add-on is the only thing open on MNQ. If other trades or contracts are open, SL/TP stay working and the bot logs it.
- Fixed bracket: no $ profit lock, no ATR trail, no past-TP trail, and no re-entry for add-on trades (none of these were in the backtest).
- Confidence shown = backtest win rate (ONL 45%, E1 51%).
- Guards still applied: $500 per-trade stop-risk cap (skip if over), session limits (day -$500, overnight -$300, daily +$2,000), max 3 open, daily loss limit when session limits are off. A main-bot signal on the same bar takes the slot.

## Backtest (1 MNQ, after $3 RT + 1 tick slippage each side, MNQ $2/pt)
Periods: P1 Nov 2024-Apr 2025 (includes the Feb-Apr 2025 selloff), P2 May-Oct 2025, P3 Nov 2025-Mar 2026, P4 Apr-Sep 2026.

| Setup | P1 n / net / avg | P2 n / net / avg | P3 n / net / avg | P4 n / net / avg | Total n / net / avg / win |
|---|---|---|---|---|---|
| ONL reclaim, only when the revamp is flat (live rule) | 31 / +$785 / +$25.3 | 29 / +$386 / +$13.3 | 29 / +$984 / +$33.9 | 30 / +$1,092 / +$36.4 | 119 / +$3,247 / +$27.3 / 45% |
| ONL reclaim, all first-of-day signals | 50 / +$440 / +$8.8 | 43 / -$52 / -$1.2 | 52 / +$1,039 / +$20.0 | 48 / +$1,009 / +$21.0 | 193 / +$2,435 / +$12.6 / 39% |
| E1 6 PM long (59.5 / 1.48) | 38 / +$239 / +$6.3 | 55 / +$743 / +$13.5 | 31 / +$784 / +$25.3 | 45 / +$1,380 / +$30.7 | 170 / +$3,078 / +$18.1 / 51% |

The E1 thresholds were mined on Nov 2024-Mar 2026 (P1-P3), so only P4 (Apr-Oct 2026 holdout: 46 trades, +$1,527, PF 2.06) is independent.
The ONL "flat" rows use the RTH revamp's trades. The live REVAMP also trades overnight, so the live E1 may be skipped a bit more often than in the backtest.
