# Saved rule: three down days and a low 2-day RSI

Turned on in `start_mnq_live_revamp.bat` (`DIP_DOWN_RSI=1`). The code default is off, so the other launchers do not trade it. It does nothing until that bat is started.

Buy 1 MNQ at the 18:00 ET open after a daily Nasdaq close that is the third lower close in a row and the 2-day RSI is under 15. Sell when the 2-day RSI closes back above 65, or on the close of the 6th session, whichever comes first. While this trade is on, the 5-minute bot does not open new trades. If the bot is already in a trade at 6pm, this one is skipped.

The tested rule has no stop. The live order uses a 2,500-point disaster stop ($5,000 on 1 MNQ) so the account is not naked. That distance was not hit from 2010 through 2026-11-06. The $500 stop cap still applies to every other trade.

| Period | Net | Trades | Profit factor |
|---|---:|---:|---:|
| 2010–2019 | +$3,372 | 88 | 1.92 |
| 2020–2022 | +$8,288 | 36 | 2.83 |
| 2023–2024 | +$1,730 | 18 | 1.42 |
| 2025–2026 (not used to pick it) | +$1,774 | 19 | — |

It lost in 2016, 2020, and 2023. A large part of the 2020–2022 gain was 2022. Reproduce with `research/build_dips_2026-11-07/run.py` (the `DOWN_RSI` row).
