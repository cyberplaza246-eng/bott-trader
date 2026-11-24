# MNQ Micro Scalp $20 (paper)

## What it does
- Watches the Lucid/Rithmic MNQ quote stream.
- Enters on a short impulse (price move) plus fast/slow EMA alignment.
- Places **1 contract** on Lucid **TEST/sim** with a **hard stop**.
- Does **not** take profit until the trade has been open **at least 10 seconds**.
- After 10 seconds, exits when unrealized P&amp;L is **$20 or more** (10 MNQ points at $2/pt).

## Start
1. Stop overnight paper and day hybrid (only one Lucid login).
2. Double-click `start_mnq_micro_scalp20_paper.bat` or run it from a terminal.
3. Leave the window open.

## Env knobs
- `MICRO_SCALP20_TP_USD` (default 20)
- `MICRO_SCALP20_MIN_HOLD_SEC` (default 10)
- `MICRO_SCALP20_SL_PTS` (default 10)
- `MICRO_SCALP20_RTH_ONLY` (default false)

## Safety
- Paper / Lucid simulator only (`live_mode=False`).
- Not the locked S16 mid ORB bot.
- Funded live stays off until you explicitly say otherwise.
