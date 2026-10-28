# Handoff for the next Grok (Oct 27, 2026)

Paste this to the next Grok: "Read HANDOFF_NEXT_GROK.md in C:\Users\shawa\Projects\bott-trader (branch research-2026-10-27) and continue."

## Setup
- PC: Paragan, repo C:\Users\shawa\Projects\bott-trader (PowerShell, no rg). Live branch: revamp-live-2026-10-06.
- Accounts: Eval LFE050-PUK50C73-TEST017, OG eval LFE050-WZK0983B-TEST016, Funded LFF050-9XP156CR-PRO005 (only ONE bot on PRO005 at a time; Lucid copier off).
- Rules: never place/cancel orders, never stop/restart a bot without Shawayne's OK. Back up before edits; new behavior behind env switches default OFF. No bypassing software licensing (Clara Core was NOT decompiled).
- Full revamp backup: C:\Users\shawa\Projects\bott-trader_revamp_backup_20261027

## Findings
- Revamp alone (1 MNQ, Nov 2024-Sep 2026, net of $4 RT + 1 tick slip each side): +$25.2k (+$14.7k train, +$10.4k 2026), PF 1.19, win 61%, max DD ~$4,800, needs ~$7k account. 17/23 months green. Robust to 2x slippage and +/-20% exit params. Caveat: params tuned Oct 2026 on same data.
- Sleeves: daytime breakout +$14.1k (best), overnight breakout +$5.4k on ~1,000 low-value trades (00:00-09:00 = only +$2.3k), 6 PM long ~+$2k, ONL ~+$2k, daytime trend -$0.9k (only loser).
- Clara Core replica + fusion with revamp: made it WORSE in every clean test (an early win was a UTC/ET look-ahead bug).
- OSS ORB/VWAP ports: only noise-band strategy profitable (+$3.1k, $5.2k DD).
- OG regime fix (src\core\ensemble_trader.py, backup .bak_regime) backtests WORSE (-$5.1k vs -$1.9k). It defaults ON at next OG restart -> set OG_REGIME_FIX=0 or restore the backup.
- Funded 12K: SIXPM_SKIP_MONDAY=1, SIXPM_TRAIL=1, RITHMIC_ORDER_AUTO_RECONNECT need a restart to load.

## Next steps
1. Backtest revamp with daytime trend sleeve OFF and 00:00-09:00 overnight breakouts trimmed (pick on train/walk-forward, test 2026 once).
2. Decide on OG_REGIME_FIX=0.
3. Research scripts are in research/clara and research/sixpm (sim.py, rank.py, solo.py, revamp/sens.py etc.). Data: data/MNQ_1m.csv.

## Update Oct 28, 2026 8:40 AM ET
- Shawayne said YES to the trim tests and asked to try to DOUBLE profit, via bigger size OR more trades per day (raise position/day caps, re-entries, looser breakout entries, more RTH breakout windows).
- Work in progress on Grok's box: /workspace/clara/revamp/next/ (scripts + results). Rule: pick variants on train (<=2025)/walk-forward, test 2026 once, report net after costs, DD, $2k breach count, account size needed.
- Nothing changed on the PC or live bots.
- Also: CRM Residential phone screen (Asst. Property Manager, $28-30/hr) - prep delivered; booking at c.nxtapply.com/schedule/crm-residential-phone-interview.
