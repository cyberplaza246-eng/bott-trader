# OG MNQ bot, sweeps-only replay (2026-10-26)
Data: MNQ 1m 2024-09-22 to 2026-09-21 + live 5m bars 2026-10-07 to 10-26 (gap 9/21-10/7; nothing before 2024-09-22).
Method: real EnsembleTrader code replayed per 5m bar (7,932 sweep bars), then trades simulated with live rules (sizing, $500 risk cap, USD profit lock, cooldowns, session limits, Tokyo block, bad hours, same-direction-in-profit extra-trade rule). Re-entries/add-ons/adopted trades are NOT modeled.
## Base (live config)
118 trades, 42.4% win, avg win ~$97, avg loss ~-$102, PF 0.70. Net gross -$1,805, bot accounting ($2.50/trade) -$2,100, realistic ($2.24/contract) -$2,110. Max drawdown ~$2.4k.
2024: 6 trades +$39 | 2025: 78 trades -$911 | 2026 YTD: 34 trades -$1,237 (35% win) | Oct 2026: 0 trades.
Long 61 trades -$1,642 (39% win), short 57 trades -$468 (46% win). Sunday 4 trades -$161.
Avg SL 45 pts / TP 57 pts: breakeven win rate ~44.2% before costs; actual 42.4%. Exits: 58% SL, 34% TP, 8% lock stop.
See base_trades.csv, base_monthly.csv, base_breakdowns.txt, funnel.json, ablations.csv, ablation_combos.csv.
