@echo off
REM FUNDED 12K dry run (live data, simulated orders) - generated on box; review before use.
cd /d C:\Users\shawa\Projects\bott-trader
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TRADING_MODE=live
set NEAT_BAR_EXPORT=1
set NEAT_BAR_EXPORT_PATH=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
set PASS_EVAL_3K_ON=0
set FLEX3_ON=0
set FUNDED_12K_ON=1
set FUNDED_ACCOUNT_ALLOWLIST=LFF050-9XP156CR-PRO005
set RITHMIC_ACCOUNT_ID=LFF050-9XP156CR-PRO005
set FUNDED_START_BALANCE=52200
set FUNDED_FLOOR=50100
set FUNDED_6PM_CUSHION=0
set FUNDED_OG_ON=0
set FUNDED_EXCLUDE_SRC=pb_ema,pb_pdc,pb_pwh,pb_vwap,pb_zone,E1
set FUNDED_SIZE=1
set FUNDED_DAILY_LOSS_STOP=500
set FUNDED_HALF_DD=1000
set FUNDED_FULL_DD=500
set FUNDED_PAUSE_DD=1500
set FUNDED_PAUSE_DAYS=2
set FUNDED_OG_MAX_PER_DAY=10
set FUNDED_FLAT_ET=16:44
set FUNDED_REOPEN_ET=18:00
set FUNDED_V2_LOCK_ACT=400
set FUNDED_V2_LOCK_KEEP=50
set FUNDED_COST_PER_CONTRACT=4
set FLEX3_1M_CSV=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
set FLEX3_BAR_LABEL=open
set FLEX3_STATE_FILE=logs/funded12k_state.json
set FLEX3_EVENT_LOG=logs\funded12k_events.jsonl
set MAX_CONCURRENT_TRADES=3
set LOCK_MODE=usd
set LOCK_ACTIVATE_USD=100
set LOCK_KEEP_USD=20
set LOCK_TRAIL_START_USD=150
set LOCK_TRAIL_KEEP_PCT=0.5
if "%FUNDED_ACCOUNT_ALLOWLIST%"=="" (echo FUNDED_ACCOUNT_ALLOWLIST is empty - edit this .bat with the funded account id & pause & exit /b 1)
title FUNDED 12K dry run (live data, simulated orders)
.\venv\Scripts\python.exe .\start_live_rithmic.py --symbols MNQ --paper-orders --yes
pause
