@echo off
REM EVAL PASSER SIM UK50C73-TEST017 - generated on box; review before use. Never point at a non-sim account.
cd /d C:\Users\shawa\Projects\bott-trader
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TRADING_MODE=live
set NEAT_BAR_EXPORT=1
set NEAT_BAR_EXPORT_PATH=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
set EVAL_PASSER_ON=1
set FUNDED_12K_ON=0
set FLEX3_ACCOUNT=LFE050-PUK50C73-TEST017
set RITHMIC_ACCOUNT_ID=LFE050-PUK50C73-TEST017
set FLEX3_SIZE=1
set FLEX3_EXCLUDE_SRC=pb_ema
set FLEX3_DAILY_LOSS_STOP=1500
set FLEX3_DAY_PROFIT_CAP=1350
set FLEX3_CAP_FLATTEN=1
set FLEX3_COST_PER_CONTRACT=4
set FLEX3_TARGET=3000
set FLEX3_CONSISTENCY=0.50
set FLEX3_6PM_DD_GATE=500
set FLEX3_6PM_FAR_TP_PTS=800

set FLEX3_OG_TREND=1
set FLEX3_FLAT_ET=16:44
set FLEX3_REOPEN_ET=18:00
set FLEX3_V2_LOCK_ACT=400
set FLEX3_V2_LOCK_KEEP=50
set FLEX3_1M_CSV=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
set FLEX3_BAR_LABEL=open
set FLEX3_STATE_FILE=logs/flex3_state.json

set FLEX3_EVENT_LOG=logs\flex3_events.jsonl
set FLEX3_STATUS=1
set FLEX3_STATUS_LOG=logs\flex3_status.log
set MAX_CONCURRENT_TRADES=3
title EVAL PASSER SIM UK50C73-TEST017
.\venv\Scripts\python.exe .\start_live_rithmic.py --symbols MNQ --yes
pause
