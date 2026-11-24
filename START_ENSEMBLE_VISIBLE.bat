@echo off
cd /d C:\Users\shawa\Projects\bott-trader
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TRADING_MODE=live
set FALLBACK_ENTRIES_ENABLED=false
REM Skip NEW OG entries in the 60 min after Tokyo open (8 PM ET in EDT, 7 PM ET in EST). 0 = off.
set OG_TOKYO_BLOCK_MIN=60
REM Plain-English bias / setup / win-rate readout each loop (read-only). 0 = off.
set OG_STATUS=1
set OG_STATUS_LOG=logs\og_status.log
set RITHMIC_ACCOUNT_ID=LFE050-WZK0983B-TEST016
set DAILY_LOSS_LIMIT_PERCENT=1.0
set TRAIL_ATR_MULT=0.35
set TRAIL_ACTIVATION_R=0.25
set NEAT_BAR_EXPORT=1
set NEAT_BAR_EXPORT_PATH=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
REM Confidence lot sizing (gated): base 1 until day green + win streak, max 4
set BASE_CONTRACTS=1
set MAX_CONTRACTS=4
set SIZE_WIN_STREAK_MIN=0
set SIZE_REQUIRE_DAY_GREEN=0
set MAX_CONCURRENT_TRADES=3
set EXTRA_TRADE_MIN_CONF=0.70
set MAX_TRADES_PER_DAY=0
REM Skip/trim entries whose stop risk x contracts > this ($). 0 = off.
set MAX_RISK_PER_TRADE_USD=500
set SIZE_CONF_2=0.72
set SIZE_CONF_3=0.78
set SIZE_CONF_4=0.85
REM Profit lock (2026-10-07): no SL move until +$100 open (whole trade), then lock +$20,
REM from +$150 peak keep 50% of peak; tighten-only, profit-only. ATR trail only after past-TP.
set LOCK_MODE=usd
set LOCK_ACTIVATE_USD=100
set LOCK_KEEP_USD=20
set LOCK_TRAIL_START_USD=150
set LOCK_TRAIL_KEEP_PCT=0.5
REM Re-hunt right after a close: 1 bar after a win, 2 bars (10 min) after a loss
set COOLDOWN_BARS_AFTER_WIN=1
set COOLDOWN_BARS_AFTER_LOSS=2
set LOSS_COOLDOWN_MINUTES=10
title bott-trader MNQ LIVE (Lucid sim)
.\venv\Scripts\python.exe .\start_live_rithmic.py --symbols MNQ --yes
pause
