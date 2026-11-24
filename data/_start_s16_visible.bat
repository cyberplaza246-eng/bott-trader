@echo off
title MNQ S16 mid PAPER - leave this window open
cd /d "C:\Users\shawa\Projects\bott-trader"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set TRADING_MODE=paper
set ALLOW_LIVE=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY=false
set RITHMIC_SKIP_HISTORY_PNL=false
set RITHMIC_SKIP_DIAGNOSTICS=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
set RITHMIC_IMMEDIATE_PROTECTIVE=true
set RITHMIC_ALLOW_PROTECTIVE_FALLBACK=true
set RITHMIC_LUCID_SL_STOP_LIMIT=true
set PAPER_DAY_HYBRID=false
set DAY_HYBRID=false
set DAY_HYBRID_GLOBEX=false
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set MICRO_SCALP20=false
set MICRO_SCALP=false
set S16_MID_SIZE=1
set EMA15_1M_SEED=data\MNQ_1m.csv
set S16_MID_LOCK=C:\Users\shawa\Projects\bott-trader\data\MNQ_PASS_s16_mid.json
echo.
echo ============================================
echo   MNQ S16 MID PAPER - Lucid TEST
echo   Look for this black window on the taskbar
echo   Leave open. Ctrl+C to stop.
echo ============================================
echo.
"C:\Users\shawa\Projects\bott-trader\venv\Scripts\python.exe" -u scripts\run_s16_mid_paper.py
echo.
echo Exited code %ERRORLEVEL%.
pause
