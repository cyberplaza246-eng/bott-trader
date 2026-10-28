@echo off
REM MNQ S16 mid HybridS15_IB ? Lucid PAPER only. Stop day hybrid / overnight first (one Lucid login).
title MNQ S16 mid paper (HybridS15_IB)
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ S16 MID PAPER
echo   HybridS15_IB ORB20+IB45 long-only
echo   Lucid TEST sim ? NOT funded live
echo   Stop day hybrid / overnight / micro-scalp first
echo   Close Lucid desktop / R|Trader if open
echo   Leave this window open. Ctrl+C to stop.
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Missing venv\Scripts\activate.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Paper / TEST only ? never funded live
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

REM Disable other bots
set PAPER_DAY_HYBRID=false
set DAY_HYBRID=false
set DAY_HYBRID_GLOBEX=false
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set MICRO_SCALP20=false
set MICRO_SCALP=false

if not defined S16_MID_SIZE set S16_MID_SIZE=1
set EMA15_1M_SEED=data\MNQ_1m.csv
set S16_MID_LOCK=%~dp0data\MNQ_PASS_s16_mid.json

REM Always venv python ? never Windows Store
venv\Scripts\python.exe -u scripts\run_s16_mid_paper.py %*
echo.
echo S16 mid paper exited with code %ERRORLEVEL%.
pause
