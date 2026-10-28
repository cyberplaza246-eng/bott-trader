@echo off
REM MNQ OpenTrend - Lucid PAPER/TEST only. Do not use for funded live trading.
title MNQ OpenTrend paper (Lucid TEST)
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ OPENTREND PAPER
echo   5m EMA9/EMA21 crossover, ATR14 brackets
echo   Lucid TEST sim - NOT funded live
echo   Long and short, RTH only, flatten 15:55 ET
echo   OpenTrend is not Trend Catcher
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

REM Paper / TEST only - never funded live
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

set OPEN_TREND_SIZE=1
set EMA15_1M_SEED=data\MNQ_1m.csv

REM Always venv python - never Windows Store
venv\Scripts\python.exe -u scripts\run_open_trend_paper.py %*
echo.
echo OpenTrend paper exited with code %ERRORLEVEL%.
pause
