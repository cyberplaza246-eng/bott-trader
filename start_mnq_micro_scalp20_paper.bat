@echo off
REM MNQ micro-scalp $20 — Lucid PAPER only. Do not run with overnight/day hybrid (one Lucid login).
title MNQ micro scalp $20 (paper)
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MICRO SCALP $20 PAPER
echo   Min hold 10s then TP when up $20+
echo   Hard stop from fill · Lucid TEST sim
echo   Stop overnight / day hybrid first
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

set MICRO_SCALP20=true
set MICRO_SCALP20_RTH_ONLY=false
set MICRO_SCALP20_LOOKBACK_SEC=25
set MICRO_SCALP20_IMPULSE_PTS=2.5
set MICRO_SCALP20_SL_PTS=15
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY=true
set RITHMIC_SKIP_HISTORY_PNL=false
set RITHMIC_SKIP_DIAGNOSTICS=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
REM Lucid TEST never books native SL/TP — place STOP + LIMIT with the fill
set RITHMIC_IMMEDIATE_PROTECTIVE=true
set RITHMIC_ALLOW_PROTECTIVE_FALLBACK=true
set RITHMIC_LUCID_SL_STOP_LIMIT=true
set TRADING_MODE=paper
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set PAPER_DAY_HYBRID=false
set DAY_HYBRID=false

python -u scripts\run_micro_scalp20_paper.py
echo.
echo Micro scalp exited with code %ERRORLEVEL%.
pause
