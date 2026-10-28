@echo off
REM Micro-scalp paper bot — $20+ TP after 10s hold. Do not run with other Lucid bots.
title MNQ micro scalp (paper)
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MICRO SCALP RTH BOT
echo   15m+5m+1m stack, quiet 1m pullback
echo   Stop: ATR clip 6-12 | TP: $20 after 10s hold
echo   Max hold 180s | Lucid TEST011 sim
echo   Desk: http://127.0.0.1:5055
echo   Stop other Lucid bots first (one login)
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

venv\Scripts\python.exe -c "import socket; s=socket.socket(); s.settimeout(0.4); raise SystemExit(0 if s.connect_ex(('127.0.0.1',5055))==0 else 1)" 1>nul 2>nul
if %ERRORLEVEL%==0 (
    echo Desk already at http://127.0.0.1:5055
) else (
    echo Starting MNQ desk in its own window...
    start "MNQ desk" cmd /k "%~dp0start_mnq_desk.bat"
    timeout /t 2 /nobreak >nul
)
echo.

set PAPER_MICRO_SCALP=true
set MICRO_SCALP=true
set PAPER_DAY_HYBRID=false
set DAY_HYBRID=false
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY_PNL=true
set RITHMIC_SKIP_DIAGNOSTICS=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
set TRADING_MODE=paper
set EMA15_1M_SEED=data\MNQ_1m.csv
set MICRO_SCALP_TP_USD=20
set MICRO_SCALP_MIN_HOLD_SEC=10
set MICRO_SCALP_MAX_HOLD_SEC=180
set MICRO_SCALP_SCAN_SEC=1

python -u start_live_mtf_scalping.py --paper --micro-scalp --symbols MNQ --duration 0
echo.
echo Micro scalp exited with code %ERRORLEVEL%.
pause
