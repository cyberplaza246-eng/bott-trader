@echo off
REM One Lucid session. Do not run this with start_mnq_live.bat. Ctrl+C this window first if you want the day ema15 recipe.
title MNQ overnight Lucid sim
cd /d "%~dp0"

echo ============================================
echo   BottTrader - OVERNIGHT LUCID SIM ORDERS
echo   Recipe: last-through last 20-minute high/low (NO fade)
echo   TP 0.65R + MACD 15m require (conf floor OFF — OVERNIGHT_MIN_CONFIDENCE=0)
echo   Keeps running through RTH once started. No flatten 09:25. No 9:30-16:00 idle.
echo   No ET-day fill cap by default (PAPER_MAX_FILLS=0). Set PAPER_MAX_FILLS^>0 to cap.
echo   Orders go to Lucid TEST011 / CME simulator
echo   NOT local fake fills. NOT funded live money
echo   Desk: http://127.0.0.1:5055
echo   Do not also run start_mnq_live.bat (one Lucid login)
echo   Close Lucid / R^|Trader desktop too
echo   CME closed Fri 17:00-Sun 18:00 ET. 5 losers in a row = 30 min cooldown.
echo   To bypass loss cooldown temporarily: set OVERNIGHT_SKIP_COOLDOWN=1 below.
echo   Leave this black window open. Ctrl+C to flatten and stop.
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Missing venv\Scripts\activate.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
set PYTHONUNBUFFERED=1

REM Independent desk window that survives this bot. Skip if already up.
venv\Scripts\python.exe -c "import socket; s=socket.socket(); s.settimeout(0.4); raise SystemExit(0 if s.connect_ex(('127.0.0.1',5055))==0 else 1)" 1>nul 2>nul
if %ERRORLEVEL%==0 (
    echo Desk already at http://127.0.0.1:5055 — attaching, not starting a second copy.
) else (
    echo Starting MNQ desk in its own window...
    start "MNQ desk" cmd /k "%~dp0start_mnq_desk.bat"
    timeout /t 2 /nobreak >nul
)
echo.

set PAPER_OVERNIGHT=true
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
if not defined PAPER_TEST_FILL set PAPER_TEST_FILL=false
if not defined OVERNIGHT_MIN_CONFIDENCE set OVERNIGHT_MIN_CONFIDENCE=0
if not defined OVERNIGHT_TP_R set OVERNIGHT_TP_R=0.65
if not defined OVERNIGHT_MACD set OVERNIGHT_MACD=true
if not defined OVERNIGHT_MACD_REQUIRE set OVERNIGHT_MACD_REQUIRE=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY_PNL=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
set RITHMIC_QUOTES_ONLY=false
set TRADING_MODE=paper
set OVERNIGHT_TRADING=false

REM Temporary override: skip 5-loss / 30m cooldown (restart required). Remove when done.
set OVERNIGHT_SKIP_COOLDOWN=1
REM Or disable the trigger entirely for paper: set OVERNIGHT_COOLDOWN_LOSSES=0

set "TEST_FILL_FLAG="
if /I "%PAPER_TEST_FILL%"=="true" set "TEST_FILL_FLAG=--paper-test-fill"
python -u start_live_mtf_scalping.py --paper --overnight-research %TEST_FILL_FLAG% --symbols MNQ --duration 0
echo.
echo Bot exited with code %ERRORLEVEL%. Window stays open so you can read the log.
pause
