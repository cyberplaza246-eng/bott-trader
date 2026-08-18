@echo off
REM One Lucid session. Stop the day bot first (Ctrl+C after 15:50). Do not run this with start_mnq_live.bat.
title MNQ overnight Lucid sim
cd /d "%~dp0"

echo ============================================
echo   BottTrader - OVERNIGHT LUCID SIM ORDERS
echo   Recipe: break settled ONH/ONL only (NO fade)
echo   1 MNQ, ATR x1.5 stop (12-40), 1R TP
echo   Flatten 09:25 ET. HARD idle 9:30-16:00 ET
echo   Orders go to Lucid TEST011 / CME simulator
echo   NOT local fake fills. NOT funded live money
echo   Desk: http://127.0.0.1:5055
echo   Day bot must be STOPPED first (one Lucid login)
echo   Close Lucid / R^|Trader desktop too
echo   First order: Globex 18:00 ET + ~60 min settle
echo   Leave this black window open
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
set PAPER_TEST_FILL=false
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY_PNL=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
set RITHMIC_QUOTES_ONLY=false
set TRADING_MODE=paper
set OVERNIGHT_TRADING=false

python -u start_live_mtf_scalping.py --paper --overnight-research --symbols MNQ --duration 0
echo.
echo Bot exited with code %ERRORLEVEL%. Window stays open so you can read the log.
pause
