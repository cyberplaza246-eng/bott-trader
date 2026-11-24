@echo off
REM Day hybrid bot — NOT the overnight bot. Do not run with start_mnq_overnight_paper.bat (one Lucid login).
title MNQ day hybrid (stacked long+short)
cd /d "%~dp0"

echo ============================================
echo   BottTrader - DAY HYBRID RTH BOT
echo   Separate from overnight Lucid bot
echo   RTH Mon-Fri 9:35-15:50 + Globex Sun 6 PM+ (DAY_HYBRID_GLOBEX=true)
echo   LONG pullback + SHORT quiet continuous — flatten 15:50 RTH only
echo   Lucid TEST011 sim — not funded live
echo   Desk: http://127.0.0.1:5055
echo   Stop overnight bot first if it is using Lucid
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

venv\Scripts\python.exe -c "import socket; s=socket.socket(); s.settimeout(0.4); raise SystemExit(0 if s.connect_ex(('127.0.0.1',5055))==0 else 1)" 1>nul 2>nul
if %ERRORLEVEL%==0 (
    echo Desk already at http://127.0.0.1:5055
) else (
    echo Starting MNQ desk in its own window...
    start "MNQ desk" cmd /k "%~dp0start_mnq_desk.bat"
    timeout /t 2 /nobreak >nul
)
echo.

set PAPER_DAY_HYBRID=true
set DAY_HYBRID=true
set DAY_HYBRID_GLOBEX=true
set DAY_HYBRID_GLOBEX_COOLDOWN=3
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY=false
set RITHMIC_SKIP_HISTORY_PNL=false
set RITHMIC_SKIP_DIAGNOSTICS=true
set RITHMIC_DISABLE_YAHOO_FALLBACK=true
set DATABENTO_REFRESH_QUIET=true
set DAY_HYBRID_RSI_GATE=true
set DAY_HYBRID_RSI_MIN_SHORT=40
set DAY_HYBRID_RSI_MAX_LONG=65
set TRADING_MODE=paper
set EMA15_1M_SEED=data\MNQ_1m.csv

REM Always use venv python — never Windows Store python (avoids dual Lucid sessions)
venv\Scripts\python.exe -u start_live_mtf_scalping.py --paper --day-hybrid --symbols MNQ --duration 0
echo.
echo Day hybrid exited with code %ERRORLEVEL%.
pause
