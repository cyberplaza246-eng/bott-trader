@echo off
if "%~1"=="--run" goto :run
title BottTrader - LIVE MES+MNQ (max 4)
cd /d "%~dp0"
cmd /c "%~f0" --run
goto :eof

:run
setlocal enabledelayedexpansion

echo ============================================
echo   BottTrader - LIVE MES+MNQ (max 4)
echo ============================================
echo.

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set "key=%%A"
    if not "!key:~0,1!"=="#" if not "!key!"=="" (
        set "%%A=%%B"
    )
)

set LIVE_MODE=true
set MAX_POSITIONS=4
set MAX_CONCURRENT_TRADES=4

echo   Account: %RITHMIC_USER_ID%
echo   System:  %RITHMIC_SYSTEM%
echo   Symbols: MES MNQ
echo   Mode:    PAPER-ORDERS (live data, simulated fills)
echo ============================================
echo.

python start_live_rithmic.py --symbols MES MNQ --paper-orders --yes

