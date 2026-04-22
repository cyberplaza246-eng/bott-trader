@echo off
if "%~1"=="--run" goto :run
title BottTrader - LIVE NQ only
cd /d "%~dp0"
cmd /c "%~f0" --run
goto :eof

:run
setlocal enabledelayedexpansion

echo ============================================
echo   BottTrader - LIVE NQ only
echo ============================================
echo.

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set "key=%%A"
    if not "!key:~0,1!"=="#" if not "!key!"=="" (
        set "%%A=%%B"
    )
)

echo   Account: %RITHMIC_USER_ID%
echo   System:  %RITHMIC_SYSTEM%
echo   Symbols: NQ
echo   Mode:    LIVE - REAL MONEY
echo ============================================
echo.

python start_live_rithmic.py --symbol NQ --yes
