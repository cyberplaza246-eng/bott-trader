@echo off
if "%~1"=="--run" goto :run
title BottTrader - LIVE MES+NQ
cd /d "%~dp0"
cmd /c "%~f0" --run
goto :eof

:run
setlocal enabledelayedexpansion

echo ============================================
echo   BottTrader - LIVE MES+NQ
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
echo   Symbols: MES NQ
echo   Mode:    LIVE - REAL MONEY
echo ============================================
echo.

python start_live_rithmic.py --symbols MES NQ --yes
