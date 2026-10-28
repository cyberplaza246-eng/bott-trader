@echo off
rem Paper only — use start_nq_live.bat or start_mnq_live.bat for real Rithmic orders

if "%~1"=="--run" goto :run

title BottTrader - Nasdaq MTF Paper

cd /d "%~dp0"

cmd /c "%~f0" --run

goto :eof



:run

setlocal enabledelayedexpansion



echo ============================================

echo   BottTrader - Nasdaq MTF Scalping (Paper)

echo   Strategy: MTF scalping (TP 1.2x SL, ~50%% WR)

echo   Pick MNQ, NQ, or both at startup

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

echo   Mode:    PAPER

echo ============================================

echo.



call venv\Scripts\activate.bat

python start_live_mtf_scalping.py --prompt --paper

