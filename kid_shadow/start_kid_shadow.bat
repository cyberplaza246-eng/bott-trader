@echo off
REM NEAT kid shadow paper trader. No Rithmic, no orders. Separate window from the revamp.
cd /d "%~dp0\.."
title NEAT kid shadow (paper, no orders)
echo ============================================
echo   NEAT kid shadow - paper only, no orders
echo   Bars: data\live_bars_MNQ_5m.csv
echo   Trades: kid_shadow\kid_trades.csv
echo ============================================
echo.
"%~dp0..\venv\Scripts\python.exe" -u "%~dp0kid_shadow.py"
echo.
echo Shadow exited with code %ERRORLEVEL%.
pause
