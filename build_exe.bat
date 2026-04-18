@echo off
REM ─────────────────────────────────────────────────────────
REM  Build BottTrader.exe locally on Windows
REM  Requires Python 3.10+ installed and on PATH
REM ─────────────────────────────────────────────────────────
echo.
echo ╔═══════════════════════════════════════════╗
echo ║   BottTrader - Building Desktop App       ║
echo ╚═══════════════════════════════════════════╝
echo.

REM Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt
pip install -r requirements_desktop.txt

REM Build
echo.
echo [2/3] Building BottTrader.exe...
pyinstaller --onefile --windowed --name BottTrader ^
    --add-data "src;src" ^
    --add-data "config;config" ^
    --add-data "data;data" ^
    --hidden-import="pandas" ^
    --hidden-import="numpy" ^
    --hidden-import="sklearn" ^
    --hidden-import="sklearn.ensemble" ^
    --hidden-import="sklearn.svm" ^
    --hidden-import="sklearn.preprocessing" ^
    --hidden-import="sklearn.model_selection" ^
    --hidden-import="scipy" ^
    --hidden-import="pandas_ta" ^
    --hidden-import="yfinance" ^
    --hidden-import="requests" ^
    --hidden-import="flask" ^
    --hidden-import="flask_cors" ^
    --hidden-import="feedparser" ^
    --hidden-import="apscheduler" ^
    --hidden-import="apscheduler.schedulers.background" ^
    --hidden-import="async_rithmic" ^
    --hidden-import="dotenv" ^
    --hidden-import="pythonjsonlogger" ^
    --hidden-import="customtkinter" ^
    --hidden-import="pystray" ^
    --hidden-import="plyer.platforms.win.notification" ^
    --collect-all pandas ^
    --collect-all numpy ^
    --collect-all sklearn ^
    --collect-all pandas_ta ^
    --collect-all customtkinter ^
    --collect-all async_rithmic ^
    launcher_app.py

echo.
if exist dist\BottTrader.exe (
    echo [3/3] ✅ Build successful!
    echo.
    echo   Output: dist\BottTrader.exe
    echo.
    echo   Copy BottTrader.exe and your .env file to the same folder,
    echo   then double-click to launch.
) else (
    echo [3/3] ❌ Build failed — check errors above
)
echo.
pause
