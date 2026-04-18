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
    --hidden-import="sklearn" ^
    --hidden-import="sklearn.ensemble" ^
    --hidden-import="sklearn.svm" ^
    --hidden-import="pandas_ta" ^
    --hidden-import="customtkinter" ^
    --hidden-import="dotenv" ^
    --hidden-import="pystray" ^
    --hidden-import="plyer.platforms.win.notification" ^
    --collect-data customtkinter ^
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
