@echo off
REM Close Lucid desktop. Do not start a second copy. Close Cursor is OK — leave this black window open.
title MNQ desk
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Missing venv\Scripts\activate.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
set PYTHONUNBUFFERED=1

venv\Scripts\python.exe -c "import socket; s=socket.socket(); s.settimeout(0.4); raise SystemExit(0 if s.connect_ex(('127.0.0.1',5055))==0 else 1)" 1>nul 2>nul
if %ERRORLEVEL%==0 (
    echo Desk already running at http://127.0.0.1:5055
    echo Overnight paper will attach. You can close this extra window.
    timeout /t 8
    exit /b 0
)

echo MNQ desk: http://127.0.0.1:5055
echo Close Cursor is OK — leave this window open.
echo Do not start a second desk if this one is already up.
echo.
python -u scripts/run_mtf_dashboard.py
echo.
echo Desk exited with code %ERRORLEVEL%. Window stays open so you can read the log.
pause
