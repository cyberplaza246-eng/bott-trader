@echo off
REM Unattended MNQ live loop — restarts on crash. Does not place a YES prompt.
REM Flatten happens at RTH / scalp-window close inside the Python process.

title BottTrader - MNQ unattended
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
if not exist logs mkdir logs

:loop
echo [%date% %time%] Starting MNQ profit mode...
call "%~dp0start_mnq_live.bat"
echo [%date% %time%] Bot exited with code %errorlevel%. Restarting in 15 seconds...
timeout /t 15 /nobreak >nul
goto loop
