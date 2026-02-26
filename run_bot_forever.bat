@echo off
REM ============================================================
REM  AI Forex Trading Bot - Auto-Restart Launcher (Windows)
REM  Keeps the bot running 24/7. Restarts on crash.
REM  Put a shortcut to this file in your Startup folder to
REM  auto-launch when Windows boots.
REM ============================================================

title AI Trading Bot - LIVE

cd /d "%~dp0"

echo.
echo ====================================
echo   AI Trading Bot - Starting...
echo ====================================
echo.

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

:loop
echo [%date% %time%] Launching bot...
REM Run bot - output to console (remove redirection to see errors)
venv\Scripts\python.exe -m scripts.run_bot
echo [%date% %time%] Bot exited with code %errorlevel%. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
