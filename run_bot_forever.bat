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

:loop
echo [%date% %time%] Launching bot...
venv\Scripts\python.exe -m scripts.run_bot >> logs\bot_supervisor.log 2>&1
echo [%date% %time%] Bot exited. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
