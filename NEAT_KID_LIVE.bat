@echo off
REM NEAT kid live PAPER trader - auto: reads main bot bars if it runs, else own data-only login. Never places orders.
cd /d "%~dp0"
title NEAT KID - live paper (no orders)
"%~dp0venv\Scripts\python.exe" -u "%~dp0tools\neat_live_kid.py"
echo.
echo Kid exited with code %ERRORLEVEL%.
pause
