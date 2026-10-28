@echo off
title Overnight drift PAPER (no orders)
REM PAPER ONLY: long at 18:00 ET reopen Sun-Thu, exit 09:25 ET next day (set ODRIFT_EXIT=0930 to change). Never holds past 16:00. No broker, no Rithmic, no orders.
REM Reads data\live_bars_MNQ_5m.csv read-only (fallback yfinance). Fills -> logs\overnight_drift_paper.csv. Ctrl+C to stop.
cd /d "%~dp0"
if not defined ODRIFT_EXIT set ODRIFT_EXIT=0925
if not defined ODRIFT_SYMBOL set ODRIFT_SYMBOL=MNQ
venv\Scripts\python.exe -u scripts\overnight_drift_paper.py
pause

