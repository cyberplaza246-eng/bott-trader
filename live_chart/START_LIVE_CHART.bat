@echo off
REM READ-ONLY live chart for EVAL PASSER (no orders, no broker). Open http://127.0.0.1:8765
REM Usage: START_LIVE_CHART.bat          -> EVAL PASSER live
REM        START_LIVE_CHART.bat funded   -> FUNDED 12K live (port 8766)
REM        START_LIVE_CHART.bat replay   -> demo replay of 2026-09-15
set BOT=C:\Users\shawa\Projects\bott-trader
set BARS=C:\Users\shawa\neat-mnq-evolution\artifacts\live_bars\MNQ_1m_live.csv
set PY=%BOT%\venv\Scripts\python.exe
cd /d %~dp0
if /i "%1"=="replay" (
  start "" http://127.0.0.1:8765
  "%PY%" server.py --replay 2026-09-15 --speed 60
) else if /i "%1"=="funded" (
  start "" http://127.0.0.1:8766
  "%PY%" server.py --mode funded --port 8766 --bars "%BARS%" --events "%BOT%\logs\funded12k_events.jsonl" --state "%BOT%\logs\funded12k_state.json" --daily-stop 500 --start-balance 52200 --floor 50100
) else (
  start "" http://127.0.0.1:8765
  "%PY%" server.py --mode eval --bars "%BARS%" --events "%BOT%\logs\flex3_events.jsonl" --state "%BOT%\logs\flex3_state.json" --daily-stop 1500
)
pause
