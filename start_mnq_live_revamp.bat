@echo off
REM ============================================================
REM  MNQ LIVE Rithmic ? REVAMP exits (overnight research winner)
REM  Places REAL Rithmic/Lucid orders via start_live_rithmic.py
REM
REM  Exits (forced ON by this bat):
REM    REVAMP_TRAIL_ATR_MULT=4.0     (was 0.6 on normal ensemble live)
REM    REVAMP_BREAKOUT_TARGET_R=2.0  fixed 2R, no trail
REM    REVAMP_MR_TARGET=bb_far
REM  Profit locks stay as in .env (+$100->+$15, +$150->+$50).
REM
REM  ORIGINAL (no revamp):  start_mnq_live.bat   (ema15_eod day bot)
REM  OR ensemble without revamp: START_ENSEMBLE_VISIBLE.bat
REM  Restore folder: C:\Users\shawa\Projects\bott-trader-ORIGINAL-2026-10-06
REM
REM  Does NOT auto-confirm beyond --yes (same as ensemble bat).
REM  Close Lucid / R|Trader desktop first (one login).
REM ============================================================
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ LIVE REVAMP EXITS
echo   Path: start_live_rithmic.py (Rithmic orders)
echo   Research 5m ADX ensemble: trend, MR, breakout
echo   Trail 4.0 ATR / Breakout 2R fixed / MR bb_far
echo   Profit locks: from .env (keep ON)
echo   REAL orders ? use with care
echo ============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Missing venv\Scripts\python.exe
    pause
    exit /b 1
)

set "PATH=%~dp0venv\Scripts;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM --- REVAMP exits (master + three winner knobs) ---
set "REVAMP_EXITS=1"
set "REVAMP_TRAIL_ATR_MULT=4.0"
set "REVAMP_BREAKOUT_TARGET_R=2.0"
set "REVAMP_MR_TARGET=bb_far"

REM Live mode (do not inherit paper)
set "TRADING_MODE=live"
set "FALLBACK_ENTRIES_ENABLED=true"

title bott-trader MNQ LIVE REVAMP (Lucid)

echo REVAMP_EXITS=%REVAMP_EXITS%
echo REVAMP_TRAIL_ATR_MULT=%REVAMP_TRAIL_ATR_MULT%
echo REVAMP_BREAKOUT_TARGET_R=%REVAMP_BREAKOUT_TARGET_R%
echo REVAMP_MR_TARGET=%REVAMP_MR_TARGET%
echo.
echo Starting LIVE revamp path (Ctrl+C to stop)...
"%~dp0venv\Scripts\python.exe" -u ".\start_live_rithmic.py" --symbols MNQ --yes
echo.
echo Bot exited with code %ERRORLEVEL%.
pause
