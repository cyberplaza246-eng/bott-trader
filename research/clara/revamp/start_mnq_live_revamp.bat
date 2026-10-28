@echo off
REM ============================================================
REM  MNQ LIVE Rithmic ? REVAMP exits (overnight research winner)
REM  Places REAL Rithmic/Lucid orders via start_live_rithmic.py
REM
REM  Exits (forced ON by this bat):
REM    REVAMP_TRAIL_ATR_MULT=4.0 (was 0.6)
REM    REVAMP_BREAKOUT_TARGET_R=2.0 fixed 2R
REM    REVAMP_MR_TARGET=bb_far
REM  $ lock (bt 2026-10-08): +$100 -> SL +$20, from +$150 keep 50%% of peak
REM  Overnight/Globex entries ON (OVERNIGHT_TRADING + SESSION_MODE=extended).
REM
REM  ORIGINAL (no revamp): start_mnq_live.bat (ema15_eod)
REM  OR ensemble without revamp: START_ENSEMBLE_VISIBLE.bat
REM  Restore folder: C:\Users\shawa\Projects\bott-trader-ORIGINAL-2026-10-06
REM
REM  No auto-confirm beyond --yes.
REM  Close Lucid / R|Trader desktop first (one login).
REM------
REM ============================================================
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ LIVE REVAMP EXITS
echo   Path: start_live_rithmic.py (Rithmic orders)
echo   Research 5m ADX ensemble: trend, MR, breakout
echo   Trail 4.0 ATR / Breakout 2R fixed / MR bb_far
echo   $ lock: +$100 locks +$20, +$150 keeps half
echo   REAL orders ? use with care
echo ============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Missing venv\Scripts\python.exe
    pause
    exit /b 1
)

set "PATH=%~dp0venv\Scripts;%SystemRoot%\System32;%SystemRoot%"
set "LOCK_MODE=usd"
set "LOCK_ACTIVATE_USD=100"
set "LOCK_KEEP_USD=20"
set "LOCK_TRAIL_START_USD=150"
set "LOCK_TRAIL_KEEP_PCT=0.5"
set "SL_MOVE_MIN_BUFFER_TICKS=4"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM --- REVAMP exits (master + three winner knobs) ---
set "REVAMP_EXITS=1"
set "REVAMP_TRAIL_ATR_MULT=4.0"
set "REVAMP_BREAKOUT_TARGET_R=2.0"
set "REVAMP_MR_TARGET=bb_far"

REM --- Overnight / Globex hunting (widen past RTH-only 09:30-16:00 ET) ---
REM OVERNIGHT_TRADING / SESSION_MODE=extended -> ensemble rth_only=False
REM Also clears UTC bad-hour blacklist.
set "OVERNIGHT_TRADING=true"
set "SESSION_MODE=extended"
set "SCALP_SESSIONS=off"
set "REVAMP_RTH_ONLY=0"
REM Overnight takes trend + breakout + MR (was trend only, 2026-10-10). Clean-data bt 2026-10-19
REM (scripts/backtest_revamp_live_stack.py): ~2x trades, +$6.9k over Nov24-Sep26, better 4/4 periods.
REM Set to trend to restore the old rule.
set "REVAMP_OVERNIGHT_SLEEVES=all"

REM --- FIXED 1 lot, conf sizing off (10-08); max 3 trades --------------
set "BASE_CONTRACTS=1"
set "MAX_CONTRACTS=1"
set "SIZE_WIN_STREAK_MIN=0"
set "SIZE_REQUIRE_DAY_GREEN=0"
set "SIZE_CONF_2=9.99"
set "SIZE_CONF_3=9.99"
set "SIZE_CONF_4=9.99"
set "MAX_CONCURRENT_TRADES=3"
set "EXTRA_TRADE_MIN_CONF=0.70"
set "MAX_TRADES_PER_DAY=0"
REM Skip/trim entries whose stop risk x contracts > this ($). 0 = off.
set "MAX_RISK_PER_TRADE_USD=500"
set "REENTRY_ENABLED=1"
set "REENTRY_WINDOW_BARS=12"
set "REENTRY_MAX=1"

REM Live mode (do not inherit paper)
set "TRADING_MODE=live"
set "FALLBACK_ENTRIES_ENABLED=false"

title bott-trader MNQ LIVE REVAMP (Lucid)

echo REVAMP_EXITS=%REVAMP_EXITS%
echo REVAMP_TRAIL_ATR_MULT=%REVAMP_TRAIL_ATR_MULT%
echo REVAMP_BREAKOUT_TARGET_R=%REVAMP_BREAKOUT_TARGET_R%
echo REVAMP_MR_TARGET=%REVAMP_MR_TARGET%
echo OVERNIGHT_TRADING=%OVERNIGHT_TRADING% SESSION_MODE=%SESSION_MODE% SCALP_SESSIONS=%SCALP_SESSIONS%
echo REVAMP_OVERNIGHT_SLEEVES=%REVAMP_OVERNIGHT_SLEEVES% (overnight: trend + breakout + mean-reversion)
echo Session: Globex Sun 6pm-Fri 5pm ET (halt 5-6pm Mon-Thu); RTH-only entry filter OFF
echo.
echo Starting LIVE revamp path (Ctrl+C to stop)...
"%~dp0venv\Scripts\python.exe" -u ".\start_live_rithmic.py" --symbols MNQ --yes
echo.
echo Bot exited with code %ERRORLEVEL%.
pause
