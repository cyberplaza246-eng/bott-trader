@echo off
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ CONTINUOUS LIVE
echo   Stack: daily + 15m + 60m EMA (long OR short when all agree)
echo   Hunt: continuous RTH 09:35-15:50 ET
echo   Entry: quiet 1m bar close with trend
echo   Stop: fixed 25 pts / Target: 500-pt cap / EOD 15:50
echo   Cooldown: 15 min / max 8 day
echo   LIVE orders on Rithmic / Lucid - one login only
echo   Close Lucid desktop first
echo ============================================
echo.

call venv\Scripts\activate.bat
REM Prefer venv python only — WindowsApps python alias can spawn a 2nd live bot.
set "PATH=%~dp0venv\Scripts;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM Force recipe (do not inherit stale Cursor/shell env)
set "SCALP_MODE=ema15_eod"
set "STRATEGY_MODE=ema15_eod"
set "SESSION_MODE=rth"
set "OVERNIGHT_TRADING=false"
set "SCALP_SESSIONS=rth=09:30-16:00"
set "MAX_HOLD_SECONDS=23400"
set "MNQ_MAX_POSITIONS=2"
set "MAX_TRADES_PER_DAY=8"
set "EMA15_MAX_TRADES_DAY=8"
set "USE_30S_BARS=false"
set "RITHMIC_SKIP_HISTORY_PNL=true"
set "VERBOSE_SKIP_REASONS=true"
set "USE_ORDER_FLOW=false"
set "EMA15_CLEAN_FORWARD=1"
set "EMA15_CONTINUOUS=1"
set "EMA15_ENTRY_COOLDOWN_MIN=15"
set "EMA15_LEVEL_MODE=0"
set "EMA15_REQUIRE_DAILY=true"
set "EMA15_REQUIRE_60M=true"
set "EMA15_REQUIRE_5M=false"
set "EMA15_REQUIRE_QUIET=true"
set "EMA15_SIDE=both"
set "EMA15_SL_FIXED=25"
set "EMA15_TP_R=0"
set "SCALP_TP_PTS=500"
set "EMA15_MIN_CONFIDENCE=0"
set "EMA15_ADX_MIN=0"
set "EMA15_1M_SEED=data\MNQ_1m.csv"
set "LOSS_COOLDOWN_MINUTES=0"
set "MTF_MAX_CONSEC_LOSSES=99"

echo Starting LIVE continuous...
"%~dp0venv\Scripts\python.exe" -u start_live_mtf_scalping.py --live --symbols MNQ --duration 0
echo.
echo Bot exited. Press any key to close.
pause >nul
