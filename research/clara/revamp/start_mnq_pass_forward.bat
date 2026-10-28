@echo off
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ CONTINUOUS FORWARD
echo   Stack: daily + 15m + 60m EMA ALL long
echo   Hunt: continuous RTH 09:35-15:50 ET
echo   Entry: quiet 1m bar close timing
echo   Stop: fixed 25 pts / Target: 500-pt cap / EOD 15:50
echo   Cooldown: 15 min between entries / max 8 day
echo   Forward check: IS+Val+Aug green (quiet 1m continuous)
echo   Lucid TEST / paper - one Lucid login only
echo ============================================
echo.

call venv\Scripts\activate.bat
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

if not defined SCALP_MODE set "SCALP_MODE=ema15_eod"
if not defined STRATEGY_MODE set "STRATEGY_MODE=ema15_eod"
if not defined SESSION_MODE set "SESSION_MODE=rth"
if not defined OVERNIGHT_TRADING set "OVERNIGHT_TRADING=false"
if not defined SCALP_SESSIONS set "SCALP_SESSIONS=rth=09:30-16:00"
if not defined MAX_HOLD_SECONDS set "MAX_HOLD_SECONDS=23400"
if not defined MNQ_MAX_POSITIONS set "MNQ_MAX_POSITIONS=2"
if not defined MAX_TRADES_PER_DAY set "MAX_TRADES_PER_DAY=8"
if not defined USE_30S_BARS set "USE_30S_BARS=false"
if not defined RITHMIC_SKIP_HISTORY_PNL set "RITHMIC_SKIP_HISTORY_PNL=true"
if not defined VERBOSE_SKIP_REASONS set "VERBOSE_SKIP_REASONS=true"
if not defined USE_ORDER_FLOW set "USE_ORDER_FLOW=false"

REM --- Continuous hunt + 1m quiet timing (NOT slotted w2_open) ---
if not defined EMA15_CLEAN_FORWARD set "EMA15_CLEAN_FORWARD=1"
if not defined EMA15_CONTINUOUS set "EMA15_CONTINUOUS=1"
if not defined EMA15_ENTRY_COOLDOWN_MIN set "EMA15_ENTRY_COOLDOWN_MIN=15"
if not defined EMA15_LEVEL_MODE set "EMA15_LEVEL_MODE=0"
if not defined EMA15_REQUIRE_DAILY set "EMA15_REQUIRE_DAILY=true"
if not defined EMA15_REQUIRE_60M set "EMA15_REQUIRE_60M=true"
if not defined EMA15_REQUIRE_5M set "EMA15_REQUIRE_5M=false"
if not defined EMA15_REQUIRE_QUIET set "EMA15_REQUIRE_QUIET=true"
if not defined EMA15_SIDE set "EMA15_SIDE=both"
if not defined EMA15_SL_FIXED set "EMA15_SL_FIXED=25"
if not defined EMA15_TP_R set "EMA15_TP_R=0"
if not defined SCALP_TP_PTS set "SCALP_TP_PTS=500"
if not defined EMA15_MIN_CONFIDENCE set "EMA15_MIN_CONFIDENCE=0"
if not defined EMA15_ADX_MIN set "EMA15_ADX_MIN=0"
if not defined EMA15_1M_SEED set "EMA15_1M_SEED=data\MNQ_1m.csv"
if not defined LOSS_COOLDOWN_MINUTES set "LOSS_COOLDOWN_MINUTES=0"
if not defined MTF_MAX_CONSEC_LOSSES set "MTF_MAX_CONSEC_LOSSES=99"

if not defined PAPER_USE_RITHMIC set "PAPER_USE_RITHMIC=true"

echo Starting PAPER continuous...
"%~dp0venv\Scripts\python.exe" -u start_live_mtf_scalping.py --paper --symbols MNQ --duration 0
