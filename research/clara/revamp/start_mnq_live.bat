@echo off
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ L6 LIVE (day bot)
echo   Stack: 15m+5m agree, quiet 1m (body^<=4)
echo   Stop: 1m ATR x1.5 clip 6-25, TP 1R
echo   EMA15_LEVEL_MODE=0 (levels on desk map only)
echo   Conf ^>=75, max 20/day, flatten 15:50, Lucid TEST011
echo   REAL orders on Rithmic — use with care
echo   See docs\PROFITABLE_LIVE.md
echo ============================================
echo.

call venv\Scripts\activate.bat

if not defined SCALP_MODE set SCALP_MODE=ema15_eod
if not defined STRATEGY_MODE set STRATEGY_MODE=ema15_eod
if not defined SESSION_MODE set SESSION_MODE=rth
if not defined OVERNIGHT_TRADING set OVERNIGHT_TRADING=false
if not defined SCALP_SESSIONS set SCALP_SESSIONS=rth=09:30-16:00
if not defined MAX_HOLD_SECONDS set MAX_HOLD_SECONDS=23400
if not defined MNQ_MAX_POSITIONS set MNQ_MAX_POSITIONS=2
if not defined MAX_TRADES_PER_DAY set MAX_TRADES_PER_DAY=20
if not defined USE_30S_BARS set USE_30S_BARS=false
if not defined RITHMIC_SKIP_HISTORY_PNL set RITHMIC_SKIP_HISTORY_PNL=true
if not defined VERBOSE_SKIP_REASONS set VERBOSE_SKIP_REASONS=true
if not defined USE_ORDER_FLOW set USE_ORDER_FLOW=false
REM --- L6 (level_strategy_search winner) ---
if not defined EMA15_LEVEL_MODE set EMA15_LEVEL_MODE=0
if not defined EMA15_REQUIRE_DAILY set EMA15_REQUIRE_DAILY=false
if not defined EMA15_REQUIRE_60M set EMA15_REQUIRE_60M=false
if not defined EMA15_REQUIRE_5M set EMA15_REQUIRE_5M=true
if not defined EMA15_SL_TF set EMA15_SL_TF=1m
if not defined EMA15_SL_ATR_MULT set EMA15_SL_ATR_MULT=1.5
if not defined EMA15_SL_MIN set EMA15_SL_MIN=6
if not defined EMA15_SL_MAX set EMA15_SL_MAX=25
if not defined EMA15_TP_R set EMA15_TP_R=1.0
if not defined EMA15_MIN_CONFIDENCE set EMA15_MIN_CONFIDENCE=75
if not defined EMA15_1M_SEED set EMA15_1M_SEED=data\MNQ_1m.csv

set "LOCK_R1=0.5"
set "LOCK_R1_KEEP=0.1"
set "LOCK_R2=1.0"
set "LOCK_R2_KEEP=0.5"
set "TRAIL_KEEP_PCT=0.5"
set "SL_MOVE_MIN_BUFFER_TICKS=4"
python start_live_mtf_scalping.py --live --symbols MNQ --duration 0
