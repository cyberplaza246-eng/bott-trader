@echo off
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ PROFIT MODE (LIVE)
echo   15m EMA + daily confirm + ATR stop
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
if not defined MAX_TRADES_PER_DAY set MAX_TRADES_PER_DAY=6
if not defined USE_30S_BARS set USE_30S_BARS=false
if not defined RITHMIC_SKIP_HISTORY_PNL set RITHMIC_SKIP_HISTORY_PNL=true
if not defined VERBOSE_SKIP_REASONS set VERBOSE_SKIP_REASONS=true
if not defined USE_ORDER_FLOW set USE_ORDER_FLOW=false
if not defined EMA15_REQUIRE_DAILY set EMA15_REQUIRE_DAILY=true
if not defined EMA15_1M_SEED set EMA15_1M_SEED=data\MNQ_1m.csv

python start_live_mtf_scalping.py --live --symbols MNQ --duration 0
