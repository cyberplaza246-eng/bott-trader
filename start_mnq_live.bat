@echo off
cd /d "%~dp0"

echo ============================================
echo   BottTrader - MNQ PROFIT MODE (LIVE)
echo   Hybrid ultra_fast | RTH scalp windows
echo   REAL orders on Rithmic — use with care
echo   --live overrides TRADING_MODE in .env
echo   See docs\PROFITABLE_LIVE.md
echo ============================================
echo.

call venv\Scripts\activate.bat

REM Locked recipe — wins only when these are unset in .env
if not defined SCALP_MODE set SCALP_MODE=hybrid
if not defined STRATEGY_MODE set STRATEGY_MODE=scalp_hybrid
if not defined SESSION_MODE set SESSION_MODE=rth
if not defined OVERNIGHT_TRADING set OVERNIGHT_TRADING=false
if not defined SCALP_SESSIONS set SCALP_SESSIONS=morning=09:30-12:00;afternoon=13:30-16:00
if not defined SCALP_AGGRESSIVE set SCALP_AGGRESSIVE=true
if not defined SCALP_FAST_MODE set SCALP_FAST_MODE=true
if not defined MAX_HOLD_SECONDS set MAX_HOLD_SECONDS=30
if not defined MNQ_MAX_POSITIONS set MNQ_MAX_POSITIONS=2
if not defined USE_ORDER_FLOW set USE_ORDER_FLOW=true
if not defined ORDER_FLOW_MODE set ORDER_FLOW_MODE=block
if not defined MTF_MAX_CONSEC_LOSSES set MTF_MAX_CONSEC_LOSSES=3
if not defined LOSS_COOLDOWN_MINUTES set LOSS_COOLDOWN_MINUTES=15
if not defined SCAN_SLEEP_OPEN_SEC set SCAN_SLEEP_OPEN_SEC=5
if not defined SCAN_SLEEP_IDLE_SEC set SCAN_SLEEP_IDLE_SEC=10
if not defined USE_30S_BARS set USE_30S_BARS=true

python start_live_mtf_scalping.py --live --symbols MNQ --duration 0
