@echo off

cd /d "%~dp0"

echo ============================================
echo   BottTrader - NQ MTF Scalping (LIVE)
echo   REAL orders on Rithmic — use with care
echo   --live overrides TRADING_MODE in .env
echo ============================================
echo.

call venv\Scripts\activate.bat

python start_live_mtf_scalping.py --live --symbols NQ
