@echo off
cd /d "%~dp0"
echo ============================================
echo   MNQ 15m EMA EOD — REAL 1m Databento
echo   No synthetic 30s
echo ============================================
echo.

call venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8

set ONE_M=data\MNQ_1m.csv
if not exist "%ONE_M%" (
    echo Missing %ONE_M%
    echo Download Databento 1m first.
    pause
    exit /b 1
)

echo 1m: %ONE_M%
echo.
python scripts/backtest_ema15_official.py %ONE_M%
echo.
echo Also saved: data\real_1m_simple_search.json
pause
