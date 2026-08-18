@echo off
cd /d "%~dp0"
echo ============================================
echo   BottTrader - Nasdaq hybrid BACKTEST
echo   MNQ + NQ  ^|  no Rithmic / no live orders
echo ============================================
echo.

call venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8

if "%~1"=="rithmic-csv" (
    echo --- MNQ  data\MNQ_1m_rithmic.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol MNQ --csv-1m data/MNQ_1m_rithmic.csv --csv-5m data/MNQ_5m_rithmic.csv
    echo.
    echo --- NQ  data\NQ_1m_rithmic.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol NQ --csv-1m data/NQ_1m_rithmic.csv --csv-5m data/NQ_5m_rithmic.csv
) else if "%~1"=="nq" (
    echo --- NQ  data\NQ_1m.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol NQ --csv-1m data/NQ_1m.csv --csv-5m data/NQ_5m.csv
) else if "%~1"=="mnq" (
    echo --- MNQ  data\MNQ_1m.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol MNQ
) else (
    echo --- MNQ  data\MNQ_1m.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol MNQ
    echo.
    echo --- NQ  data\NQ_1m.csv ---
    python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows --symbol NQ --csv-1m data/NQ_1m.csv --csv-5m data/NQ_5m.csv
)

echo.
echo Saved: data\scalp_hybrid_backtest_MNQ.json  and/or  data\scalp_hybrid_backtest_NQ.json
echo Optional: start_mnq_backtest.bat rithmic-csv ^| nq ^| mnq
pause
