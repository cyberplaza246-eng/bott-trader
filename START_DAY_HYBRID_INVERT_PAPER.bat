@echo off
cd /d C:\Users\shawa\Projects\bott-trader
call .\venv\Scripts\activate.bat
set PAPER_DAY_HYBRID=true
set DAY_HYBRID=true
set DAY_HYBRID_GLOBEX=true
set DAY_HYBRID_RSI_GATE=true
set DAY_HYBRID_RSI_MIN_SHORT=40
set DAY_HYBRID_RSI_MAX_LONG=65
set DAY_HYBRID_INVERT=true
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY=false
set TRADING_MODE=paper
set EMA15_1M_SEED=data\MNQ_1m.csv
echo INVERT ON — long signals become shorts and vice versa
.\venv\Scripts\python.exe -u start_live_mtf_scalping.py --paper --day-hybrid --symbols MNQ --duration 0
pause
