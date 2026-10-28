@echo off
REM ============================================================
REM  NQ LIVE Rithmic - REVAMP exits (copy of start_mnq_live_revamp.bat for NQ)
REM  Places REAL Rithmic/Lucid orders via start_live_rithmic.py --symbols NQ
REM
REM  NQ = $20/pt, tick 0.25 = $5 (10x MNQ). 1 contract, 1 trade at a time.
REM  Profit lock: dollar lock scaled x10 for NQ (LOCK_MODE=usd):
REM    +$1000 open profit -> SL locks +$200; from +$1500 keep 50%% of peak.
REM  Daily loss limit stays as configured (.env: 1%% of $50k = $500). It only
REM  blocks NEW entries - it CANNOT stop one NQ trade from losing more than
REM  $500 (typical revamp stop is ~100+ pts = $2000+ on NQ).
REM
REM  ONE Rithmic login only: do NOT run this together with the MNQ bot
REM  (start_mnq_live_revamp.bat / START_ENSEMBLE_VISIBLE.bat) or with the
REM  Lucid / R|Trader desktop logged in. This bat refuses to start if any
REM  start_live_rithmic.py process is already running.
REM
REM  Contract: NQZ6 (Dec 2026). At the December roll change
REM  RITHMIC_CONTRACT_OVERRIDES below to NQ=NQH7 (or remove it to use the
REM  broker front-month lookup).
REM ============================================================
cd /d "%~dp0"

echo ============================================
echo   BottTrader - NQ LIVE REVAMP EXITS
echo   NQ = $20/pt, tick $5 - 10x MNQ risk
echo   REAL orders - use with care
echo ============================================
echo.

REM --- Single Rithmic connection guard ---
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'start_live_rithmic' }); if ($p.Count -gt 0) { $p | ForEach-Object { Write-Host ('   PID ' + $_.ProcessId + ': ' + $_.CommandLine) }; exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo.
    echo !!! ANOTHER start_live_rithmic.py BOT IS RUNNING ^(listed above^). !!!
    echo !!! Rithmic/Lucid allows ONE connection. Stop the other bot first. !!!
    echo !!! Not starting NQ. !!!
    echo.
    pause
    exit /b 1
)
echo No other start_live_rithmic.py running. Make sure Lucid / R^|Trader desktop is logged out.
echo.

if not exist "venv\Scripts\python.exe" (
    echo Missing venv\Scripts\python.exe
    pause
    exit /b 1
)

set "PATH=%~dp0venv\Scripts;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM --- Instrument: NQ front month on CME ---
set "TRADING_SYMBOLS=NQ"
set "SYMBOL_PRIORITY=NQ"
set "RITHMIC_CONTRACT_OVERRIDES=NQ=NQZ6"
set "COMMISSION_RT_NQ=4.50"
REM NEAT sidecar export is MNQ-only; keep it off here.
set "NEAT_BAR_EXPORT=0"

REM --- Profit lock: dollar lock scaled x10 for NQ ---
set "PROFIT_LOCK_ENABLED=true"
set "LOCK_MODE=usd"
set "LOCK_ACTIVATE_USD=1000"
set "LOCK_KEEP_USD=200"
set "LOCK_TRAIL_START_USD=1500"
set "LOCK_TRAIL_KEEP_PCT=0.5"
set "SL_MOVE_MIN_BUFFER_TICKS=4"

REM --- REVAMP exits (same as MNQ revamp bat) ---
set "REVAMP_EXITS=1"
set "REVAMP_TRAIL_ATR_MULT=4.0"
set "REVAMP_BREAKOUT_TARGET_R=2.0"
set "REVAMP_MR_TARGET=bb_far"

REM --- Session (same as MNQ revamp bat: Globex/overnight entries on) ---
set "OVERNIGHT_TRADING=true"
set "SESSION_MODE=extended"
set "SCALP_SESSIONS=off"
set "REVAMP_RTH_ONLY=0"

REM --- Size: 1 NQ contract, 1 trade at a time (1-4 confidence sizing OFF) ---
set "BASE_CONTRACTS=1"
set "MAX_CONTRACTS=1"
set "SIZE_WIN_STREAK_MIN=0"
set "SIZE_REQUIRE_DAY_GREEN=0"
set "SIZE_CONF_2=9.99"
set "SIZE_CONF_3=9.99"
set "SIZE_CONF_4=9.99"
set "MAX_CONCURRENT_TRADES=1"
set "EXTRA_TRADE_MIN_CONF=9.99"
set "MAX_TRADES_PER_DAY=0"

REM --- Daily loss: as configured (1%% of INITIAL_BALANCE 50000 = $500) ---
set "DAILY_LOSS_LIMIT_PERCENT=1.0"

REM Live mode (do not inherit paper)
set "TRADING_MODE=live"
set "FALLBACK_ENTRIES_ENABLED=false"

title bott-trader NQ LIVE REVAMP (Lucid)

echo Symbol: NQ (contract override %RITHMIC_CONTRACT_OVERRIDES%), 1 contract max, 1 trade at a time
echo Lock: +$%LOCK_ACTIVATE_USD% -^> +$%LOCK_KEEP_USD%, trail from +$%LOCK_TRAIL_START_USD% keep %LOCK_TRAIL_KEEP_PCT% of peak
echo Daily loss limit: 1%% of balance ($500) - blocks new entries only; one NQ stop can exceed it.
echo REVAMP_TRAIL_ATR_MULT=%REVAMP_TRAIL_ATR_MULT% BREAKOUT_TARGET_R=%REVAMP_BREAKOUT_TARGET_R% MR_TARGET=%REVAMP_MR_TARGET%
echo.
echo Starting LIVE NQ revamp (Ctrl+C to stop)...
"%~dp0venv\Scripts\python.exe" -u ".\start_live_rithmic.py" --symbols NQ --yes
echo.
echo Bot exited with code %ERRORLEVEL%.
pause
