@echo off
REM ============================================================
REM  AI Forex Trading Bot - Windows Setup Script
REM  Run this on your Windows PC where MetaTrader 5 is installed
REM ============================================================

echo.
echo ====================================
echo   AI Forex Trading Bot - Setup
echo ====================================
echo.

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.10+ from python.org
    echo         Make sure to check "Add Python to PATH" during install
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Try installing TensorFlow (optional - requires Python 3.10-3.12)
echo.
echo Installing TensorFlow (optional - for LSTM AI model)...
pip install tensorflow>=2.13.0 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] TensorFlow could not be installed.
    echo           This is OK! The bot will run with 7 AI models instead of 8.
    echo           To enable LSTM, install Python 3.12 from:
    echo           https://www.python.org/downloads/release/python-31210/
    echo.
) else (
    echo [OK] TensorFlow installed - all 8 AI models available
)

REM Install MetaTrader5 (Windows only)
echo.
echo Installing MetaTrader5 package...
pip install MetaTrader5

REM Install yfinance for data downloads
pip install yfinance

REM Check if .env exists
if not exist ".env" (
    echo.
    echo [SETUP] Creating .env from .env.example
    copy .env.example .env
    echo.
    echo *** IMPORTANT: Edit .env with your MT5 credentials! ***
    echo     Open .env in Notepad and fill in:
    echo       - MT5_ACCOUNT
    echo       - MT5_PASSWORD
    echo       - MT5_SERVER
    echo       - NEWSAPI_KEY
    echo.
    notepad .env
)

echo.
echo ====================================
echo   Setup Complete!
echo ====================================
echo.
echo Next steps:
echo   1. Make sure MetaTrader 5 is open and logged in
echo   2. Edit .env with your credentials (if not done)
echo   3. Train the AI model:
echo        python -m scripts.train_lstm
echo   4. Start paper trading:
echo        python -m scripts.run_bot
echo   5. When ready for live trading, change TRADING_MODE=live in .env
echo.
pause
