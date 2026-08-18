@echo off

cd /d "%~dp0"

call venv\Scripts\activate.bat

REM Sim testing: use MNQ (10x smaller risk). For NQ paper: --symbols NQ
python start_live_mtf_scalping.py --symbols MNQ --paper

