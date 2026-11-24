@echo off
cd /d "%~dp0"
echo ============================================
echo   Validate MNQ hybrid edge
echo   Ablation + walk-forward + paper journal
echo   No Rithmic / no live orders
echo ============================================
echo.
call venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
python scripts/validate_hybrid_edge.py --symbol MNQ
echo.
pause
