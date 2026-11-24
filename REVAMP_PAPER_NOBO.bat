@echo off
title REVAMP PAPER (breakout OFF) - NO ORDERS
cd /d %~dp0
venv\Scripts\python.exe tools\revamp_paper_nobo.py %*
pause
