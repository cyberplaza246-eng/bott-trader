@echo off
REM Close Lucid desktop. Do not start a second copy. Close Cursor is OK — leave this black window open.
cd /d "%~dp0"
start "MNQ overnight paper" cmd /k "%~dp0start_mnq_overnight_paper.bat"
