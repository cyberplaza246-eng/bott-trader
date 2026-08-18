@echo off
REM ============================================================
REM  Add AI Trading Bot to Windows Startup
REM  Run this ONCE to make the bot start automatically when
REM  your laptop boots up.
REM ============================================================

echo.
echo Creating startup shortcut...

set "SCRIPT_DIR=%~dp0"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\AI-Trading-Bot.lnk"
set "TARGET=%SCRIPT_DIR%start_mnq_forever.bat"

REM Create VBS script to make shortcut (Windows doesn't have a native way)
set "VBS=%TEMP%\create_shortcut.vbs"
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo Set oLink = oWS.CreateShortcut^("%SHORTCUT%"^)
echo oLink.TargetPath = "%TARGET%"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%"
echo oLink.Description = "BottTrader MNQ unattended"
echo oLink.WindowStyle = 7
echo oLink.Save
) > "%VBS%"
cscript //nologo "%VBS%"
del "%VBS%"

if exist "%SHORTCUT%" (
    echo.
    echo [OK] Startup shortcut created!
    echo     Location: %SHORTCUT%
    echo.
    echo The bot will now start automatically when Windows boots.
    echo To remove, delete the shortcut from:
    echo     %STARTUP%
    echo.
) else (
    echo.
    echo [ERROR] Failed to create shortcut.
    echo You can manually create one:
    echo   1. Press Win+R, type: shell:startup
    echo   2. Create a shortcut to: %TARGET%
    echo.
)

pause
