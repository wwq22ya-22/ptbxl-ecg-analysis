@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_stm32cubemx_ascii.ps1"
if errorlevel 1 (
    echo.
    echo STM32CubeMX special launcher failed. See the message above.
    pause
)
endlocal

