@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\ecg_stm32_viz.py" %*
if errorlevel 1 pause
