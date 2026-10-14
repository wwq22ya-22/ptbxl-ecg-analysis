@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\ecg_stm32_viz.py" --manifest "%~dp0outputs\cubeai\resnet\serial_manifest.json" %*
if errorlevel 1 pause
