@echo off
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\ecg_stm32_viz.py" --manifest "%~dp0outputs\distillation\distilled_export\serial_manifest.json" %*
