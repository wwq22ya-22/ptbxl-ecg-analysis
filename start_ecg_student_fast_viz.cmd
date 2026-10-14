@echo off
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\ecg_stm32_fast_viz.py" %*
