@echo off
title MB Cloning Voice
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0launch_mb_cloning_voice.py"
if errorlevel 1 pause
