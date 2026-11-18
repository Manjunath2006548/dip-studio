@echo off
title DIP Studio - Histogram Equalization Workbench
cd /d "%~dp0"

echo.
echo  DIP Studio - ECE3401 CA-3
echo  -------------------------
echo  Installing dependencies (first run only)...
python -m pip install -q -r requirements.txt

echo.
echo  Starting on http://localhost:8000
echo  Press Ctrl+C to stop.
echo.
python app.py
pause
