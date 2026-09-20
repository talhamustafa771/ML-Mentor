@echo off
REM ---------------------------------------------------------------------
REM  Install everything ML Mentor needs. Run this once.
REM ---------------------------------------------------------------------
cd /d "%~dp0"

echo Installing packages. This takes a few minutes the first time.
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Checking that it all works...
python tests\test_smoke.py
echo.
echo Done. Double-click run.bat to start.
pause
