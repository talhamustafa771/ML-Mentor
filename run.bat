@echo off
REM ---------------------------------------------------------------------
REM  Start ML Mentor on this laptop.
REM  Double-click this file. A browser tab opens by itself.
REM ---------------------------------------------------------------------
cd /d "%~dp0"

echo Starting ML Mentor...
echo.

python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Streamlit did not start. The usual cause is that the packages are not
  echo installed yet. Run install.bat once, then try this again.
  echo.
)
pause
