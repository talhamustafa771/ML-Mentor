@echo off
REM ---------------------------------------------------------------------
REM  Start ML Mentor so your phone can reach it over the same Wi-Fi.
REM
REM  Do NOT type the 0.0.0.0 address into your phone. That is an
REM  instruction to this laptop, not an address a phone can reach.
REM  Use the IPv4 address printed below.
REM ---------------------------------------------------------------------
cd /d "%~dp0"

echo Your laptop's address on this network:
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:8501
echo.
echo Open one of those on your phone, with both on the same Wi-Fi.
echo Leave this window open while you study.
echo.

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
