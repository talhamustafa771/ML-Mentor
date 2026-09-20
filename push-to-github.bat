@echo off
REM ---------------------------------------------------------------------
REM  Send this folder to GitHub, so Streamlit Cloud picks up the changes.
REM
REM  THIS FOLDER IS THE MASTER COPY. Edit here, never on the GitHub
REM  website: editing there creates a commit this folder does not have,
REM  and the next push is rejected until it is forced.
REM ---------------------------------------------------------------------
cd /d "%~dp0"

set REPO=https://github.com/muhammad-talha04/ml-mentor.git

REM A stale lock file makes "git add" fail silently and report that zero
REM files are ready to send. Clear it, but only when git is not running.
tasklist /fi "imagename eq git.exe" | find /i "git.exe" >nul
if errorlevel 1 (
  if exist ".git\index.lock" (
    echo Clearing a leftover lock file...
    del /f /q ".git\index.lock"
  )
) else (
  echo Git is already running. Close it and try again.
  pause
  exit /b 1
)

if not exist ".git" (
  git init
  git branch -M main
  git remote add origin %REPO%
)

git config user.name "muhammad-talha04"
git config user.email "princetalha772@gmail.com"

REM Refuse to send anything that looks like a secret.
findstr /s /i /m /c:"gsk_" /c:"nvapi-" *.py *.toml *.txt *.md 2>nul | findstr /v "secrets.toml.example" >nul
if not errorlevel 1 (
  echo.
  echo STOPPED: a file in this folder appears to contain an API key.
  echo Keys belong in Streamlit's Secrets, never in the code.
  echo.
  pause
  exit /b 1
)

git add -A
git status --short
echo.

git commit -m "Update ML Mentor"
git push --force origin main

echo.
echo Sent. Streamlit Cloud redeploys on its own within a minute or so.
pause
