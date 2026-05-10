@echo off
echo ============================================================
echo 🤖 AI Stock GPT - Starting Application
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed or not in PATH
    echo Please install npm with Node.js
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

REM Start the application
python start_app.py %*

if errorlevel 1 (
    echo.
    echo ❌ Failed to start AI Stock GPT
    pause
    exit /b 1
)

pause
