@echo off
REM Run Script for STT Backend (Windows)
REM Author: Debarun Lahiri

setlocal enabledelayedexpansion

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup first: setup.bat
    exit /b 1
)

REM Use venv Python directly (avoids PATH issues)
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"

REM Check if venv Python exists
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python not found in virtual environment.
    echo Please run setup first: setup.bat
    exit /b 1
)

REM Verify Python and packages
echo Verifying Python environment...
"%VENV_PYTHON%" -c "import pydantic_settings" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pydantic_settings not found.
    echo Please run: setup.bat
    exit /b 1
)

echo [OK] Starting STT Backend server...
echo.

REM Run the application using venv Python directly
"%VENV_PYTHON%" run.py

endlocal

