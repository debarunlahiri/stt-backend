@echo off
REM Setup Script for STT Backend (Windows)
REM Author: Debarun Lahiri

setlocal enabledelayedexpansion

echo ==========================================
echo STT Backend Setup and Run Script (Windows)
echo ==========================================
echo.

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Step 1: Check Python version
echo Step 1: Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11 or higher.
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found: Python %PYTHON_VERSION%
echo.

REM Step 2: Check if virtual environment exists
echo Step 2: Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM Step 3: Activate virtual environment
echo Step 3: Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [ERROR] Could not find virtual environment activation script.
    exit /b 1
)
echo.

REM Step 4: Upgrade pip
echo Step 4: Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing...
) else (
    echo [OK] pip upgraded.
)
echo.

REM Step 5: Install requirements
echo Step 5: Installing Python dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo [OK] All dependencies installed.
echo.

REM Step 6: Verify installation
echo Step 6: Verifying installation...
python -c "import pydantic_settings; print('OK pydantic_settings')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pydantic_settings not found
    exit /b 1
)

python -c "import fastapi; print('OK fastapi')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] fastapi not found
    exit /b 1
)

python -c "import faster_whisper; print('OK faster-whisper')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] faster-whisper not found
    exit /b 1
)

python -c "import argostranslate; print('OK argostranslate')" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] argostranslate not found (used as fallback)
)

python -c "import transformers; print('OK transformers (NLLB-200 support)')" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] transformers not found (NLLB-200 will not work, but Argos Translate fallback available)
)

python -c "import sentencepiece; print('OK sentencepiece (NLLB-200 support)')" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] sentencepiece not found (NLLB-200 will not work)
)
echo [OK] Core packages verified.
echo.

REM Step 7: Check FFmpeg
echo Step 7: Checking FFmpeg installation...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found. Audio processing will not work.
    echo Please install FFmpeg:
    echo   Download from: https://ffmpeg.org/download.html
    echo   Add to PATH after installation
) else (
    for /f "delims=" %%i in ('ffmpeg -version 2^>^&1 ^| findstr /i "ffmpeg version"') do set FFMPEG_VERSION=%%i
    echo [OK] Found: %FFMPEG_VERSION%
)
echo.

REM Step 8: Create necessary directories
echo Step 8: Creating necessary directories...
echo Creating models directory for Whisper model cache...
if not exist "models" mkdir models
echo Creating audio_recordings directory for saved audio files...
if not exist "audio_recordings" mkdir audio_recordings
echo [OK] Directories created:
echo   - models\ (for Whisper model cache)
echo   - audio_recordings\ (for saved audio files)
echo.

REM Step 9: Summary
echo ==========================================
echo [OK] Setup Complete!
echo ==========================================
echo.
echo To run the server, use one of these methods:
echo.
echo Method 1: Use the run script (recommended)
echo   run.bat
echo.
echo Method 2: Use this script to run
echo   setup.bat run
echo.
echo Method 3: Activate venv and run manually
echo   venv\Scripts\activate
echo   python run.py
echo.

REM Check if user wants to run immediately
if "%1"=="run" (
    echo Starting server...
    echo.
    python run.py
)

endlocal

