@echo off
REM Create Offline Installation Zip Package (Windows)
REM This script creates zip files of all required components for offline installation

echo ==========================================
echo Creating Offline Installation Package
echo ==========================================
echo.

set SCRIPT_DIR=%~dp0
set OUTPUT_DIR=%SCRIPT_DIR%offline-package
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
cd /d "%SCRIPT_DIR%"

echo Step 1: Collecting Whisper Models...
set WHISPER_MODEL_PATH=.\models\models--Systran--faster-whisper-large-v3
if not exist "%WHISPER_MODEL_PATH%" (
    echo ERROR: Whisper models not found at %WHISPER_MODEL_PATH%
    echo Please ensure models are downloaded first.
    exit /b 1
)

if not exist "%OUTPUT_DIR%\whisper-models" mkdir "%OUTPUT_DIR%\whisper-models"
echo Copying Whisper models...
xcopy /E /I /Y "%WHISPER_MODEL_PATH%" "%OUTPUT_DIR%\whisper-models\models--Systran--faster-whisper-large-v3\"
echo [OK] Whisper models copied
echo.

echo Step 2: Collecting Translation Packages...
set TRANSLATION_PACKAGES=%USERPROFILE%\.local\share\argos-translate\packages
if not exist "%TRANSLATION_PACKAGES%" (
    echo WARNING: Translation packages not found at %TRANSLATION_PACKAGES%
    echo Translation will not work until packages are installed.
) else (
    if not exist "%OUTPUT_DIR%\translation-packages" mkdir "%OUTPUT_DIR%\translation-packages"
    echo Copying translation packages...
    xcopy /E /I /Y "%TRANSLATION_PACKAGES%\*" "%OUTPUT_DIR%\translation-packages\"
    echo [OK] Translation packages copied
)
echo.

echo Step 3: Collecting NLLB-200 Translation Models...
set HF_CACHE=%USERPROFILE%\.cache\huggingface
set NLLB_MODEL_PATH=%HF_CACHE%\models--facebook--nllb-200-distilled-600M
if exist "%NLLB_MODEL_PATH%" (
    if not exist "%OUTPUT_DIR%\nllb-models" mkdir "%OUTPUT_DIR%\nllb-models"
    echo Copying NLLB-200 models...
    xcopy /E /I /Y "%NLLB_MODEL_PATH%" "%OUTPUT_DIR%\nllb-models\models--facebook--nllb-200-distilled-600M\"
    echo [OK] NLLB-200 models copied
) else (
    echo WARNING: NLLB-200 models not found in cache
    echo Note: Download models with: python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
)
echo.

echo Step 4: Collecting Stanza Models (Optional)...
set STANZA_MODELS=%USERPROFILE%\stanza_resources
if exist "%STANZA_MODELS%" (
    if not exist "%OUTPUT_DIR%\stanza-models" mkdir "%OUTPUT_DIR%\stanza-models"
    echo Copying Stanza models...
    xcopy /E /I /Y "%STANZA_MODELS%\*" "%OUTPUT_DIR%\stanza-models\"
    echo [OK] Stanza models copied
) else (
    echo Stanza models not found (optional - not required)
)
echo.

echo Step 5: Creating Archive Files...
cd /d "%OUTPUT_DIR%"

echo Creating whisper-models.zip...
powershell Compress-Archive -Path whisper-models -DestinationPath "%SCRIPT_DIR%\whisper-models-%TIMESTAMP%.zip" -Force
echo [OK] Created whisper-models-%TIMESTAMP%.zip

if exist "translation-packages" (
    echo Creating translation-packages.zip...
    powershell Compress-Archive -Path translation-packages -DestinationPath "%SCRIPT_DIR%\translation-packages-%TIMESTAMP%.zip" -Force
    echo [OK] Created translation-packages-%TIMESTAMP%.zip
)

if exist "nllb-models" (
    echo Creating nllb-models.zip...
    powershell Compress-Archive -Path nllb-models -DestinationPath "%SCRIPT_DIR%\nllb-models-%TIMESTAMP%.zip" -Force
    echo [OK] Created nllb-models-%TIMESTAMP%.zip
)

if exist "stanza-models" (
    echo Creating stanza-models.zip...
    powershell Compress-Archive -Path stanza-models -DestinationPath "%SCRIPT_DIR%\stanza-models-%TIMESTAMP%.zip" -Force
    echo [OK] Created stanza-models-%TIMESTAMP%.zip
)

echo.
echo ==========================================
echo Package Creation Complete!
echo ==========================================
echo.
echo Created files in: %SCRIPT_DIR%
echo.
echo Next steps:
echo 1. Transfer these .zip files to your offline server
echo 2. Extract them following instructions in OFFLINE_INSTALLATION_ZIP_GUIDE.md
echo.
echo Installation instructions: See OFFLINE_INSTALLATION_ZIP_GUIDE.md
pause

