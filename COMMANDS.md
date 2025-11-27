# STT Backend - Setup and Run Commands

This file contains all the commands needed to set up and run the STT Backend project.

## Quick Start (Recommended)

### macOS/Linux

#### Option 1: Automated Setup and Run
```bash
# Make scripts executable (first time only)
chmod +x setup.sh run.sh

# Run setup (installs all dependencies)
./setup.sh

# Run the server
./run.sh
```

#### Option 2: Setup and Run in One Command
```bash
chmod +x setup.sh
./setup.sh run
```

#### Option 3: Quick Run (if already set up)
```bash
./run.sh
```

### Windows

#### Option 1: Automated Setup and Run
```cmd
REM Run setup (installs all dependencies)
setup.bat

REM Run the server
run.bat
```

#### Option 2: Setup and Run in One Command
```cmd
setup.bat run
```

#### Option 3: Quick Run (if already set up)
```cmd
run.bat
```

## Manual Setup Steps

### Step 1: Create Virtual Environment
```bash
# Using Python 3.11 (recommended)
python3.11 -m venv venv

# OR if Python 3.11+ is your default python3
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
venv\Scripts\activate
```

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 4: Install All Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation
```bash
# Check if all critical packages are installed
python -c "import pydantic_settings; print('✓ pydantic_settings')"
python -c "import fastapi; print('✓ fastapi')"
python -c "import faster_whisper; print('✓ faster-whisper')"
python -c "import argostranslate; print('✓ argostranslate')"
python -c "import torch; print('✓ torch')"
```

### Step 6: Create Required Directories

**On macOS/Linux:**
```bash
mkdir -p models
mkdir -p audio_recordings
```

**On Windows:**
```cmd
if not exist models mkdir models
if not exist audio_recordings mkdir audio_recordings
```

### Step 7: Run the Server
```bash
python run.py
```

## Troubleshooting Commands

### If Python Can't Find Packages

**Check which Python is being used:**
```bash
which python
python --version
```

**Check Python path:**
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

**Reinstall packages in correct environment:**
```bash
# Make sure venv is activated first!
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Then reinstall
pip install --force-reinstall -r requirements.txt
```

### If Virtual Environment Issues

**Remove and recreate venv:**
```bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Check Package Installation
```bash
# List all installed packages
pip list

# Check specific package
pip show pydantic-settings

# Check if package is importable
python -c "import pydantic_settings; print(pydantic_settings.__version__)"
```

## System Dependencies

### Check FFmpeg Installation
```bash
ffmpeg -version
```

### Install FFmpeg (if missing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

## Environment Variables (Optional)

### Create .env file (optional)
```bash
cp .env.example .env  # If .env.example exists
# OR create manually
```

### Set Proxy (if needed for office networks)
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

## Running the Server

### Recommended: Use Run Script

**On macOS/Linux:**
```bash
./run.sh
```

**On Windows:**
```cmd
run.bat
```

### Alternative: Standard Run

**On macOS/Linux:**
```bash
source venv/bin/activate
python run.py
```

**On Windows:**
```cmd
venv\Scripts\activate
python run.py
```

### Alternative: Use Venv Python Directly
```bash
./venv/bin/python run.py
```

### Run with Custom Port
```bash
source venv/bin/activate
PORT=8080 python run.py
```

### Run in Background
```bash
source venv/bin/activate
nohup python run.py > server.log 2>&1 &
```

## Testing the Server

### Check Health Endpoint
```bash
curl http://localhost:8000/health
```

### Test Transcription (after server is running)
```bash
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "audio_file=@sample.wav" \
  -F "language=auto"
```

## Model Download Commands

### Check Model Cache
```bash
ls -lh models/
```

### Clear Model Cache (force re-download)
```bash
rm -rf models/
```

### Manual Model Download (if automatic fails)
```bash
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir ./models/faster-whisper-large-v3
```

## Translation Package Commands

### Check Translation Packages
```bash
python -c "import argostranslate.package; print(argostranslate.package.get_installed_packages())"
```

### Update Translation Package Index
```bash
python -c "import argostranslate.package; argostranslate.package.update_package_index()"
```

## Development Commands

### Install Development Dependencies
```bash
pip install pytest black flake8 mypy
```

### Run Tests (if available)
```bash
pytest
```

### Format Code
```bash
black app/
```

### Check Code Style
```bash
flake8 app/
```

## Docker Commands

### Build Docker Image
```bash
docker build -t stt-backend .
```

### Run with Docker Compose
```bash
docker-compose up --build
```

### Run in Background
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Docker
```bash
docker-compose down
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError: No module named 'pydantic_settings'

**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Verify you're using the right Python
which python  # Should point to venv/bin/python

# Reinstall the package
pip install --force-reinstall pydantic-settings
```

### Issue: Python version mismatch

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.11+

# If wrong version, recreate venv with correct Python
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: FFmpeg not found

**Solution:**
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
# OR
sudo apt-get install ffmpeg  # Linux

# Verify installation
ffmpeg -version
```

### Issue: Model download fails

**Solution:**
```bash
# Set proxy if behind corporate firewall
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Or download manually
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir ./models/
```

## Notes

- Always activate the virtual environment before running commands
- Use `python3.11` explicitly if you have multiple Python versions
- Check `which python` to verify you're using the venv Python
- Models are cached in `./models/` directory after first download
- Translation packages are installed in Python site-packages

