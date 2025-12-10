#!/bin/bash
# Setup and Run Script for STT Backend
# Author: Debarun Lahiri

set -e  # Exit on error

echo "=========================================="
echo "STT Backend Setup and Run Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check Python version
echo -e "${YELLOW}Step 1: Checking Python version...${NC}"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python 3.11+ is required. Found: $(python3 --version)${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Python 3.11+ not found. Please install Python 3.11 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"
echo ""

# Step 2: Check if virtual environment exists
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi
echo ""

# Step 3: Activate virtual environment
echo -e "${YELLOW}Step 3: Activating virtual environment...${NC}"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}Virtual environment activated.${NC}"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo -e "${GREEN}Virtual environment activated (Windows).${NC}"
else
    echo -e "${RED}Error: Could not find virtual environment activation script.${NC}"
    exit 1
fi
echo ""

# Step 4: Upgrade pip
echo -e "${YELLOW}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}pip upgraded.${NC}"
echo ""

# Step 5: Install requirements
echo -e "${YELLOW}Step 5: Installing Python dependencies...${NC}"
echo "This may take several minutes..."
pip install -r requirements.txt
echo -e "${GREEN}All dependencies installed.${NC}"
echo ""

# Step 6: Verify installation
echo -e "${YELLOW}Step 6: Verifying installation...${NC}"
python -c "import pydantic_settings; print('✓ pydantic_settings')" || { echo -e "${RED}Error: pydantic_settings not found${NC}"; exit 1; }
python -c "import fastapi; print('✓ fastapi')" || { echo -e "${RED}Error: fastapi not found${NC}"; exit 1; }
python -c "import faster_whisper; print('✓ faster-whisper')" || { echo -e "${RED}Error: faster-whisper not found${NC}"; exit 1; }
python -c "import argostranslate; print('✓ argostranslate')" || { echo -e "${YELLOW}Warning: argostranslate not found (used as fallback)${NC}"; }
python -c "import transformers; print('✓ transformers (NLLB-200 support)')" || { echo -e "${YELLOW}Warning: transformers not found (NLLB-200 will not work, but Argos Translate fallback available)${NC}"; }
python -c "import sentencepiece; print('✓ sentencepiece (NLLB-200 support)')" || { echo -e "${YELLOW}Warning: sentencepiece not found (NLLB-200 will not work)${NC}"; }
echo -e "${GREEN}Core packages verified.${NC}"
echo ""

# Step 7: Check FFmpeg
echo -e "${YELLOW}Step 7: Checking FFmpeg installation...${NC}"
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n 1)
    echo -e "${GREEN}Found: $FFMPEG_VERSION${NC}"
else
    echo -e "${RED}Warning: FFmpeg not found. Audio processing will not work.${NC}"
    echo "Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
fi
echo ""

# Step 8: Create necessary directories
echo -e "${YELLOW}Step 8: Creating necessary directories...${NC}"
echo "Creating models directory for Whisper model cache..."
mkdir -p models
echo "Creating audio_recordings directory for saved audio files..."
mkdir -p audio_recordings
echo -e "${GREEN}Directories created:${NC}"
echo "  - models/ (for Whisper model cache)"
echo "  - audio_recordings/ (for saved audio files)"
echo ""

# Step 9: Summary
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To run the server, use one of these methods:"
echo ""
echo "Method 1: Use the run script (recommended)"
echo "  ./run.sh              # macOS/Linux"
echo "  run.bat               # Windows"
echo ""
echo "Method 2: Use this script to run"
echo "  ./setup.sh run        # macOS/Linux"
echo "  setup.bat run         # Windows"
echo ""
echo "Method 3: Activate venv and run manually"
echo "  source venv/bin/activate  # macOS/Linux"
echo "  venv\\Scripts\\activate     # Windows"
echo "  python run.py"
echo ""

# Check if user wants to run immediately
if [ "$1" == "run" ]; then
    echo -e "${YELLOW}Starting server...${NC}"
    echo ""
    python run.py
fi

