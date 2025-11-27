#!/bin/bash
# Run Script for STT Backend
# Author: Debarun Lahiri

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    echo "Please run setup first: ./setup.sh"
    exit 1
fi

# Use venv Python directly (avoids shell alias issues)
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

# Check if venv Python exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}Error: Python not found in virtual environment.${NC}"
    echo "Please run setup first: ./setup.sh"
    exit 1
fi

# Verify Python and packages
echo -e "${YELLOW}Verifying Python environment...${NC}"
$VENV_PYTHON -c "import pydantic_settings" 2>/dev/null || {
    echo -e "${RED}Error: pydantic_settings not found.${NC}"
    echo "Please run: ./setup.sh"
    exit 1
}

echo -e "${GREEN}Starting STT Backend server...${NC}"
echo ""

# Run the application using venv Python directly
$VENV_PYTHON run.py
