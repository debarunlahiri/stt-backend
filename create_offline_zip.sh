#!/bin/bash
# Create Offline Installation Zip Package
# This script creates zip files of all required components for offline installation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Creating Offline Installation Package"
echo "=========================================="
echo ""

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${SCRIPT_DIR}/offline-package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"

echo -e "${GREEN}Step 1: Collecting Whisper Models...${NC}"

# Check if Whisper models exist
WHISPER_MODEL_PATH="./models/models--Systran--faster-whisper-large-v3"
if [ ! -d "${WHISPER_MODEL_PATH}" ]; then
    echo -e "${RED}ERROR: Whisper models not found at ${WHISPER_MODEL_PATH}${NC}"
    echo "Please ensure models are downloaded first."
    exit 1
fi

# Copy Whisper models
mkdir -p "${OUTPUT_DIR}/whisper-models"
echo "Copying Whisper models..."
cp -r "${WHISPER_MODEL_PATH}" "${OUTPUT_DIR}/whisper-models/"
WHISPER_SIZE=$(du -sh "${OUTPUT_DIR}/whisper-models" | cut -f1)
echo -e "${GREEN}✓ Whisper models copied (${WHISPER_SIZE})${NC}"
echo ""

echo -e "${GREEN}Step 2: Collecting Translation Packages...${NC}"

# Check if translation packages exist
TRANSLATION_PACKAGES="${HOME}/.local/share/argos-translate/packages"
if [ ! -d "${TRANSLATION_PACKAGES}" ]; then
    echo -e "${YELLOW}WARNING: Translation packages not found at ${TRANSLATION_PACKAGES}${NC}"
    echo "Translation will not work until packages are installed."
    echo "See DOWNLOAD_TRANSLATION_MODELS.md for instructions."
else
    # Copy translation packages
    mkdir -p "${OUTPUT_DIR}/translation-packages"
    echo "Copying translation packages..."
    cp -r "${TRANSLATION_PACKAGES}"/* "${OUTPUT_DIR}/translation-packages/" 2>/dev/null || true
    TRANSLATION_SIZE=$(du -sh "${OUTPUT_DIR}/translation-packages" | cut -f1)
    echo -e "${GREEN}✓ Translation packages copied (${TRANSLATION_SIZE})${NC}"
fi
echo ""

echo -e "${GREEN}Step 3: Collecting NLLB-200 Translation Models...${NC}"

# Check if NLLB-200 models exist in HuggingFace cache
HF_CACHE="${HOME}/.cache/huggingface"
NLLB_MODEL_PATH="${HF_CACHE}/models--facebook--nllb-200-distilled-600M"
if [ -d "${NLLB_MODEL_PATH}" ]; then
    mkdir -p "${OUTPUT_DIR}/nllb-models"
    echo "Copying NLLB-200 models..."
    cp -r "${NLLB_MODEL_PATH}" "${OUTPUT_DIR}/nllb-models/" 2>/dev/null || true
    NLLB_SIZE=$(du -sh "${OUTPUT_DIR}/nllb-models" | cut -f1)
    echo -e "${GREEN}✓ NLLB-200 models copied (${NLLB_SIZE})${NC}"
else
    echo -e "${YELLOW}NLLB-200 models not found in cache${NC}"
    echo -e "${YELLOW}Note: Download models with: python -c \"from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')\"${NC}"
fi
echo ""

echo -e "${GREEN}Step 4: Collecting Stanza Models (Optional)...${NC}"

# Check if Stanza models exist
STANZA_MODELS="${HOME}/stanza_resources"
if [ -d "${STANZA_MODELS}" ]; then
    mkdir -p "${OUTPUT_DIR}/stanza-models"
    echo "Copying Stanza models..."
    cp -r "${STANZA_MODELS}"/* "${OUTPUT_DIR}/stanza-models/" 2>/dev/null || true
    STANZA_SIZE=$(du -sh "${OUTPUT_DIR}/stanza-models" | cut -f1)
    echo -e "${GREEN}✓ Stanza models copied (${STANZA_SIZE})${NC}"
else
    echo -e "${YELLOW}Stanza models not found (optional - not required)${NC}"
fi
echo ""

echo -e "${GREEN}Step 5: Collecting HuggingFace Cache (Optional)...${NC}"

# Check if HuggingFace cache exists
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "${HF_CACHE}" ]; then
    HF_SIZE=$(du -sh "${HF_CACHE}" | cut -f1)
    echo -e "${YELLOW}HuggingFace cache found (${HF_SIZE})${NC}"
    echo -e "${YELLOW}Note: This is large but may be needed for some functionality${NC}"
    echo -e "${YELLOW}Skipping by default. Manually copy if needed: ${HF_CACHE}${NC}"
    echo ""
    read -p "Include HuggingFace cache? This is a large file. (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "${OUTPUT_DIR}/huggingface-cache"
        echo "Copying HuggingFace cache (${HF_SIZE})... This may take a while..."
        cp -r "${HF_CACHE}"/* "${OUTPUT_DIR}/huggingface-cache/" 2>/dev/null || true
        echo -e "${GREEN}✓ HuggingFace cache copied (${HF_SIZE})${NC}"
    else
        echo -e "${YELLOW}Skipping HuggingFace cache${NC}"
        echo "You can manually copy if needed: ${HF_CACHE}"
    fi
else
    echo -e "${YELLOW}HuggingFace cache not found (optional)${NC}"
fi
echo ""

echo -e "${GREEN}Step 6: Creating Archive Files...${NC}"

# Create individual archives
cd "${OUTPUT_DIR}"

echo "Creating whisper-models.tar.gz..."
tar -czf "${SCRIPT_DIR}/whisper-models-${TIMESTAMP}.tar.gz" whisper-models/
echo -e "${GREEN}✓ Created whisper-models-${TIMESTAMP}.tar.gz${NC}"

if [ -d "translation-packages" ] && [ "$(ls -A translation-packages)" ]; then
    echo "Creating translation-packages.tar.gz..."
    tar -czf "${SCRIPT_DIR}/translation-packages-${TIMESTAMP}.tar.gz" translation-packages/
    echo -e "${GREEN}✓ Created translation-packages-${TIMESTAMP}.tar.gz${NC}"
fi

if [ -d "nllb-models" ] && [ "$(ls -A nllb-models)" ]; then
    echo "Creating nllb-models.tar.gz..."
    tar -czf "${SCRIPT_DIR}/nllb-models-${TIMESTAMP}.tar.gz" nllb-models/
    echo -e "${GREEN}✓ Created nllb-models-${TIMESTAMP}.tar.gz${NC}"
fi

if [ -d "stanza-models" ] && [ "$(ls -A stanza-models)" ]; then
    echo "Creating stanza-models.tar.gz..."
    tar -czf "${SCRIPT_DIR}/stanza-models-${TIMESTAMP}.tar.gz" stanza-models/
    echo -e "${GREEN}✓ Created stanza-models-${TIMESTAMP}.tar.gz${NC}"
fi

if [ -d "huggingface-cache" ] && [ "$(ls -A huggingface-cache)" ]; then
    echo "Creating huggingface-cache.tar.gz..."
    tar -czf "${SCRIPT_DIR}/huggingface-cache-${TIMESTAMP}.tar.gz" huggingface-cache/
    echo -e "${GREEN}✓ Created huggingface-cache-${TIMESTAMP}.tar.gz${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Package Creation Complete!${NC}"
echo "=========================================="
echo ""
echo "Created files in: ${SCRIPT_DIR}/"
echo ""
ls -lh "${SCRIPT_DIR}"/*.tar.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Total package size:"
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
echo "  ${TOTAL_SIZE}"
echo ""
echo "Next steps:"
echo "1. Transfer these .tar.gz files to your offline server"
echo "2. Extract them following instructions in OFFLINE_INSTALLATION_ZIP_GUIDE.md"
echo ""
echo "Installation instructions: See OFFLINE_INSTALLATION_ZIP_GUIDE.md"

