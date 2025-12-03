# Complete Offline Installation - What to Zip

This guide lists ALL files and directories that need to be downloaded and zipped for fully offline installation.

## Required Components (Must Include)

### 1. Whisper Models (Speech-to-Text) - REQUIRED

**Location**: `./models/` (in project directory)

**Path**: 
- Full path: `/Users/debarunlahiri/Development/PythonProjects/stt-backend/models`
- Relative to project: `./models/`

**Size**: ~2.9 GB

**Contents**:
```
models/
└── models--Systran--faster-whisper-large-v3/
    └── snapshots/
        └── edaa852ec7e145841d8ffdb056a99866b5f0a478/
            ├── model.bin          (REQUIRED)
            ├── config.json        (REQUIRED)
            ├── tokenizer.json     (REQUIRED)
            ├── vocabulary.json    (REQUIRED)
            └── preprocessor_config.json
```

**Required Files**:
- `model.bin` - The actual Whisper model
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer for text processing
- `vocabulary.json` - Vocabulary file

**Download Source**: Pre-downloaded from HuggingFace Systran/faster-whisper-large-v3

---

### 2. Argos Translate Packages (Translation) - REQUIRED

**Location**: `~/.local/share/argos-translate/packages/`

**Platform-specific paths**:
- **macOS**: `/Users/username/.local/share/argos-translate/packages/`
- **Linux**: `/home/username/.local/share/argos-translate/packages/`
- **Windows**: `C:\Users\username\.local\share\argos-translate\packages\`

**Size**: ~481 MB

**Contents** (6 packages):
```
packages/
├── en_hi/          # English to Hindi
├── hi_en/          # Hindi to English
├── en_ko/          # English to Korean
├── ko_en/          # Korean to English
├── hi_ko/          # Hindi to Korean
└── ko_hi/          # Korean to Hindi
```

**Each package directory contains**:
- Model files (`.pt` or `.onnx` files)
- Configuration files
- Vocabulary files

**Download Source**: GitHub releases - https://github.com/argosopentech/argos-translate/releases

**Download Links**:
- `translate-en_hi-1_8.argosmodel`
- `translate-hi_en-1_8.argosmodel`
- `translate-en_ko-1_10.argosmodel`
- `translate-ko_en-1_10.argosmodel`
- `translate-hi_ko-1_10.argosmodel`
- `translate-ko_hi-1_10.argosmodel`

---

### 3. Hugging Face Cache (May be Used) - RECOMMENDED

**Location**: `~/.cache/huggingface/`

**Platform-specific paths**:
- **macOS**: `/Users/username/.cache/huggingface/`
- **Linux**: `/home/username/.cache/huggingface/`
- **Windows**: `C:\Users\username\.cache\huggingface\`

**Size**: ~5.9 GB (on your system)

**Purpose**: Contains cached models and tokenizers that faster-whisper might reference

**Note**: This is large, but may be needed if the application references HuggingFace cache

---

## Optional Components (Can Include but Not Required)

### 4. Stanza Models (Optional)

**Location**: `~/stanza_resources/`

**Platform-specific paths**:
- **macOS**: `/Users/username/stanza_resources/`
- **Linux**: `/home/username/stanza_resources/`
- **Windows**: `C:\Users\username\stanza_resources\`

**Size**: ~3.1 MB

**Status**: OPTIONAL - Not required for translation to work

**Contents**:
- English, Hindi, Korean tokenization models
- Used by Argos Translate but not critical

---

### 5. spaCy Models (Not Present) - OPTIONAL

**Location**: `~/.local/share/spacy/` or `~/.local/lib/python3.x/site-packages/`

**Status**: NOT REQUIRED - spaCy models don't need to be installed

**Note**: Argos Translate works without spaCy language models

---

## Complete Zip File Structure

Create a zip file with this structure:

```
offline-installation-package.zip
├── whisper-models/
│   └── models--Systran--faster-whisper-large-v3/
│       └── snapshots/
│           └── edaa852ec7e145841d8ffdb056a99866b5f0a478/
│               ├── model.bin
│               ├── config.json
│               ├── tokenizer.json
│               ├── vocabulary.json
│               └── preprocessor_config.json
│
├── translation-packages/
│   ├── en_hi/
│   ├── hi_en/
│   ├── en_ko/
│   ├── ko_en/
│   ├── hi_ko/
│   └── ko_hi/
│
├── stanza-models/  (OPTIONAL)
│   ├── en/
│   ├── hi/
│   ├── ko/
│   └── resources.json
│
└── huggingface-cache/  (OPTIONAL - Large!)
    └── [cache contents]
```

---

## Step-by-Step: Creating the Zip File

### On a Machine with Internet Access

#### Step 1: Download Whisper Models

The models should already be in `./models/` directory. If not:

```bash
# Models are already downloaded in your project
# Just copy the models directory
```

#### Step 2: Download Translation Packages

See `DOWNLOAD_TRANSLATION_MODELS.md` for detailed instructions.

Quick version:
```bash
# Use the Python script or direct download from GitHub
# Files will be in translation_packages/ directory
```

#### Step 3: Create Zip File

**Option A: Create separate zips (Recommended)**

```bash
# Zip Whisper models (2.9 GB)
cd /Users/debarunlahiri/Development/PythonProjects/stt-backend
tar -czf whisper-models.tar.gz models/models--Systran--faster-whisper-large-v3/

# Zip Translation packages (481 MB)
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/

# Zip Stanza models (Optional, 3.1 MB)
tar -czf stanza-models.tar.gz ~/stanza_resources/

# Zip HuggingFace cache (Optional, 5.9 GB - Large!)
tar -czf huggingface-cache.tar.gz ~/.cache/huggingface/
```

**Option B: Create single zip**

```bash
# Create directory structure
mkdir offline-package
cp -r models/models--Systran--faster-whisper-large-v3 offline-package/whisper-models/
cp -r ~/.local/share/argos-translate/packages offline-package/translation-packages/
cp -r ~/stanza_resources offline-package/stanza-models/  # Optional

# Create zip
tar -czf offline-installation.tar.gz offline-package/
```

---

## Installation on Offline Server

### Step 1: Extract Whisper Models

```bash
# Extract to project directory
tar -xzf whisper-models.tar.gz -C ./models/
```

### Step 2: Extract Translation Packages

**macOS/Linux**:
```bash
# Create directory
mkdir -p ~/.local/share/argos-translate/

# Extract packages
tar -xzf translation-packages.tar.gz -C ~/.local/share/argos-translate/
```

**Windows**:
```powershell
# Extract to
# C:\Users\username\.local\share\argos-translate\packages\
```

### Step 3: Extract Stanza Models (Optional)

```bash
# Extract to home directory
tar -xzf stanza-models.tar.gz -C ~/
```

### Step 4: Extract HuggingFace Cache (Optional)

```bash
# Create cache directory
mkdir -p ~/.cache/

# Extract cache
tar -xzf huggingface-cache.tar.gz -C ~/.cache/
```

---

## File Sizes Summary

| Component | Size | Required | Location |
|-----------|------|----------|----------|
| Whisper Models | ~2.9 GB | YES | `./models/` |
| Translation Packages | ~481 MB | YES | `~/.local/share/argos-translate/packages/` |
| HuggingFace Cache | ~5.9 GB | Recommended | `~/.cache/huggingface/` |
| Stanza Models | ~3.1 MB | NO | `~/stanza_resources/` |
| spaCy Models | N/A | NO | Not needed |

**Total Required Size**: ~3.4 GB (Whisper + Translation)

**Total with Optional**: ~9.4 GB (if including HuggingFace cache)

---

## Quick Reference: What to Include

### Minimum Required (3.4 GB):
1. ✅ Whisper models directory
2. ✅ Argos Translate packages directory

### Recommended (9.4 GB):
1. ✅ Whisper models directory
2. ✅ Argos Translate packages directory
3. ✅ HuggingFace cache directory

### Optional (3.1 MB):
1. ⚠️ Stanza models (not required but helps)

---

## Verification Script

After extracting, verify installation:

```bash
# Check Whisper models
ls -la models/models--Systran--faster-whisper-large-v3/snapshots/*/model.bin

# Check Translation packages
ls -la ~/.local/share/argos-translate/packages/

# Should see: en_hi, hi_en, en_ko, ko_en, hi_ko, ko_hi
```

---

## Platform-Specific Notes

### macOS
All paths use `~` which expands to `/Users/username/`

### Linux
All paths use `~` which expands to `/home/username/`

### Windows
Use Windows-style paths:
- `C:\Users\username\.local\share\argos-translate\packages\`
- `C:\Users\username\stanza_resources\`
- `C:\Users\username\.cache\huggingface\`

---

## Download Locations Summary

| Component | Download Source | Notes |
|-----------|----------------|-------|
| Whisper Models | Already in `./models/` | From Systran/faster-whisper-large-v3 |
| Translation Packages | GitHub: argosopentech/argos-translate/releases | See DOWNLOAD_TRANSLATION_MODELS.md |
| Stanza Models | Automatically (if needed) | Optional - not required |
| HuggingFace Cache | Already in `~/.cache/huggingface/` | May be referenced |

---

## Complete Installation Checklist

- [ ] Whisper models extracted to `./models/`
- [ ] Translation packages extracted to `~/.local/share/argos-translate/packages/`
- [ ] (Optional) Stanza models extracted to `~/stanza_resources/`
- [ ] (Optional) HuggingFace cache extracted to `~/.cache/huggingface/`
- [ ] Verify all required files exist
- [ ] Restart server
- [ ] Test transcription and translation

---

For detailed download instructions, see:
- `DOWNLOAD_TRANSLATION_MODELS.md` - Translation packages
- `TRANSLATION_PACKAGES_LOCATION.md` - Package management
- `STANZA_STORAGE_LOCATION.md` - Stanza (optional)

