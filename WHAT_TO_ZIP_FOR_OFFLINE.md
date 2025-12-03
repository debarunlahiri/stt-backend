# What to Zip for Offline Installation - Complete List

## Quick Answer

You need to zip these **2 REQUIRED** components:

1. **Whisper Models** - `./models/models--Systran--faster-whisper-large-v3/` (~2.9 GB)
2. **Translation Packages** - `~/.local/share/argos-translate/packages/` (~481 MB)

**Total Required: ~3.4 GB**

---

## Complete List of What Requires Download

### ✅ REQUIRED (Must Zip)

#### 1. Whisper Models (Speech-to-Text)

**Location**: 
- Project directory: `./models/models--Systran--faster-whisper-large-v3/`
- Full path: `/Users/debarunlahiri/Development/PythonProjects/stt-backend/models/models--Systran--faster-whisper-large-v3/`

**Size**: ~2.9 GB

**What's inside**:
```
models--Systran--faster-whisper-large-v3/
└── snapshots/
    └── edaa852ec7e145841d8ffdb056a99866b5f0a478/
        ├── model.bin              (LARGE - the actual model)
        ├── config.json            (REQUIRED)
        ├── tokenizer.json         (REQUIRED)
        ├── vocabulary.json        (REQUIRED)
        └── preprocessor_config.json
```

**Zip Command**:
```bash
tar -czf whisper-models.tar.gz models/models--Systran--faster-whisper-large-v3/
```

---

#### 2. Argos Translate Packages (Translation)

**Location**:
- macOS: `/Users/debarunlahiri/.local/share/argos-translate/packages/`
- Linux: `/home/username/.local/share/argos-translate/packages/`
- Windows: `C:\Users\username\.local\share\argos-translate\packages\`

**Size**: ~481 MB

**What's inside**:
```
packages/
├── en_hi/          (English → Hindi)
├── hi_en/          (Hindi → English)
├── en_ko/          (English → Korean)
├── ko_en/          (Korean → English)
├── hi_ko/          (Hindi → Korean)
└── ko_hi/          (Korean → Hindi)
```

**Zip Command**:
```bash
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/
```

---

### ⚠️ OPTIONAL (Can Include)

#### 3. HuggingFace Cache (May be Needed)

**Location**: `~/.cache/huggingface/`

**Size**: ~5.9 GB (on your system)

**Status**: Large but may be referenced by faster-whisper

**Zip Command**:
```bash
tar -czf huggingface-cache.tar.gz ~/.cache/huggingface/
```

**Note**: This is large. Only include if you have space and want to be safe.

---

#### 4. Stanza Models (Optional - Not Required)

**Location**: `~/stanza_resources/`

**Size**: ~3.1 MB

**Status**: Optional - Argos Translate works without it

**Zip Command**:
```bash
tar -czf stanza-models.tar.gz ~/stanza_resources/
```

---

## Easy Way: Use the Script

I've created automated scripts to zip everything:

**macOS/Linux:**
```bash
./create_offline_zip.sh
```

**Windows:**
```cmd
create_offline_zip.bat
```

This will create separate zip files for each component.

---

## Manual Zip Commands

### Minimum Required (3.4 GB):

```bash
# Whisper models
tar -czf whisper-models.tar.gz models/models--Systran--faster-whisper-large-v3/

# Translation packages
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/
```

### Complete Package (9.4 GB):

```bash
# Required
tar -czf whisper-models.tar.gz models/models--Systran--faster-whisper-large-v3/
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/

# Optional
tar -czf stanza-models.tar.gz ~/stanza_resources/
tar -czf huggingface-cache.tar.gz ~/.cache/huggingface/
```

---

## File Size Summary

Based on your system:

| Component | Location | Size | Required? |
|-----------|----------|------|-----------|
| Whisper Models | `./models/models--Systran--faster-whisper-large-v3/` | ~2.9 GB | ✅ YES |
| Translation Packages | `~/.local/share/argos-translate/packages/` | 481 MB | ✅ YES |
| HuggingFace Cache | `~/.cache/huggingface/` | ~5.9 GB | ⚠️ Recommended |
| Stanza Models | `~/stanza_resources/` | 3.1 MB | ⚪ Optional |

---

## What Does NOT Need to be Downloaded

These are Python packages installed via pip (included in `requirements.txt`):
- fastapi, uvicorn
- torch, torchvision, torchaudio (PyTorch)
- numpy, librosa, soundfile
- faster-whisper (Python package, not models)
- argostranslate (Python package, not translation models)
- langdetect
- spacy, stanza (Python packages, models are separate)

**Note**: Python packages are installed via `pip install -r requirements.txt` - they don't need to be zipped.

---

## Installation on Offline Server

After transferring zip files:

### 1. Install Python Dependencies (if not already done)
```bash
pip install -r requirements.txt
```

### 2. Extract Whisper Models
```bash
tar -xzf whisper-models.tar.gz -C ./models/
```

### 3. Extract Translation Packages

**macOS/Linux:**
```bash
mkdir -p ~/.local/share/argos-translate/
tar -xzf translation-packages.tar.gz -C ~/.local/share/argos-translate/
```

**Windows:**
```powershell
# Extract to: C:\Users\username\.local\share\argos-translate\packages\
```

### 4. (Optional) Extract Stanza Models
```bash
tar -xzf stanza-models.tar.gz -C ~/
```

### 5. (Optional) Extract HuggingFace Cache
```bash
mkdir -p ~/.cache/
tar -xzf huggingface-cache.tar.gz -C ~/.cache/
```

---

## Quick Reference

**Minimum Package** (Required only):
- ✅ whisper-models.tar.gz (2.9 GB)
- ✅ translation-packages.tar.gz (481 MB)
- **Total: ~3.4 GB**

**Recommended Package**:
- ✅ whisper-models.tar.gz (2.9 GB)
- ✅ translation-packages.tar.gz (481 MB)
- ⚠️ huggingface-cache.tar.gz (5.9 GB)
- **Total: ~9.4 GB**

---

## Verification Checklist

After extracting, verify:

- [ ] `models/models--Systran--faster-whisper-large-v3/snapshots/*/model.bin` exists
- [ ] `~/.local/share/argos-translate/packages/en_hi/` exists
- [ ] `~/.local/share/argos-translate/packages/hi_en/` exists
- [ ] `~/.local/share/argos-translate/packages/en_ko/` exists
- [ ] `~/.local/share/argos-translate/packages/ko_en/` exists
- [ ] `~/.local/share/argos-translate/packages/hi_ko/` exists
- [ ] `~/.local/share/argos-translate/packages/ko_hi/` exists

---

## Summary

**For fully offline operation, you need to zip:**

1. **Whisper models directory** (2.9 GB) - REQUIRED
2. **Translation packages directory** (481 MB) - REQUIRED

**Optional but recommended:**
3. HuggingFace cache (5.9 GB) - Large but may help

**That's it!** Everything else (Python packages) is installed via pip and doesn't require separate downloads.

See also:
- `OFFLINE_INSTALLATION_ZIP_GUIDE.md` - Detailed guide
- `QUICK_OFFLINE_GUIDE.md` - Quick reference
- `create_offline_zip.sh` / `create_offline_zip.bat` - Automated scripts

