# Quick Guide: What to Zip for Offline Installation

## Summary

**Total Required Size**: ~3.4 GB
**Total with Optional**: ~9.4 GB

---

## REQUIRED Components (Must Include)

### 1. Whisper Models (Speech-to-Text)
- **Location**: `./models/models--Systran--faster-whisper-large-v3/`
- **Size**: ~2.9 GB
- **Required Files**: `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`

### 2. Translation Packages (Translation)
- **Location**: `~/.local/share/argos-translate/packages/`
- **Size**: ~481 MB
- **Required**: 6 packages (en_hi, hi_en, en_ko, ko_en, hi_ko, ko_hi)

---

## OPTIONAL Components (Can Include)

### 3. HuggingFace Cache
- **Location**: `~/.cache/huggingface/`
- **Size**: ~5.9 GB (on your system)
- **Status**: Recommended but large

### 4. Stanza Models
- **Location**: `~/stanza_resources/`
- **Size**: ~3.1 MB
- **Status**: Optional - not required

---

## Quick Commands to Create Zip

### Using the Automated Script (Recommended)

**macOS/Linux:**
```bash
./create_offline_zip.sh
```

**Windows:**
```cmd
create_offline_zip.bat
```

### Manual Method

**macOS/Linux:**
```bash
# Required components
tar -czf whisper-models.tar.gz models/models--Systran--faster-whisper-large-v3/
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/

# Optional components
tar -czf stanza-models.tar.gz ~/stanza_resources/
tar -czf huggingface-cache.tar.gz ~/.cache/huggingface/  # Large!
```

**Windows (PowerShell):**
```powershell
# Required components
Compress-Archive -Path "models\models--Systran--faster-whisper-large-v3" -DestinationPath "whisper-models.zip"
Compress-Archive -Path "$env:USERPROFILE\.local\share\argos-translate\packages" -DestinationPath "translation-packages.zip"

# Optional components
Compress-Archive -Path "$env:USERPROFILE\stanza_resources" -DestinationPath "stanza-models.zip"
Compress-Archive -Path "$env:USERPROFILE\.cache\huggingface" -DestinationPath "huggingface-cache.zip"
```

---

## Installation Locations on Offline Server

After transferring and extracting:

| Component | Extract To |
|-----------|-----------|
| Whisper Models | `./models/models--Systran--faster-whisper-large-v3/` |
| Translation Packages | `~/.local/share/argos-translate/packages/` |
| Stanza Models | `~/stanza_resources/` (optional) |
| HuggingFace Cache | `~/.cache/huggingface/` (optional) |

---

## File Size Breakdown

| Component | Size | Priority |
|-----------|------|----------|
| Whisper Models | 2.9 GB | ✅ REQUIRED |
| Translation Packages | 481 MB | ✅ REQUIRED |
| HuggingFace Cache | 5.9 GB | ⚠️ RECOMMENDED |
| Stanza Models | 3.1 MB | ⚪ OPTIONAL |

---

## Minimum Package (Required Only)

If you need to minimize size, zip only:
1. Whisper models (2.9 GB)
2. Translation packages (481 MB)

**Total**: ~3.4 GB

---

## Complete Package (Recommended)

Include everything:
1. Whisper models (2.9 GB)
2. Translation packages (481 MB)
3. HuggingFace cache (5.9 GB)
4. Stanza models (3.1 MB)

**Total**: ~9.4 GB

---

## Detailed Instructions

For complete instructions, see: **OFFLINE_INSTALLATION_ZIP_GUIDE.md**

