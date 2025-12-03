# Stanza Storage Location

## Default Storage Directory

**Stanza stores its models in the user's home directory by default:**

### Default Location:
```
~/stanza_resources
```

**Full paths:**
- **macOS**: `/Users/username/stanza_resources`
- **Linux**: `/home/username/stanza_resources`
- **Windows**: `C:\Users\username\stanza_resources`

### Actual Location on Your System:
Based on your system, Stanza models are stored at:
```
/Users/debarunlahiri/stanza_resources
```

**Size**: ~3.1 MB

**Contents**:
- `en/` - English models (tokenize, mwt models)
- `hi/` - Hindi models (tokenize models)
- `ko/` - Korean models (tokenize models)
- `resources.json` - Model metadata file

**Model files found**:
- `/Users/debarunlahiri/stanza_resources/hi/tokenize/hdtb.pt`
- `/Users/debarunlahiri/stanza_resources/ko/tokenize/kaist.pt`
- `/Users/debarunlahiri/stanza_resources/en/tokenize/combined.pt`
- `/Users/debarunlahiri/stanza_resources/en/mwt/combined.pt`

## Current Configuration

In this application, we've configured Stanza to:
1. Use existing models if `~/stanza_resources` exists (your case)
2. Otherwise set custom directories to prevent downloads

**Configuration logic**:
- If `~/stanza_resources` exists → Use it (models already downloaded)
- If not → Set `~/.stanza` (prevents downloads)

**Environment variables set**:
- **STANZA_RESOURCES_DIR**: 
  - `~/stanza_resources` (if it exists - your case)
  - `~/.stanza` (if it doesn't exist)
- **STANZA_CACHE_DIR**: `~/.stanza_cache` (cache location)

These are set in:
- `run.py`
- `app/main.py`
- `app/services/translation_service.py`

## Important Notes

1. **Stanza is NOT required to work** - It's only a dependency of Argos Translate but doesn't need to be initialized or have models downloaded
2. **You don't need to download Stanza models** - Argos Translate works without Stanza being fully functional
3. **The directories may not exist** - This is normal and fine. Stanza won't download models because we've disabled it.

## Check Stanza Storage Location

### Check default location:
```bash
# macOS/Linux
ls -la ~/stanza_resources
# or
ls -la ~/.stanza_resources

# Windows
dir %USERPROFILE%\stanza_resources
```

### Check configured location:
```bash
# macOS/Linux
ls -la ~/.stanza
ls -la ~/.stanza_cache

# Windows
dir %USERPROFILE%\.stanza
dir %USERPROFILE%\.stanza_cache
```

## If You Need to Download Stanza Models (NOT Required)

**Note: This is NOT necessary for the application to work!** Stanza is just a dependency, but Argos Translate doesn't require it to be initialized.

If you want to download Stanza models anyway (for other purposes):

1. Set the environment variable:
   ```bash
   export STANZA_RESOURCES_DIR=~/stanza_resources
   ```

2. Download models on a machine with internet:
   ```bash
   python -c "import stanza; stanza.download('en')"
   ```

3. Models will be stored in `~/stanza_resources` directory

## Why We Configure It

We set `STANZA_RESOURCES_DIR` and `STANZA_CACHE_DIR` to:
1. Prevent Stanza from trying to download models (which requires internet)
2. Avoid certificate errors when it tries to connect through office proxy
3. Ensure offline operation

The actual location doesn't matter much because **Stanza doesn't need to work** for Argos Translate to function. The translation models (`.argosmodel` files) are stored separately.

## Summary

- **Default Location**: `~/stanza_resources`
- **Your Current Location**: `/Users/debarunlahiri/stanza_resources` (3.1 MB, models installed)
- **Configured in Code**: Uses existing location if found, otherwise prevents downloads
- **Required**: NO - Stanza models are NOT needed for translation to work
- **Status**: Models exist on your system, but they're optional

**Important**: Having Stanza models installed is fine, but they're not required. The application will work whether Stanza models exist or not.

For translation to work, you only need the **Argos Translate packages** installed in:
- `~/.local/share/argos-translate/packages/`

See `DOWNLOAD_TRANSLATION_MODELS.md` for Argos Translate package installation.

## Why Stanza Might Still Show Errors

Even though you have Stanza models installed, you might still see initialization errors because:
1. Stanza tries to check for updates online (blocked by office proxy)
2. Certificate verification fails (443 SSL errors)
3. Initialization may fail but this doesn't affect Argos Translate

**Solution**: All these errors are suppressed and handled gracefully. Translation will work regardless.

