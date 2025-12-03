# Fix spaCy and Stanza Initialization Errors

## Problem

Argos Translate depends on both **spaCy** and **Stanza**, which may try to:
1. Download models when imported (requires internet)
2. Initialize models that aren't installed
3. Connect to the internet through office proxy (causing SSL certificate 443 errors)

In offline environments or office networks with proxy restrictions, this causes errors.

## Solution

**The fix has already been applied in the code!** The application now automatically:
1. Disables spaCy and Stanza model downloads
2. Suppresses SSL/certificate warnings for office proxy
3. Handles initialization errors gracefully

### Automatic Fix Applied

The following environment variables are automatically set:
- `SPACY_DISABLE_MODEL_DOWNLOAD=1` - Prevents spaCy from downloading models
- `STANZA_RESOURCES_DIR` - Sets local directory for Stanza (prevents downloads)
- `STANZA_CACHE_DIR` - Sets cache directory for Stanza

SSL certificate warnings are also suppressed to handle office proxy issues.

### Manual Fix (If Needed)

If you still see errors, you can manually set these in your `.env` file:

```env
SPACY_DISABLE_MODEL_DOWNLOAD=1
STANZA_RESOURCES_DIR=~/.stanza
STANZA_CACHE_DIR=~/.stanza_cache
```

Or set in your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):
```bash
export SPACY_DISABLE_MODEL_DOWNLOAD=1
export STANZA_RESOURCES_DIR=~/.stanza
export STANZA_CACHE_DIR=~/.stanza_cache
```

### Office Proxy / SSL Certificate Issues

The code automatically suppresses SSL certificate errors that occur when dependencies try to check for updates through office proxies. These warnings are non-critical and don't affect functionality.

## Verification

After applying the fix, restart your server and check if the error is resolved:

```bash
python run.py
```

If you still see errors, check the specific error message and we can address it further.

## Common Errors and Solutions

### 1. spaCy "not initialized" Error
**Solution:** Already fixed in code. spaCy doesn't need to be initialized - Argos Translate handles this internally.

### 2. Stanza Certificate 443 Error (Office Proxy)
**Solution:** Already fixed in code. SSL warnings are suppressed and Stanza downloads are disabled.

### 3. Model Download Errors
**Solution:** Already fixed. Both spaCy and Stanza are configured to not download models.

### 4. Import Errors
**Solution:** Ensure dependencies are installed:
```bash
pip install spacy stanza
```

However, **spaCy and Stanza don't need to be initialized or have models installed** - Argos Translate works without them being fully functional.

## Note

Argos Translate uses spaCy internally for some processing but doesn't require specific language models to be downloaded. The translation models (`.argosmodel` files) are separate and must be installed separately as described in `DOWNLOAD_TRANSLATION_MODELS.md`.

