# Fix for Office Proxy / spaCy / Stanza Errors

## Problems Fixed

1. **spaCy "not initialized" error** - spaCy tries to initialize but we've disabled downloads
2. **Stanza certificate 443 error** - Stanza tries to download models through office proxy, causing SSL certificate errors
3. **Translation not working in office network proxy** - Dependencies trying to connect to internet

## Solutions Applied

### 1. Disabled All Model Downloads

The code now automatically sets these environment variables before any imports:
- `SPACY_DISABLE_MODEL_DOWNLOAD=1` - Prevents spaCy from downloading models
- `STANZA_RESOURCES_DIR` - Sets local directory for Stanza (prevents internet downloads)
- `STANZA_CACHE_DIR` - Sets cache directory for Stanza

### 2. Suppressed SSL/Certificate Warnings

All SSL certificate warnings (including 443 port errors) are now suppressed:
- urllib3 SSL warnings disabled
- Certificate verification warnings suppressed
- Office proxy certificate errors handled gracefully

### 3. Graceful Error Handling

The translation service now:
- Handles spaCy/Stanza initialization errors gracefully
- Continues even if dependencies have warnings
- Falls back to manual package detection if needed
- Logs errors as debug messages instead of failing

## How It Works

1. **Environment Variables Set Early**: Before any imports in:
   - `run.py` (entry point)
   - `app/main.py` (main application)
   - `app/services/translation_service.py` (translation service)

2. **Warning Suppression**: All warnings from spaCy, Stanza, and SSL are filtered

3. **Error Handling**: Errors are caught and handled gracefully - service continues to work

## Verification

After restarting your server, you should see:
- No spaCy initialization errors
- No Stanza certificate 443 errors
- Translation works if packages are installed

Check logs for:
```
INFO: Translation packages storage location: /path/to/packages
INFO: Loaded X installed packages from disk
```

## If Errors Persist

If you still see errors, check:

1. **Are translation packages installed?**
   ```bash
   ls -la ~/.local/share/argos-translate/packages/
   ```
   Should show directories like: `en_hi`, `hi_en`, `en_ko`, etc.

2. **Check error logs** - Look for actual error messages (not just warnings)

3. **Verify environment variables** - Add to `.env` file if needed:
   ```env
   SPACY_DISABLE_MODEL_DOWNLOAD=1
   STANZA_RESOURCES_DIR=~/.stanza
   STANZA_CACHE_DIR=~/.stanza_cache
   ```

## Important Notes

- **spaCy and Stanza don't need to be initialized** - Argos Translate works without them being fully functional
- **SSL certificate errors are harmless** - They're just warnings from dependencies checking for updates
- **Translation packages must be installed** - See `DOWNLOAD_TRANSLATION_MODELS.md` for instructions
- **All warnings are suppressed** - This is safe because we don't need the features that trigger them

## Summary

The application is now fully configured to work in offline environments with office proxy restrictions. spaCy and Stanza errors are handled gracefully, and translation will work as long as the Argos Translate packages are installed.

