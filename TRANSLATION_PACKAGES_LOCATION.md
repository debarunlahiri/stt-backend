# Translation Packages Storage Location

## Package Storage Directory

Argos Translate stores all installed translation packages in the following locations:

### macOS/Linux
```
~/.local/share/argos-translate/packages
```

**Full path example:**
```
/Users/debarunlahiri/.local/share/argos-translate/packages
```

### Windows
```
C:\Users\username\.local\share\argos-translate\packages
```

## Package Directory Structure

Each translation package is stored in its own subdirectory named with the language pair:

```
~/.local/share/argos-translate/packages/
├── en_hi/          # English to Hindi
├── hi_en/          # Hindi to English
├── en_ko/          # English to Korean
├── ko_en/          # Korean to English
├── hi_ko/          # Hindi to Korean
└── ko_hi/          # Korean to Hindi
```

## Current Installed Packages

Based on your system, the following packages are installed:

- `en_hi` - English to Hindi
- `hi_en` - Hindi to English
- `en_ko` - English to Korean
- `ko_en` - Korean to English

## How to Check Installed Packages

### Using Python
```bash
python -c "import argostranslate.package; packages = argostranslate.package.get_installed_packages(); [print(f'{p.from_code} -> {p.to_code}') for p in packages]"
```

### Using File System
```bash
# macOS/Linux
ls -la ~/.local/share/argos-translate/packages/

# Windows
dir %USERPROFILE%\.local\share\argos-translate\packages
```

## Important Notes

1. **Offline Operation**: Once packages are installed, they are stored locally and work completely offline
2. **Persistence**: Packages persist across server restarts and Python environment changes
3. **Shared Location**: Packages are stored in the user's home directory, not in the project directory
4. **Backup**: To backup packages, copy the entire `~/.local/share/argos-translate/packages/` directory
5. **Size**: Each package is typically 50-200 MB in size

## Package Contents

Each package directory contains:
- Model files (`.pt` or `.onnx` files)
- Configuration files
- Vocabulary files
- Other translation model assets

## Viewing Package Location in Logs

When the translation service starts, it logs the package storage location:

```
INFO: Translation packages storage location: /Users/debarunlahiri/.local/share/argos-translate/packages
INFO: Translation package already installed: en -> hi (location: /Users/debarunlahiri/.local/share/argos-translate/packages/en_hi)
```

## Backup and Restore

### Backup Packages
```bash
# macOS/Linux
tar -czf translation-packages-backup.tar.gz ~/.local/share/argos-translate/packages/

# Windows
# Use 7-Zip or similar to compress the packages directory
```

### Restore Packages
```bash
# Extract to the packages directory
tar -xzf translation-packages-backup.tar.gz -C ~/.local/share/argos-translate/
```

## Troubleshooting

### Check if packages directory exists
```bash
ls -la ~/.local/share/argos-translate/packages/
```

### Check package size
```bash
du -sh ~/.local/share/argos-translate/packages/*
```

### Remove a specific package
```bash
rm -rf ~/.local/share/argos-translate/packages/en_hi
```

### Clear all packages
```bash
rm -rf ~/.local/share/argos-translate/packages/*
```

## Related Files

- `translation_packages.json` - Manifest file in project root listing installed packages
- This file is created automatically and tracks which packages are installed

