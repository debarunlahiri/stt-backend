# Translation Packages Storage Location

## Offline Mode

The translation service runs in **fully offline mode** - it does NOT download packages from the internet. All translation packages must be pre-installed on the system before running the server.

## Package Storage Directory

Argos Translate stores all installed translation packages in platform-specific locations:

### macOS
```
~/.local/share/argos-translate/packages
```

**Full path example:**
```
/Users/debarunlahiri/.local/share/argos-translate/packages
```

### Linux
```
~/.local/share/argos-translate/packages
```

**Full path example:**
```
/home/username/.local/share/argos-translate/packages
```

### Windows
```
C:\Users\username\.local\share\argos-translate\packages
```

**Full path example:**
```
C:\Users\john\.local\share\argos-translate\packages
```

**Note:** The application automatically detects the correct path based on your operating system.

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

**macOS:**
```bash
ls -la ~/.local/share/argos-translate/packages/
# Or full path:
ls -la /Users/$(whoami)/.local/share/argos-translate/packages/
```

**Linux:**
```bash
ls -la ~/.local/share/argos-translate/packages/
# Or full path:
ls -la /home/$(whoami)/.local/share/argos-translate/packages/
```

**Windows (Command Prompt):**
```cmd
dir %USERPROFILE%\.local\share\argos-translate\packages
```

**Windows (PowerShell):**
```powershell
Get-ChildItem "$env:USERPROFILE\.local\share\argos-translate\packages"
```

## Important Notes

1. **Fully Offline Mode**: The application runs in fully offline mode - it does NOT download packages from the internet
2. **Pre-installation Required**: Translation packages must be installed before running the server
3. **Persistence**: Packages persist across server restarts and Python environment changes
4. **Shared Location**: Packages are stored in the user's home directory, not in the project directory
5. **Platform Independent**: Works on macOS, Linux, and Windows - paths are automatically detected
6. **Backup**: To backup packages, copy the entire packages directory for your platform
7. **Size**: Each package is typically 50-200 MB in size

## Installing Translation Packages (One-Time Setup)

### On a Machine with Internet Access

Before deploying to an offline server, install packages on a machine with internet:

```bash
# Activate your virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install packages using Python
python -c "
import argostranslate.package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
packages_to_install = ['en_hi', 'hi_en', 'en_ko', 'ko_en', 'hi_ko', 'ko_hi']
for pkg_code in packages_to_install:
    for pkg in available_packages:
        if f'{pkg.from_code}_{pkg.to_code}' == pkg_code:
            print(f'Installing {pkg_code}...')
            package_path = pkg.download()
            argostranslate.package.install_from_path(package_path)
            print(f'Installed {pkg_code}')
            break
"
```

### Transferring Packages to Offline Server

After installation, copy the packages directory to your offline server:

**macOS/Linux:**
```bash
# On machine with internet (after installing packages)
tar -czf translation-packages.tar.gz ~/.local/share/argos-translate/packages/

# Transfer to offline server and extract:
tar -xzf translation-packages.tar.gz -C ~/.local/share/argos-translate/
```

**Windows:**
```powershell
# On machine with internet (after installing packages)
# Use 7-Zip or similar to compress:
# C:\Users\username\.local\share\argos-translate\packages

# Transfer to offline server and extract to:
# C:\Users\username\.local\share\argos-translate\packages
```

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

**macOS/Linux:**
```bash
tar -czf translation-packages-backup.tar.gz ~/.local/share/argos-translate/packages/
```

**Windows (PowerShell):**
```powershell
Compress-Archive -Path "$env:USERPROFILE\.local\share\argos-translate\packages" -DestinationPath "translation-packages-backup.zip"
```

**Windows (7-Zip):**
```cmd
"C:\Program Files\7-Zip\7z.exe" a translation-packages-backup.7z "%USERPROFILE%\.local\share\argos-translate\packages"
```

### Restore Packages

**macOS/Linux:**
```bash
# Extract to the packages directory
tar -xzf translation-packages-backup.tar.gz -C ~/.local/share/argos-translate/
```

**Windows (PowerShell):**
```powershell
Expand-Archive -Path "translation-packages-backup.zip" -DestinationPath "$env:USERPROFILE\.local\share\argos-translate\"
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

