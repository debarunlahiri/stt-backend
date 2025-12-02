# Offline Setup Summary - Quick Reference

## The Problem

Argos Translate translation packages need to be downloaded. Since your server has no internet, you must download them on another machine and transfer them.

## Quick Solution

### Step 1: Download on Machine with Internet

Use one of these methods:

**Option A: Direct Download Links (Easiest)**

Go to: https://github.com/argosopentech/argos-translate/releases

Download these 6 files (each ~50-200 MB):
1. `translate-en_hi-1_8.argosmodel`
2. `translate-hi_en-1_8.argosmodel`
3. `translate-en_ko-1_10.argosmodel`
4. `translate-ko_en-1_10.argosmodel`
5. `translate-hi_ko-1_10.argosmodel`
6. `translate-ko_hi-1_10.argosmodel`

**Option B: Use Python Script**

See `DOWNLOAD_TRANSLATION_MODELS.md` for automated download script.

### Step 2: Transfer to Offline Server

Copy all 6 `.argosmodel` files to your offline server (USB drive, network share, etc.)

### Step 3: Install on Offline Server

Place all 6 files in a folder, then run:

```bash
# Create packages directory
mkdir -p ~/.local/share/argos-translate/packages

# Install each package (replace with actual filenames)
python -c "
import argostranslate.package
import glob

# Install all .argosmodel files in current directory
for pkg_file in glob.glob('*.argosmodel'):
    print(f'Installing {pkg_file}...')
    argostranslate.package.install_from_path(pkg_file)
    print(f'✓ Installed {pkg_file}')
"

# Verify
python -c "import argostranslate.package; packages = argostranslate.package.get_installed_packages(); print(f'Installed: {len(packages)} packages'); [print(f'  - {p.from_code} -> {p.to_code}') for p in packages]"
```

You should see all 6 packages installed.

### Step 4: Restart Server

After installation, restart your server. The translation service will now work offline.

## Package Storage Location

After installation, packages are stored at:

- **macOS**: `/Users/yourusername/.local/share/argos-translate/packages/`
- **Linux**: `/home/yourusername/.local/share/argos-translate/packages/`
- **Windows**: `C:\Users\yourusername\.local\share\argos-translate\packages\`

## Verification

Check if packages are installed:

```bash
ls -la ~/.local/share/argos-translate/packages/
```

You should see 6 directories: `en_hi`, `hi_en`, `en_ko`, `ko_en`, `hi_ko`, `ko_hi`

## Full Instructions

For detailed instructions with all methods and troubleshooting, see:
- **DOWNLOAD_TRANSLATION_MODELS.md** - Complete download and installation guide
- **TRANSLATION_PACKAGES_LOCATION.md** - Package management and locations

## Important Notes

1. **No internet needed** - Once packages are installed, the server works completely offline
2. **One-time setup** - Packages only need to be installed once
3. **Shared location** - Packages are stored in your user directory, not project directory
4. **Size** - Total download size: ~600-1200 MB (compressed: ~300-600 MB)

