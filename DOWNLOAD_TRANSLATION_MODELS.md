# How to Download Translation Models Offline

## Overview

The translation service uses Argos Translate, which requires translation model packages. These packages must be downloaded **manually on a machine with internet** and then transferred to your offline server.

## Step 1: Download Packages on a Machine with Internet

You need to download 6 translation packages (all bidirectional pairs):

1. English ↔ Hindi (en ↔ hi)
2. English ↔ Korean (en ↔ ko)
3. Hindi ↔ Korean (hi ↔ ko)

### Method 1: Using Python Script (Recommended)

On a machine with internet, create and run this script:

```python
#!/usr/bin/env python3
"""
Download Argos Translate packages for offline installation.
Run this script on a machine with internet access.
"""

import argostranslate.package
import os
from pathlib import Path

# Packages to download
packages_to_install = [
    'en_hi',  # English to Hindi
    'hi_en',  # Hindi to English
    'en_ko',  # English to Korean
    'ko_en',  # Korean to English
    'hi_ko',  # Hindi to Korean
    'ko_hi',  # Korean to Hindi
]

print("Updating package index...")
argostranslate.package.update_package_index()

print("\nFetching available packages...")
available_packages = argostranslate.package.get_available_packages()

# Create download directory
download_dir = Path("translation_packages")
download_dir.mkdir(exist_ok=True)

print(f"\nDownloading packages to: {download_dir.absolute()}\n")

downloaded_files = []

for pkg_code in packages_to_install:
    from_code, to_code = pkg_code.split('_')
    
    # Find the package
    package = None
    for pkg in available_packages:
        if pkg.from_code == from_code and pkg.to_code == to_code:
            package = pkg
            break
    
    if package:
        print(f"Downloading {pkg_code} ({from_code} -> {to_code})...")
        try:
            package_path = package.download()
            
            # Copy to our download directory
            import shutil
            dest_path = download_dir / os.path.basename(package_path)
            shutil.copy2(package_path, dest_path)
            downloaded_files.append(str(dest_path))
            
            file_size = os.path.getsize(dest_path) / (1024 * 1024)  # MB
            print(f"  ✓ Downloaded: {dest_path.name} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Error downloading {pkg_code}: {e}")
    else:
        print(f"  ✗ Package not found: {pkg_code}")

print(f"\n✓ Downloaded {len(downloaded_files)} packages")
print(f"\nPackages saved in: {download_dir.absolute()}")
print("\nNext steps:")
print("1. Compress the 'translation_packages' directory")
print("2. Transfer it to your offline server")
print("3. Follow installation instructions in this file")
```

Save as `download_translation_packages.py` and run:

```bash
python download_translation_packages.py
```

### Method 2: Direct Download URLs

Argos Translate packages are hosted on GitHub. Here are direct download links:

**English ↔ Hindi:**
- English to Hindi: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_hi-1_8.argosmodel`
- Hindi to English: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_en-1_8.argosmodel`

**English ↔ Korean:**
- English to Korean: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_ko-1_10.argosmodel`
- Korean to English: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_en-1_10.argosmodel`

**Hindi ↔ Korean:**
- Hindi to Korean: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_ko-1_10.argosmodel`
- Korean to Hindi: `https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_hi-1_10.argosmodel`

**Download with wget/curl:**

```bash
# Create download directory
mkdir translation_packages
cd translation_packages

# Download all packages
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_hi-1_8.argosmodel
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_en-1_8.argosmodel
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_ko-1_10.argosmodel
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_en-1_10.argosmodel
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_ko-1_10.argosmodel
wget https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_hi-1_10.argosmodel

# Or use curl
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_hi-1_8.argosmodel
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_en-1_8.argosmodel
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-en_ko-1_10.argosmodel
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_en-1_10.argosmodel
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-hi_ko-1_10.argosmodel
curl -L -O https://github.com/argosopentech/argos-translate/releases/download/v1.9.0/translate-ko_hi-1_10.argosmodel
```

**Note:** Version numbers may change. Check latest releases at:
https://github.com/argosopentech/argos-translate/releases

## Step 2: Transfer Packages to Offline Server

After downloading, compress the packages directory:

```bash
# Compress
tar -czf translation_packages.tar.gz translation_packages/

# Or for Windows, create a ZIP file
```

Transfer the compressed file to your offline server using USB drive, network share, or any other method.

## Step 3: Install Packages on Offline Server

### macOS/Linux:

```bash
# Extract packages
tar -xzf translation_packages.tar.gz

# Create packages directory if it doesn't exist
mkdir -p ~/.local/share/argos-translate/packages

# Install each package
cd translation_packages

python -c "
import argostranslate.package
import sys
import os

package_files = [
    'translate-en_hi-1_8.argosmodel',
    'translate-hi_en-1_8.argosmodel',
    'translate-en_ko-1_10.argosmodel',
    'translate-ko_en-1_10.argosmodel',
    'translate-hi_ko-1_10.argosmodel',
    'translate-ko_hi-1_10.argosmodel'
]

for pkg_file in package_files:
    if os.path.exists(pkg_file):
        print(f'Installing {pkg_file}...')
        argostranslate.package.install_from_path(pkg_file)
        print(f'✓ Installed {pkg_file}')
    else:
        print(f'✗ File not found: {pkg_file}')
"

# Verify installation
python -c "import argostranslate.package; packages = argostranslate.package.get_installed_packages(); [print(f'{p.from_code} -> {p.to_code}') for p in packages]"
```

### Windows:

```powershell
# Extract packages (use 7-Zip or Windows built-in extraction)

# Create packages directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.local\share\argos-translate\packages"

# Install each package
cd translation_packages

python -c "
import argostranslate.package
import sys
import os

package_files = [
    'translate-en_hi-1_8.argosmodel',
    'translate-hi_en-1_8.argosmodel',
    'translate-en_ko-1_10.argosmodel',
    'translate-ko_en-1_10.argosmodel',
    'translate-hi_ko-1_10.argosmodel',
    'translate-ko_hi-1_10.argosmodel'
]

for pkg_file in package_files:
    if os.path.exists(pkg_file):
        print(f'Installing {pkg_file}...')
        argostranslate.package.install_from_path(pkg_file)
        print(f'✓ Installed {pkg_file}')
    else:
        print(f'✗ File not found: {pkg_file}')
"

# Verify installation
python -c "import argostranslate.package; packages = argostranslate.package.get_installed_packages(); [print(f'{p.from_code} -> {p.to_code}') for p in packages]"
```

## Step 4: Verify Installation

Check that all packages are installed:

```bash
python -c "import argostranslate.package; packages = argostranslate.package.get_installed_packages(); print(f'Installed packages: {len(packages)}'); [print(f'  - {p.from_code} -> {p.to_code}') for p in packages]"
```

You should see all 6 packages listed.

## Package Storage Locations

After installation, packages are stored at:

- **macOS/Linux**: `~/.local/share/argos-translate/packages/`
- **Windows**: `C:\Users\username\.local\share\argos-translate\packages\`

## Troubleshooting

### Packages not found after installation

1. Check the packages directory exists:
   ```bash
   ls -la ~/.local/share/argos-translate/packages/
   ```

2. Verify package files are present (each package creates a directory)

3. Ensure you're using the same Python environment where packages were installed

### Package files have different names

If downloaded files have different names, list them first:

```bash
ls -la translation_packages/
```

Then update the installation script with the actual filenames.

### Offline Installation Script

Create `install_translation_packages_offline.py`:

```python
#!/usr/bin/env python3
"""
Install Argos Translate packages offline.
Place all .argosmodel files in the same directory as this script.
"""

import argostranslate.package
import os
from pathlib import Path

# Get current directory
current_dir = Path(__file__).parent

# Find all .argosmodel files
package_files = list(current_dir.glob("*.argosmodel"))

if not package_files:
    print("No .argosmodel files found in current directory!")
    print("Please place the downloaded package files here and run again.")
    exit(1)

print(f"Found {len(package_files)} package files\n")

for pkg_file in package_files:
    print(f"Installing {pkg_file.name}...")
    try:
        argostranslate.package.install_from_path(str(pkg_file))
        print(f"  ✓ Successfully installed {pkg_file.name}\n")
    except Exception as e:
        print(f"  ✗ Error installing {pkg_file.name}: {e}\n")

print("\nVerifying installation...")
packages = argostranslate.package.get_installed_packages()
print(f"Installed {len(packages)} packages:")
for pkg in packages:
    print(f"  - {pkg.from_code} -> {pkg.to_code}")
```

Save this script, place it in the same directory as your downloaded `.argosmodel` files, and run it.

## Alternative: Pre-downloaded Packages Archive

If you have access to the packages directory from another installation, you can directly copy it:

**From another machine (that has packages installed):**

```bash
# On source machine
tar -czf argos-translate-packages.tar.gz ~/.local/share/argos-translate/packages/

# Transfer and extract on offline server
tar -xzf argos-translate-packages.tar.gz -C ~/.local/share/argos-translate/
```

This is the fastest method if you have access to a machine that already has the packages installed.

## Package Sizes

Each package is typically 50-200 MB:
- Total size for all 6 packages: ~600-1200 MB
- Compressed: ~300-600 MB

Make sure you have enough disk space before downloading.

