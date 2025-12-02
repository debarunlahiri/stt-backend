#!/usr/bin/env python3
"""
Install Argos Translate packages offline.

This script installs translation packages from .argosmodel files that were
previously downloaded. Place all .argosmodel files in the same directory
as this script and run it.

Usage:
    python install_translation_packages_offline.py

The script will:
1. Find all .argosmodel files in the current directory
2. Install each package
3. Verify installation
"""

import argostranslate.package
import os
import sys
from pathlib import Path

def main():
    """Install translation packages from local .argosmodel files."""
    
    # Get current directory
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    
    # Find all .argosmodel files
    package_files = list(current_dir.glob("*.argosmodel"))
    
    if not package_files:
        print("ERROR: No .argosmodel files found in current directory!")
        print(f"\nDirectory searched: {current_dir.absolute()}")
        print("\nPlease:")
        print("1. Download translation packages from:")
        print("   https://github.com/argosopentech/argos-translate/releases")
        print("2. Place all .argosmodel files in the same directory as this script")
        print("3. Run this script again")
        print("\nSee DOWNLOAD_TRANSLATION_MODELS.md for download instructions.")
        sys.exit(1)
    
    print(f"Found {len(package_files)} package file(s):\n")
    for pkg_file in package_files:
        size_mb = pkg_file.stat().st_size / (1024 * 1024)
        print(f"  - {pkg_file.name} ({size_mb:.1f} MB)")
    
    print("\n" + "="*60)
    print("Installing packages...")
    print("="*60 + "\n")
    
    installed_count = 0
    failed_count = 0
    
    for pkg_file in package_files:
        print(f"Installing {pkg_file.name}...")
        try:
            argostranslate.package.install_from_path(str(pkg_file))
            installed_count += 1
            print(f"  ✓ Successfully installed {pkg_file.name}\n")
        except Exception as e:
            failed_count += 1
            print(f"  ✗ ERROR installing {pkg_file.name}: {e}\n")
    
    print("="*60)
    print("Installation Summary")
    print("="*60)
    print(f"Successfully installed: {installed_count}/{len(package_files)}")
    if failed_count > 0:
        print(f"Failed: {failed_count}/{len(package_files)}")
    
    # Verify installation
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60 + "\n")
    
    try:
        packages = argostranslate.package.get_installed_packages()
        print(f"Total installed packages: {len(packages)}\n")
        
        if packages:
            print("Installed translation pairs:")
            for pkg in sorted(packages, key=lambda x: (x.from_code, x.to_code)):
                print(f"  ✓ {pkg.from_code} -> {pkg.to_code}")
        
        # Check for required packages
        required_pairs = [
            ("en", "hi"), ("hi", "en"),
            ("en", "ko"), ("ko", "en"),
            ("hi", "ko"), ("ko", "hi")
        ]
        
        installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in packages]
        missing_pairs = [pair for pair in required_pairs if pair not in installed_pairs]
        
        if missing_pairs:
            print(f"\n⚠ WARNING: Missing {len(missing_pairs)} required package(s):")
            for from_code, to_code in missing_pairs:
                print(f"  ✗ {from_code} -> {to_code}")
            print("\nSome translation features may not work.")
        else:
            print("\n✓ All required translation packages are installed!")
            
    except Exception as e:
        print(f"ERROR verifying installation: {e}")
    
    # Show package storage location
    print("\n" + "="*60)
    print("Package Storage Location")
    print("="*60)
    
    import platform
    if platform.system() == "Windows":
        storage_path = Path.home() / ".local" / "share" / "argos-translate" / "packages"
    else:
        storage_path = Path.home() / ".local" / "share" / "argos-translate" / "packages"
    
    print(f"Packages are stored at: {storage_path}")
    
    if storage_path.exists():
        package_dirs = [d.name for d in storage_path.iterdir() if d.is_dir()]
        if package_dirs:
            print(f"\nPackage directories found: {len(package_dirs)}")
            for pkg_dir in sorted(package_dirs):
                print(f"  - {pkg_dir}")
    else:
        print("\n⚠ Storage directory does not exist yet.")
    
    print("\n" + "="*60)
    print("Installation complete!")
    print("="*60)
    print("\nYou can now restart your server and translation will work offline.")
    print("\nFor more information, see:")
    print("  - DOWNLOAD_TRANSLATION_MODELS.md")
    print("  - TRANSLATION_PACKAGES_LOCATION.md")

if __name__ == "__main__":
    main()

