"""
Translation Service Module

This module provides offline translation capabilities using Argos Translate and language
detection using langdetect. It supports translation between English, Hindi, and Korean.

The service automatically installs required translation packages on first use and
provides thread-safe translation operations.

Author: Debarun Lahiri
"""

import os
import warnings
import ssl

# CRITICAL: Set environment variables BEFORE any imports
# Prevent spacy and stanza from trying to download models (requires internet)
os.environ.setdefault('SPACY_DISABLE_MODEL_DOWNLOAD', '1')
# Stanza default location is ~/stanza_resources, but we can set custom to prevent downloads
# If stanza_resources exists, use it; otherwise set a non-existent path to prevent downloads
stanza_default = os.path.expanduser('~/stanza_resources')
if os.path.exists(stanza_default):
    os.environ.setdefault('STANZA_RESOURCES_DIR', stanza_default)
else:
    os.environ.setdefault('STANZA_RESOURCES_DIR', os.path.expanduser('~/.stanza'))
os.environ.setdefault('STANZA_CACHE_DIR', os.path.expanduser('~/.stanza_cache'))

# Disable SSL verification warnings for office proxy/certificate issues
# This prevents certificate errors when dependencies try to check for updates
warnings.filterwarnings('ignore', message='.*SSL.*')
warnings.filterwarnings('ignore', message='.*certificate.*')
warnings.filterwarnings('ignore', message='.*443.*')
warnings.filterwarnings('ignore', category=ssl.SSLError)

# Suppress spacy warnings - it's a dependency of argostranslate but we don't need its models
warnings.filterwarnings('ignore', category=UserWarning, module='spacy')
warnings.filterwarnings('ignore', category=UserWarning, module='en_core_web')
warnings.filterwarnings('ignore', message='.*spacy.*')
warnings.filterwarnings('ignore', message='.*spacy.*not.*initialized.*', category=UserWarning)

# Suppress stanza warnings - it's also a dependency
warnings.filterwarnings('ignore', category=UserWarning, module='stanza')
warnings.filterwarnings('ignore', message='.*stanza.*')
warnings.filterwarnings('ignore', message='.*stanza.*download.*')
warnings.filterwarnings('ignore', message='.*stanza.*certificate.*')

import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

from langdetect import detect, detect_langs, LangDetectException

# Import argostranslate after suppressing warnings and setting environment variables
# Wrap in try-except to handle any spacy/stanza initialization errors gracefully
# These libraries may try to download models or check internet even though we don't need them
try:
    # Temporarily disable urllib3 SSL warnings for office proxy
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    urllib3.disable_warnings(urllib3.exceptions.SubjectAltNameWarning)
except:
    pass

try:
    import argostranslate.package
    import argostranslate.translate
except Exception as e:
    # If there's an import error, log it but continue - translation may still work
    import logging as log_module
    log_module.basicConfig(level=log_module.WARNING)
    error_msg = str(e)
    # Don't fail on spacy/stanza initialization errors - they're not critical
    if 'spacy' in error_msg.lower() or 'stanza' in error_msg.lower() or 'certificate' in error_msg.lower() or 'ssl' in error_msg.lower() or '443' in error_msg:
        log_module.warning(f"Argos Translate dependency warning (non-critical): {error_msg[:200]}")
    else:
        log_module.warning(f"Argos Translate import warning: {error_msg[:200]}")
    # Try to import anyway - argostranslate may work even if spacy/stanza have issues
    try:
        import argostranslate.package
        import argostranslate.translate
    except:
        # If it still fails, we'll handle it in the service initialization
        pass

from app.config import settings

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Translation service with language detection and offline translation.
    
    This service provides:
    - Automatic language detection from input text
    - Offline translation between supported language pairs
    - Automatic installation of translation packages
    - Thread-safe operations for concurrent requests
    
    Supported languages:
    - English (en)
    - Hindi (hi)
    - Korean (ko)
    
    Supported translation pairs:
    - English ↔ Hindi
    - English ↔ Korean
    - Hindi ↔ Korean
    
    Author: Debarun Lahiri
    """
    
    # Language code mapping: Maps language codes to their display names
    # Used for validation and display purposes
    LANGUAGE_MAP = {
        "en": "English",
        "hi": "Hindi",
        "ko": "Korean",
        "auto": "auto"  # Special value for auto-detection
    }
    
    # Supported translation pairs: All bidirectional translation pairs supported by the service
    # Each tuple represents (source_language, target_language)
    SUPPORTED_PAIRS = [
        ("en", "hi"),  # English to Hindi
        ("hi", "en"),  # Hindi to English
        ("en", "ko"),  # English to Korean
        ("ko", "en"),  # Korean to English
        ("hi", "ko"),  # Hindi to Korean
        ("ko", "hi"),  # Korean to Hindi
    ]
    
    def __init__(self):
        """
        Initialize the TranslationService in fully offline mode.
        
        This service works completely offline - it does NOT connect to the internet.
        Translation packages must be pre-installed on the system before running.
        
        Packages are loaded from local disk storage:
        - macOS/Linux: ~/.local/share/argos-translate/packages
        - Windows: C:\\Users\\username\\.local\\share\\argos-translate\\packages
        
        If packages are not found, the service will still start but translation
        will fail when requested. See DOWNLOAD_TRANSLATION_MODELS.md for
        instructions on downloading and installing packages offline.
        """
        self._installed_packages = []  # List of (from_code, to_code) tuples for installed packages
        self._initialization_attempted = False  # Track if initialization has been attempted
        self._packages_manifest_file = Path("./translation_packages.json")  # File to save installed packages list
        
        # Step 1: Load installed packages from disk (fully offline)
        # This reads from local storage only - no internet required
        # Error handling is done inside _load_installed_packages_from_disk
        self._load_installed_packages_from_disk()
        
        # Step 2: Initialize packages from local disk only (fully offline)
        # This does NOT attempt to connect to the internet
        try:
            self._initialize_translation_packages()
        except Exception as e:
            error_str = str(e).lower()
            # Don't fail on spacy/stanza/ssl errors - these are non-critical dependencies
            if any(x in error_str for x in ['spacy', 'stanza', 'ssl', 'certificate', '443', 'download', 'not initialized']):
                logger.debug(f"Non-critical dependency warning during initialization: {str(e)[:200]}")
                # Continue anyway - translation may still work
                if not self._installed_packages:
                    logger.info("Note: Translation packages should be pre-installed. See DOWNLOAD_TRANSLATION_MODELS.md")
            else:
                logger.warning(f"Failed to initialize translation packages from disk: {str(e)[:200]}")
                logger.info("Translation packages must be pre-installed. See DOWNLOAD_TRANSLATION_MODELS.md for instructions.")
            
            # Service will still start but translation may fail if packages missing
            if self._installed_packages:
                logger.info(f"Found {len(self._installed_packages)} pre-installed packages from disk")
            else:
                logger.warning("No translation packages found. Translation features will not work until packages are installed.")
        finally:
            self._initialization_attempted = True
    
    def _get_packages_storage_path(self) -> str:
        """
        Get the translation packages storage path for the current platform.
        
        Returns platform-specific path:
        - macOS/Linux: ~/.local/share/argos-translate/packages
        - Windows: C:\\Users\\username\\.local\\share\\argos-translate\\packages
        
        Returns:
            Full path to translation packages directory
        """
        import platform
        system = platform.system()
        
        if system == "Windows":
            # Windows: C:\Users\username\.local\share\argos-translate\packages
            packages_path = os.path.join(
                os.path.expanduser("~"),
                ".local",
                "share",
                "argos-translate",
                "packages"
            )
        else:
            # macOS/Linux: ~/.local/share/argos-translate/packages
            packages_path = os.path.expanduser("~/.local/share/argos-translate/packages")
        
        return packages_path
    
    def _load_installed_packages_from_disk(self) -> None:
        """
        Load installed packages list from disk (works offline).
        
        This method checks for already-installed packages from Argos Translate's
        local storage and loads them. This allows the service to work offline
        if packages were previously installed.
        
        Packages are stored in:
        - macOS/Linux: ~/.local/share/argos-translate/packages
        - Windows: C:\\Users\\username\\.local\\share\\argos-translate\\packages
        """
        try:
            # Get currently installed packages from Argos Translate (reads from disk, no internet needed)
            # This may trigger spacy/stanza initialization, but we've disabled their downloads
            installed_packages_list = argostranslate.package.get_installed_packages()
            installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages_list]
            
            # Get package storage location for logging (platform-specific)
            packages_storage_path = self._get_packages_storage_path()
            
            # Filter to only include supported pairs
            self._installed_packages = [
                (from_code, to_code) for from_code, to_code in installed_pairs
                if (from_code, to_code) in self.SUPPORTED_PAIRS
            ]
            
            if self._installed_packages:
                logger.info(
                    f"Loaded {len(self._installed_packages)} installed packages from disk: {self._installed_packages}"
                )
                logger.info(f"Translation packages storage location: {packages_storage_path}")
                # Save to manifest file for reference
                self._save_packages_manifest()
            else:
                logger.info("No installed translation packages found on disk")
                logger.info(f"Translation packages will be stored at: {packages_storage_path}")
                
        except Exception as e:
            error_str = str(e).lower()
            # Handle spacy/stanza/ssl errors gracefully - they're non-critical
            if any(x in error_str for x in ['spacy', 'stanza', 'ssl', 'certificate', '443', 'download', 'not initialized']):
                logger.debug(f"Non-critical dependency warning (spacy/stanza/ssl): {str(e)[:200]}")
                # Try to continue - packages may still be readable
                try:
                    packages_storage_path = self._get_packages_storage_path()
                    # Check if packages directory exists manually
                    if os.path.exists(packages_storage_path):
                        # Try to list packages manually
                        pkg_dirs = [d for d in os.listdir(packages_storage_path) 
                                   if os.path.isdir(os.path.join(packages_storage_path, d))]
                        if pkg_dirs:
                            logger.info(f"Found package directories manually: {pkg_dirs}")
                            # Try to extract language pairs from directory names
                            for pkg_dir in pkg_dirs:
                                if '_' in pkg_dir:
                                    parts = pkg_dir.split('_')
                                    if len(parts) == 2:
                                        pair = (parts[0], parts[1])
                                        if pair in self.SUPPORTED_PAIRS:
                                            if pair not in self._installed_packages:
                                                self._installed_packages.append(pair)
                            if self._installed_packages:
                                logger.info(f"Loaded {len(self._installed_packages)} packages from directory listing")
                except:
                    pass
            
            if not self._installed_packages:
                logger.warning(f"Failed to load installed packages: {str(e)[:200]}")
                self._installed_packages = []
    
    def _save_packages_manifest(self) -> None:
        """
        Save installed packages list to a manifest file for persistence.
        
        This file can be used to verify installed packages and works offline.
        """
        try:
            manifest_data = {
                "installed_packages": self._installed_packages,
                "supported_pairs": self.SUPPORTED_PAIRS,
                "timestamp": time.time()
            }
            with open(self._packages_manifest_file, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            logger.debug(f"Saved packages manifest to {self._packages_manifest_file}")
        except Exception as e:
            logger.warning(f"Failed to save packages manifest: {str(e)}")
    
    def _initialize_translation_packages(self) -> None:
        """
        Initialize translation packages from local disk only (fully offline).
        
        This method:
        1. Checks for already installed packages from disk (offline)
        2. Updates the internal list and saves to manifest file
        
        This method works completely offline and does NOT attempt to connect
        to the internet. Packages must be pre-installed on the system.
        
        If you need to install new packages, do so manually on a machine with
        internet access before deploying to an offline server.
        """
        try:
            # Step 1: Get currently installed packages from disk (works offline)
            installed_packages_list = argostranslate.package.get_installed_packages()
            installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages_list]
            packages_storage_path = self._get_packages_storage_path()
            logger.info(f"Found {len(installed_pairs)} already installed packages on disk: {installed_pairs}")
            logger.info(f"Translation packages storage location: {packages_storage_path}")
            
            # Log details of installed packages for debugging
            for pkg in installed_packages_list:
                logger.debug(f"Installed package: {pkg.from_code} -> {pkg.to_code} (package: {pkg})")
            
            # Step 2: Filter to only include supported pairs (no internet access)
            installed_packages = []
            for from_code, to_code in self.SUPPORTED_PAIRS:
                # Check if already installed
                if (from_code, to_code) in installed_pairs:
                    installed_packages.append((from_code, to_code))
                    package_dir = os.path.join(packages_storage_path, f"{from_code}_{to_code}")
                    logger.info(
                        f"Translation package available: {from_code} -> {to_code} "
                        f"(location: {package_dir})"
                    )
                else:
                    # Package not installed - log warning but don't try to download
                    logger.warning(
                        f"Translation package not installed: {from_code} -> {to_code}. "
                        f"Install manually if needed."
                    )
            
            # Update internal tracking of installed packages
            self._installed_packages = installed_packages
            # Save to manifest file for persistence
            self._save_packages_manifest()
            logger.info(f"Translation service initialized (offline mode) with {len(installed_packages)} packages: {installed_packages}")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation packages: {str(e)}", exc_info=True)
            logger.warning("Translation will work with limited language pairs")
            # Don't set to empty list if we already have some packages installed
            if not hasattr(self, '_installed_packages') or not self._installed_packages:
                self._installed_packages = []
    
    def _check_package_installed(self, from_code: str, to_code: str) -> bool:
        """
        Check if a translation package is installed locally (offline only).
        
        This method checks if a package is already installed on the local system.
        It does NOT attempt to download packages from the internet.
        
        Args:
            from_code: Source language code (e.g., "en", "hi", "ko")
            to_code: Target language code (e.g., "en", "hi", "ko")
            
        Returns:
            True if package is installed, False otherwise
        """
        try:
            # Check if package is already installed from disk (works offline)
            installed_packages_list = argostranslate.package.get_installed_packages()
            installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages_list]
            
            if (from_code, to_code) in installed_pairs:
                # Update internal tracking if not already tracked
                if (from_code, to_code) not in self._installed_packages:
                    self._installed_packages.append((from_code, to_code))
                    self._save_packages_manifest()
                packages_storage_path = self._get_packages_storage_path()
                package_dir = os.path.join(packages_storage_path, f"{from_code}_{to_code}")
                logger.info(
                    f"Translation package found: {from_code} -> {to_code} "
                    f"(location: {package_dir})"
                )
                return True
            
            # Package not installed - return False (do NOT try to download)
            logger.warning(
                f"Translation package not installed: {from_code} -> {to_code}. "
                f"This server runs in offline mode. Install packages manually if needed."
            )
            return False
                
        except Exception as e:
            logger.error(f"Failed to check package {from_code}->{to_code}: {str(e)}", exc_info=True)
            return False
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Tuple of (detected_language_code, confidence)
            
        Raises:
            ValueError: If language detection fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Step 1: Get primary language detection using langdetect
            detected_lang = detect(text)
            
            # Step 2: Get confidence scores for all detected languages
            lang_probs = detect_langs(text)
            # Extract confidence for the detected language
            confidence = next(
                (prob.prob for prob in lang_probs if prob.lang == detected_lang),
                0.0
            )
            
            # Step 3: Map detected language to supported languages if needed
            # langdetect may return language codes with regional variants (e.g., "en-US", "hi-IN")
            if detected_lang not in self.LANGUAGE_MAP:
                # Try to find closest match by checking language prefix
                if detected_lang.startswith("en"):
                    detected_lang = "en"
                elif detected_lang.startswith("hi"):
                    detected_lang = "hi"
                elif detected_lang.startswith("ko"):
                    detected_lang = "ko"
                else:
                    # Default to English if language is completely unknown
                    detected_lang = "en"
                    confidence = 0.5  # Low confidence for unknown language
            
            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
            return detected_lang, confidence
            
        except LangDetectException as e:
            # Handle language detection library exceptions gracefully
            logger.warning(f"Language detection failed: {str(e)}, defaulting to English")
            return "en", 0.5
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error in language detection: {str(e)}")
            raise ValueError(f"Language detection failed: {str(e)}")
    
    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: str = "en"
    ) -> Tuple[str, str, Dict]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code (en, hi, ko) or None for auto-detect
            target_language: Target language code (en, hi, ko)
            
        Returns:
            Tuple of (translated_text, detected_source_language, metadata)
            
        Raises:
            ValueError: If translation fails or language pair is not supported
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Step 1: Validate target language
        if target_language not in self.LANGUAGE_MAP:
            raise ValueError(f"Unsupported target language: {target_language}")
        
        # Step 2: Detect source language if not provided or set to "auto"
        if source_language is None or source_language == "auto":
            detected_lang, confidence = self.detect_language(text)
            source_language = detected_lang
        else:
            # Validate source language if explicitly provided
            if source_language not in self.LANGUAGE_MAP:
                raise ValueError(f"Unsupported source language: {source_language}")
            confidence = 1.0  # Full confidence when language is explicitly specified
        
        # Step 3: Check if source and target are the same (no translation needed)
        if source_language == target_language:
            logger.info("Source and target languages are the same, returning original text")
            return text, source_language, {
                "translation_applied": False,
                "confidence": confidence,
                "processing_time_sec": 0.0
            }
        
        # Step 4: Check if translation pair is supported
        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            raise ValueError(
                f"Translation from {source_language} to {target_language} is not supported. "
                f"Supported pairs: {self.SUPPORTED_PAIRS}"
            )
        
        # Step 5: Ensure translation packages are initialized
        # Reload packages from disk if not already initialized
        if not self._installed_packages:
            logger.info("No translation packages found. Checking disk for installed packages...")
            try:
                self._initialize_translation_packages()
            except Exception as e:
                logger.error(f"Failed to load translation packages: {str(e)}")
                raise ValueError(
                    f"Translation packages are not installed. "
                    f"This server runs in offline mode. Please install packages manually. "
                    f"Error: {str(e)}"
                )
        
        # Step 6: Verify the specific pair is installed (offline mode - no download attempts)
        if (source_language, target_language) not in self._installed_packages:
            logger.info(f"Translation package for {source_language} -> {target_language} not in cache. Checking disk...")
            if not self._check_package_installed(source_language, target_language):
                raise ValueError(
                    f"Translation package for {source_language} -> {target_language} is not installed. "
                    f"Installed packages: {self._installed_packages}. "
                    f"This server runs in offline mode. Install packages manually on a machine with internet access."
                )
        
        try:
            start_time = time.time()
            
            # Step 7: Get installed packages from argostranslate
            # This ensures we have the latest list after potential installations
            installed_packages = argostranslate.package.get_installed_packages()
            
            # Step 8: Find the package object for this specific translation pair
            installed_packages_list = argostranslate.package.get_installed_packages()
            package = next(
                (pkg for pkg in installed_packages_list 
                 if pkg.from_code == source_language and pkg.to_code == target_language),
                None
            )
            
            # Step 9: If package not found, refresh the list and try again
            # This handles cases where packages were just installed
            if package is None:
                logger.warning(f"Package {source_language} -> {target_language} not found in installed packages. Refreshing...")
                installed_packages_list = argostranslate.package.get_installed_packages()
                package = next(
                    (pkg for pkg in installed_packages_list 
                     if pkg.from_code == source_language and pkg.to_code == target_language),
                    None
                )
                
                if package is None:
                    raise ValueError(
                        f"Translation package for {source_language} -> {target_language} not found. "
                        f"Installed packages: {[(p.from_code, p.to_code) for p in installed_packages_list]}. "
                        f"Please install the required package."
                    )
            
            # Step 10: Perform the actual translation
            try:
                # Try using the package's translate method first (preferred method)
                translated_text = package.translate(text)
            except AttributeError as e:
                # Fallback to using argostranslate.translate.translate() if package.translate doesn't work
                # This handles different versions of argostranslate that may have different APIs
                logger.warning(f"package.translate() failed, trying argostranslate.translate.translate(): {str(e)}")
                try:
                    translated_text = argostranslate.translate.translate(text, source_language, target_language)
                except Exception as e2:
                    raise ValueError(
                        f"Translation failed: {str(e2)}. "
                        f"Package: {package}, from_code: {source_language}, to_code: {target_language}"
                    )
            
            # Step 11: Calculate processing time and build metadata
            processing_time = time.time() - start_time
            
            metadata = {
                "source_language": source_language,
                "target_language": target_language,
                "detection_confidence": confidence,
                "processing_time_sec": processing_time,
                "translation_applied": True,
                "original_length": len(text),
                "translated_length": len(translated_text)
            }
            
            logger.info(
                f"Translated text from {source_language} to {target_language} "
                f"({processing_time:.3f}s)"
            )
            
            return translated_text, source_language, metadata
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise ValueError(f"Translation failed: {str(e)}")
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported language codes.
        
        Returns:
            List of supported language codes (e.g., ["en", "hi", "ko", "auto"])
        """
        return list(self.LANGUAGE_MAP.keys())
    
    def get_supported_pairs(self) -> list:
        """
        Get list of supported translation pairs.
        
        Returns:
            List of tuples representing (source_language, target_language) pairs
        """
        return self.SUPPORTED_PAIRS.copy()
    
    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if a translation pair is supported.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if the translation pair is supported, False otherwise
        """
        return (source_lang, target_lang) in self.SUPPORTED_PAIRS

