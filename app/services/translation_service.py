"""
Translation Service Module

This module provides offline translation capabilities using Argos Translate and language
detection using langdetect. It supports translation between English, Hindi, and Korean.

The service automatically installs required translation packages on first use and
provides thread-safe translation operations.

Author: Debarun Lahiri
"""

import logging
import time
from typing import Optional, Dict, Tuple
from langdetect import detect, detect_langs, LangDetectException
import argostranslate.package
import argostranslate.translate

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
        Initialize the TranslationService.
        
        Attempts to initialize and install translation packages during startup.
        If initialization fails, packages will be installed on first use.
        This allows the service to start even if network is unavailable during startup.
        """
        self._installed_packages = []  # List of (from_code, to_code) tuples for installed packages
        self._initialization_attempted = False  # Track if initialization has been attempted
        
        # Try to initialize, but don't fail if it doesn't work
        # This allows the service to start even if packages can't be installed immediately
        try:
            self._initialize_translation_packages()
        except Exception as e:
            logger.error(f"Failed to initialize translation packages during startup: {str(e)}")
            logger.info("Translation packages will be installed on first use")
        finally:
            self._initialization_attempted = True
    
    def _initialize_translation_packages(self) -> None:
        """
        Initialize and install required translation packages.
        
        This method:
        1. Updates the Argos Translate package index
        2. Checks for already installed packages
        3. Downloads and installs missing packages for all supported language pairs
        4. Updates the internal list of installed packages
        
        The installation process may take time depending on network speed and
        package sizes. Packages are cached locally after installation.
        
        Raises:
            Exception: If package index update fails or critical installation errors occur
        """
        try:
            # Step 1: Update package index to get latest available packages
            logger.info("Updating argostranslate package index...")
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            logger.info(f"Found {len(available_packages)} available translation packages")
            
            # Step 2: Get currently installed packages to avoid re-installation
            installed_packages_list = argostranslate.package.get_installed_packages()
            installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages_list]
            logger.info(f"Found {len(installed_pairs)} already installed packages: {installed_pairs}")
            
            # Log details of installed packages for debugging
            for pkg in installed_packages_list:
                logger.debug(f"Installed package: {pkg.from_code} -> {pkg.to_code} (package: {pkg})")
            
            # Step 3: Install required packages for all supported language pairs
            installed_packages = []
            for from_code, to_code in self.SUPPORTED_PAIRS:
                # Check if already installed to avoid unnecessary downloads
                if (from_code, to_code) in installed_pairs:
                    installed_packages.append((from_code, to_code))
                    logger.info(f"Translation package already installed: {from_code} -> {to_code}")
                    continue
                
                # Try to find the package in available packages
                package_to_install = next(
                    (pkg for pkg in available_packages 
                     if pkg.from_code == from_code and pkg.to_code == to_code),
                    None
                )
                
                if package_to_install:
                    try:
                        logger.info(f"Installing translation package: {from_code} -> {to_code}...")
                        # Download the package file
                        package_path = package_to_install.download()
                        # Install the downloaded package
                        argostranslate.package.install_from_path(package_path)
                        installed_packages.append((from_code, to_code))
                        logger.info(f"Successfully installed translation package: {from_code} -> {to_code}")
                    except Exception as e:
                        logger.error(f"Failed to install package {from_code}->{to_code}: {str(e)}", exc_info=True)
                else:
                    logger.warning(f"Translation package not available: {from_code} -> {to_code}")
            
            # Update internal tracking of installed packages
            self._installed_packages = installed_packages
            logger.info(f"Translation service initialized with {len(installed_packages)} packages: {installed_packages}")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation packages: {str(e)}", exc_info=True)
            logger.warning("Translation will work with limited language pairs")
            # Don't set to empty list if we already have some packages installed
            if not hasattr(self, '_installed_packages') or not self._installed_packages:
                self._installed_packages = []
    
    def _install_single_package(self, from_code: str, to_code: str) -> bool:
        """
        Install a single translation package for a specific language pair.
        
        This method is called when a translation is requested for a pair that
        doesn't have an installed package yet. It attempts to download and
        install the package on-demand.
        
        Args:
            from_code: Source language code (e.g., "en", "hi", "ko")
            to_code: Target language code (e.g., "en", "hi", "ko")
            
        Returns:
            True if package was successfully installed or already exists, False otherwise
            
        Raises:
            Exception: If package installation fails
        """
        try:
            # Update package index to ensure we have latest package information
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            
            # Check if package is already installed to avoid redundant installation
            installed_packages_list = argostranslate.package.get_installed_packages()
            installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages_list]
            
            if (from_code, to_code) in installed_pairs:
                # Update internal tracking if not already tracked
                if (from_code, to_code) not in self._installed_packages:
                    self._installed_packages.append((from_code, to_code))
                logger.info(f"Translation package already installed: {from_code} -> {to_code}")
                return True
            
            # Find the package in available packages
            package_to_install = next(
                (pkg for pkg in available_packages 
                 if pkg.from_code == from_code and pkg.to_code == to_code),
                None
            )
            
            if package_to_install:
                logger.info(f"Installing translation package: {from_code} -> {to_code}...")
                # Download the package file to local cache
                package_path = package_to_install.download()
                # Install the downloaded package
                argostranslate.package.install_from_path(package_path)
                # Update internal tracking
                self._installed_packages.append((from_code, to_code))
                logger.info(f"Successfully installed translation package: {from_code} -> {to_code}")
                return True
            else:
                logger.warning(f"Translation package not available: {from_code} -> {to_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install package {from_code}->{to_code}: {str(e)}", exc_info=True)
            raise
    
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
        
        # Step 5: Ensure translation packages are installed
        # Try to install packages if not already installed
        if not self._installed_packages:
            logger.info("No translation packages found. Attempting to install required packages...")
            try:
                self._initialize_translation_packages()
            except Exception as e:
                logger.error(f"Failed to install translation packages: {str(e)}")
                raise ValueError(
                    f"Translation packages are not installed and installation failed: {str(e)}. "
                    "Please check server logs for details."
                )
        
        # Step 6: Verify the specific pair is installed, try to install if not
        if (source_language, target_language) not in self._installed_packages:
            logger.info(f"Translation package for {source_language} -> {target_language} not found. Attempting to install...")
            try:
                self._install_single_package(source_language, target_language)
            except Exception as e:
                logger.error(f"Failed to install package {source_language} -> {target_language}: {str(e)}")
                raise ValueError(
                    f"Translation package for {source_language} -> {target_language} is not installed. "
                    f"Installed packages: {self._installed_packages}. "
                    f"Installation error: {str(e)}"
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

