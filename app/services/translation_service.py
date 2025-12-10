"""
Translation Service Module

This module provides offline translation capabilities using Argos Translate and language
detection using langdetect. It supports translation between English, Hindi, Korean, and Urdu.

For Urdu, Hindi, and Korean sources, translations are done through English:
1. First translate source language to English
2. Then translate English to all three target languages (English, Hindi, Korean)

The service works fully offline and requires pre-installed translation packages.

Author: Debarun Lahiri
"""

import os
import warnings

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
# Note: ssl.SSLError is an Exception, not a Warning, so can't be filtered here
# SSL errors will be caught in try-except blocks instead

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

# Initialize logger first
logger = logging.getLogger(__name__)

# Import transformers for NLLB-200 (primary translation engine)
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    logger.warning("transformers library not available. NLLB-200 translation will not work. Install with: pip install transformers sentencepiece")

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

ARGOS_AVAILABLE = False
try:
    import argostranslate.package
    import argostranslate.translate
    ARGOS_AVAILABLE = True
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
        ARGOS_AVAILABLE = True
    except:
        # If it still fails, we'll handle it in the service initialization
        ARGOS_AVAILABLE = False
        pass

from app.config import settings


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
    - Urdu (ur)
    
    Supported translation pairs:
    - English ↔ Hindi
    - English ↔ Korean
    - English ↔ Urdu
    - Hindi ↔ English
    - Korean ↔ English
    - Urdu ↔ English
    - Hindi ↔ Korean (via English)
    - Korean ↔ Hindi (via English)
    
    Author: Debarun Lahiri
    """
    
    # Language code mapping: Maps language codes to their display names
    # Used for validation and display purposes
    LANGUAGE_MAP = {
        "en": "English",
        "hi": "Hindi",
        "ko": "Korean",
        "ur": "Urdu",
        "auto": "auto"  # Special value for auto-detection
    }
    
    # NLLB-200 language code mapping: Maps our language codes to NLLB-200 language codes
    # NLLB-200 uses specific language codes with script indicators
    NLLB_LANGUAGE_MAP = {
        "en": "eng_Latn",  # English (Latin script)
        "hi": "hin_Deva",  # Hindi (Devanagari script)
        "ko": "kor_Hang",  # Korean (Hangul script)
        "ur": "urd_Arab",  # Urdu (Arabic script)
    }
    
    # Supported translation pairs: All bidirectional translation pairs supported by the service
    # Each tuple represents (source_language, target_language)
    # Note: For ur/hi/ko sources, translations go through English first
    SUPPORTED_PAIRS = [
        ("en", "hi"),  # English to Hindi
        ("hi", "en"),  # Hindi to English
        ("en", "ko"),  # English to Korean
        ("ko", "en"),  # Korean to English
        ("en", "ur"),  # English to Urdu
        ("ur", "en"),  # Urdu to English
        ("hi", "ko"),  # Hindi to Korean (via English)
        ("ko", "hi"),  # Korean to Hindi (via English)
    ]
    
    def __init__(self):
        """
        Initialize the TranslationService in fully offline mode.
        
        This service works completely offline - it does NOT connect to the internet.
        Translation packages must be pre-installed on the system before running.
        
        Uses NLLB-200 as primary translation engine (more accurate) with Argos Translate as fallback.
        
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
        
        # NLLB-200 model initialization
        self._nllb_pipeline = None
        self._nllb_tokenizer = None
        self._nllb_model = None
        self._use_nllb = settings.use_nllb_translation and TRANSFORMERS_AVAILABLE
        
        # Initialize NLLB-200 if enabled (WARNING: Resource intensive!)
        if self._use_nllb:
            logger.warning(
                "NLLB-200 is enabled. This requires 8GB+ free RAM and 2GB+ disk space. "
                "If you experience issues, set use_nllb_translation=False in config."
            )
            try:
                self._initialize_nllb_model()
                logger.info("NLLB-200 initialized successfully and will be used for translations")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Failed to initialize NLLB-200 model: {error_msg[:300]}. "
                    f"Falling back to Argos Translate automatically."
                )
                logger.info("To permanently disable NLLB-200, set use_nllb_translation=False in config.py")
                self._use_nllb = False
        else:
            logger.info("NLLB-200 is disabled. Using Argos Translate for translations (lighter, more suitable for 16GB RAM systems).")
        
        # Step 1: Load installed packages from disk (fully offline) - for Argos fallback
        # This reads from local storage only - no internet required
        # Error handling is done inside _load_installed_packages_from_disk
        if not self._use_nllb or ARGOS_AVAILABLE:
            self._load_installed_packages_from_disk()
        
        # Step 2: Initialize packages from local disk only (fully offline) - for Argos fallback
        # This does NOT attempt to connect to the internet
        if not self._use_nllb or ARGOS_AVAILABLE:
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
    
    def _initialize_nllb_model(self) -> None:
        """
        Initialize NLLB-200 model for offline translation.
        
        This method loads the NLLB-200 model from local path or HuggingFace cache.
        Works completely offline if model is already downloaded.
        
        WARNING: NLLB-200 requires significant resources:
        - RAM: 8GB+ free memory recommended (model size ~2.4GB in memory)
        - Disk: 2GB+ for model files
        - Not recommended for systems with less than 32GB total RAM
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is not installed. Install with: pip install transformers sentencepiece")
        
        # Check available memory (warning only, don't block)
        # psutil is optional - if not installed, skip memory check
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_memory_gb < 32:
                logger.warning(
                    f"System has {total_memory_gb:.1f}GB total RAM. "
                    f"NLLB-200 requires 8GB+ free RAM and is not recommended for systems with <32GB total RAM. "
                    f"Consider using Argos Translate instead (set use_nllb_translation=False)."
                )
            
            if available_memory_gb < 8:
                logger.error(
                    f"Only {available_memory_gb:.1f}GB free RAM available. "
                    f"NLLB-200 requires 8GB+ free RAM. "
                    f"Falling back to Argos Translate."
                )
                raise RuntimeError(f"Insufficient memory: {available_memory_gb:.1f}GB free (requires 8GB+)")
                
            logger.info(f"System memory check passed: {available_memory_gb:.1f}GB free of {total_memory_gb:.1f}GB total")
        except ImportError:
            # psutil not available, skip memory check (optional dependency)
            logger.debug("psutil not available, skipping memory check. Install with: pip install psutil")
        except RuntimeError:
            # Re-raise memory errors
            raise
        except Exception as e:
            logger.warning(f"Memory check failed: {str(e)}. Proceeding with caution.")
        
        model_path = settings.nllb_model_path
        cache_dir = settings.nllb_cache_dir
        
        # Check if model_path is a local path
        local_path = Path(model_path)
        if local_path.exists() and local_path.is_dir():
            # Use local path directly
            logger.info(f"Loading NLLB-200 model from local path: {model_path}")
            model_path = str(local_path)
            use_local_files_only = True
        else:
            # Check if model exists in cache
            try:
                from transformers.utils import TRANSFORMERS_CACHE
                default_cache = TRANSFORMERS_CACHE
            except ImportError:
                # Fallback to default HuggingFace cache location
                default_cache = os.path.expanduser("~/.cache/huggingface")
            
            cache_path = Path(cache_dir) if cache_dir else Path(default_cache)
            model_cache_path = cache_path / "models--facebook--nllb-200-distilled-600M"
            
            if model_cache_path.exists():
                logger.info(f"Loading NLLB-200 model from cache: {model_cache_path}")
                # Find the snapshot directory
                snapshots_dir = model_cache_path / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        model_path = str(snapshots[0])
                        use_local_files_only = True
                    else:
                        use_local_files_only = False
                else:
                    use_local_files_only = False
            else:
                logger.warning(f"NLLB-200 model not found in cache. Will attempt to load from HuggingFace (may require internet).")
                logger.warning(f"Cache directory: {cache_path}")
                use_local_files_only = False
        
        try:
            logger.info(f"Initializing NLLB-200 tokenizer and model...")
            logger.warning(
                "NLLB-200 is resource-intensive. If you experience issues, "
                "set use_nllb_translation=False in config to use Argos Translate instead."
            )
            
            # Load tokenizer with low_cpu_mem_usage to reduce memory footprint
            self._nllb_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=use_local_files_only
            )
            
            # Load model with optimizations for lower memory usage
            # Use low_cpu_mem_usage and torch_dtype to reduce memory footprint
            try:
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    local_files_only=use_local_files_only,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            except Exception as load_error:
                # If loading with optimizations fails, try standard loading
                logger.warning(f"Optimized loading failed, trying standard loading: {str(load_error)}")
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    local_files_only=use_local_files_only
                )
            
            # Move model to CPU explicitly (M2 Mac doesn't have CUDA, use CPU)
            if settings.device == "cpu" or not torch.cuda.is_available():
                self._nllb_model = self._nllb_model.to("cpu")
                logger.info("NLLB-200 model loaded on CPU")
            else:
                self._nllb_model = self._nllb_model.to("cuda")
                logger.info("NLLB-200 model loaded on CUDA")
            
            # Set pipeline to True to indicate model is loaded (we'll use model directly)
            self._nllb_pipeline = True  # Use as a flag to indicate model is loaded
            
            logger.info("NLLB-200 model initialized successfully")
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "memory" in error_msg.lower():
                logger.error(
                    f"NLLB-200 failed due to insufficient memory: {error_msg}. "
                    f"Please set use_nllb_translation=False in config to use Argos Translate instead."
                )
            raise
        except Exception as e:
            logger.error(
                f"Failed to initialize NLLB-200 model: {str(e)}. "
                f"Falling back to Argos Translate. To disable NLLB-200, set use_nllb_translation=False in config."
            )
            raise
    
    def _translate_with_nllb(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """
        Translate text using NLLB-200 model.
        
        Args:
            text: Text to translate
            source_language: Source language code (en, hi, ko, ur)
            target_language: Target language code (en, hi, ko, ur)
            
        Returns:
            Translated text
        """
        if not self._nllb_pipeline:
            raise ValueError("NLLB-200 model is not initialized")
        
        # Map language codes to NLLB-200 codes
        src_nllb = self.NLLB_LANGUAGE_MAP.get(source_language)
        tgt_nllb = self.NLLB_LANGUAGE_MAP.get(target_language)
        
        if not src_nllb or not tgt_nllb:
            raise ValueError(f"Unsupported language pair for NLLB-200: {source_language} -> {target_language}")
        
        try:
            # Use model and tokenizer directly for translation
            # Set the source and target language tokens
            self._nllb_tokenizer.src_lang = src_nllb
            
            # Tokenize the input text
            inputs = self._nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to the same device as the model
            if settings.device == "cuda" and next(self._nllb_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate translation
            if torch is None:
                raise ImportError("torch is not available")
            with torch.no_grad():
                generated_tokens = self._nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=self._nllb_tokenizer.lang_code_to_id[tgt_nllb],
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the generated tokens
            translated_text = self._nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            return translated_text
        except Exception as e:
            logger.error(f"NLLB-200 translation failed: {str(e)}")
            raise ValueError(f"NLLB-200 translation failed: {str(e)}")
    
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
            from_code: Source language code (e.g., "en", "hi", "ko", "ur")
            to_code: Target language code (e.g., "en", "hi", "ko", "ur")
            
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
                elif detected_lang.startswith("ur"):
                    detected_lang = "ur"
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
            source_language: Source language code (en, hi, ko, ur) or None for auto-detect
            target_language: Target language code (en, hi, ko, ur)
            
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
        
        start_time = time.time()
        translation_engine = None
        
        # Step 5: Try NLLB-200 first (if enabled and available)
        if self._use_nllb and self._nllb_pipeline:
            try:
                logger.info(f"Translating with NLLB-200: {source_language} -> {target_language}")
                translated_text = self._translate_with_nllb(text, source_language, target_language)
                translation_engine = "nllb-200"
                logger.info(f"NLLB-200 translation successful")
            except Exception as e:
                logger.warning(f"NLLB-200 translation failed: {str(e)}. Falling back to Argos Translate.")
                translation_engine = None
        
        # Step 6: Fallback to Argos Translate if NLLB-200 failed or is not available
        if translation_engine is None:
            if not ARGOS_AVAILABLE:
                raise ValueError(
                    "Neither NLLB-200 nor Argos Translate is available. "
                    "Please ensure at least one translation engine is properly configured."
                )
            
            # Ensure translation packages are initialized
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
            
            # Verify the specific pair is installed (offline mode - no download attempts)
            if (source_language, target_language) not in self._installed_packages:
                logger.info(f"Translation package for {source_language} -> {target_language} not in cache. Checking disk...")
                if not self._check_package_installed(source_language, target_language):
                    raise ValueError(
                        f"Translation package for {source_language} -> {target_language} is not installed. "
                        f"Installed packages: {self._installed_packages}. "
                        f"This server runs in offline mode. Install packages manually on a machine with internet access."
                    )
            
            try:
                logger.info(f"Translating with Argos Translate: {source_language} -> {target_language}")
                
                # Get installed packages from argostranslate
                installed_packages_list = argostranslate.package.get_installed_packages()
                package = next(
                    (pkg for pkg in installed_packages_list 
                     if pkg.from_code == source_language and pkg.to_code == target_language),
                    None
                )
                
                # If package not found, refresh the list and try again
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
                
                # Perform the actual translation
                try:
                    # Try using the package's translate method first (preferred method)
                    translated_text = package.translate(text)
                except AttributeError as e:
                    # Fallback to using argostranslate.translate.translate() if package.translate doesn't work
                    logger.warning(f"package.translate() failed, trying argostranslate.translate.translate(): {str(e)}")
                    try:
                        translated_text = argostranslate.translate.translate(text, source_language, target_language)
                    except Exception as e2:
                        raise ValueError(
                            f"Translation failed: {str(e2)}. "
                            f"Package: {package}, from_code: {source_language}, to_code: {target_language}"
                        )
                
                translation_engine = "argos-translate"
                logger.info(f"Argos Translate translation successful")
                
            except Exception as e:
                logger.error(f"Argos Translate translation failed: {str(e)}")
                raise ValueError(f"Translation failed: {str(e)}")
        
        # Step 7: Calculate processing time and build metadata
        processing_time = time.time() - start_time
        
        metadata = {
            "source_language": source_language,
            "target_language": target_language,
            "detection_confidence": confidence,
            "processing_time_sec": processing_time,
            "translation_applied": True,
            "translation_engine": translation_engine,
            "original_length": len(text),
            "translated_length": len(translated_text)
        }
        
        logger.info(
            f"Translated text from {source_language} to {target_language} "
            f"using {translation_engine} ({processing_time:.3f}s)"
        )
        
        return translated_text, source_language, metadata
    
    def translate_to_all_languages(
        self,
        text: str,
        source_language: Optional[str] = None
    ) -> Tuple[str, str, str, str, Dict]:
        """
        Translate text to all three target languages (English, Hindi, Korean).
        
        If the source language is Urdu, Hindi, or Korean, this method will:
        1. First translate the text to English
        2. Then translate English to English, Hindi, and Korean
        
        If the source language is English, it will translate directly to all three languages.
        
        Args:
            text: Text to translate
            source_language: Source language code (en, hi, ko, ur) or None for auto-detect
            
        Returns:
            Tuple of (english_text, hindi_text, korean_text, detected_source_language, metadata)
            
        Raises:
            ValueError: If translation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        start_time = time.time()
        
        # Step 1: Detect source language if not provided
        if source_language is None or source_language == "auto":
            detected_lang, confidence = self.detect_language(text)
            source_language = detected_lang
        else:
            # Validate source language
            if source_language not in self.LANGUAGE_MAP:
                raise ValueError(f"Unsupported source language: {source_language}")
            confidence = 1.0
        
        # Step 2: If source is Urdu, Hindi, or Korean, translate to English first
        intermediate_text = text
        intermediate_lang = source_language
        
        if source_language in ["ur", "hi", "ko"]:
            logger.info(f"Source language is {source_language}, translating to English first...")
            try:
                intermediate_text, _, _ = self.translate(
                    text=text,
                    source_language=source_language,
                    target_language="en"
                )
                intermediate_lang = "en"
                logger.info(f"Successfully translated {source_language} -> English")
            except Exception as e:
                logger.error(f"Failed to translate {source_language} to English: {str(e)}")
                raise ValueError(f"Failed to translate {source_language} to English: {str(e)}")
        
        # Step 3: Translate from English to all three languages
        # At this point, intermediate_text is in English (either original or translated)
        # and intermediate_lang is "en"
        en_text = intermediate_text  # English is already the intermediate text
        
        try:
            # Translate English to Hindi
            hi_text, _, _ = self.translate(
                text=intermediate_text,
                source_language="en",
                target_language="hi"
            )
            
            # Translate English to Korean
            ko_text, _, _ = self.translate(
                text=intermediate_text,
                source_language="en",
                target_language="ko"
            )
            
        except Exception as e:
            logger.error(f"Failed to translate to all languages: {str(e)}")
            raise ValueError(f"Failed to translate to all languages: {str(e)}")
        
        processing_time = time.time() - start_time
        
        metadata = {
            "source_language": source_language,
            "intermediate_language": intermediate_lang,
            "detection_confidence": confidence,
            "processing_time_sec": processing_time,
            "translation_applied": True,
            "original_length": len(text),
            "english_length": len(en_text),
            "hindi_length": len(hi_text),
            "korean_length": len(ko_text)
        }
        
        logger.info(
            f"Translated text from {source_language} to all languages "
            f"(en, hi, ko) in {processing_time:.3f}s"
        )
        
        return en_text, hi_text, ko_text, source_language, metadata
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported language codes.
        
        Returns:
            List of supported language codes (e.g., ["en", "hi", "ko", "ur", "auto"])
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

