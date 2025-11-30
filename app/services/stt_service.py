"""
Speech-to-Text Service Module

This module provides offline speech-to-text transcription capabilities using the
faster-whisper library (an optimized implementation of OpenAI's Whisper model).

The service implements a model pool pattern to enable concurrent processing of
multiple audio files simultaneously, improving throughput and resource utilization.

Features:
- Multi-language support (English, Hindi, Korean)
- Word-level timestamps
- Automatic language detection
- GPU acceleration support
- Thread-safe concurrent processing

Author: Debarun Lahiri
"""

import os
import logging
import time
import threading
from typing import Optional, List, Tuple, Dict
import numpy as np
from faster_whisper import WhisperModel
import torch

from app.config import settings
from app.models.schemas import Segment, WordTimestamp, LanguageCode

logger = logging.getLogger(__name__)


class ModelPool:
    """
    Thread-safe pool of WhisperModel instances for concurrent processing.
    
    This class manages a pool of pre-loaded Whisper model instances to enable
    concurrent transcription requests. Models are distributed using round-robin
    selection, ensuring load balancing across available model instances.
    
    The pool pattern significantly improves throughput by allowing multiple
    requests to be processed simultaneously without blocking each other.
    
    Author: Debarun Lahiri
    """
    
    def __init__(self, pool_size: int, model_size: str, device: str, compute_type: str, model_cache_dir: str, model_local_path: Optional[str] = None):
        """
        Initialize the ModelPool.
        
        Args:
            pool_size: Number of model instances to create in the pool
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to run models on ("cpu" or "cuda")
            compute_type: Compute type (int8 for CPU, int8_float16/float16/float32 for GPU)
            model_cache_dir: Directory to cache downloaded models
            model_local_path: Optional path to local model directory (for offline use without HuggingFace)
        """
        self.pool_size = pool_size
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_cache_dir = model_cache_dir
        self.model_local_path = model_local_path
        self.models: List[WhisperModel] = []  # List of loaded model instances
        self.current_index = 0  # Current index for round-robin selection
        self.index_lock = threading.Lock()  # Lock for thread-safe index access
        self._initialized = False  # Track initialization status
        
    def initialize(self) -> None:
        """
        Initialize all models in the pool.
        
        This method loads all Whisper model instances into memory. The initialization
        process may take significant time, especially on first run when models need
        to be downloaded. Subsequent runs will be faster as models are cached.
        
        The method automatically detects and configures device settings (CPU/GPU)
        and optimizes CPU thread allocation for best performance.
        
        Raises:
            RuntimeError: If model loading fails for any instance
        """
        if self._initialized:
            return
            
        logger.info(f"Initializing model pool with {self.pool_size} instances...")
        start_time = time.time()
        
        # Step 1: Auto-detect and validate device configuration
        device = self.device
        if device == "cpu":
            if torch.cuda.is_available():
                logger.warning(
                    "CUDA is available but device is set to CPU. "
                    "Set device=cuda in config to use GPU."
                )
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        
        # Step 2: Create model cache directory if it doesn't exist
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Step 3: Optimize CPU threads for best performance
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use optimal CPU threads: for CPU inference, use all cores minus 1 for system
        # For each model instance, we want good parallelism but not oversubscription
        # For GPU inference, use fewer threads (4) as GPU handles most computation
        optimal_threads = max(1, cpu_count - 1) if device == "cpu" else 4
        if settings.cpu_threads > 0:
            optimal_threads = settings.cpu_threads
        
        logger.info(f"Using {optimal_threads} CPU threads per model instance (total CPU cores: {cpu_count})")
        
        # Step 4: Determine model path - use local path if available, otherwise use model size
        model_path_or_size = self.model_size
        if self.model_local_path:
            # Check if local model path exists
            if os.path.isdir(self.model_local_path):
                model_path_or_size = self.model_local_path
                logger.info(f"Using local model from: {self.model_local_path}")
                
                # Verify required model files exist
                required_files = ["model.bin", "config.json", "tokenizer.json", "vocabulary.json"]
                missing_files = []
                for f in required_files:
                    if not os.path.exists(os.path.join(self.model_local_path, f)):
                        missing_files.append(f)
                
                if missing_files:
                    logger.warning(
                        f"Some model files may be missing in {self.model_local_path}: {missing_files}. "
                        f"Model loading may fail."
                    )
            else:
                logger.error(
                    f"Local model path not found: {self.model_local_path}. "
                    f"Please ensure the model files are present in this directory. "
                    f"Required files: model.bin, config.json, tokenizer.json, vocabulary.json"
                )
                raise RuntimeError(
                    f"Local model path not found: {self.model_local_path}. "
                    f"Download the model files and place them in this directory."
                )
        else:
            logger.info(f"No local model path specified. Using model size: {self.model_size}")
            logger.warning(
                "This will attempt to download from HuggingFace. "
                "If you have no internet access, set model_local_path in config."
            )
        
        # Step 5: Load all model instances into the pool
        for i in range(self.pool_size):
            try:
                logger.info(f"Loading model instance {i+1}/{self.pool_size}...")
                model = WhisperModel(
                    model_path_or_size,
                    device=device,
                    compute_type=self.compute_type,
                    download_root=self.model_cache_dir,
                    cpu_threads=optimal_threads,
                    num_workers=1,
                    local_files_only=True if self.model_local_path else False
                )
                self.models.append(model)
                logger.info(f"Model instance {i+1}/{self.pool_size} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model instance {i+1}: {str(e)}")
                raise RuntimeError(f"Model pool initialization failed: {str(e)}")
        
        load_time = time.time() - start_time
        self._initialized = True
        logger.info(f"Model pool initialized successfully in {load_time:.2f} seconds with {self.pool_size} instances")
    
    def get_model(self) -> WhisperModel:
        """
        Get a model from the pool using round-robin selection (thread-safe).
        
        This method uses a round-robin algorithm to distribute requests evenly
        across all model instances in the pool. The selection is thread-safe,
        allowing concurrent requests to safely access different model instances.
        
        Returns:
            A WhisperModel instance from the pool
            
        Raises:
            RuntimeError: If the pool is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model pool not initialized. Call initialize() first.")
        
        # Thread-safe round-robin selection
        with self.index_lock:
            index = self.current_index
            self.current_index = (self.current_index + 1) % self.pool_size
        
        return self.models[index]
    
    def is_initialized(self) -> bool:
        """
        Check if pool is initialized.
        
        Returns:
            True if the pool is initialized and ready to use, False otherwise
        """
        return self._initialized


class STTService:
    """
    Speech-to-Text service using faster-whisper for offline transcription.
    
    This service provides high-quality speech-to-text transcription using OpenAI's
    Whisper model via the faster-whisper implementation. It supports multiple
    languages, word-level timestamps, and automatic language detection.
    
    The service uses a model pool to enable concurrent processing of multiple
    audio files, significantly improving throughput and resource utilization.
    
    Features:
    - Multi-language support (English, Hindi, Korean, and more)
    - Word-level timestamps for precise timing information
    - Automatic language detection
    - GPU acceleration support (CUDA)
    - Thread-safe concurrent processing
    
    Author: Debarun Lahiri
    """
    
    def __init__(self):
        """
        Initialize the STTService.
        
        The service is initialized with configuration from settings but does not
        load models until initialize_model() is called. This allows for lazy
        initialization to avoid long startup times.
        """
        self.model_pool: Optional[ModelPool] = None  # Model pool instance
        self.model_size = settings.model_size  # Whisper model size
        self.device = settings.device  # Device (cpu/cuda)
        self.compute_type = settings.compute_type  # Compute type for optimization
        self.model_cache_dir = settings.model_cache_dir  # Model cache directory
        self.model_local_path = settings.model_local_path  # Local model path for offline use
        self.pool_size = settings.model_pool_size  # Number of model instances
        self._is_loaded = False  # Track if models are loaded
        
    def initialize_model(self) -> None:
        """
        Initialize the model pool. This may take time on first run.
        
        This method creates and initializes the model pool with the configured
        number of Whisper model instances. On first run, models will be downloaded
        and cached. Subsequent runs will load from cache, which is faster.
        
        The initialization process:
        1. Creates a ModelPool instance
        2. Loads all model instances into memory
        3. Configures device and compute settings
        4. Optimizes CPU thread allocation
        
        Raises:
            RuntimeError: If model pool initialization fails
        """
        if self._is_loaded and self.model_pool is not None:
            logger.info("Model pool already initialized")
            return
            
        try:
            model_info = self.model_local_path if self.model_local_path else self.model_size
            logger.info(
                f"Initializing model pool: {model_info} "
                f"(device: {self.device}, compute_type: {self.compute_type}, pool_size: {self.pool_size})"
            )
            
            # Create and initialize model pool
            self.model_pool = ModelPool(
                pool_size=self.pool_size,
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                model_cache_dir=self.model_cache_dir,
                model_local_path=self.model_local_path
            )
            self.model_pool.initialize()
            self._is_loaded = True
            
            logger.info("Model pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model pool: {str(e)}")
            raise RuntimeError(f"Model pool initialization failed: {str(e)}")
    
    def transcribe(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = None,
        enable_word_timestamps: bool = True,
        **kwargs
    ) -> Tuple[str, List[Segment], str, Dict]:
        """
        Transcribe audio array to text.
        
        Args:
            audio_array: Audio array as numpy array (float32, mono, 16kHz)
            language: Language code (en, hi, ko) or None for auto-detect
            enable_word_timestamps: Whether to include word-level timestamps
            **kwargs: Additional transcription parameters
            
        Returns:
            Tuple of (full_text, segments, detected_language, metadata)
        """
        if not self._is_loaded or self.model_pool is None:
            raise RuntimeError("Model pool not initialized. Call initialize_model() first.")
        
        # Get a model from the pool (thread-safe round-robin selection)
        model = self.model_pool.get_model()
        
        try:
            # Prepare transcription parameters
            transcription_kwargs = {
                "beam_size": kwargs.get("beam_size", settings.beam_size),
                "best_of": kwargs.get("best_of", settings.best_of),
                "patience": kwargs.get("patience", settings.patience),
                "temperature": kwargs.get("temperature", settings.temperature),
                "condition_on_previous_text": kwargs.get(
                    "condition_on_previous_text",
                    settings.condition_on_previous_text
                ),
                "compression_ratio_threshold": kwargs.get(
                    "compression_ratio_threshold",
                    settings.compression_ratio_threshold
                ),
                "log_prob_threshold": kwargs.get(
                    "log_prob_threshold",
                    settings.log_prob_threshold
                ),
                "no_speech_threshold": kwargs.get(
                    "no_speech_threshold",
                    settings.no_speech_threshold
                ),
                "word_timestamps": enable_word_timestamps,
            }
            
            # Set language if specified
            if language and language != "auto":
                transcription_kwargs["language"] = language
                if language not in settings.supported_languages:
                    logger.warning(
                        f"Language {language} not in supported list. "
                        f"Proceeding with auto-detection."
                    )
                    transcription_kwargs.pop("language")
            
            # Set initial prompt if provided
            if settings.initial_prompt:
                transcription_kwargs["initial_prompt"] = settings.initial_prompt
            
            # Perform transcription (each model instance is independent, no lock needed)
            start_time = time.time()
            logger.info(f"Starting transcription (language={language}, word_timestamps={enable_word_timestamps})")
            
            segments_list, info = model.transcribe(
                audio_array,
                **transcription_kwargs
            )
            
            # Extract detected language
            detected_language = info.language or "unknown"
            language_probability = getattr(info, "language_probability", None)
            
            # CRITICAL: Handle Urdu detection - Urdu and Hindi share similar characteristics
            # If Urdu is detected and language was auto, force re-transcription with Hindi
            # This ensures we get proper Hindi transcription, not just Hindi language code
            # Only do this if language was not explicitly set to Hindi
            original_language_was_auto = (language is None or language == "auto")
            if (detected_language == "ur" or detected_language.startswith("ur")) and original_language_was_auto:
                logger.warning(
                    f"Urdu detected ({detected_language}). Urdu is not supported. "
                    f"Re-transcribing with Hindi (hi) forced."
                )
                
                # Force Hindi language for re-transcription
                transcription_kwargs["language"] = "hi"
                
                # Re-transcribe with Hindi forced to get proper Hindi transcription
                logger.info("Re-transcribing with Hindi language forced")
                segments_list, info = model.transcribe(
                    audio_array,
                    **transcription_kwargs
                )
                
                # Update detected language to Hindi
                detected_language = "hi"
                language_probability = getattr(info, "language_probability", None)
                logger.info("Re-transcription complete with Hindi language")
            
            # Map similar languages to supported ones
            # Whisper may detect language variants (e.g., "en-US", "hi-IN") that need
            # to be mapped to our supported base language codes
            if detected_language not in settings.supported_languages:
                # Map common language variants to supported base languages
                language_mapping = {
                    "ur": "hi",  # Urdu -> Hindi (should not happen after re-transcription)
                    "ur-PK": "hi",  # Urdu (Pakistan) -> Hindi
                    "ur-IN": "hi",  # Urdu (India) -> Hindi
                    "hi-IN": "hi",  # Hindi (India) -> Hindi
                    "en-US": "en",  # English (US) -> English
                    "en-GB": "en",  # English (UK) -> English
                    "ko-KR": "ko",  # Korean (South Korea) -> Korean
                }
                
                if detected_language in language_mapping:
                    mapped_lang = language_mapping[detected_language]
                    logger.info(f"Mapping detected language {detected_language} to {mapped_lang}")
                    detected_language = mapped_lang
                else:
                    # If still not supported, keep detected language but log warning
                    logger.warning(
                        f"Detected language {detected_language} not in supported list {settings.supported_languages}. "
                        f"Keeping as detected but may cause issues."
                    )
            
            # Process segments
            segments = []
            full_text_parts = []
            
            for segment in segments_list:
                segment_text = segment.text.strip()
                if not segment_text:
                    continue
                
                full_text_parts.append(segment_text)
                
                # Build word timestamps if available
                words = None
                if enable_word_timestamps and segment.words:
                    words = [
                        WordTimestamp(
                            word=word.word.strip(),
                            start=word.start,
                            end=word.end,
                            confidence=getattr(word, "probability", None)
                        )
                        for word in segment.words
                        if word.word.strip()
                    ]
                
                segments.append(
                    Segment(
                        start=segment.start,
                        end=segment.end,
                        text=segment_text,
                        words=words,
                        language=detected_language  # Use mapped language
                    )
                )
            
            full_text = " ".join(full_text_parts)
            processing_time = time.time() - start_time
            
            # Calculate metadata
            audio_duration = len(audio_array) / 16000.0  # Assuming 16kHz
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            metadata = {
                "processing_time_sec": processing_time,
                "real_time_factor": real_time_factor,
                "audio_duration_sec": audio_duration,
                "language_probability": language_probability,
                "word_count": len(full_text.split())
            }
            
            logger.info(
                f"Transcription completed: {detected_language}, "
                f"RTF: {real_time_factor:.3f}, words: {metadata['word_count']}"
            )
            
            return full_text, segments, detected_language, metadata
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """
        Check if model pool is loaded and ready to use.
        
        Returns:
            True if models are loaded and initialized, False otherwise
        """
        return self._is_loaded and self.model_pool is not None and self.model_pool.is_initialized()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model pool.
        
        Returns a dictionary containing:
        - Model size and configuration
        - Device information (CPU/GPU)
        - Pool size and initialization status
        - GPU information if available (name, memory usage)
        
        Returns:
            Dictionary with model pool information
        """
        info = {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "is_loaded": self._is_loaded,
            "pool_size": self.pool_size if self.model_pool else 0,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        
        return info

