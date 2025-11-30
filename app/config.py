"""
Configuration Module

This module defines all application settings and configuration options using
Pydantic's BaseSettings. Settings can be loaded from environment variables,
a .env file, or use default values.

The configuration includes:
- Server settings (host, port, workers, reload)
- Model settings (size, device, compute type, pool size)
- Audio processing settings (sample rate, channels, duration limits)
- Transcription parameters (beam size, temperature, etc.)
- Logging and other service settings

Author: Debarun Lahiri
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """
    Application settings and configuration.
    
    This class defines all configurable parameters for the STT backend service.
    Settings can be overridden via environment variables or a .env file.
    
    All settings have sensible defaults that work out of the box, but can be
    customized based on system resources and requirements.
    
    Author: Debarun Lahiri
    """
    app_name: str = "Offline STT Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True  # Enable auto-reload on code changes
    reload_dirs: Optional[List[str]] = None  # Directories to watch for changes (defaults to app directory)
    
    # Model settings
    # Model size speed comparison (approximate RTF on CPU with optimized settings):
    # - tiny: ~0.1x (fastest, lower accuracy) - Recommended for speed
    # - base: ~0.2x (fast, good accuracy) - Good balance
    # - small: ~0.5x (medium speed, good accuracy)
    # - medium: ~1.0x (slower, better accuracy)
    # - large-v2/v3: ~2.0x (slowest, best accuracy) - Current default
    model_size: str = "large-v3"  # Options: tiny, base, small, medium, large-v2, large-v3
    model_cache_dir: str = "./models"
    # Local model path - Use this to load model from a local directory instead of downloading
    # Set to the path containing model files (model.bin, config.json, tokenizer.json, vocabulary.json)
    # If set, this takes priority over model_size and no internet connection is required
    # Example: "./models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
    model_local_path: Optional[str] = "./models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
    device: str = "cpu"  # cpu or cuda (use cuda for GPU acceleration - 5-10x faster)
    compute_type: str = "int8"  # int8 (CPU), int8_float16/float16/float32 (GPU)
    model_pool_size: int = 4  # Number of model instances for concurrent processing
    
    # Supported languages
    supported_languages: List[str] = ["en", "hi", "ko"]
    
    # Audio settings
    max_file_size_mb: int = 500
    max_audio_duration_seconds: int = 60  # 1 minute
    audio_sample_rate: int = 16000
    audio_channels: int = 1  # mono
    audio_storage_dir: str = "./audio_recordings"  # Directory to save uploaded audio files
    save_audio_files: bool = True  # Whether to save uploaded audio files
    audio_mp3_bitrate: str = "192k"  # MP3 bitrate (128k, 192k, 256k, 320k) - all files saved as MP3
    
    # Processing settings (optimized for speed)
    # Note: For even faster transcription, consider using a smaller model (e.g., "base" or "small")
    # Speed vs Accuracy tradeoff:
    # - beam_size=1 (greedy): Fastest, slightly lower accuracy
    # - beam_size=5: Slower, better accuracy (original setting)
    # - condition_on_previous_text=False: Faster, slightly lower context awareness
    enable_word_timestamps: bool = True
    enable_diarization: bool = False
    beam_size: int = 1  # Reduced from 5 for faster inference (1=greedy, 5=beam search)
    best_of: int = 1  # Reduced from 5 for faster inference
    patience: float = 1.0
    temperature: float = 0.0
    condition_on_previous_text: bool = False  # Disabled for faster inference
    initial_prompt: Optional[str] = None
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    cpu_threads: int = 0  # 0 = auto-detect based on CPU count, set to specific number to override
    
    # Rate limiting (requests per minute per IP)
    rate_limit_per_minute: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    class Config:
        """
        Pydantic configuration for settings.
        
        This configuration enables:
        - Loading settings from .env file
        - Case-insensitive environment variable matching
        - UTF-8 encoding for .env file
        """
        env_file = ".env"  # Load settings from .env file if present
        env_file_encoding = "utf-8"  # Use UTF-8 encoding for .env file
        case_sensitive = False  # Allow case-insensitive environment variables
        protected_namespaces = ('settings_',)  # Protect settings_ namespace


settings = Settings()

