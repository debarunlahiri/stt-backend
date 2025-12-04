"""
Main Application Module

This module contains the FastAPI application and all API endpoints for the
Offline Speech-to-Text Backend service.

The application provides RESTful API endpoints for:
- Audio transcription (speech-to-text)
- Text translation between multiple languages
- Language detection from text
- Health checks and system status

The service uses a model pool pattern for concurrent processing and supports
multiple audio formats through comprehensive audio processing.

Author: Debarun Lahiri
"""

import os
import warnings

# CRITICAL: Set environment variables BEFORE any imports
# Prevent spacy and stanza (dependencies of argostranslate) from trying to download models
os.environ.setdefault('SPACY_DISABLE_MODEL_DOWNLOAD', '1')
# Stanza default location is ~/stanza_resources - use it if it exists, otherwise prevent downloads
stanza_default = os.path.expanduser('~/stanza_resources')
if os.path.exists(stanza_default):
    os.environ.setdefault('STANZA_RESOURCES_DIR', stanza_default)
else:
    os.environ.setdefault('STANZA_RESOURCES_DIR', os.path.expanduser('~/.stanza'))
os.environ.setdefault('STANZA_CACHE_DIR', os.path.expanduser('~/.stanza_cache'))

# Disable SSL verification warnings for office proxy/certificate issues
warnings.filterwarnings('ignore', message='.*SSL.*')
warnings.filterwarnings('ignore', message='.*certificate.*')
warnings.filterwarnings('ignore', message='.*443.*')
# Note: ssl.SSLError is an Exception, not a Warning, so can't be filtered here
# SSL errors will be caught in try-except blocks instead

import logging
import time
import asyncio
import uuid
import io
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from app.config import settings
from app.models.schemas import (
    TranscriptionResponse,
    HealthResponse,
    LanguageCode,
    TranslationRequest,
    TranslationResponse,
    LanguageDetectionRequest,
    LanguageDetectionResponse
)
from app.services.stt_service import STTService
from app.services.audio_processor import AudioProcessor
from app.services.translation_service import TranslationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=settings.log_file if settings.log_file else None
)
logger = logging.getLogger(__name__)

# Global service instances
stt_service: Optional[STTService] = None
audio_processor: Optional[AudioProcessor] = None
translation_service: Optional[TranslationService] = None

# Thread pool executor for blocking operations
executor: Optional[ThreadPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    
    This function handles the application lifecycle:
    - Startup: Initializes all services, loads models, creates directories
    - Shutdown: Gracefully shuts down thread pools and cleans up resources
    
    The startup process:
    1. Creates audio storage directory if enabled
    2. Initializes thread pool executor for concurrent processing
    3. Creates service instances (STT, AudioProcessor, TranslationService)
    4. Loads Whisper models (may take time on first run)
    
    Author: Debarun Lahiri
    """
    global stt_service, audio_processor, translation_service, executor
    
    # Startup phase
    logger.info("Starting Offline STT Backend Server...")
    logger.info(f"Configuration: model={settings.model_size}, device={settings.device}")
    
    try:
        # Step 1: Create audio storage directory if saving is enabled
        if settings.save_audio_files:
            audio_dir = Path(settings.audio_storage_dir)
            audio_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Audio storage directory created/verified: {audio_dir.absolute()}")
        
        # Step 2: Initialize thread pool executor for blocking operations
        # The executor allows concurrent processing of multiple requests
        # Use max_workers based on CPU count, but limit to reasonable number to avoid oversubscription
        max_workers = min(8, (os.cpu_count() or 4) + 4)
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stt-worker")
        logger.info(f"Initialized thread pool executor with {max_workers} workers for concurrent processing")
        
        # Step 3: Initialize service instances
        stt_service = STTService()
        audio_processor = AudioProcessor()
        translation_service = TranslationService()
        
        # Step 4: Load Whisper models (this may take time, especially on first run)
        logger.info("Loading Whisper model (this may take a while on first run)...")
        stt_service.initialize_model()
        logger.info("Model loaded successfully. Server ready to accept requests.")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown phase
    logger.info("Shutting down server...")
    if executor:
        executor.shutdown(wait=True)  # Wait for all tasks to complete
        logger.info("Thread pool executor shut down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Offline Multilingual Speech-to-Text Backend API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio recordings if saving is enabled
if settings.save_audio_files:
    audio_dir = Path(settings.audio_storage_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing service information and available endpoints.
    
    Returns basic service information including name, version, status, and
    a list of available API endpoints for easy discovery.
    
    Returns:
        Dictionary with service information and endpoint list
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/transcribe",
            "translate": "/v1/translate",
            "detect_language": "/v1/detect-language",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint showing model status and system information.
    
    This endpoint provides comprehensive system health information including:
    - Service status (healthy/unhealthy)
    - Model loading status
    - Device information (CPU/GPU)
    - Supported languages and audio formats
    - GPU information if available
    
    Returns:
        HealthResponse with system status and configuration information
        
    Raises:
        HTTPException: If STT service is not initialized
    """
    if stt_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT service not initialized"
        )
    
    model_info = stt_service.get_model_info()
    supported_formats = audio_processor.get_supported_formats() if audio_processor else []
    
    return HealthResponse(
        status="healthy" if stt_service.is_model_loaded() else "unhealthy",
        model_loaded=stt_service.is_model_loaded(),
        device=model_info["device"],
        supported_languages=settings.supported_languages,
        supported_audio_formats=supported_formats,
        model_size=model_info["model_size"],
        gpu_available=model_info["gpu_available"],
        gpu_name=model_info.get("gpu_name")
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_audio(
    request: Request,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[LanguageCode] = None,
    enable_word_timestamps: Optional[bool] = None,
    enable_diarization: Optional[bool] = None
):
    """
    Transcribe audio file to text and automatically translate to all 3 languages.
    
    Supports all major audio formats including:
    - Common formats: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM
    - Additional formats: AIFF, AMR, WMA, MP2, MP4, 3GP, GS by ffmpeg
    
    All formats supported by ffmpeg are automatically detected and processed.
    
    The transcribed text is automatically translated to English, Hindi, and Korean,
    and all translations are included in the response.
    
    **Limitations:**
    - Maximum audio duration: 1 minute (60 seconds). Audio files longer than 1 minute will be rejected.
    
    - **language**: Language code (en, hi, ko) or 'auto' for auto-detection (default: auto)
    - **enable_word_timestamps**: Include word-level timestamps (default: true)
    - **enable_diarization**: Enable speaker diarization (default: false, not yet implemented)
    """
    if stt_service is None or audio_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT service not initialized"
        )
    
    if not stt_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Read audio file (this is async, so it doesn't block)
        audio_bytes = await audio_file.read()
        original_filename = audio_file.filename or "audio"
        
        # Validate file size (quick operation, can be synchronous)
        audio_processor.validate_audio_file(len(audio_bytes))
        
        # Save audio file if enabled (convert to MP3)
        audio_file_url = None
        saved_filename = None
        if settings.save_audio_files:
            try:
                # Generate unique filename with timestamp and UUID
                # Always save as MP3 format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                saved_filename = f"{timestamp}_{unique_id}.mp3"
                
                # Convert and save file in thread pool to avoid blocking
                def convert_and_save_mp3():
                    """Convert audio to MP3 and save it."""
                    audio_dir = Path(settings.audio_storage_dir)
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    file_path = audio_dir / saved_filename
                    
                    # Load audio from bytes
                    audio_io = io.BytesIO(audio_bytes)
                    
                    # Detect format from filename extension, or let pydub auto-detect
                    file_ext = Path(original_filename).suffix.lower()
                    if file_ext:
                        # Remove the dot and use as format
                        audio_format = file_ext[1:]  # Remove leading dot
                    else:
                        # Let pydub auto-detect
                        audio_format = None
                    
                    # Load audio segment
                    audio_segment = AudioSegment.from_file(audio_io, format=audio_format)
                    
                    # Export as MP3 with specified bitrate
                    audio_segment.export(
                        str(file_path),
                        format="mp3",
                        bitrate=settings.audio_mp3_bitrate
                    )
                    
                    logger.info(f"Audio file converted to MP3 and saved: {file_path} (bitrate: {settings.audio_mp3_bitrate})")
                
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    convert_and_save_mp3
                )
                
                # Generate full URL
                base_url = str(request.base_url).rstrip('/')
                audio_file_url = f"{base_url}/audio/{saved_filename}"
                logger.info(f"Audio file URL: {audio_file_url}")
                
            except Exception as e:
                logger.warning(f"Failed to save audio file as MP3: {str(e)}")
                # Continue without saving - don't fail the request
        
        # Get request parameters
        lang_code = language.value if language else "auto"
        if lang_code == "auto":
            lang_code = None
        
        word_ts = enable_word_timestamps if enable_word_timestamps is not None else settings.enable_word_timestamps
        
        # Run blocking operations in thread pool to allow concurrent requests
        # This prevents blocking the async event loop and allows multiple requests
        # to be processed simultaneously
        mime_type = audio_file.content_type
        
        def process_and_transcribe():
            """
            Process audio and transcribe - runs in thread pool.
            
            This function runs in a separate thread to avoid blocking the async event loop.
            It performs:
            1. Audio loading and format conversion
            2. Audio normalization
            3. Speech-to-text transcription
            
            Returns:
                Tuple of (full_text, segments, detected_language, metadata)
            """
            # Step 1: Process audio (with MIME type for better format detection)
            audio_array, sample_rate = audio_processor.load_audio(audio_bytes, original_filename, mime_type)
            # Step 2: Normalize audio to improve quality
            audio_array = audio_processor.normalize_audio(audio_array)
            
            # Step 3: Transcribe audio to text
            return stt_service.transcribe(
                audio_array,
                language=lang_code,
                enable_word_timestamps=word_ts
            )
        
        # Execute blocking operations in thread pool
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Thread pool executor not initialized"
            )
        
        full_text, segments, detected_language, metadata = await asyncio.get_event_loop().run_in_executor(
            executor,
            process_and_transcribe
        )
        
        # Calculate overall confidence (average of segment confidences if available)
        confidence = None
        if segments:
            confidences = []
            for seg in segments:
                if seg.words:
                    word_confs = [w.confidence for w in seg.words if w.confidence is not None]
                    if word_confs:
                        confidences.extend(word_confs)
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        # Automatically translate transcribed text to all 3 languages
        # Uses two-step translation: if source is ur/hi/ko, first converts to English, then to all languages
        en_text = full_text
        hi_text = full_text
        ko_text = full_text
        
        if translation_service is not None and full_text.strip():
            try:
                # Use translate_to_all_languages which handles two-step translation automatically
                # This method translates ur/hi/ko to English first, then English to all 3 languages
                en_text, hi_text, ko_text, detected_translation_lang, translation_metadata = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: translation_service.translate_to_all_languages(
                        text=full_text,
                        source_language=detected_language
                    )
                )
                logger.info(f"Transcribed text automatically translated to all 3 languages (source: {detected_translation_lang})")
            except Exception as e:
                logger.warning(f"Failed to translate transcribed text: {str(e)}. Using original text for all languages.")
                # Continue with original text if translation fails
                en_text = hi_text = ko_text = full_text
        
        # Build response
        response = TranscriptionResponse(
            text=full_text,
            language=detected_language,
            detected_language=detected_language,
            segments=segments,
            english_text=en_text,
            hindi_text=hi_text,
            korean_text=ko_text,
            processing_time_sec=metadata["processing_time_sec"],
            real_time_factor=metadata["real_time_factor"],
            audio_duration_sec=metadata["audio_duration_sec"],
            confidence=confidence,
            word_count=metadata["word_count"],
            audio_file_url=audio_file_url
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/v1/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_text(request: TranslationRequest):
    """
    Translate text to all 3 languages (English, Hindi, Korean).
    
    Returns translations in all supported languages:
    - English (en)
    - Hindi (hi)
    - Korean (ko)
    
    Translation logic:
    - If source is Urdu/Hindi/Korean: First converts to English, then English to all 3 languages
    - If source is English: Translates directly to all 3 languages
    
    - **text**: Text to translate
    - **source_language**: Source language code (en, hi, ko, ur) or 'auto' for auto-detection (default: auto)
    - **target_language**: (Deprecated - translations are always returned in all languages)
    
    If source_language is 'auto' or not provided, the language will be automatically detected.
    The response will contain translations in all 3 languages regardless of the target_language parameter.
    """
    if translation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation service not initialized"
        )
    
    try:
        source_lang = request.source_language.value if request.source_language else None
        
        # Run translations in thread pool to allow concurrent requests
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Thread pool executor not initialized"
            )
        
        # Use the new translate_to_all_languages method which handles two-step translation
        # This method automatically translates ur/hi/ko to English first, then to all languages
        en_text, hi_text, ko_text, detected_lang, metadata = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: translation_service.translate_to_all_languages(
                text=request.text,
                source_language=source_lang
            )
        )
        
        response = TranslationResponse(
            english_text=en_text,
            hindi_text=hi_text,
            korean_text=ko_text,
            source_language=detected_lang,
            detected_language=detected_lang,
            detection_confidence=metadata.get("detection_confidence", 1.0),
            processing_time_sec=metadata.get("processing_time_sec", 0.0),
            translation_applied=metadata.get("translation_applied", True)
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Translation validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@app.post("/v1/detect-language", response_model=LanguageDetectionResponse, tags=["Translation"])
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of the input text.
    
    - **text**: Text to detect language for
    
    Returns the detected language code, language name, and confidence score.
    Supports: English (en), Hindi (hi), Korean (ko)
    """
    if translation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation service not initialized"
        )
    
    try:
        # Run language detection in thread pool to allow concurrent requests
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Thread pool executor not initialized"
            )
        
        detected_lang, confidence = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: translation_service.detect_language(request.text)
        )
        
        # Get language name
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "ko": "Korean"
        }
        language_name = language_names.get(detected_lang, "Unknown")
        
        response = LanguageDetectionResponse(
            detected_language=detected_lang,
            language_name=language_name,
            confidence=confidence
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Language detection validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.
    
    This handler catches any exceptions that are not explicitly handled by
    endpoint handlers and returns a standardized error response. All errors
    are logged with full traceback for debugging.
    
    Args:
        request: The FastAPI request object
        exc: The exception that was raised
        
    Returns:
        JSONResponse with error details and 500 status code
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Determine reload directories
    reload_dirs = settings.reload_dirs if settings.reload_dirs else ["app"]
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload or settings.debug,
        reload_dirs=reload_dirs if (settings.reload or settings.debug) else None,
        workers=1 if (settings.reload or settings.debug) else settings.workers
    )

