"""
Audio Processing Service Module

This module provides comprehensive audio file processing capabilities for the STT backend.
It supports a wide variety of audio formats through multiple fallback methods:
1. pydub with ffmpeg (primary method - supports most formats)
2. librosa (fallback for many formats)
3. soundfile (fallback for WAV, FLAC, OGG, etc.)
4. ffmpeg-python (final fallback)

The service handles format conversion, sample rate conversion, channel conversion,
and audio normalization to prepare audio for speech-to-text processing.

Supported formats include: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM, AIFF, AMR,
and many other formats supported by ffmpeg.

Author: Debarun Lahiri
"""

import os
import io
import logging
import tempfile
from typing import Tuple, Optional, List
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import librosa
import soundfile as sf

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None

from app.config import settings

logger = logging.getLogger(__name__)

# Comprehensive list of supported audio formats (via ffmpeg and librosa)
SUPPORTED_AUDIO_FORMATS = [
    # Common formats
    "wav", "wave", "mp3", "m4a", "aac", "flac", "ogg", "opus", "webm",
    # Additional formats supported by ffmpeg
    "aiff", "aif", "au", "ra", "rm", "wma", "mp2", "mp4", "m4p", "m4r",
    "3gp", "3g2", "amr", "gsm", "dvf", "wv", "mpc", "ape", "tta", "tak",
    "dsf", "dff", "ac3", "dts", "mka", "mkv", "avi", "mov", "qt", "flv",
    # Audio codecs in containers
    "pcm", "alaw", "mulaw", "g722", "g726", "adpcm", "gsm", "vorbis"
]

SUPPORTED_MIME_TYPES = [
    "audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/aac", "audio/flac", "audio/ogg",
    "audio/vorbis", "audio/opus", "audio/webm", "audio/x-m4a",
    "video/webm", "video/mp4", "audio/aiff", "audio/x-aiff", "audio/amr"
]


class AudioProcessor:
    """
    Handles audio format conversion and preprocessing for STT.
    
    This class provides comprehensive audio processing capabilities including:
    - Format detection and conversion
    - Sample rate conversion to target rate (16kHz)
    - Channel conversion to mono
    - Audio normalization
    - Duration and file size validation
    
    Supports all major audio formats through multiple fallback methods:
    - pydub with ffmpeg (primary - supports most formats)
    - librosa (fallback for many formats)
    - soundfile (fallback for WAV, FLAC, OGG, etc.)
    - ffmpeg-python (final fallback)
    
    Formats include: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM, AIFF, AMR, etc.
    
    Author: Debarun Lahiri
    """
    
    def __init__(self):
        """
        Initialize the AudioProcessor with configuration from settings.
        
        Sets up target audio parameters:
        - Sample rate: 16kHz (required by Whisper)
        - Channels: Mono (1 channel)
        - Max duration: 60 seconds (configurable)
        """
        self.target_sample_rate = settings.audio_sample_rate  # Target: 16kHz
        self.target_channels = settings.audio_channels  # Target: Mono (1 channel)
        self.max_duration = settings.max_audio_duration_seconds  # Max: 60 seconds
        
    def detect_format(self, filename: str, mime_type: Optional[str] = None) -> Optional[str]:
        """
        Detect audio format from filename extension or MIME type.
        
        Args:
            filename: Original filename
            mime_type: MIME type if available
            
        Returns:
            Detected format string or None
        """
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else None
            if ext and ext in SUPPORTED_AUDIO_FORMATS:
                return ext
        
        if mime_type:
            mime_lower = mime_type.lower()
            for supported_mime in SUPPORTED_MIME_TYPES:
                if supported_mime in mime_lower:
                    return mime_lower.split('/')[-1]
        
        return None
        
    def load_audio(self, audio_file: bytes, filename: str, mime_type: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file from bytes and convert to required format.
        Supports all formats that ffmpeg, librosa, and soundfile can decode.
        
        Args:
            audio_file: Audio file as bytes
            filename: Original filename (for format detection)
            mime_type: Optional MIME type for format detection
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If audio format is unsupported or audio is invalid
        """
        detected_format = self.detect_format(filename, mime_type)
        audio_io = io.BytesIO(audio_file)
        
        # Method 1: Try pydub with ffmpeg (supports most formats)
        # This is the primary method as it supports the widest variety of formats
        try:
            logger.debug(f"Attempting to load {filename} with pydub/ffmpeg (format: {detected_format})")
            
            # Try with detected format first for better performance, then auto-detect if that fails
            audio = None
            if detected_format:
                try:
                    audio = AudioSegment.from_file(audio_io, format=detected_format)
                except (CouldntDecodeError, Exception):
                    # If format-specific loading fails, try auto-detection
                    audio_io.seek(0)
                    audio = AudioSegment.from_file(audio_io, format=None)
            else:
                # No format detected, use auto-detection
                audio = AudioSegment.from_file(audio_io, format=None)
            
            # Calculate duration in seconds
            duration_seconds = len(audio) / 1000.0  # pydub length is in milliseconds
            
            # Validate duration against maximum allowed
            if duration_seconds > self.max_duration:
                raise ValueError(
                    f"Audio duration ({duration_seconds:.1f}s) exceeds maximum "
                    f"allowed duration ({self.max_duration}s)"
                )
            
            # Convert to mono if needed (Whisper requires mono audio)
            if audio.channels != self.target_channels:
                audio = audio.set_channels(self.target_channels)
            
            # Convert to target sample rate (16kHz required by Whisper)
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)
            
            # Convert AudioSegment to numpy array (float32, normalized to [-1.0, 1.0])
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 1:
                audio_array = audio_array.reshape(-1, 1)
            else:
                audio_array = audio_array.reshape(-1, audio.channels)
                
            # Normalize to [-1.0, 1.0] range based on sample width
            # This ensures consistent audio levels regardless of source format
            sample_width = audio.sample_width
            if sample_width == 1:
                # 8-bit unsigned: range [0, 255] -> [-1.0, 1.0]
                audio_array = (audio_array.astype(np.float32) - 128.0) / 128.0
            elif sample_width == 2:
                # 16-bit signed: range [-32768, 32767] -> [-1.0, 1.0]
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif sample_width == 4:
                # 32-bit signed: range [-2147483648, 2147483647] -> [-1.0, 1.0]
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            
            # Ensure mono output (take first channel if stereo)
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array[:, 0]
            else:
                audio_array = audio_array.flatten()
            
            logger.info(
                f"Loaded audio with pydub: {filename}, format: {detected_format or 'auto'}, "
                f"duration: {duration_seconds:.2f}s, sample_rate: {audio.frame_rate}, "
                f"channels: {audio.channels}"
            )
            
            return audio_array, self.target_sample_rate
                
        except (CouldntDecodeError, Exception) as e:
            logger.warning(f"Pydub failed to decode {filename}, trying librosa: {str(e)}")
            
            # Method 2: Try librosa (supports many formats)
            try:
                audio_io.seek(0)
                # Load without duration limit first to check actual duration
                audio_array, sample_rate = librosa.load(
                    audio_io,
                    sr=self.target_sample_rate,
                    mono=True
                )
                
                duration_seconds = len(audio_array) / sample_rate
                
                # Check duration limit
                if duration_seconds > self.max_duration:
                    raise ValueError(
                        f"Audio duration ({duration_seconds:.1f}s) exceeds maximum "
                        f"allowed duration ({self.max_duration}s)"
                    )
                
                logger.info(
                    f"Loaded audio with librosa: {filename}, duration: {duration_seconds:.2f}s, "
                    f"sample_rate: {sample_rate}"
                )
                
                return audio_array, sample_rate
                
            except Exception as e2:
                logger.warning(f"Librosa failed to decode {filename}, trying soundfile: {str(e2)}")
                
                # Method 3: Try soundfile (handles WAV, FLAC, OGG, etc.)
                try:
                    audio_io.seek(0)
                    # Save to temporary file for soundfile (it doesn't handle BytesIO well)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{detected_format or 'wav'}") as tmp_file:
                        tmp_file.write(audio_file)
                        tmp_path = tmp_file.name
                    
                    try:
                        audio_array, sample_rate = sf.read(
                            tmp_path,
                            dtype='float32',
                            always_2d=False
                        )
                        
                        # Convert to mono if needed
                        if len(audio_array.shape) > 1:
                            audio_array = np.mean(audio_array, axis=1)
                        
                        # Resample if needed
                        if sample_rate != self.target_sample_rate:
                            import librosa
                            audio_array = librosa.resample(
                                audio_array,
                                orig_sr=sample_rate,
                                target_sr=self.target_sample_rate
                            )
                            sample_rate = self.target_sample_rate
                        
                        duration_seconds = len(audio_array) / sample_rate
                        if duration_seconds > self.max_duration:
                            raise ValueError(
                                f"Audio duration ({duration_seconds:.1f}s) exceeds maximum "
                                f"allowed duration ({self.max_duration}s)"
                            )
                        
                        logger.info(
                            f"Loaded audio with soundfile: {filename}, duration: {duration_seconds:.2f}s, "
                            f"sample_rate: {sample_rate}"
                        )
                        
                        return audio_array, sample_rate
                        
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                            
                except Exception as e3:
                    logger.warning(f"Soundfile failed to decode {filename}, trying ffmpeg-python: {str(e3)}")
                    
                    # Method 4: Try ffmpeg-python directly (as final fallback)
                    if FFMPEG_AVAILABLE:
                        try:
                            audio_io.seek(0)
                            # Save to temporary file for ffmpeg
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{detected_format or 'wav'}") as tmp_file:
                                tmp_file.write(audio_file)
                                tmp_path = tmp_file.name
                            
                            try:
                                # Use ffmpeg-python to convert to WAV with target sample rate and mono
                                out_path = tmp_path + "_converted.wav"
                                stream = ffmpeg.input(tmp_path)
                                stream = ffmpeg.output(
                                    stream,
                                    out_path,
                                    acodec='pcm_s16le',
                                    ac=1,
                                    ar=self.target_sample_rate
                                )
                                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                                
                                # Load the converted file with soundfile
                                audio_array, sample_rate = sf.read(
                                    out_path,
                                    dtype='float32',
                                    always_2d=False
                                )
                                
                                # Ensure mono
                                if len(audio_array.shape) > 1:
                                    audio_array = np.mean(audio_array, axis=1)
                                
                                duration_seconds = len(audio_array) / sample_rate
                                if duration_seconds > self.max_duration:
                                    raise ValueError(
                                        f"Audio duration ({duration_seconds:.1f}s) exceeds maximum "
                                        f"allowed duration ({self.max_duration}s)"
                                    )
                                
                                logger.info(
                                    f"Loaded audio with ffmpeg-python: {filename}, duration: {duration_seconds:.2f}s, "
                                    f"sample_rate: {sample_rate}"
                                )
                                
                                # Clean up converted file
                                try:
                                    os.unlink(out_path)
                                except Exception:
                                    pass
                                
                                return audio_array, sample_rate
                                
                            finally:
                                # Clean up temporary files
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass
                                
                        except Exception as e4:
                            # All methods including ffmpeg-python failed
                            error_msg = (
                                f"Failed to decode audio file '{filename}'. "
                                f"Supported formats include: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM, "
                                f"AIFF, AMR, and other formats supported by ffmpeg. "
                                f"Errors: pydub={str(e)}, librosa={str(e2)}, soundfile={str(e3)}, ffmpeg={str(e4)}"
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    else:
                        # All methods failed and ffmpeg-python not available
                        error_msg = (
                            f"Failed to decode audio file '{filename}'. "
                            f"Supported formats include: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM, "
                            f"AIFF, AMR, and other formats supported by ffmpeg. "
                            f"Errors: pydub={str(e)}, librosa={str(e2)}, soundfile={str(e3)}. "
                            f"ffmpeg-python not available as fallback."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
    
    def validate_audio_file(self, file_size: int) -> None:
        """
        Validate audio file size.
        
        Args:
            file_size: File size in bytes
            
        Raises:
            ValueError: If file size exceeds maximum allowed
        """
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum "
                f"allowed size ({settings.max_file_size_mb} MB)"
            )
    
    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Normalize audio array to prevent clipping and improve quality.
        
        This method performs two normalization steps:
        1. Removes DC offset (removes any constant bias in the signal)
        2. Normalizes amplitude to prevent clipping (scales to 95% of max range)
        
        Args:
            audio_array: Audio array as numpy array (float32, mono, 16kHz)
            
        Returns:
            Normalized audio array with DC offset removed and amplitude normalized
        """
        # Step 1: Remove DC offset (any constant bias in the signal)
        # This improves audio quality by centering the signal around zero
        audio_array = audio_array - np.mean(audio_array)
        
        # Step 2: Normalize to prevent clipping
        # Scale to 95% of maximum range to leave headroom and prevent distortion
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95
        
        return audio_array
    
    def get_audio_info(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """
        Get information about audio array.
        
        Args:
            audio_array: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with audio information
        """
        duration = len(audio_array) / sample_rate
        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": 1 if len(audio_array.shape) == 1 else audio_array.shape[1],
            "samples": len(audio_array),
            "dtype": str(audio_array.dtype)
        }
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported audio formats.
        
        Returns:
            List of supported format extensions
        """
        return sorted(set(SUPPORTED_AUDIO_FORMATS))

