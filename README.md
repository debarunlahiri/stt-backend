# Offline Multilingual Speech-to-Text and Translation Backend

A production-ready, 100% offline Python-based REST API server for accurate speech-to-text transcription and text translation in Hindi, English, and Korean. Uses faster-whisper (OpenAI Whisper with CTranslate2 acceleration) for transcription and Argos Translate for offline translation.

## Features

- **100% Offline Operation** - No external API calls required, works completely offline
- **Multilingual Support** - English (en), Hindi (hi), and Korean (ko) with automatic language detection
- **Text Translation** - Translate text between English, Hindi, and Korean (fully offline)
- **Language Detection** - Automatically detect the language of input text
- **Word-Level Timestamps** - Accurate timing information for each word in the transcript
- **Universal Audio Format Support** - Supports all major audio formats:
  - Common: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, WEBM
  - Additional: AIFF, AMR, WMA, MP2, MP4, 3GP, GSM, and more
  - All formats supported by FFmpeg are automatically handled
- **GPU/CPU Support** - Automatic GPU acceleration with CPU fallback
- **Fast Processing** - Optimized with CTranslate2 for real-time transcription
- **Docker Ready** - Containerized deployment with Docker and Docker Compose
- **Production Grade** - Error handling, logging, health checks, and validation

## Requirements

- Python 3.11+
- FFmpeg (for audio format conversion)
- 4-8 GB RAM (CPU mode) or 4 GB GPU memory (CUDA mode)
- Disk space: ~3-10 GB for models (depending on model size)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stt-backend
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-pip
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

Copy `.env.example` to `.env` and modify settings:

```bash
cp .env.example .env
```

Edit `.env` to customize:
- Model size (tiny, base, small, medium, large-v2, large-v3)
- Device (cpu or cuda)
- File size limits
- Processing parameters

## Quick Start

### Run with Python

```bash
python run.py
```

The server will start on `http://localhost:8000`. On first run, the Whisper model will be downloaded automatically (this may take several minutes).

### Run with Docker

```bash
docker-compose up --build
```

The server will be available at `http://localhost:8000`.

## Configuration

### Model Sizes

Choose the appropriate model size based on your accuracy and speed requirements:

| Model Size | Parameters | Disk Size | Speed (CPU) | Accuracy |
|------------|------------|-----------|-------------|----------|
| tiny       | 39M        | ~150 MB   | Fastest     | Good     |
| base       | 74M        | ~300 MB   | Fast        | Better   |
| small      | 244M       | ~1 GB     | Medium      | Good     |
| medium     | 769M       | ~3 GB     | Slow        | Better   |
| large-v2   | 1550M      | ~6 GB     | Slower      | Best     |
| large-v3   | 1550M      | ~6 GB     | Slower      | Best     |

Default: `large-v3` (best accuracy)

### Environment Variables

Key configuration options in `.env`:

```env
# Model Settings
MODEL_SIZE=large-v3              # tiny, base, small, medium, large-v2, large-v3
DEVICE=cpu                       # cpu or cuda
COMPUTE_TYPE=int8                # int8, int8_float16, float16, float32

# Audio Settings
MAX_FILE_SIZE_MB=500             # Maximum file size in MB
MAX_AUDIO_DURATION_SECONDS=60     # Maximum audio duration (1 minute)
AUDIO_SAMPLE_RATE=16000          # Target sample rate (16kHz for Whisper)

# Server Settings
HOST=0.0.0.0                     # Server host
PORT=8000                        # Server port
WORKERS=1                        # Number of workers (use 1 for CPU)

# Processing Settings
ENABLE_WORD_TIMESTAMPS=true      # Include word-level timestamps
BEAM_SIZE=5                      # Beam search size
BEST_OF=5                        # Best of N candidates
```

## API Documentation

### Base URL

```
http://localhost:8000
```

### Interactive API Docs

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### 1. Health Check

**GET** `/health`

Check server status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "supported_languages": ["en", "hi", "ko"],
  "supported_audio_formats": ["aac", "ac3", "aif", "aiff", "amr", ...],
  "model_size": "large-v3",
  "gpu_available": false,
  "gpu_name": null
}
```

#### 2. Transcribe Audio

**POST** `/v1/transcribe`

Transcribe audio file to text.

**Parameters:**
- `audio_file` (file, required): Audio file to transcribe
- `language` (query, optional): Language code (en, hi, ko) or "auto" for auto-detection
- `enable_word_timestamps` (query, optional): Include word-level timestamps (default: true)
- `enable_diarization` (query, optional): Enable speaker diarization (default: false, not yet implemented)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/v1/transcribe?language=auto" \
  -F "audio_file=@sample.wav"
```

**Response:**
```json
{
  "text": "नमस्ते, hello, 안녕하세요",
  "language": "hi",
  "detected_language": "hi",
  "segments": [
    {
      "start": 0.0,
      "end": 2.1,
      "text": "नमस्ते",
      "words": [
        {
          "word": "नमस्ते",
          "start": 0.0,
          "end": 1.5,
          "confidence": 0.95
        }
      ],
      "language": "hi"
    }
  ],
  "processing_time_sec": 3.4,
  "real_time_factor": 0.42,
  "audio_duration_sec": 8.1,
  "confidence": 0.92,
  "word_count": 5
}
```

#### 3. Translate Text

**POST** `/v1/translate`

Translate text from one language to another. Supports translation between English, Hindi, and Korean.

**Request Body:**
```json
{
  "text": "नमस्ते, आप कैसे हैं?",
  "source_language": "hi",
  "target_language": "en"
}
```

**Parameters:**
- `text` (string, required): Text to translate
- `source_language` (string, optional): Source language code (en, hi, ko) or "auto" for auto-detection. Default: "auto"
- `target_language` (string, required): Target language code (en, hi, ko). Default: "en"

**Supported Translation Pairs:**
- English ↔ Hindi (en ↔ hi)
- English ↔ Korean (en ↔ ko)
- Hindi ↔ Korean (hi ↔ ko)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "source_language": "hi",
    "target_language": "en"
  }'
```

**Response:**
```json
{
  "translated_text": "Hello, how are you?",
  "source_language": "hi",
  "target_language": "en",
  "detected_language": "hi",
  "detection_confidence": 0.99,
  "processing_time_sec": 0.15,
  "translation_applied": true
}
```

**Auto-detect Source Language:**
```bash
curl -X POST "http://localhost:8000/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_language": "auto",
    "target_language": "hi"
  }'
```

#### 4. Detect Language

**POST** `/v1/detect-language`

Detect the language of input text. Supports English, Hindi, and Korean.

**Request Body:**
```json
{
  "text": "नमस्ते"
}
```

**Parameters:**
- `text` (string, required): Text to detect language for

**Example Request:**
```bash
curl -X POST "http://localhost:8000/v1/detect-language" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?"
  }'
```

**Response:**
```json
{
  "detected_language": "hi",
  "language_name": "Hindi",
  "confidence": 0.99
}
```

## Usage Examples

### Python

```python
import requests

url = "http://localhost:8000/v1/transcribe"
files = {"audio_file": open("sample.wav", "rb")}
params = {"language": "auto", "enable_word_timestamps": True}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Transcribed text: {result['text']}")
print(f"Detected language: {result['detected_language']}")
print(f"Processing time: {result['processing_time_sec']:.2f}s")
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('audio_file', fs.createReadStream('sample.wav'));

axios.post('http://localhost:8000/v1/transcribe', form, {
  headers: form.getHeaders(),
  params: {
    language: 'auto',
    enable_word_timestamps: true
  }
})
.then(response => {
  console.log('Transcribed text:', response.data.text);
  console.log('Detected language:', response.data.detected_language);
})
.catch(error => console.error('Error:', error));
```

### cURL

```bash
# Transcribe with auto language detection
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "audio_file=@sample.wav" \
  -F "language=auto"

# Transcribe with specific language
curl -X POST "http://localhost:8000/v1/transcribe?language=hi" \
  -F "audio_file=@hindi_audio.mp3"

# Transcribe without word timestamps (faster)
curl -X POST "http://localhost:8000/v1/transcribe?enable_word_timestamps=false" \
  -F "audio_file=@sample.wav"
```

### Translation Examples

#### Python

```python
import requests

# Translate Hindi to English
url = "http://localhost:8000/v1/translate"
data = {
    "text": "नमस्ते, आप कैसे हैं?",
    "source_language": "hi",
    "target_language": "en"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Original: {data['text']}")
print(f"Translated: {result['translated_text']}")
print(f"Detected language: {result['detected_language']}")
```

#### Auto-detect and Translate

```python
import requests

# Auto-detect source language and translate to Hindi
url = "http://localhost:8000/v1/translate"
data = {
    "text": "Hello, how are you?",
    "source_language": "auto",
    "target_language": "hi"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Translated: {result['translated_text']}")
```

#### Detect Language

```python
import requests

# Detect language of text
url = "http://localhost:8000/v1/detect-language"
data = {
    "text": "नमस्ते"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Detected: {result['language_name']} ({result['detected_language']})")
print(f"Confidence: {result['confidence']:.2%}")
```

#### JavaScript/Node.js

```javascript
const axios = require('axios');

// Translate text
axios.post('http://localhost:8000/v1/translate', {
  text: 'नमस्ते, आप कैसे हैं?',
  source_language: 'hi',
  target_language: 'en'
})
.then(response => {
  console.log('Translated:', response.data.translated_text);
  console.log('Detected language:', response.data.detected_language);
})
.catch(error => console.error('Error:', error));

// Detect language
axios.post('http://localhost:8000/v1/detect-language', {
  text: 'नमस्ते'
})
.then(response => {
  console.log('Language:', response.data.language_name);
  console.log('Confidence:', response.data.confidence);
})
.catch(error => console.error('Error:', error));
```

#### cURL

```bash
# Translate Hindi to English
curl -X POST "http://localhost:8000/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "source_language": "hi",
    "target_language": "en"
  }'

# Auto-detect and translate to Hindi
curl -X POST "http://localhost:8000/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_language": "auto",
    "target_language": "hi"
  }'

# Detect language
curl -X POST "http://localhost:8000/v1/detect-language" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते"
  }'
```

## Deployment

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. View logs:
```bash
docker-compose logs -f
```

3. Stop the server:
```bash
docker-compose down
```

### Production Deployment

For production, use a process manager like Gunicorn:

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

**Note:** For CPU mode, use `-w 1` (single worker) since model loading per worker uses significant memory.

### Environment Variables for Production

```env
MODEL_SIZE=large-v3
DEVICE=cpu
WORKERS=1
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=500
```

## Performance

### CPU Performance

| Model Size | RTF (Real-Time Factor) | Memory Usage |
|------------|------------------------|--------------|
| tiny       | ~0.1-0.2              | ~500 MB      |
| base       | ~0.2-0.4              | ~1 GB        |
| small      | ~0.4-0.8              | ~2 GB        |
| medium     | ~0.8-1.5              | ~5 GB        |
| large-v3   | ~1.5-3.0              | ~8 GB        |

RTF < 1.0 means faster than real-time.

### GPU Performance

With CUDA and appropriate compute_type, RTF can be significantly lower (~0.1-0.5 for large-v3).

## Troubleshooting

### Model Download Issues

If model download fails, manually download:
```bash
# Models are cached in ./models directory
# Delete and restart to re-download
rm -rf ./models
python run.py
```

### Out of Memory Errors

- Use a smaller model size (tiny, base, or small)
- Reduce `MAX_FILE_SIZE_MB` and `MAX_AUDIO_DURATION_SECONDS`
- Use CPU mode with single worker

### Audio Format Not Supported

The system supports all formats that FFmpeg can decode. If a format fails:
1. Check FFmpeg installation: `ffmpeg -version`
2. Verify file is not corrupted
3. Try converting to WAV first: `ffmpeg -i input.xyz output.wav`

### Slow Processing

- Use smaller model size for faster processing
- Disable word timestamps: `enable_word_timestamps=false`
- Use GPU acceleration if available (set `DEVICE=cuda`)
- Reduce beam_size and best_of parameters

### Language Detection Issues

- Specify language explicitly: `language=hi`, `language=en`, or `language=ko`
- Ensure audio contains clear speech
- Use larger model for better language detection

### Translation Issues

- **Translation packages not installing**: On first run, translation packages are downloaded automatically. This may take a few minutes. Ensure you have internet connection for the initial package download.
- **Unsupported language pair**: Only bidirectional translation between en, hi, and ko is supported. Check the supported pairs in the API documentation.
- **Low translation quality**: Translation quality depends on the text complexity. For better results, ensure the source text is clear and well-formed.
- **Language detection errors**: If auto-detection fails, explicitly specify the source language in the translation request.

## Project Structure

```
stt-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── stt_service.py        # STT service with faster-whisper
│       ├── audio_processor.py    # Audio format handling
│       └── translation_service.py # Translation and language detection
├── models/                  # Cached Whisper models
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── run.py                  # Server entry point
└── README.md              # This file
```

## Supported Languages

### Speech-to-Text

- **English (en)** - Highest accuracy
- **Hindi (hi)** - Devanagari script output
- **Korean (ko)** - Hangul script output

Language is automatically detected if not specified.

### Translation

The translation API supports bidirectional translation between:
- **English ↔ Hindi** (en ↔ hi)
- **English ↔ Korean** (en ↔ ko)
- **Hindi ↔ Korean** (hi ↔ ko)

Translation is fully offline and uses Argos Translate for high-quality translations. Source language can be auto-detected if not specified.

## License

This project uses:
- **faster-whisper**: MIT License
- **OpenAI Whisper**: MIT License
- **FastAPI**: MIT License
- **Argos Translate**: MIT License
- **langdetect**: Apache License 2.0

Check individual package licenses for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass (if applicable)
- Documentation is updated

## Support

For issues, feature requests, or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Open an issue on the repository

## Acknowledgments

- OpenAI for the Whisper model
- faster-whisper for CTranslate2 acceleration
- FFmpeg for universal audio format support

