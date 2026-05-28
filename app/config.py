"""
Central configuration module for the application.

- Loads all environment-based settings (TTS, STT, LLM, audio pipeline)
- Defines default values for speech synthesis and recognition
- Configures AI model settings and conversation limits
- Sets audio processing thresholds and timing controls
- Defines file system paths for data, vectors, and logs
- Provides logging configuration constants

Acts as the single source of truth for system-wide configuration,
making the application flexible, environment-driven, and easy to deploy.
"""


import os


# Main
APP_TITLE   = "Technodysis Voice Agent"
APP_VERSION = "2.4.0"


# Routes
HEALTH_STT_LABEL = os.getenv("HEALTH_STT_LABEL", "faster-whisper")
HEALTH_VAD_LABEL = "silero"


# Paths
_BASE              = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR           = os.path.join(_BASE, "data")
INPUT_DIR          = os.path.join(DATA_DIR, "input")
VECTOR_DB_DIR      = os.path.join(DATA_DIR, "vector_database")
KNOWLEDGE_BASE_DIR = os.path.join(INPUT_DIR, "knowledge_base.txt")
FAISS_INDEX_DIR    = os.path.join(VECTOR_DB_DIR, "meetings_index.faiss")
CHUNKS_NPY_DIR     = os.path.join(VECTOR_DB_DIR, "chunks.npy")
STATIC_DIR         = os.path.join(_BASE, "static")
UPLOADS_DIR        = os.path.join(_BASE, "uploads")
MEETINGS_INPUT_DIR = os.path.join(INPUT_DIR, "meetings")


# User Interface
UI_FILE_NAME = "index.html"


# STT  (faster-whisper)
# Model size: tiny, base, small, medium, large-v2, large-v3
STT_MODEL_NAME         = os.environ.get("STT_MODEL_NAME", "base")
STT_DEVICE             = os.environ.get("STT_DEVICE", "auto")   # auto | cpu | cuda
STT_COMPUTE_TYPE       = os.environ.get("STT_COMPUTE_TYPE", "auto")  # auto | int8 | float16 | float32
STT_LANGUAGE           = os.environ.get("STT_LANGUAGE", "en")
STT_REJECT_NON_ENGLISH = os.environ.get("STT_REJECT_NON_ENGLISH", "1") == "1"
STT_BEAM_SIZE          = int(os.environ.get("STT_BEAM_SIZE", "5"))


# LLM Model
BC_COOLDOWN                = float(os.getenv("BC_COOLDOWN", "5.0"))
MAX_CACHE_SIZE             = int(os.getenv("MAX_CACHE_SIZE", "100"))
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL                  = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_API_TIMEOUT            = int(os.getenv("LLM_API_TIMEOUT", "10"))
LLM_WARMUP_TIMEOUT         = int(os.getenv("LLM_WARMUP_TIMEOUT", "15"))
LLM_WARMUP_NUM_PREDICT     = int(os.getenv("LLM_WARMUP_NUM_PREDICT", "5"))
LLM_STREAM_REQUEST_TIMEOUT = int(os.getenv("LLM_STREAM_REQUEST_TIMEOUT", "30"))
LLM_MAX_GENERATION_TOKENS  = int(os.getenv("LLM_MAX_GENERATION_TOKENS", "250"))
LLM_CONTEXT_WINDOW_TOKENS  = int(os.getenv("LLM_CONTEXT_WINDOW_TOKENS", "2048"))
LLM_TEMPERATURE            = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_CHAT_HISTORY_LIMIT     = int(os.getenv("LLM_CHAT_HISTORY_LIMIT", "5"))
CHUNK_SIZE                 = int(os.getenv("CHUNK_SIZE", "150"))
RETRIEVAL_TOP_K            = int(os.getenv("RETRIEVAL_TOP_K", "5"))
FUZZY_MATCH_THRESHOLD      = int(os.getenv("FUZZY_MATCH_THRESHOLD", "80"))
MAX_HISTORY_TURNS          = int(os.getenv("MAX_HISTORY_TURNS", "20"))
ENTITY_CORRECTIONS         = {}


# Audio pipeline
SAMPLE_RATE          = 16_000
SPEECH_THRESHOLD     = 0.60
SILENCE_LIMIT        = 1.00
MIN_SPEECH_SECS      = 0.70
BARGE_IN_HOLD        = 0.35
MAX_AUDIO_SECS       = 8
PARTIAL_INTERVAL     = 1.20
TTS_FLUSH_CHARS      = 140
BC_MIN_SPEECH_BEFORE = 1.50
BC_PAUSE_THRESHOLD   = 0.30


# Audio preprocessing
NOISE_CANCEL_ENABLED    = True
NOISE_CANCEL_STRENGTH   = 0.80
NOISE_RMS_FLOOR         = 0.003
NOISE_SAMPLE_SECS       = 0.20
NOISE_FRAME_SIZE        = 512
NOISE_HOP_SIZE          = 256
NOISE_QUIET_FRAME_RATIO = 0.10
NOISE_ALPHA_BASE        = 1.2
NOISE_ALPHA_SCALE       = 1.0
NOISE_NORM_EPS          = 1e-8


# WebSocket
WS_PING_INTERVAL           = 25.0
WS_PING_TIMEOUT            = 10.0
WS_PING_MAX_MISSED         = 2
VAD_FRAME_SIZE             = 512
MIN_SPEAKER_CHECK_SECS     = 0.8
MIN_INTERRUPT_CHECK_SECS   = 0.8
BC_MIN_SPEECH_FALLBACK     = 1.2
BC_PAUSE_FALLBACK          = 0.35
PRIMARY_ENROLLMENT_COUNT   = 3
PRIMARY_UPDATE_WINDOW      = 5
PRIMARY_UPDATE_MIN_SCORE   = 0.55
SESSION_UPDATE_DELAY       = 1.2
SPEECH_SAMPLE_RATE         = 16000
SPEAKER_ENROLLMENT_SAMPLES = 3
SPEECH_SILENCE_THRESHOLD   = float(os.getenv("SPEECH_SILENCE_THRESHOLD", "0.01"))
TARGET_SPEAKER_VERIFICATION = True
if TARGET_SPEAKER_VERIFICATION:
    SPEAKER_THRESHOLD          = float(os.getenv("SPEAKER_THRESHOLD", "0.45"))
else:
    SPEAKER_THRESHOLD          = float(os.getenv("SPEAKER_THRESHOLD", "0.01"))
    


# TTS
TTS_VOICE  = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_RATE   = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")
TTS_PITCH  = os.getenv("TTS_PITCH", "+0Hz")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "kokoro").strip().lower()   # kokoro | edge | auto
TTS_FALLBACK_PROVIDER = os.getenv("TTS_FALLBACK_PROVIDER", "edge").strip().lower()
TTS_DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "af_heart")
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto").strip().lower()
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))



# Logging
LOGS_DIR         = os.path.join(_BASE, "logs")
LOG_FILE         = os.path.join(LOGS_DIR, "app.log")
LOG_MAX_BYTES    = 5 * 1024 * 1024                    # 5 MB per file
LOG_BACKUP_COUNT = 5                                  # keep last 5 files


# ── Teams Meeting Assistant ────────────────────────────────────────────────

# Microsoft Azure / Graph API
AZURE_TENANT_ID     = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID     = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")

# Teams webhook
TEAMS_WEBHOOK_SECRET = os.getenv("TEAMS_WEBHOOK_SECRET", "changeme")
WEBHOOK_BASE_URL     = os.getenv("WEBHOOK_BASE_URL", "https://your-domain.ngrok.io")

# PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/meetingdb")

# MinIO (S3-compatible)
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "meeting-recordings")

# SMTP email
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", "noreply@technodysis.com")
ADMIN_EMAIL   = os.getenv("ADMIN_EMAIL", "admin@technodysis.com")

# Ollama (meeting pipeline uses same base URL)
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MEETING_LLM_MODEL  = os.getenv("MEETING_LLM_MODEL", "llama3.2:3b")
MEETING_LLM_TIMEOUT = int(os.getenv("MEETING_LLM_TIMEOUT", "600"))

# Jira (optional)
JIRA_ENABLED     = os.getenv("JIRA_ENABLED", "false").lower() == "true"
JIRA_URL         = os.getenv("JIRA_URL", "")
JIRA_USER        = os.getenv("JIRA_USER", "")
JIRA_API_TOKEN   = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "MEET")

# Templates
TEMPLATES_DIR = os.path.join(_BASE, "templates")

# Transcription
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_LANGUAGE   = os.getenv("WHISPER_LANGUAGE", "en")


# ── Redis ──────────────────────────────────────────────────────────────────────
REDIS_URL             = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
REDIS_SOCKET_TIMEOUT  = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))

# Redis key prefixes
REDIS_JOB_PREFIX        = "job:"
REDIS_SESSION_PREFIX    = "session:"
REDIS_SUMMARY_PREFIX    = "summary:"
REDIS_STREAM_PIPELINE   = "stream:pipeline"
REDIS_STREAM_JOBS       = "stream:jobs"

# Redis TTLs (seconds)
REDIS_JOB_TTL           = int(os.getenv("REDIS_JOB_TTL", "86400"))     # 24 h
REDIS_SESSION_TTL       = int(os.getenv("REDIS_SESSION_TTL", "3600"))   # 1 h
REDIS_CACHE_TTL         = int(os.getenv("REDIS_CACHE_TTL", "1800"))     # 30 min
REDIS_SUMMARY_TTL       = int(os.getenv("REDIS_SUMMARY_TTL", "604800")) # 7 days

# Pipeline worker config
PIPELINE_WORKER_COUNT   = int(os.getenv("PIPELINE_WORKER_COUNT", "2"))
PIPELINE_STREAM_MAXLEN  = int(os.getenv("PIPELINE_STREAM_MAXLEN", "1000"))
PIPELINE_CONSUMER_GROUP = os.getenv("PIPELINE_CONSUMER_GROUP", "pipeline_workers")
PIPELINE_CONSUMER_NAME  = os.getenv("PIPELINE_CONSUMER_NAME", "worker_1")
PIPELINE_BLOCK_MS       = int(os.getenv("PIPELINE_BLOCK_MS", "5000"))


# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_HOST            = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT            = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT       = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY", "")
QDRANT_PREFER_GRPC     = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"

# Qdrant collection names
QDRANT_COL_RAW         = os.getenv("QDRANT_COL_RAW", "meetings_raw")
QDRANT_COL_SEMANTIC    = os.getenv("QDRANT_COL_SEMANTIC", "meetings_semantic")
QDRANT_COL_SUMMARY     = os.getenv("QDRANT_COL_SUMMARY", "meetings_summaries")

# Vector dimensions
EMBED_DIM_SMALL        = int(os.getenv("EMBED_DIM_SMALL", "384"))   # all-MiniLM-L6-v2
EMBED_DIM_LARGE        = int(os.getenv("EMBED_DIM_LARGE", "768"))   # all-mpnet-base-v2

# Embedding models
EMBED_MODEL_FAST       = os.getenv("EMBED_MODEL_FAST", "all-MiniLM-L6-v2")
EMBED_MODEL_QUALITY    = os.getenv("EMBED_MODEL_QUALITY", "all-mpnet-base-v2")

# Semantic chunking
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))
CHUNK_MAX_SENTENCES     = int(os.getenv("CHUNK_MAX_SENTENCES", "8"))
CHUNK_MIN_CHARS         = int(os.getenv("CHUNK_MIN_CHARS", "80"))

# Incremental summarizer
SUMMARIZER_CHUNK_LINES  = int(os.getenv("SUMMARIZER_CHUNK_LINES", "30"))
SUMMARIZER_MAX_TOKENS   = int(os.getenv("SUMMARIZER_MAX_TOKENS", "512"))

# Retry config for LLM calls
LLM_RETRY_MAX_ATTEMPTS  = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "4"))
LLM_RETRY_MIN_WAIT      = float(os.getenv("LLM_RETRY_MIN_WAIT", "1.0"))
LLM_RETRY_MAX_WAIT      = float(os.getenv("LLM_RETRY_MAX_WAIT", "30.0"))

