# ─────────────────────────────────────────────────────────────────────────────
#  Technodysis Meeting Assistant — Dockerfile
#
#  Default (slim): video upload + transcription only  →  ~1.5 GB image
#  Full voice assistant:
#    docker compose build --build-arg MODE=full
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ARG MODE=slim

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        libsndfile1 \
        libsndfile-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt requirements.slim.txt requirements_meeting.txt ./

# PyTorch CPU-only — prevents pulling the 2.5 GB CUDA wheel
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install slim or full requirements depending on MODE build arg
RUN if [ "$MODE" = "full" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.slim.txt; \
    fi

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p uploads logs data/input data/vector_database templates

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8096", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
