# ─────────────────────────────────────────────────────────────────────────────
#  Technodysis Meeting Assistant — Dockerfile
#
#  GPU build (default via docker-compose):
#    docker compose up --build
#
#  CPU-only build:
#    docker build --build-arg CUDA=cpu -t meeting-app .
# ─────────────────────────────────────────────────────────────────────────────

ARG CUDA=cu121

# Use CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG MODE=full
ARG CUDA=cu121

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        gcc \
        g++ \
        libsndfile1 \
        libsndfile-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt requirements_meeting.txt ./

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121 \
        torchaudio==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt -r requirements_meeting.txt

# ── NLP model downloads (baked into image — no runtime download needed) ────────
# spaCy  : en_core_web_sm for NER + PyTextRank phrase extraction — ~12 MB
# NLTK   : vader_lexicon (sentiment) + punkt/punkt_tab (Sumy tokenizer) — ~5 MB
RUN python -m spacy download en_core_web_sm \
    && python -c "\
import nltk; \
nltk.download('vader_lexicon', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
print('NLTK data downloaded')"

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p uploads logs data/input data/vector_database data/topic_model templates

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
