#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Technodysis Meeting Assistant — Automated GPU Setup Script (Ubuntu)
#  Usage: bash setup.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Technodysis Meeting Assistant — Setup              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Detect CUDA version ───────────────────────────────────────────────────────
detect_cuda() {
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        info "Detected CUDA $CUDA_VER"

        if   [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then CUDA_TAG="cu124"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then CUDA_TAG="cu121"
        elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then CUDA_TAG="cu118"
        else warn "Unsupported CUDA $CUDA_VER — falling back to CPU install"; CUDA_TAG="cpu"
        fi
    else
        warn "nvidia-smi not found — installing CPU-only PyTorch"
        CUDA_TAG="cpu"
    fi
}

# ── Step 1: System packages ───────────────────────────────────────────────────
step1_system_packages() {
    info "Step 1/8 — Installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        postgresql postgresql-contrib \
        docker.io curl git ffmpeg \
        build-essential libpq-dev
    sudo systemctl enable --now docker
    sudo systemctl enable --now postgresql
    success "System packages installed."
}

# ── Step 2: Python virtual environment ───────────────────────────────────────
step2_venv() {
    info "Step 2/8 — Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install --upgrade pip -q
    success "Virtual environment ready."
}

# ── Step 3: Install PyTorch (GPU or CPU) ─────────────────────────────────────
step3_pytorch() {
    info "Step 3/8 — Installing PyTorch ($CUDA_TAG)..."
    source venv/bin/activate

    if [ "$CUDA_TAG" == "cpu" ]; then
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -q
    else
        pip install \
            torch==2.5.1+${CUDA_TAG} \
            torchvision==0.20.1+${CUDA_TAG} \
            torchaudio==2.5.1 \
            --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" -q
    fi
    success "PyTorch installed."
}

# ── Step 4: Install Python dependencies ──────────────────────────────────────
step4_pip_deps() {
    info "Step 4/8 — Installing Python dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt -r requirements_meeting.txt -q
    success "Python dependencies installed."
}

# ── Step 5: Set up .env ───────────────────────────────────────────────────────
step5_env() {
    info "Step 5/8 — Configuring environment..."

    if [ ! -f ".env" ]; then
        cp .env.example .env
        info "Created .env from .env.example"
    else
        warn ".env already exists — skipping copy"
    fi

    # Auto-set GPU-specific values
    if [ "$CUDA_TAG" != "cpu" ]; then
        sed -i 's/^WHISPER_DEVICE=.*/WHISPER_DEVICE=cuda/'   .env
        sed -i 's/^STT_DEVICE=.*/STT_DEVICE=cuda/'           .env
        sed -i 's/^TTS_DEVICE=.*/TTS_DEVICE=cuda/'           .env
        sed -i 's/^STT_COMPUTE_TYPE=.*/STT_COMPUTE_TYPE=float16/' .env 2>/dev/null || true
        success "GPU device flags set in .env (WHISPER_DEVICE=cuda, STT_DEVICE=cuda, TTS_DEVICE=cuda)"
    fi

    # Prompt for required secrets if still at placeholder values
    prompt_if_placeholder() {
        local KEY="$1" PROMPT="$2"
        local CURRENT
        CURRENT=$(grep "^${KEY}=" .env | cut -d= -f2-)
        if [[ "$CURRENT" == *"your-"* ]] || [[ "$CURRENT" == *"replace-"* ]] || [[ "$CURRENT" == *"hf_your"* ]]; then
            read -rp "  Enter $PROMPT (or press Enter to skip): " VAL
            if [ -n "$VAL" ]; then
                sed -i "s|^${KEY}=.*|${KEY}=${VAL}|" .env
            fi
        fi
    }

    echo ""
    echo -e "${YELLOW}Please provide required credentials (press Enter to skip and fill manually later):${NC}"
    prompt_if_placeholder "AZURE_TENANT_ID"     "Azure Tenant ID"
    prompt_if_placeholder "AZURE_CLIENT_ID"     "Azure Client ID"
    prompt_if_placeholder "AZURE_CLIENT_SECRET" "Azure Client Secret"
    prompt_if_placeholder "TEAMS_WEBHOOK_SECRET" "Teams Webhook Secret (random string)"
    prompt_if_placeholder "WEBHOOK_BASE_URL"    "Webhook Public URL (e.g. https://xxx.ngrok.io)"
    prompt_if_placeholder "HF_TOKEN"            "HuggingFace Token (for diarization)"
    prompt_if_placeholder "SMTP_USER"           "SMTP Email address"
    prompt_if_placeholder "SMTP_PASSWORD"       "SMTP App Password"

    success ".env configured."
}

# ── Step 6: PostgreSQL setup ──────────────────────────────────────────────────
step6_postgres() {
    info "Step 6/8 — Setting up PostgreSQL..."

    DB_NAME="meetingdb"
    DB_USER="meetuser"
    DB_PASS="meetpass"

    # Update .env with correct DB URL
    sed -i "s|^DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}|" .env

    # Create user and database if they don't exist
    sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

    sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" -q

    # Run Alembic migrations
    source venv/bin/activate
    export DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
    alembic upgrade head

    success "PostgreSQL database '${DB_NAME}' ready, migrations applied."
}

# ── Step 7: MinIO (Docker) ────────────────────────────────────────────────────
step7_minio() {
    info "Step 7/8 — Starting MinIO (Docker)..."

    if docker ps --format '{{.Names}}' | grep -q "^minio$"; then
        warn "MinIO container already running — skipping"
    elif docker ps -a --format '{{.Names}}' | grep -q "^minio$"; then
        docker start minio
        success "MinIO container restarted."
    else
        docker run -d \
            --name minio \
            --restart unless-stopped \
            -p 9000:9000 \
            -p 9001:9001 \
            -v "$HOME/minio_data:/data" \
            -e "MINIO_ROOT_USER=minioadmin" \
            -e "MINIO_ROOT_PASSWORD=minioadmin" \
            minio/minio server /data --console-address ":9001"
        success "MinIO started (API: :9000, Console: :9001)"
    fi
}

# ── Step 8: Ollama + LLM model ────────────────────────────────────────────────
step8_ollama() {
    info "Step 8/8 — Setting up Ollama and LLM model..."

    if ! command -v ollama &>/dev/null; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        success "Ollama already installed."
    fi

    # Start Ollama service if not running
    if ! pgrep -x "ollama" &>/dev/null; then
        ollama serve &>/dev/null &
        sleep 3
    fi

    # Pull models
    LLM_MODEL=$(grep "^LLM_MODEL=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
    LLM_MODEL="${LLM_MODEL:-llama3.2:3b}"

    MEETING_MODEL=$(grep "^MEETING_LLM_MODEL=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
    MEETING_MODEL="${MEETING_MODEL:-llama3.2:3b}"

    info "Pulling $LLM_MODEL..."
    ollama pull "$LLM_MODEL"

    if [ "$LLM_MODEL" != "$MEETING_MODEL" ]; then
        info "Pulling $MEETING_MODEL..."
        ollama pull "$MEETING_MODEL"
    fi

    success "Ollama models ready."
}

# ── Final: Start the server ───────────────────────────────────────────────────
start_server() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   Setup complete! Starting the server...             ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
    info "Server starting at http://0.0.0.0:8000"
    info "Health check: http://localhost:8000/health"
    echo ""

    source venv/bin/activate
    set -a; source .env; set +a
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_SYSTEM=false
SKIP_OLLAMA=false
START_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-system)   SKIP_SYSTEM=true ;;
        --skip-ollama)   SKIP_OLLAMA=true ;;
        --start-only)    START_ONLY=true  ;;
        --help|-h)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-system   Skip apt-get system package installation"
            echo "  --skip-ollama   Skip Ollama + model download"
            echo "  --start-only    Skip all setup, just start the server"
            exit 0 ;;
    esac
done

# ── Main ──────────────────────────────────────────────────────────────────────
if $START_ONLY; then
    detect_cuda
    start_server
    exit 0
fi

detect_cuda
$SKIP_SYSTEM || step1_system_packages
step2_venv
step3_pytorch
step4_pip_deps
step5_env
step6_postgres
step7_minio
$SKIP_OLLAMA || step8_ollama
start_server
