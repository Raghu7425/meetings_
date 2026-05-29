# ═══════════════════════════════════════════════════════════════════════════════
#  Technodysis Meeting Assistant — Automated Setup (Windows PowerShell)
#  Usage: .\setup.ps1
#  Flags: -SkipEnvPrompt   skip credential prompts (fill .env manually later)
#         -StartOnly       skip setup, just start docker compose
#         -Down            stop and remove all containers
# ═══════════════════════════════════════════════════════════════════════════════
param(
    [switch]$SkipEnvPrompt,
    [switch]$StartOnly,
    [switch]$Down
)

$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Info    ($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Success ($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn    ($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Err     ($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

function Wait-Healthy {
    param([string]$Container, [int]$TimeoutSec = 60)
    Info "Waiting for $Container to be healthy..."
    $elapsed = 0
    while ($elapsed -lt $TimeoutSec) {
        $status = docker inspect --format "{{.State.Health.Status}}" $Container 2>$null
        if ($status -eq "healthy") { Success "$Container is healthy."; return }
        Start-Sleep -Seconds 3
        $elapsed += 3
        Write-Host "  ...waiting ($elapsed/$TimeoutSec s)" -ForegroundColor DarkGray
    }
    Warn "$Container did not become healthy in ${TimeoutSec}s — continuing anyway."
}

function Prompt-IfPlaceholder {
    param([string]$Key, [string]$Label, [string]$EnvFile)
    $line  = Select-String -Path $EnvFile -Pattern "^$Key=" | Select-Object -First 1
    $value = if ($line) { ($line.Line -split "=", 2)[1] } else { "" }
    if ($value -match "your-|replace-|hf_your|your_") {
        $input = Read-Host "  Enter $Label (Enter to skip)"
        if ($input) {
            (Get-Content $EnvFile) -replace "^$Key=.*", "$Key=$input" |
                Set-Content $EnvFile -Encoding utf8
        }
    }
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Technodysis Meeting Assistant — Docker Setup       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── --Down: stop everything ───────────────────────────────────────────────────
if ($Down) {
    Info "Stopping and removing all containers..."
    docker compose down
    Success "All containers stopped."
    exit 0
}

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
Info "Step 1/5 — Checking prerequisites..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Err "Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop"
}

$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Err "Docker Desktop is not running. Please start it and try again."
}

$gpuAvailable = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nvidiaSmi = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        Success "GPU detected: $nvidiaSmi"
        $gpuAvailable = $true
    }
} else {
    Warn "nvidia-smi not found — GPU acceleration will not be used."
}

Success "Prerequisites OK."

# ── Step 2: Configure .env ────────────────────────────────────────────────────
Info "Step 2/5 — Configuring .env..."

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Info "Created .env from .env.example"
} else {
    Warn ".env already exists — skipping copy."
}

# Auto-set GPU device flags
if ($gpuAvailable) {
    (Get-Content ".env") -replace "^WHISPER_DEVICE=.*",  "WHISPER_DEVICE=cuda"  |
        Set-Content ".env" -Encoding utf8
    (Get-Content ".env") -replace "^STT_DEVICE=.*",      "STT_DEVICE=cuda"      |
        Set-Content ".env" -Encoding utf8
    (Get-Content ".env") -replace "^TTS_DEVICE=.*",      "TTS_DEVICE=cuda"      |
        Set-Content ".env" -Encoding utf8
    Success "GPU flags set in .env"
}

# Prompt for secrets
if (-not $SkipEnvPrompt -and -not $StartOnly) {
    Write-Host ""
    Write-Host "  Enter credentials (press Enter to skip, fill .env manually later):" -ForegroundColor Yellow
    Prompt-IfPlaceholder "AZURE_TENANT_ID"      "Azure Tenant ID"            ".env"
    Prompt-IfPlaceholder "AZURE_CLIENT_ID"      "Azure Client ID"            ".env"
    Prompt-IfPlaceholder "AZURE_CLIENT_SECRET"  "Azure Client Secret"        ".env"
    Prompt-IfPlaceholder "TEAMS_WEBHOOK_SECRET" "Teams Webhook Secret"       ".env"
    Prompt-IfPlaceholder "WEBHOOK_BASE_URL"     "Webhook Public URL (ngrok)" ".env"
    Prompt-IfPlaceholder "HF_TOKEN"             "HuggingFace Token"          ".env"
    Prompt-IfPlaceholder "SMTP_USER"            "SMTP Email address"         ".env"
    Prompt-IfPlaceholder "SMTP_PASSWORD"        "SMTP App Password"          ".env"
}

Success ".env configured."

# ── Step 3: Start Docker services ─────────────────────────────────────────────
Info "Step 3/5 — Starting Docker services (postgres, minio, ollama)..."

# Start all infrastructure first (not the app — migrations need to run first)
docker compose up -d postgres minio ollama qdrant

Wait-Healthy "meeting_postgres" 90
Wait-Healthy "meeting_minio"    90
Wait-Healthy "meeting_qdrant"   120

Success "Infrastructure containers running."

# ── Step 4: Run database migrations ───────────────────────────────────────────
Info "Step 4/5 — Running Alembic database migrations..."

# Run migrations inside a temporary container using the app image
# Build the image first if needed
docker compose build app

docker compose run --rm `
    -e DATABASE_URL="postgresql+asyncpg://meetuser:meetpass@postgres:5432/meetingdb" `
    app `
    alembic upgrade head

Success "Database migrations applied."

# ── Step 5: Pull Ollama model ──────────────────────────────────────────────────
Info "Step 5/5 — Pulling LLM model into Ollama..."

# Read model name from .env, default to llama3.2:3b
$llmModel = "llama3.2:3b"
$envLine  = Select-String -Path ".env" -Pattern "^LLM_MODEL=" | Select-Object -First 1
if ($envLine) {
    $llmModel = ($envLine.Line -split "=", 2)[1].Trim().Trim('"').Trim("'")
}

Info "Pulling $llmModel (this may take a few minutes on first run)..."
docker exec meeting_ollama ollama pull $llmModel
Success "Model '$llmModel' ready."

# Also pull meeting model if different
$meetingModel = "llama3.2:3b"
$meetingLine  = Select-String -Path ".env" -Pattern "^MEETING_LLM_MODEL=" | Select-Object -First 1
if ($meetingLine) {
    $meetingModel = ($meetingLine.Line -split "=", 2)[1].Trim().Trim('"').Trim("'")
}
if ($meetingModel -ne $llmModel) {
    Info "Pulling meeting model $meetingModel..."
    docker exec meeting_ollama ollama pull $meetingModel
    Success "Model '$meetingModel' ready."
}

# ── Start the app ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Setup complete! Starting the app...                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  App         →  http://localhost:8000" -ForegroundColor White
Write-Host "  MinIO UI    →  http://localhost:9001  (admin/minioadmin)" -ForegroundColor White
Write-Host "  Ollama API  →  http://localhost:11434" -ForegroundColor White
Write-Host ""
Write-Host "  To stop:  docker compose down" -ForegroundColor DarkGray
Write-Host "  To logs:  docker compose logs -f app" -ForegroundColor DarkGray
Write-Host ""

docker compose up -d app

Info "Tailing app logs (Ctrl+C to exit — containers keep running)..."
docker compose logs -f app
