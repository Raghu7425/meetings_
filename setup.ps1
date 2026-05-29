# ==============================================================================
#  Technodysis Meeting Assistant - Automated Setup (Windows PowerShell)
#
#  Usage:
#     .\setup.ps1
#
#  Flags:
#     -SkipEnvPrompt   Skip credential prompts
#     -StartOnly       Skip setup and only start services
#     -Down            Stop and remove containers
# ==============================================================================

param(
    [switch]$SkipEnvPrompt,
    [switch]$StartOnly,
    [switch]$Down
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

function Info ($msg) {
    Write-Host "[INFO]  $msg" -ForegroundColor Cyan
}

function Success ($msg) {
    Write-Host "[OK]    $msg" -ForegroundColor Green
}

function Warn ($msg) {
    Write-Host "[WARN]  $msg" -ForegroundColor Yellow
}

function Err ($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 1
}

function Wait-Healthy {
    param(
        [string]$Container,
        [int]$TimeoutSec = 60
    )

    Info "Waiting for $Container to become healthy..."

    $elapsed = 0

    while ($elapsed -lt $TimeoutSec) {

        $status = docker inspect `
            --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}" `
            $Container 2>$null

        if ($status -eq "healthy" -or $status -eq "running") {
            Success "$Container is ready."
            return
        }

        Start-Sleep -Seconds 3
        $elapsed += 3

        Write-Host "  ...waiting ($elapsed/$TimeoutSec s)" -ForegroundColor DarkGray
    }

    Warn "$Container did not become healthy in ${TimeoutSec}s - continuing anyway."
}

function Prompt-IfPlaceholder {
    param(
        [string]$Key,
        [string]$Label,
        [string]$EnvFile
    )

    $line = Select-String `
        -Path $EnvFile `
        -Pattern "^$Key=" |
        Select-Object -First 1

    $value = ""

    if ($line) {
        $value = ($line.Line -split "=", 2)[1]
    }

    if (
        $value -match "your-" -or
        $value -match "replace-" -or
        $value -match "hf_your" -or
        $value -match "your_"
    ) {

        $userValue = Read-Host "  Enter $Label (press Enter to skip)"

        if ($userValue) {

            (Get-Content $EnvFile) `
                -replace "^$Key=.*", "$Key=$userValue" |
                Set-Content $EnvFile -Encoding utf8
        }
    }
}

# ------------------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------------------

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Technodysis Meeting Assistant - Docker Setup " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------------------
# Handle --Down
# ------------------------------------------------------------------------------

if ($Down) {

    Info "Stopping and removing containers..."

    docker compose down

    Success "Containers stopped."

    exit 0
}

# ------------------------------------------------------------------------------
# Step 1 - Check prerequisites
# ------------------------------------------------------------------------------

Info "Step 1/5 - Checking prerequisites..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {

    Err "Docker not found. Install Docker Desktop first."
}

try {
    docker info | Out-Null
}
catch {
    Err "Docker Desktop is not running. Start Docker and try again."
}

$gpuAvailable = $false

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {

    try {

        $gpuName = nvidia-smi `
            --query-gpu=name `
            --format=csv,noheader 2>$null

        if ($gpuName) {

            Success "GPU detected: $gpuName"

            $gpuAvailable = $true
        }
    }
    catch {
        Warn "GPU check failed."
    }
}
else {

    Warn "nvidia-smi not found - GPU acceleration disabled."
}

Success "Prerequisites OK."

# ------------------------------------------------------------------------------
# Step 2 - Configure .env
# ------------------------------------------------------------------------------

Info "Step 2/5 - Configuring .env..."

if (-not (Test-Path ".env")) {

    if (-not (Test-Path ".env.example")) {
        Err ".env.example not found."
    }

    Copy-Item ".env.example" ".env"

    Success "Created .env from .env.example"
}
else {

    Warn ".env already exists - keeping existing file."
}

# GPU flags
if ($gpuAvailable) {

    (Get-Content ".env") `
        -replace "^WHISPER_DEVICE=.*", "WHISPER_DEVICE=cuda" |
        Set-Content ".env" -Encoding utf8

    (Get-Content ".env") `
        -replace "^STT_DEVICE=.*", "STT_DEVICE=cuda" |
        Set-Content ".env" -Encoding utf8

    (Get-Content ".env") `
        -replace "^TTS_DEVICE=.*", "TTS_DEVICE=cuda" |
        Set-Content ".env" -Encoding utf8

    Success "GPU device flags updated."
}

# Prompt for credentials
if (-not $SkipEnvPrompt -and -not $StartOnly) {

    Write-Host ""
    Write-Host "Enter credentials (press Enter to skip):" -ForegroundColor Yellow

    Prompt-IfPlaceholder "AZURE_TENANT_ID"      "Azure Tenant ID"            ".env"
    Prompt-IfPlaceholder "AZURE_CLIENT_ID"      "Azure Client ID"            ".env"
    Prompt-IfPlaceholder "AZURE_CLIENT_SECRET"  "Azure Client Secret"        ".env"
    Prompt-IfPlaceholder "TEAMS_WEBHOOK_SECRET" "Teams Webhook Secret"       ".env"
    Prompt-IfPlaceholder "WEBHOOK_BASE_URL"     "Webhook Public URL"         ".env"
    Prompt-IfPlaceholder "HF_TOKEN"             "HuggingFace Token"          ".env"
    Prompt-IfPlaceholder "SMTP_USER"            "SMTP Email"                 ".env"
    Prompt-IfPlaceholder "SMTP_PASSWORD"        "SMTP App Password"          ".env"
}

Success ".env configured."

# ------------------------------------------------------------------------------
# StartOnly mode
# ------------------------------------------------------------------------------

if ($StartOnly) {

    Info "Starting services only..."

    docker compose up -d

    Success "Services started."

    exit 0
}

# ------------------------------------------------------------------------------
# Step 3 - Start infrastructure
# ------------------------------------------------------------------------------

Info "Step 3/5 - Starting infrastructure containers..."

docker compose up -d postgres redis minio ollama qdrant memgraph

Wait-Healthy "meeting_postgres"  90
Wait-Healthy "meeting_redis"     60
Wait-Healthy "meeting_minio"     90
Wait-Healthy "meeting_qdrant"    120
Wait-Healthy "meeting_memgraph"  120

Success "Infrastructure services are running."

# ------------------------------------------------------------------------------
# Step 4 - Build app and run migrations
# ------------------------------------------------------------------------------

Info "Step 4/5 - Building app image (NLP bake-in: spaCy en_core_web_sm + NLTK vader/punkt + PyTextRank + Sumy)..."

docker compose build app

Info "Running database migrations..."

docker compose run --rm `
    -e DATABASE_URL="postgresql+asyncpg://meetuser:meetpass@postgres:5432/meetingdb" `
    app `
    alembic upgrade head

Success "Database migrations completed."

# ------------------------------------------------------------------------------
# Step 5 - Pull Ollama models
# ------------------------------------------------------------------------------

Info "Step 5/5 - Pulling Ollama models..."

$llmModel = "llama3.2:3b"

$envLine = Select-String `
    -Path ".env" `
    -Pattern "^LLM_MODEL=" |
    Select-Object -First 1

if ($envLine) {

    $llmModel = ($envLine.Line -split "=", 2)[1].
        Trim().
        Trim('"').
        Trim("'")
}

Info "Pulling model: $llmModel"

docker exec meeting_ollama ollama pull $llmModel

Success "Model '$llmModel' ready."

# Meeting model
$meetingModel = "llama3.2:3b"

$meetingLine = Select-String `
    -Path ".env" `
    -Pattern "^MEETING_LLM_MODEL=" |
    Select-Object -First 1

if ($meetingLine) {

    $meetingModel = ($meetingLine.Line -split "=", 2)[1].
        Trim().
        Trim('"').
        Trim("'")
}

if ($meetingModel -ne $llmModel) {

    Info "Pulling meeting model: $meetingModel"

    docker exec meeting_ollama ollama pull $meetingModel

    Success "Model '$meetingModel' ready."
}

# ------------------------------------------------------------------------------
# Start application
# ------------------------------------------------------------------------------

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host " Setup complete! Starting the application... " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""

Write-Host " App URL        : http://localhost:8000" -ForegroundColor White
Write-Host " Intelligence   : http://localhost:8000/intelligence/topics" -ForegroundColor White
Write-Host " MinIO UI       : http://localhost:9001" -ForegroundColor White
Write-Host " Ollama API     : http://meeting_ollama:11434" -ForegroundColor White
Write-Host " Qdrant UI      : http://localhost:6333/dashboard" -ForegroundColor White
Write-Host " Memgraph Bolt  : bolt://localhost:7687  (enable: MEMGRAPH_ENABLED=true)" -ForegroundColor White
Write-Host ""

Write-Host " NLP Stack    : spaCy + PyTextRank + Sumy + BERTopic + SentenceTransformers" -ForegroundColor DarkGray
Write-Host " Graph        : Memgraph (disabled by default — set MEMGRAPH_ENABLED=true)" -ForegroundColor DarkGray
Write-Host " MinIO Login  : admin / minioadmin" -ForegroundColor DarkGray
Write-Host " Stop Stack   : docker compose down" -ForegroundColor DarkGray
Write-Host " View Logs    : docker compose logs -f app" -ForegroundColor DarkGray
Write-Host " Rebuild      : docker compose build --no-cache app" -ForegroundColor DarkGray
Write-Host ""

docker compose up -d app

Info "Tailing app logs (Ctrl+C to exit - containers keep running)..."

docker compose logs -f app
