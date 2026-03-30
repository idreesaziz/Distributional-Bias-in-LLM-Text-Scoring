# =============================================================
# No One Gets an A — Remote Machine Setup
# =============================================================
# Run once on the Windows machine with the NVIDIA GPU.
# Does everything in order:
#   1. Install Ollama (if missing)
#   2. Start the Ollama server
#   3. Pull all local models defined in the study
#   4. Install Python dependencies
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_remote.ps1
# =============================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step($msg) {
    Write-Host "`n$(('=' * 60))" -ForegroundColor DarkCyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ('=' * 60) -ForegroundColor DarkCyan
}

function Test-OllamaRunning {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 3 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# ── 1. Install Ollama ──────────────────────────────────────────

Write-Step "Step 1 / 4 — Checking Ollama"

$ollamaExe = Get-Command "ollama" -ErrorAction SilentlyContinue
if ($ollamaExe) {
    $ver = & ollama --version 2>&1
    Write-Host "  Ollama already installed: $ver" -ForegroundColor Green
} else {
    Write-Host "  Ollama not found. Downloading installer..." -ForegroundColor Yellow
    $installerPath = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-WebRequest `
        -Uri "https://ollama.com/download/OllamaSetup.exe" `
        -OutFile $installerPath `
        -UseBasicParsing

    Write-Host "  Running silent install..."
    $proc = Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Error "  Ollama installer exited with code $($proc.ExitCode). Install manually: https://ollama.com/download"
        exit 1
    }

    # Refresh PATH so ollama is immediately available in this session
    $machinePath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath    = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $env:PATH    = "$machinePath;$userPath"

    if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
        Write-Error "  Ollama installed but not on PATH. Open a fresh terminal and re-run this script."
        exit 1
    }
    Write-Host "  Ollama installed successfully." -ForegroundColor Green
}

# ── 2. Start Ollama Server ─────────────────────────────────────

Write-Step "Step 2 / 4 — Starting Ollama server"

if (Test-OllamaRunning) {
    Write-Host "  Server already running at http://localhost:11434" -ForegroundColor Green
} else {
    Write-Host "  Launching 'ollama serve' in the background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden

    Write-Host "  Waiting for server to become ready..."
    $timeout = 60   # seconds
    $elapsed = 0
    $ready   = $false
    while ($elapsed -lt $timeout) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        if (Test-OllamaRunning) {
            $ready = $true
            break
        }
        Write-Host "    ...${elapsed}s" -NoNewline
    }
    Write-Host ""
    if (-not $ready) {
        Write-Error "  Ollama server did not start within ${timeout}s. Check for port conflicts."
        exit 1
    }
    Write-Host "  Server is ready." -ForegroundColor Green
}

# ── 3. Pull Models ─────────────────────────────────────────────

Write-Step "Step 3 / 4 — Pulling models"
Write-Host "  Note: total download is roughly 40-60 GB depending on quantisation."
Write-Host "  Each model shows its own progress bar. Skipping models that fail"
Write-Host "  is non-fatal — you can retry individually with: ollama pull <name>"
Write-Host ""

# Ordered small → large so you can test quickly before committing to 32B
$models = @(
    [PSCustomObject]@{ Tag = "llama3.2:3b";       Label = "LLaMA 3.2  3B"         },
    [PSCustomObject]@{ Tag = "deepseek-r1:7b";    Label = "DeepSeek-R1  7B"       },
    [PSCustomObject]@{ Tag = "llama3.1:8b";       Label = "LLaMA 3.1  8B"         },
    [PSCustomObject]@{ Tag = "deepseek-r1:14b";   Label = "DeepSeek-R1 14B"       },
    [PSCustomObject]@{ Tag = "deepseek-r1:32b";   Label = "DeepSeek-R1 32B (slow)"}
)

$failed = @()
foreach ($m in $models) {
    Write-Host "  Pulling $($m.Label) ($($m.Tag))..." -ForegroundColor Yellow
    & ollama pull $m.Tag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] Failed to pull $($m.Tag). Continuing..." -ForegroundColor Red
        $failed += $m.Tag
    } else {
        Write-Host "  $($m.Label) ready." -ForegroundColor Green
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`n  The following models failed to download:" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "    ollama pull $_" -ForegroundColor Yellow }
}

# ── 4. Python Dependencies ─────────────────────────────────────

Write-Step "Step 4 / 4 — Installing Python dependencies"

if (-not (Get-Command "pip" -ErrorAction SilentlyContinue)) {
    Write-Error "  pip not found. Install Python 3.9+ and make sure it is on PATH."
    exit 1
}

# Run from repo root (this script is in scripts/, so go up one level)
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "  pip install failed."
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host "  Dependencies installed." -ForegroundColor Green

# ── Done ───────────────────────────────────────────────────────

Write-Host ""
Write-Host ('=' * 60) -ForegroundColor Green
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Make sure your .env file has the API keys:" -ForegroundColor White
Write-Host "    OPENAI_API_KEY=sk-..." -ForegroundColor DarkGray
Write-Host "    GOOGLE_API_KEY=..."    -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Then run the local scoring step:" -ForegroundColor White
Write-Host "    python -m src.main --step llm" -ForegroundColor DarkGray
Write-Host ('=' * 60) -ForegroundColor Green