#Requires -Version 5.0
$ErrorActionPreference = "Stop"
$APP_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PY = "$APP_DIR\.venv\Scripts\python.exe"
$LOG     = "$APP_DIR\run.log"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

"Run started: $(Get-Date)" | Out-File $LOG -Encoding UTF8

Write-Host ""
Write-Host "========================================"
Write-Host "  Korean STT Transcription App"
Write-Host "========================================"
Write-Host ""

# -- env vars --
$env:HF_TOKEN         = $env:HF_TOKEN  # set HF_TOKEN in your environment before running
$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH             = "$APP_DIR\ffmpeg;$env:PATH"

# -- Find Python --
$PY_CMD = $null
@(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
) | ForEach-Object { if (-not $PY_CMD -and (Test-Path $_)) { $PY_CMD = $_ } }

if (-not $PY_CMD) {
    Log "[Error] Python not found. Install Python 3.12 from https://www.python.org"
    Read-Host "Press Enter to exit"
    exit 1
}
Log "[1/4] Python: $PY_CMD"

# -- Create venv --
if (-not (Test-Path $VENV_PY)) {
    Log "[2/4] Creating virtual environment..."
    & $PY_CMD -m venv "$APP_DIR\.venv"
    if ($LASTEXITCODE -ne 0) {
        Log "[Error] Failed to create virtual environment."
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Log "[2/4] Virtual environment OK"
}

# -- Check packages --
Log "[3/4] Checking packages..."
$SP = "$APP_DIR\.venv\Lib\site-packages"
$pkgOk = (Test-Path "$SP\torch\__init__.py") -and
         (Test-Path "$SP\transformers\__init__.py") -and
         (Test-Path "$SP\gradio\__init__.py") -and
         (Test-Path "$SP\pyannote\audio\__init__.py")

if (-not $pkgOk) {
    Log "[3/4] Installing packages (5-10 min)..."
    & $VENV_PY -m pip install --quiet --upgrade pip 2>&1 | Add-Content $LOG
    & $VENV_PY -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | Add-Content $LOG
    & $VENV_PY -m pip install --quiet transformers accelerate gradio 2>&1 | Add-Content $LOG
    & $VENV_PY -m pip install --quiet pyannote.audio soundfile resampy huggingface_hub 2>&1 | Add-Content $LOG
    Log "[3/4] Install complete"
} else {
    Log "[3/4] Packages OK"
}
# torchcodec은 Windows에서 DLL 오류를 일으키므로 제거 (pyannote는 soundfile로 대체)
try { & $VENV_PY -m pip uninstall torchcodec -y | Out-Null } catch {}

# -- ffmpeg --
if (-not (Test-Path "$APP_DIR\ffmpeg\ffmpeg.exe")) {
    Log "[4/4] Downloading ffmpeg..."
    if (-not (Test-Path "$APP_DIR\ffmpeg")) { New-Item -ItemType Directory "$APP_DIR\ffmpeg" | Out-Null }
    & powershell -ExecutionPolicy Bypass -File "$APP_DIR\download_ffmpeg.ps1" "$APP_DIR\ffmpeg"
    if ($LASTEXITCODE -ne 0) { Log "[Warning] ffmpeg download failed." }
} else {
    Log "[4/4] ffmpeg OK"
}

# -- Launch app --
Log ""
Log "Launching app.py..."
Write-Host ""
Write-Host "========================================"
Write-Host "  Open browser: http://localhost:7860"
Write-Host "  Press Ctrl+C to stop."
Write-Host "========================================"
Write-Host ""

Set-Location $APP_DIR
try {
    & $VENV_PY "$APP_DIR\app.py"
    $code = $LASTEXITCODE
} catch {
    Log "[Exception] $_"
    $code = 1
}

Log "app.py exited with code: $code"

if ($code -ne 0) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  [Error] App failed (code $code)"
    Write-Host "  Log: $LOG"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "--- Last 40 lines of log ---"
    Get-Content $LOG | Select-Object -Last 40
    Write-Host ""
}

Read-Host "Press Enter to close"
