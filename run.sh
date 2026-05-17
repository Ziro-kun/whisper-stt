#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"
LOG="$APP_DIR/run.log"

log() { echo "[$( date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

echo "" | tee "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  Korean STT Transcription App (macOS)"  | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo ""

# ── [1/4] Python 탐색 ──────────────────────────────────────────────────────────
PY_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; v=sys.version_info; print(v.major, v.minor)")
        major=$(echo "$ver" | cut -d' ' -f1)
        minor=$(echo "$ver" | cut -d' ' -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PY_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    log "[Error] Python 3.11 이상을 찾을 수 없습니다."
    log "  설치: brew install python@3.12"
    exit 1
fi
log "[1/4] Python: $PY_CMD ($($PY_CMD --version 2>&1))"

# ── [2/4] 가상환경 생성 ────────────────────────────────────────────────────────
if [ ! -f "$VENV_PY" ]; then
    log "[2/4] 가상환경 생성 중..."
    "$PY_CMD" -m venv "$APP_DIR/.venv"
else
    log "[2/4] 가상환경 OK"
fi

# ── [3/4] 패키지 설치 확인 ────────────────────────────────────────────────────
if ! "$VENV_PY" -c "import torch, transformers, gradio, pyannote" 2>/dev/null; then
    log "[3/4] 패키지 설치 중... (최초 실행 시 5~10분 소요)"
    "$VENV_PY" -m pip install --quiet --upgrade pip
    "$VENV_PY" -m pip install --quiet -r "$APP_DIR/requirements.txt"
    log "[3/4] 설치 완료"
else
    log "[3/4] 패키지 OK"
fi

# ── [4/4] ffmpeg 확인 ─────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    log "[4/4] ffmpeg를 찾을 수 없습니다."
    log "  설치: brew install ffmpeg"
    exit 1
else
    log "[4/4] ffmpeg OK ($(ffmpeg -version 2>&1 | head -1))"
fi

# ── .env 로드 ─────────────────────────────────────────────────────────────────
if [ -f "$APP_DIR/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$APP_DIR/.env"
    set +o allexport
fi

# ── HF_TOKEN 확인 ─────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "  HF_TOKEN이 설정되지 않았습니다."
    echo "  실행 방법: HF_TOKEN=hf_xxxx ./run.sh"
    echo "  또는:      export HF_TOKEN=hf_xxxx && ./run.sh"
    exit 1
fi
export HF_TOKEN

echo ""
echo "========================================"
echo "  브라우저에서 접속: http://localhost:7860"
echo "  종료: Ctrl+C"
echo "========================================"
echo ""

cd "$APP_DIR"
exec "$VENV_PY" "$APP_DIR/app.py"
