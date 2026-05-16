#!/usr/bin/env python3
"""
한국어 STT 녹취록 앱 런처
run.bat에서 호출하거나 직접 실행 가능.
HF_TOKEN은 환경변수로 전달받거나 run.bat에서 설정됨.
"""

import os
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

os.environ["PYTHONIOENCODING"] = "utf-8"

APP_DIR   = Path(__file__).parent
FFMPEG    = APP_DIR / "ffmpeg"
APP_PY    = APP_DIR / "app.py"

if FFMPEG.exists():
    os.environ["PATH"] = f"{FFMPEG};{os.environ.get('PATH', '')}"

(APP_DIR / "outputs").mkdir(exist_ok=True)

if not APP_PY.exists():
    print("[오류] app.py를 찾을 수 없습니다.")
    sys.exit(1)

os.chdir(APP_DIR)
try:
    subprocess.run([sys.executable, str(APP_PY)], check=False)
except KeyboardInterrupt:
    print("\n앱을 종료합니다.")
