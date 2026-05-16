#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

APP_DIR = Path(__file__).parent
APP_PY  = APP_DIR / "app.py"

(APP_DIR / "outputs").mkdir(exist_ok=True)

if not APP_PY.exists():
    print("[오류] app.py를 찾을 수 없습니다.")
    sys.exit(1)

os.chdir(APP_DIR)
try:
    subprocess.run([sys.executable, str(APP_PY)], check=False)
except KeyboardInterrupt:
    print("\n앱을 종료합니다.")
