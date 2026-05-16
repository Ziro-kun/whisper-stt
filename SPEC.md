# 한국어 STT 녹취록 — 현재 개발 명세

## 개요

한국어 음성 파일을 업로드하면 **전사(STT) + 화자 분리(Diarization)** 를 수행하고,
타임스탬프·화자 레이블이 포함된 텍스트 파일을 출력하는 로컬 실행 웹 앱.

---

## 사용 모델

| 역할 | 모델 |
|------|------|
| STT | `o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2` |
| 화자 분리 | `pyannote/speaker-diarization-3.1` |

- 두 모델 모두 HuggingFace에서 최초 실행 시 자동 다운로드 (약 3~5 GB)
- `pyannote/speaker-diarization-3.1` 은 HuggingFace 모델 페이지에서 **사용 동의(gating)** 필요
- 인증 토큰은 `HF_TOKEN` 환경변수로 주입 (코드에 하드코딩 금지)

---

## 실행 구조

```
run.bat
  └─ run.ps1          # 가상환경 생성·패키지 설치·ffmpeg 다운로드·환경변수 설정
       └─ launcher.py # ffmpeg PATH 설정, outputs/ 디렉토리 생성
            └─ app.py # Gradio 앱 본체
```

### run.ps1 담당 작업
1. Python 3.11 / 3.12 자동 탐색
2. `.venv` 가상환경 생성 (없을 경우)
3. 패키지 설치 — torch(CPU), transformers, gradio, pyannote.audio 등
4. `ffmpeg/ffmpeg.exe` 다운로드 (`download_ffmpeg.ps1` 호출)
5. `HF_TOKEN` 환경변수 설정 후 `launcher.py` 실행

---

## 디바이스 자동 선택

```
CUDA → MPS(Apple Silicon) → CPU
```

- CUDA / MPS: `float16`
- CPU: `float32`

---

## 처리 파이프라인 (`app.py`)

### 1. 모델 백그라운드 로딩
- 앱 시작 즉시 별도 스레드에서 두 모델 로딩
- Gradio UI 상태바가 3초 간격으로 로딩 진행 상황 폴링
- 로딩 전 변환 요청 시 현재 진행 로그 표시 후 중단

### 2. 오디오 전처리
- `soundfile` 로 읽기 → 모노 변환 → 16 kHz 리샘플(`resampy`)
- 실패 시 `ffmpeg` 로 fallback 변환
- 임시 WAV 파일 생성, 처리 후 자동 삭제

### 3. STT (Whisper)
- `chunk_length_s=30`, `stride_length_s=5` 청킹
- `return_timestamps=True` — 청크 단위 타임스탬프 반환
- 언어 고정: `korean`
- 별도 스레드에서 실행, 2초 간격 진행률 폴링
- 예상 소요 시간: 오디오 길이 × 2.0 (CPU 기준 보수적 추정)

### 4. 화자 분리 (pyannote)
- 화자 수 입력 가능 (0 입력 시 자동 감지)
- torchcodec DLL 오류 우회: 파일 경로 대신 `{"waveform": tensor, "sample_rate": sr}` dict 전달
- 청크별 타임스탬프 구간과 다이어리제이션 구간의 **겹침(overlap) 최대 화자** 배정

### 5. 결과 포맷
```
SPEAKER_00
[0:00:00 -> 0:00:05]  안녕하세요.
[0:00:05 -> 0:00:09]  반갑습니다.

SPEAKER_01
[0:00:10 -> 0:00:14]  네, 안녕하세요.
```
- 연속된 동일 화자는 헤더 중복 생략
- `outputs/{원본파일명}_{YYYYMMDD_HHMMSS}.txt` 로 자동 저장

---

## UI (Gradio)

- **상태바**: 모델 로딩 상태 실시간 표시 (3초 폴링)
- **오디오 업로드**: 파일 업로드 전용 (마이크 입력 미지원)
- **화자 수 입력**: 숫자 입력 (0 = 자동)
- **변환 시작 버튼**: 스트리밍 진행률 표시
- **녹취록 미리보기**: 텍스트박스 (20줄)
- **파일 다운로드**: 완료된 `.txt` 파일 직접 다운로드
- 접속 주소: `http://localhost:7860` (외부 공유 비활성화)

---

## 의존 패키지

| 패키지 | 용도 |
|--------|------|
| `torch` | 모델 연산 |
| `transformers` | Whisper 모델 로딩 |
| `accelerate` | 모델 최적화 로딩 |
| `gradio` | 웹 UI |
| `pyannote.audio` | 화자 분리 |
| `soundfile` | 오디오 읽기/쓰기 |
| `resampy` | 리샘플링 |
| `huggingface_hub` | HF 모델 다운로드 |
| `ffmpeg` (바이너리) | 오디오 포맷 변환 fallback |

---

## 알려진 제약 및 특이사항

- `torchcodec` Windows DLL 오류 → 설치 후 강제 제거, pyannote 입력을 텐서 dict로 우회
- CPU 환경에서 긴 오디오는 수 분 소요 (CUDA 권장)
- 실시간 스트리밍 전사 미지원 — 파일 업로드 후 일괄 처리
- 마이크 입력 미지원 (Gradio `sources=["upload"]` 고정)
- 출력 디렉토리 `outputs/` 는 git 추적 제외
