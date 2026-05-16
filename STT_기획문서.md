# 한국어 STT 녹취록 앱 — 개발 기획 문서

> 최초 작성일: 2026-05-16  
> 현재 버전: v0.1 (로컬 실행 · Gradio UI)  
> 플랫폼: macOS (Apple Silicon 우선) / Windows 보조

---

## 1. 프로젝트 개요

### 1.1 목적

오디오 파일을 로컬에서 완전히 처리하여 **타임스탬프 + 화자 분리가 포함된 한국어 녹취록**을 자동 생성한다. 외부 서버 전송 없이 개인 기기에서 동작하는 것이 핵심 요건이다.

### 1.2 핵심 요건

| 항목 | 내용 |
|------|------|
| 실행 환경 | 로컬 (인터넷 불필요, 최초 모델 다운로드 제외) |
| 보안 | 파일이 외부로 전송되지 않음 |
| 입력 형식 | mp3, wav, m4a 등 주요 오디오 포맷 |
| 출력 형식 | 타임스탬프 + 화자 레이블 포함 .txt |
| UI | 브라우저 기반 웹 UI (Gradio) |

### 1.3 제외 범위

- 클라우드 STT API 사용 (OpenAI Whisper API, Clova Speech 등)
- 실시간 마이크 녹음 (파일 업로드만 지원)
- 다국어 전사 (한국어 특화)

---

## 2. 시스템 아키텍처

```
[사용자 브라우저]
       │  오디오 파일 업로드
       ▼
[Gradio Web UI]  ← localhost:7860
       │
       ▼
[오디오 전처리]
  soundfile / ffmpeg
  → 16kHz mono WAV 변환
       │
       ├──────────────────────────────┐
       ▼                              ▼
[STT 파이프라인]             [화자 분리 파이프라인]
Whisper Large v3 turbo       pyannote/speaker-diarization-3.1
(타임스탬프 청크 단위 전사)   (화자 구간 타임라인 생성)
       │                              │
       └──────────────┬───────────────┘
                      ▼
              [청크 ↔ 화자 매핑]
              (구간 겹침 기반 할당)
                      │
                      ▼
              [.txt 파일 저장]
              outputs/파일명_타임스탬프.txt
                      │
                      ▼
              [Gradio UI 미리보기 + 다운로드]
```

---

## 3. 활용 모델

### 3.1 STT 모델 — Whisper 기반 한국어 특화

| 항목 | 내용 |
|------|------|
| 모델 ID | `o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2` |
| 기반 모델 | OpenAI Whisper Large v3 turbo |
| 특화 | 한국어 STT 파인튜닝 (Zeroth-Korean 데이터셋) |
| 허브 | Hugging Face |
| 출력 | 텍스트 + 청크별 타임스탬프 (chunk_length_s=30) |
| 라이선스 | 모델 카드 참조 |

**선택 이유**

Whisper Large v3 turbo는 large 대비 추론 속도가 대폭 개선되어 로컬 CPU 환경에서도 실용적이다. 해당 파인튜닝 모델은 한국어 구어체 인식률이 기본 Whisper 대비 향상되어 있다.

**추론 설정**

```python
chunk_length_s = 30      # 30초 단위 청크 처리
stride_length_s = 5      # 청크 간 5초 오버랩 (경계 손실 방지)
language = "korean"
return_timestamps = True
```

---

### 3.2 화자 분리 모델 — pyannote

| 항목 | 내용 |
|------|------|
| 모델 ID | `pyannote/speaker-diarization-3.1` |
| 의존 모델 | `pyannote/segmentation-3.0` |
| 허브 | Hugging Face |
| 출력 | 화자별 발화 구간 타임라인 (RTTM 형식 내부 처리) |
| 라이선스 | CC BY 4.0 (상업 이용 시 별도 확인 필요) |
| HF 동의 | 사용 전 두 모델 모두 접근 동의 필수 |

**선택 이유**

pyannote 3.x는 현재 오픈소스 화자 분리 모델 중 정확도가 가장 높은 축에 속한다. 별도 학습 없이 한국어 오디오에도 적용 가능하며, Whisper 청크 타임스탬프와 구간 매핑이 용이하다.

**화자 할당 로직**

```
각 Whisper 청크 [start, end]에 대해
→ pyannote 타임라인의 모든 발화 구간과 겹침 시간 계산
→ 겹침이 가장 긴 화자(SPEAKER_XX)를 해당 청크에 할당
```

---

## 4. 기술 스택

| 레이어 | 라이브러리 | 버전 |
|--------|-----------|------|
| UI | Gradio | ≥ 4.31 |
| STT | transformers (pipeline) | ≥ 4.40 |
| 화자 분리 | pyannote.audio | ≥ 3.1 |
| 오디오 처리 | soundfile, resampy | - |
| 오디오 변환 | ffmpeg (외부 의존) | - |
| ML 프레임워크 | PyTorch | ≥ 2.1 |
| 가속 | accelerate | ≥ 0.27 |
| 모델 허브 | huggingface_hub | ≥ 0.22 |

### 디바이스 우선순위

```
CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
```

Apple Silicon(M1~M4)에서는 MPS가 자동 감지되며 float16 연산을 활용한다.

---

## 5. 출력 형식

```
SPEAKER_00
[00:00:01 → 00:00:05]  안녕하세요, 오늘 회의를 시작하겠습니다.
[00:00:05 → 00:00:09]  먼저 지난주 결과를 공유드리겠습니다.

SPEAKER_01
[00:00:10 → 00:00:14]  네, 감사합니다. 말씀하신 내용 잘 들었습니다.

SPEAKER_00
[00:00:15 → 00:00:20]  그럼 본론으로 들어가겠습니다.
```

저장 경로: `outputs/{원본파일명}_{YYYYMMDD_HHMMSS}.txt`

---

## 6. 파일 구성

```
stt_app/
├── app.py                  # 메인 앱 (Gradio UI + 전사 로직)
├── requirements.txt        # Python 패키지 목록
├── 설치_및_실행.bat         # Windows용 자동 설치 + 실행 스크립트
├── STT_기획문서.md          # 본 문서
└── outputs/                # 녹취록 저장 디렉토리 (자동 생성)
```

---

## 7. 환경 설정

### 7.1 사전 요건

- Python 3.11
- ffmpeg (`brew install ffmpeg` / `winget install Gyan.FFmpeg`)
- Hugging Face 계정 및 토큰 (Read 권한)
- 아래 두 모델 접근 동의 완료
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

### 7.2 설치 (macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 7.3 실행

```bash
HF_TOKEN=hf_xxxxxxxxxx python app.py
```

브라우저에서 http://localhost:7860 접속.

### 7.4 첫 실행 유의사항

- Whisper 모델: 약 1.5GB 다운로드
- pyannote 모델: 약 1~2GB 다운로드
- 합계 약 3~4GB, 최초 1회만 소요 (이후 HF 캐시에서 로딩)

---

## 8. 성능 참고

| 환경 | 오디오 1분 처리 시간 (추정) |
|------|--------------------------|
| Apple M4 (MPS) | 30~60초 |
| NVIDIA GPU (CUDA) | 10~20초 |
| CPU only (Intel/AMD) | 3~8분 |

> CPU 전용 환경(업무용 Windows 노트북 등)에서는 긴 파일 처리에 상당한 시간이 소요된다. macOS(MPS)로 이관한 이유.

---

## 9. 향후 개발 계획

### v0.2 — 사용성 개선

- [ ] 진행률 표시 (Progress bar)
- [ ] 여러 파일 일괄 처리 (Batch)
- [ ] 화자 이름 수동 변경 기능 (SPEAKER_00 → 홍길동)

### v0.3 — 출력 형식 확장

- [ ] SRT 자막 파일 (.srt) 출력
- [ ] JSON 출력 (타임스탬프 + 화자 구조화)
- [ ] 화자별 발화 시간 통계

### v0.4 — 품질 개선

- [ ] WhisperX 연동 검토 (단어 단위 정렬 정확도 향상)
- [ ] 화자 분리 후처리 (짧은 구간 병합 등)
- [ ] 노이즈 전처리 옵션

---

## 10. 알려진 제약사항

- **화자 자동 감지 정확도**: 화자 수를 직접 지정하는 것이 자동 감지보다 정확하다.
- **겹치는 발화**: 두 화자가 동시에 말하는 구간은 한 명만 할당된다.
- **배경 음악 / 소음**: 전사 품질에 영향을 미친다. 전처리 미포함.
- **CPU 환경**: 1시간 이상 오디오는 처리 시간이 매우 길다.
- **m4a 처리**: ffmpeg 미설치 시 변환 불가.
