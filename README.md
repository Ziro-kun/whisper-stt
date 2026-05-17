# 한국어 STT 녹취록

클로바노트처럼 음성 파일을 업로드하면 **타임스탬프 + 화자 분리**가 포함된 한국어 녹취록을 자동 생성하는 로컬 실행 앱.  
파일이 외부 서버로 전송되지 않으며, 최초 모델 다운로드 이후에는 인터넷 없이도 동작한다.

---

## 의의

| 항목 | 내용 |
|------|------|
| 보안 | 음성 파일이 외부로 전송되지 않음 |
| 비용 | 무료 (클라우드 STT API 불필요) |
| 기능 | 타임스탬프 + 화자 분리 — 클로바노트와 동일 수준 |
| 플랫폼 | macOS Apple Silicon (MPS 가속) |

---

## 시스템 구조

```
오디오 파일 업로드
       │
       ▼
[오디오 전처리]
 soundfile / ffmpeg → 16kHz mono WAV
       │
       ├──────────────────────────────┐
       ▼                              ▼
[STT]                        [화자 분리]
Whisper Large v3 turbo       pyannote/speaker-diarization-3.1
(한국어 파인튜닝, MPS)         (MPS)
타임스탬프 청크 단위 전사       화자별 발화 구간 생성
       │                              │
       └──────────────┬───────────────┘
                      ▼
              [청크 ↔ 화자 매핑]
              구간 겹침 기반 할당
                      │
                      ▼
              [결과 저장 + 미리보기]
              outputs/파일명_YYYYMMDD_HHMMSS.txt
```

---

## 사용 모델

| 역할 | 모델 |
|------|------|
| STT | `o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2` |
| 화자 분리 | `pyannote/speaker-diarization-3.1` |

- 두 모델 모두 첫 실행 시 HuggingFace에서 자동 다운로드 (합계 약 3~5GB)
- 이후 실행부터는 로컬 캐시에서 즉시 로딩

### STT 모델

OpenAI Whisper Large v3 turbo를 Zeroth-Korean 데이터셋으로 파인튜닝한 모델.  
기본 turbo 모델 대비 한국어 구어체 인식률이 향상되어 있다.

### 화자 분리 모델

pyannote 3.x는 오픈소스 화자 분리 모델 중 정확도가 높은 편이며,  
별도 학습 없이 한국어 오디오에 바로 적용 가능하다.

---

## 출력 형식

```
SPEAKER_00
[0:00:01 -> 0:00:05]  안녕하세요, 오늘 회의를 시작하겠습니다.
[0:00:05 -> 0:00:09]  먼저 지난주 결과를 공유드리겠습니다.

SPEAKER_01
[0:00:10 -> 0:00:14]  네, 감사합니다. 말씀하신 내용 잘 들었습니다.

SPEAKER_00
[0:00:15 -> 0:00:20]  그럼 본론으로 들어가겠습니다.
```

저장 경로: `outputs/{원본파일명}_{YYYYMMDD_HHMMSS}.txt`

---

## 사전 요건

- macOS (Apple Silicon)
- Python 3.11 이상
- ffmpeg (`brew install ffmpeg`)
- HuggingFace 계정 및 토큰 (Read 권한)
- 아래 두 모델 접근 동의 완료
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

---

## 설치 및 실행

### 1. HuggingFace 토큰 설정

프로젝트 폴더에 `.env` 파일 생성:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

토큰 발급: https://huggingface.co/settings/tokens

### 2. 실행

`STT 녹취록.command` 파일을 더블클릭.

처음 실행 시 자동으로:
1. Python 가상환경 생성
2. 패키지 설치 (5~10분)
3. 모델 다운로드 (3~5GB, 이후 생략)
4. 브라우저에서 `http://localhost:7860` 자동 오픈

### 3. 사용법

1. 브라우저에서 오디오 파일 업로드 (mp3, wav, m4a 등)
2. 화자 수 입력 (모를 경우 0 → 자동 감지)
3. **변환 시작** 클릭
4. 완료 후 미리보기 확인 및 `.txt` 파일 다운로드

---

## 성능 참고

| 환경 | 오디오 1시간 처리 시간 (참고) |
|------|---------------------------|
| Apple M4 (MPS) | 약 30~60분 |
| NVIDIA GPU (CUDA) | 약 10~20분 |

처리 시간의 대부분은 화자 분리(pyannote) 단계에서 소요된다.

---

## 알려진 제약

- **처리 속도**: 로컬 모델 특성상 클라우드 서비스보다 느림
- **겹치는 발화**: 두 화자가 동시에 말하는 구간은 한 명만 할당됨
- **배경 소음**: 전사 품질에 영향을 미침 (별도 노이즈 제거 없음)
- **실시간 처리 불가**: 파일 업로드 후 일괄 처리 방식
- **마이크 입력 미지원**: 파일 업로드만 지원
