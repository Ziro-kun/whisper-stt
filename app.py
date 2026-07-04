import os
import sys
import subprocess
import tempfile
import datetime
import warnings
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import torch
import gradio as gr
import soundfile as sf
from pathlib import Path
import numpy as np

# PyTorch 2.6 weights_only 호환성 (pyannote 모델 로드)
import torch.serialization
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
import torch._utils
for name in dir(torch._utils):
    if name.startswith('_rebuild'):
        torch.serialization.add_safe_globals([getattr(torch._utils, name)])

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    try:
        kwargs_copy = kwargs.copy()
        kwargs_copy['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs_copy)
    except:
        return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load


def _load_dotenv():
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────────────

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL = "o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2"
DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# ── 모델 상태 (백그라운드 로딩) ────────────────────────────────────────────────

_model_ready  = False
_model_error  = ""
_load_log     = []
asr_pipe      = None
diarize_pipe  = None
vad_model     = None
vad_get_speech_timestamps = None


def _log(msg: str):
    print(msg, flush=True)
    _load_log.append(msg)


def _log_timed(msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _load_models():
    global _model_ready, _model_error, asr_pipe, diarize_pipe, vad_model, vad_get_speech_timestamps

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from pyannote.audio import Pipeline as DiarizationPipeline

    try:
        _log(f"[로딩] 디바이스: {DEVICE}")
        _log("[로딩] Whisper 모델 로딩 중... (첫 실행 시 수 GB 다운로드)")

        processor = AutoProcessor.from_pretrained(WHISPER_MODEL, token=HF_TOKEN or None)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            token=HF_TOKEN or None,
        )
        if DEVICE in ("cuda", "mps"):
            model = model.to(DEVICE)

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE if DEVICE in ("cuda", "mps") else -1,
        )
        _log("[로딩] Whisper 모델 준비 완료.")

        _log("[로딩] VAD(음성 감지) 모델 로딩 중...")
        try:
            vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
            vad_get_speech_timestamps = vad_utils[0]
        except:
            # torch.hub 실패 시 직접 로드
            from silero_vad import load_silero_vad, get_speech_timestamps
            vad_model = load_silero_vad()
            vad_get_speech_timestamps = get_speech_timestamps
        if DEVICE == "cuda":  # MPS는 silero-vad JIT 모델의 그래프 퓨저를 지원하지 않아 CPU 고정
            vad_model = vad_model.to(DEVICE)
        _log("[로딩] VAD 모델 준비 완료.")

        _log("[로딩] 화자 분리 모델 로딩 중...")
        diarize_pipe = DiarizationPipeline.from_pretrained(
            DIARIZE_MODEL,
            token=HF_TOKEN or True,
        )
        if DEVICE in ("cuda", "mps"):
            diarize_pipe.to(torch.device(DEVICE))

        _model_ready = True
        _log("[로딩] 모든 모델 준비 완료.")

    except Exception as e:
        _model_error = str(e)
        _log(f"[오류] 모델 로딩 실패: {e}")
        _log("")
        _log("확인사항:")
        _log("  1. 인터넷 연결 상태 확인")
        _log(f"  2. HuggingFace 모델 사용 동의 필요: https://huggingface.co/{DIARIZE_MODEL}")
        _log("  3. 디스크 공간 확인 (모델 약 3~5GB)")


# 백그라운드에서 모델 로딩 시작
threading.Thread(target=_load_models, daemon=True).start()


# ── VAD 전처리 ────────────────────────────────────────────────────────────────

def extract_voice_segments(wav_path: str, threshold: float = 0.5) -> tuple[str, list]:
    """
    VAD로 음성 구간만 추출하고, 구간 정보를 반환

    Returns:
        (processed_wav_path, segments_info)
        segments_info: [(start_sec, end_sec), ...]
    """
    try:
        wav_data, sr = sf.read(wav_path)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)

        # 16kHz로 변환 (VAD 모델 요구사항)
        if sr != 16000:
            import resampy
            wav_data = resampy.resample(wav_data, sr, 16000)
            sr = 16000

        # VAD 처리
        wav_tensor = torch.from_numpy(wav_data).float()
        if DEVICE == "cuda":
            wav_tensor = wav_tensor.to(DEVICE)

        speech_timestamps = vad_get_speech_timestamps(
            wav_tensor, vad_model, threshold=threshold, sampling_rate=sr, return_seconds=False
        )
        segments = [(ts["start"] / sr, ts["end"] / sr) for ts in speech_timestamps]

        # 음성 구간만 추출해서 이어 붙이기
        if not segments:
            # 음성이 없으면 전체 반환 (에러 방지)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sf.write(tmp.name, wav_data, sr)
            return tmp.name, [(0, len(wav_data) / sr)]

        processed_data = []
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            processed_data.append(wav_data[start_idx:end_idx])

        processed_wav = np.concatenate(processed_data)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, processed_wav, sr)

        return tmp.name, segments

    except Exception as e:
        _log(f"[경고] VAD 처리 실패: {e}. 원본 오디오 사용합니다.")
        return wav_path, [(0, 0)]


# ── 환각 필터 ─────────────────────────────────────────────────────────────────

def is_hallucination(text: str) -> bool:
    """
    반복되거나 무의미한 텍스트 감지
    """
    if not text or len(text.strip()) == 0:
        return True

    words = text.split()
    if not words:
        return True

    # 같은 어절이 3회 이상 반복
    for word in set(words):
        if words.count(word) >= 3:
            return True

    # 단일 문자가 10회 이상 반복 (예: "... 아아아아아...")
    for char in set(text):
        if char.isalpha() and text.count(char) >= 10:
            return True

    return False


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def seconds_to_hms(sec: float) -> str:
    return str(datetime.timedelta(seconds=int(sec)))


def _progress_bar(frac: float, width: int = 25) -> str:
    filled = int(frac * width)
    return f"[{'■' * filled}{'□' * (width - filled)}]"


def assign_speaker(start: float, end: float, diarization) -> str:
    overlap: dict[str, float] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        o = min(turn.end, end) - max(turn.start, start)
        if o > 0:
            overlap[speaker] = overlap.get(speaker, 0) + o
    return max(overlap, key=overlap.get) if overlap else "UNKNOWN"


def load_audio_as_wav(file_path: str) -> str:
    try:
        data, sr = sf.read(file_path, always_2d=True)
        data = data.mean(axis=1)
        if sr != 16000:
            import resampy
            data = resampy.resample(data, sr, 16000)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, data, 16000)
        return tmp.name
    except Exception:
        pass

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", tmp.name, "-loglevel", "error"],
        capture_output=True,
    )
    if result.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(
            "오디오 변환 실패.\n"
            f"ffmpeg 오류: {result.stderr.decode(errors='replace').strip()}"
        )
    return tmp.name


# ── 핵심 처리 ─────────────────────────────────────────────────────────────────

def transcribe(audio_path: str, num_speakers: int | None, vad_threshold: float, progress=None):
    if not _model_ready:
        if _model_error:
            yield f"모델 로딩 실패:\n{_model_error}\n\nhttps://huggingface.co/{DIARIZE_MODEL} 에서 사용 동의 후 재시작하세요.", None
        else:
            log_tail = "\n".join(_load_log[-5:]) if _load_log else "(로딩 시작 중...)"
            yield f"모델 로딩 중입니다. 잠시 후 다시 시도해주세요.\n\n진행 상황 (터미널 확인):\n{log_tail}", None
        return

    if not audio_path:
        yield "파일을 선택해주세요.", None
        return

    wav_path = None
    job_t0 = time.time()
    try:
        _log_timed(f"[전사 시작] 파일: {Path(audio_path).name}")

        if progress:
            progress(0.0, desc="오디오 변환 중...")
        yield "오디오 변환 중...", None
        wav_path = load_audio_as_wav(audio_path)

        audio_info = sf.info(wav_path)
        duration_s = audio_info.duration
        dur_str = f"{int(duration_s // 60)}분 {int(duration_s % 60)}초"
        _log_timed(f"[오디오 변환 완료] 길이: {dur_str} ({duration_s:.1f}초)")

        # ── VAD 전처리 ────────────────────────────────────────────────────────────
        if progress:
            progress(0.05, desc="음성 구간 감지 중...")
        yield "음성 구간 감지 중...", None

        vad_t0 = time.time()
        processed_wav_path, voice_segments = extract_voice_segments(wav_path, threshold=vad_threshold)
        _log_timed(f"[VAD 완료] 소요: {time.time() - vad_t0:.1f}초")

        # ── STT ──────────────────────────────────────────────────────────────────
        yield f"전사(STT) 진행 중... (오디오 {dur_str})\n{_progress_bar(0)}", None

        result_holder: list = [None]
        error_holder:  list = [None]
        asr_done = threading.Event()

        def _run_asr():
            try:
                result = asr_pipe(
                    processed_wav_path,
                    chunk_length_s=30,
                    stride_length_s=5,
                    generate_kwargs={"language": "ko", "task": "transcribe"},
                    return_timestamps=True,
                )

                chunks = []
                if isinstance(result, dict):
                    if "chunks" in result:
                        chunks = result["chunks"]
                    elif "text" in result:
                        chunks = [{
                            "timestamp": (0, 0),
                            "text": result["text"],
                        }]
                result_holder[0] = {"chunks": chunks}
            except Exception as e:
                error_holder[0] = e
            finally:
                asr_done.set()

        threading.Thread(target=_run_asr, daemon=True).start()

        CYCLE = 60.0
        t0 = time.time()
        while not asr_done.wait(timeout=5):
            elapsed = time.time() - t0
            frac = (elapsed % CYCLE) / CYCLE
            elapsed_str = f"{int(elapsed // 60)}분 {int(elapsed % 60)}초 경과"
            if progress:
                progress(0.1 + min(elapsed / 1800, 0.55), desc=f"전사 중... {elapsed_str}")
            yield f"전사(STT) 진행 중... (오디오 {dur_str})\n{_progress_bar(frac)}  {elapsed_str}", None

        if error_holder[0]:
            raise error_holder[0]

        _log_timed(f"[STT 완료] 소요: {time.time() - t0:.1f}초")

        chunks = result_holder[0].get("chunks", [])

        # 환각 필터링
        filtered_chunks = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text and not is_hallucination(text):
                filtered_chunks.append(chunk)

        chunks = filtered_chunks

        if not chunks:
            yield "전사 결과가 없습니다. 오디오를 확인해 주세요.", None
            return

        # ── 화자 분리 ─────────────────────────────────────────────────────────────
        diarize_holder: list = [None]
        diarize_error:  list = [None]
        diarize_done = threading.Event()

        def _run_diarize():
            try:
                diarize_kwargs = {}
                if num_speakers and num_speakers > 0:
                    diarize_kwargs["num_speakers"] = num_speakers
                wav_data, wav_sr = sf.read(wav_path)
                if wav_data.ndim > 1:
                    wav_data = wav_data.mean(axis=1)
                wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
                result = diarize_pipe(
                    {"waveform": wav_tensor, "sample_rate": wav_sr},
                    **diarize_kwargs,
                )
                if hasattr(result, "itertracks"):
                    diarize_holder[0] = result
                elif hasattr(result, "speaker_diarization"):
                    diarize_holder[0] = result.speaker_diarization
                else:
                    diarize_holder[0] = result
            except Exception as e:
                diarize_error[0] = e
            finally:
                diarize_done.set()

        threading.Thread(target=_run_diarize, daemon=True).start()

        t0 = time.time()
        while not diarize_done.wait(timeout=5):
            elapsed = time.time() - t0
            frac = (elapsed % CYCLE) / CYCLE
            elapsed_str = f"{int(elapsed // 60)}분 {int(elapsed % 60)}초 경과"
            if progress:
                progress(0.65 + min(elapsed / 1800, 0.3), desc=f"화자 분리 중... {elapsed_str}")
            yield f"화자 분리 중...\n{_progress_bar(frac)}  {elapsed_str}", None

        if diarize_error[0]:
            raise diarize_error[0]

        _log_timed(f"[화자분리 완료] 소요: {time.time() - t0:.1f}초")

        diarization = diarize_holder[0]

        lines = []
        prev_speaker = None

        for chunk in chunks:
            ts    = chunk.get("timestamp", (0, 0))
            start = ts[0] if ts[0] is not None else 0
            end   = ts[1] if ts[1] is not None else start + 1
            text  = chunk.get("text", "").strip()
            if not text:
                continue

            speaker = assign_speaker(start, end, diarization)
            ts_str  = f"[{seconds_to_hms(start)} -> {seconds_to_hms(end)}]"

            if speaker != prev_speaker:
                lines.append(f"\n{speaker}")
                prev_speaker = speaker

            lines.append(f"{ts_str}  {text}")

        output_text = "\n".join(lines).strip()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = Path(audio_path).stem
        save_path = OUTPUT_DIR / f"{stem}_{timestamp}.txt"
        save_path.write_text(output_text, encoding="utf-8")

        total_elapsed = time.time() - job_t0
        ratio = f"{total_elapsed / duration_s * 100:.1f}%" if duration_s else "N/A"
        _log_timed(f"[전체 완료] 총 소요: {total_elapsed:.1f}초 (오디오 길이 대비 {ratio}) → {save_path.name}")

        if progress:
            progress(1.0, desc="완료")
        yield output_text, str(save_path)

    except Exception as e:
        _log_timed(f"[오류] {e} (실패 전까지 소요: {time.time() - job_t0:.1f}초)")
        yield f"[오류] {e}", None

    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ── 로딩 상태 폴링 ─────────────────────────────────────────────────────────────

def get_status():
    if _model_error:
        return f"모델 로딩 실패: {_model_error}"
    if _model_ready:
        return "준비 완료 - 파일을 업로드하고 변환을 시작하세요."
    log_tail = _load_log[-1] if _load_log else "모델 로딩 중..."
    return f"로딩 중... {log_tail}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="한국어 STT 녹취록", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("## 한국어 STT 녹취록\n타임스탬프 · 화자 분리 포함 텍스트 변환")

    status_bar = gr.Markdown(value=get_status, every=3)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="오디오 파일 업로드",
                type="filepath",
                sources=["upload"],
            )
            num_speakers = gr.Number(
                label="화자 수 (모를 경우 0 → 자동 감지)",
                value=0,
                precision=0,
                minimum=0,
                maximum=20,
            )

            with gr.Accordion("VAD 고급 설정", open=False):
                vad_threshold = gr.Slider(
                    label="VAD 민감도 (0.0~1.0)",
                    value=0.5,
                    minimum=0.1,
                    maximum=0.9,
                    step=0.1,
                    info="높을수록 음성만 추출, 낮을수록 노이즈도 포함"
                )

            run_btn = gr.Button("변환 시작", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="녹취록 미리보기",
                lines=20,
            )
            file_output = gr.File(label=".txt 파일 다운로드")

    result_status = gr.Markdown("")

    def run(audio, n_spk, vad_thresh, progress=gr.Progress()):
        n = int(n_spk) if n_spk and int(n_spk) > 0 else None
        for preview, path in transcribe(audio, n, vad_thresh, progress=progress):
            if path is None:
                yield preview, None, ""
            else:
                yield preview, path, f"저장 완료: `{path}`"

    run_btn.click(
        fn=run,
        inputs=[audio_input, num_speakers, vad_threshold],
        outputs=[output_text, file_output, result_status],
    )

if __name__ == "__main__":
    print(f"Gradio 서버 시작 중... http://localhost:7860", flush=True)
    os.environ["GRADIO_QUEUE_CONCURRENCY_COUNT"] = "1"
    os.environ["GRADIO_QUEUE_DEFAULT_CONCURRENCY"] = "1"
    demo.queue(max_size=20, api_open=False)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True, show_error=True)
