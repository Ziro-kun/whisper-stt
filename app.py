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


def _log(msg: str):
    print(msg, flush=True)
    _load_log.append(msg)


def _load_models():
    global _model_ready, _model_error, asr_pipe, diarize_pipe

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from pyannote.audio import Pipeline as DiarizationPipeline

    try:
        _log(f"[로딩] 디바이스: {DEVICE}")
        _log("[로딩] Whisper 모델 로딩 중... (첫 실행 시 수 GB 다운로드, 수 분 소요)")

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=HF_TOKEN or None,
        )
        whisper_model.to(DEVICE)

        processor = AutoProcessor.from_pretrained(WHISPER_MODEL, token=HF_TOKEN or None)

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
            return_timestamps=True,
            generate_kwargs={"language": "korean"},
        )

        _log("[로딩] 화자 분리 모델 로딩 중...")
        diarize_pipe = DiarizationPipeline.from_pretrained(
            DIARIZE_MODEL,
            token=HF_TOKEN or None,
        )
        if DEVICE == "cuda":
            diarize_pipe.to(torch.device("cuda"))

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


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def seconds_to_hms(sec: float) -> str:
    return str(datetime.timedelta(seconds=int(sec)))


def _progress_bar(frac: float, width: int = 25) -> str:
    filled = int(frac * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {int(frac * 100)}%"


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

def transcribe(audio_path: str, num_speakers: int | None, progress=None):
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
    try:
        if progress:
            progress(0.0, desc="오디오 변환 중...")
        yield "오디오 변환 중...", None
        wav_path = load_audio_as_wav(audio_path)

        audio_info = sf.info(wav_path)
        duration_s = audio_info.duration
        dur_str = f"{int(duration_s // 60)}분 {int(duration_s % 60)}초"

        if progress:
            progress(0.05, desc=f"전사(STT) 진행 중... (오디오 {dur_str})")
        yield f"전사(STT) 진행 중... (오디오 {dur_str})\n{_progress_bar(0)}", None

        result_holder: list = [None]
        error_holder:  list = [None]
        done = threading.Event()

        def _run_asr():
            try:
                result_holder[0] = asr_pipe(
                    wav_path,
                    chunk_length_s=30,
                    stride_length_s=5,
                    return_timestamps=True,
                )
            except Exception as e:
                error_holder[0] = e
            finally:
                done.set()

        threading.Thread(target=_run_asr, daemon=True).start()

        # CPU 기준 약 2× 실시간 소요로 예상 (보수적)
        estimated_s = max(duration_s * 2.0, 20)
        t0 = time.time()

        while not done.wait(timeout=2):
            elapsed = time.time() - t0
            frac = min(elapsed / estimated_s, 0.95)
            remaining = max(0, int(estimated_s - elapsed))
            bar = _progress_bar(frac)
            if progress:
                progress(0.05 + frac * 0.60, desc=f"전사 중... {int(frac*100)}%")
            yield f"전사(STT) 진행 중... (오디오 {dur_str})\n{bar}  약 {remaining}초 남음", None

        if error_holder[0]:
            raise error_holder[0]

        result = result_holder[0]

        chunks = result.get("chunks", [])
        if not chunks:
            yield "전사 결과가 없습니다. 오디오를 확인해 주세요.", None
            return

        if progress:
            progress(0.65, desc="화자 분리 중...")
        yield "화자 분리 중...", None

        diarize_kwargs = {}
        if num_speakers and num_speakers > 0:
            diarize_kwargs["num_speakers"] = num_speakers

        # torchcodec 우회: 경로 대신 메모리 텐서로 전달
        wav_data, wav_sr = sf.read(wav_path)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
        diarization = diarize_pipe({"waveform": wav_tensor, "sample_rate": wav_sr}, **diarize_kwargs)

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

        if progress:
            progress(1.0, desc="완료")
        yield output_text, str(save_path)

    except Exception as e:
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
            run_btn = gr.Button("변환 시작", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="녹취록 미리보기",
                lines=20,
            )
            file_output = gr.File(label=".txt 파일 다운로드")

    result_status = gr.Markdown("")

    def run(audio, n_spk, progress=gr.Progress()):
        n = int(n_spk) if n_spk and int(n_spk) > 0 else None
        for preview, path in transcribe(audio, n, progress=progress):
            if path is None:
                yield preview, None, ""
            else:
                yield preview, path, f"저장 완료: `{path}`"

    run_btn.click(
        fn=run,
        inputs=[audio_input, num_speakers],
        outputs=[output_text, file_output, result_status],
    )

if __name__ == "__main__":
    print(f"Gradio 서버 시작 중... http://localhost:7860", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
