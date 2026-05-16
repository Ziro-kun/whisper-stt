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
WHISPER_REPO  = "ggerganov/whisper.cpp"
WHISPER_GGUF  = "ggml-large-v3-turbo-q5_0.bin"
DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"

# whisper-cli 바이너리 탐색 (brew install whisper-cpp)
import shutil as _shutil
WHISPER_BIN = _shutil.which("whisper-cli") or _shutil.which("whisper-main") or "whisper-cli"
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# ── 모델 상태 (백그라운드 로딩) ────────────────────────────────────────────────

_model_ready     = False
_model_error     = ""
_load_log        = []
asr_model_path   = None
diarize_pipe     = None


def _log(msg: str):
    print(msg, flush=True)
    _load_log.append(msg)


def _load_models():
    global _model_ready, _model_error, asr_model_path, diarize_pipe

    from huggingface_hub import hf_hub_download
    from pyannote.audio import Pipeline as DiarizationPipeline

    try:
        _log(f"[로딩] 디바이스: {DEVICE}")
        _log(f"[로딩] Whisper GGUF 모델 다운로드 중... ({WHISPER_GGUF}, 첫 실행 시 약 574MB)")

        asr_model_path = hf_hub_download(
            repo_id=WHISPER_REPO,
            filename=WHISPER_GGUF,
            token=HF_TOKEN or None,
        )
        _log("[로딩] Whisper GGUF 모델 준비 완료.")

        _log("[로딩] 화자 분리 모델 로딩 중...")
        diarize_pipe = DiarizationPipeline.from_pretrained(
            DIARIZE_MODEL,
            token=HF_TOKEN or None,
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
        _log("  3. 디스크 공간 확인 (모델 약 2~3GB)")


# 백그라운드에서 모델 로딩 시작
threading.Thread(target=_load_models, daemon=True).start()


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

        # STT(CPU)와 화자분리(MPS)를 동시에 실행 — Apple Silicon 병렬 처리
        yield f"STT + 화자 분리 병렬 처리 중... (오디오 {dur_str})\n{_progress_bar(0)}", None

        asr_result:     list = [None]
        asr_error:      list = [None]
        diarize_result: list = [None]
        diarize_error:  list = [None]
        asr_done     = threading.Event()
        diarize_done = threading.Event()

        def _run_asr():
            import json, tempfile
            out_dir = tempfile.mkdtemp()
            out_prefix = os.path.join(out_dir, "out")
            try:
                _log(f"[STT] 시작: {WHISPER_BIN}")
                proc = subprocess.run(
                    [
                        WHISPER_BIN,
                        "-m", asr_model_path,
                        "-f", wav_path,
                        "-l", "ko",
                        "-oj",
                        "-of", out_prefix,
                    ],
                    capture_output=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"whisper-cli 실패: {proc.stderr.decode(errors='replace').strip()}"
                    )
                out_json = out_prefix + ".json"
                if not os.path.exists(out_json):
                    raise RuntimeError("whisper-cli가 JSON을 생성하지 않았습니다.")
                with open(out_json, encoding="utf-8") as f:
                    data = json.load(f)
                transcription = data.get("transcription", [])
                asr_result[0] = {
                    "chunks": [
                        {
                            "timestamp": (
                                item["offsets"]["from"] / 1000.0,
                                item["offsets"]["to"]  / 1000.0,
                            ),
                            "text": item["text"].strip(),
                        }
                        for item in transcription
                        if item.get("text", "").strip()
                    ]
                }
                _log(f"[STT] 완료: {len(asr_result[0]['chunks'])}개 청크")
            except Exception as e:
                asr_error[0] = e
                _log(f"[STT] 오류: {e}")
            finally:
                try:
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
                except Exception:
                    pass
                asr_done.set()

        def _run_diarize():
            try:
                _log("[화자분리] 시작")
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
                    diarize_result[0] = result
                elif hasattr(result, "speaker_diarization"):
                    diarize_result[0] = result.speaker_diarization
                else:
                    diarize_result[0] = result
                _log("[화자분리] 완료")
            except Exception as e:
                diarize_error[0] = e
                _log(f"[화자분리] 오류: {e}")
            finally:
                diarize_done.set()

        threading.Thread(target=_run_asr,     daemon=True).start()
        threading.Thread(target=_run_diarize, daemon=True).start()

        t0 = time.time()
        CYCLE = 60.0  # 바가 한 번 채워지는 주기(초) — 시각적 피드백용

        while not (asr_done.is_set() and diarize_done.is_set()):
            elapsed  = time.time() - t0
            frac     = (elapsed % CYCLE) / CYCLE   # 주기적으로 리셋 → 절대 멈추지 않음
            bar      = _progress_bar(frac)
            elapsed_str = f"{int(elapsed // 60)}분 {int(elapsed % 60)}초 경과"
            stt_tag  = "완료" if asr_done.is_set()     else "진행 중"
            diar_tag = "완료" if diarize_done.is_set() else "진행 중"
            msg = (
                f"STT: {stt_tag}  |  화자 분리: {diar_tag}\n"
                f"{bar}  {elapsed_str}"
            )
            if progress:
                progress(0.1 + min(elapsed / 600, 0.8), desc=msg)
            yield msg, None
            asr_done.wait(timeout=1)
            diarize_done.wait(timeout=1)

        if asr_error[0]:
            raise asr_error[0]
        if diarize_error[0]:
            raise diarize_error[0]

        chunks = asr_result[0].get("chunks", [])
        if not chunks:
            yield "전사 결과가 없습니다. 오디오를 확인해 주세요.", None
            return

        diarization = diarize_result[0]

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
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
