#!/usr/bin/env python3
"""
Transcription pipeline built on faster-whisper.

Takes one or more audio (or video) files, puts them in a job queue, and runs a
single Whisper model over them one at a time. For each file it writes:

  <name>.json   segments with start/end timestamps (seconds), text, confidence,
                optional word-level timestamps, plus file metadata
  <name>.srt    subtitles, ready for a video player
  <name>.txt    clean plain text for downstream use (search, LLMs, NLP)

The queue state lives in <out>/queue.json, so an interrupted run can be resumed
and files that already finished are skipped.

Examples:
  python transcribe.py meeting.mp3
  python transcribe.py a.wav b.m4a podcasts/ --model small.en --out results
  python transcribe.py incoming/ --watch            # keep picking up new files
  python transcribe.py --status --out results       # show the queue
  python transcribe.py a.wav --model D:/whisper/faster-whisper-base.en   # any model folder

Runs fully offline: the model must already be on this PC, in
models/faster-whisper-<name>/ (python scripts/download_model.py base.en puts it
there), in the Hugging Face cache, or in a folder passed to --model. If it is
missing, the run stops with instructions; nothing is downloaded unless
--allow-download is given.
"""

from __future__ import annotations

import os

# Offline by default. Must be set before faster_whisper / huggingface_hub are
# imported, since huggingface_hub reads these once at import time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import json
import logging
import queue
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

log = logging.getLogger("transcribe")

# Anything FFmpeg (via PyAV) can decode works; this list only decides which
# files are picked up when a folder is passed in.
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aiff", ".aif", ".amr", ".webm", ".mp4", ".mkv", ".mov", ".avi",
}

PENDING, PROCESSING, DONE, FAILED = "pending", "processing", "done", "failed"


# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------

class AudioError(Exception):
    """The file is missing, unreadable, or has no audio stream."""


def probe_audio(path: Path) -> dict:
    """
    Check the file is decodable audio before spending model time on it.

    faster-whisper decodes with PyAV (FFmpeg bindings) and resamples to
    16 kHz mono float32 itself, so no separate ffmpeg conversion step or temp
    WAV file is needed. Any container/codec FFmpeg supports works: wav, mp3,
    m4a/aac, flac, ogg/opus, wma, and the audio track of video files.
    """
    import av

    if not path.exists():
        raise AudioError(f"file not found: {path}")
    if not path.is_file():
        raise AudioError(f"not a file: {path}")
    if path.stat().st_size == 0:
        raise AudioError(f"file is empty: {path}")
    try:
        with av.open(str(path)) as container:
            streams = container.streams.audio
            if not streams:
                raise AudioError(f"no audio stream in {path.name}")
            s = streams[0]
            # A file can parse as a container yet hold nothing decodable
            # (random bytes, a header with no samples): decode the first frame.
            if not _has_decodable_audio(container, s):
                raise AudioError(f"no decodable audio in {path.name}")
            duration = None
            if s.duration is not None and s.time_base is not None:
                duration = float(s.duration * s.time_base)
            elif container.duration is not None:
                duration = container.duration / 1_000_000
            else:  # e.g. Ogg: no duration in the header, so add up packet durations
                container.seek(0)
                total = sum(p.duration or 0 for p in container.demux(s) if p.duration)
                duration = float(total * s.time_base) if total and s.time_base else None
            return {
                "container": container.format.name,
                "codec": s.codec_context.name,
                "sample_rate": s.codec_context.sample_rate,
                "channels": s.codec_context.channels,
                "duration": round(duration, 3) if duration else None,
            }
    except AudioError:
        raise
    except Exception as e:  # av.error.InvalidDataError and friends
        raise AudioError(f"cannot decode {path.name}: {e}") from e


def _has_decodable_audio(container, stream, max_packets: int = 50) -> bool:
    try:
        for i, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                if frame.samples > 0:
                    return True
            if i >= max_packets:
                break
    except Exception:
        return False
    return False


def collect_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*")
                                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS))
        else:
            files.append(p)  # explicit files are queued even with odd extensions
    return files


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@dataclass
class TranscriberConfig:
    model: str = "base.en"         # English-only model: the pipeline targets English audio
    device: str = "auto"          # "cpu", "cuda" or "auto"
    compute_type: str = "default"  # e.g. "int8" on CPU, "float16" on GPU
    language: Optional[str] = "en"  # None = auto-detect (multilingual models only)
    beam_size: int = 5
    vad: bool = True               # skip silence with Silero VAD
    word_timestamps: bool = False
    batch_size: int = 0            # >0 uses BatchedInferencePipeline (faster on GPU)
    allow_download: bool = False   # False = only use a model already on this PC


class ModelNotFoundError(Exception):
    """The model is not on this PC (or is incomplete) and downloading is not allowed."""


# Where models live: <project>/models/faster-whisper-<name>/ (override with TRANSCRIBE_MODELS_DIR).
MODELS_DIR = Path(os.environ.get("TRANSCRIBE_MODELS_DIR") or Path(__file__).resolve().parent / "models")
# model.bin, config.json and tokenizer.json are all needed: without tokenizer.json
# faster-whisper would try to fetch a tokenizer from the internet.
MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")
MODEL_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def model_dir(name: str) -> Path:
    """The folder a model called `name` is expected in (and downloaded to)."""
    return MODELS_DIR / f"faster-whisper-{name.split('/')[-1].removeprefix('faster-whisper-')}"


def model_repo(name: str) -> str:
    from faster_whisper.utils import _MODELS
    return name if "/" in name else _MODELS.get(name, f"Systran/faster-whisper-{name}")


def missing_model_message(name: str, reason: str = "") -> str:
    folder = model_dir(name)
    repo = model_repo(name)
    return (
        f"Whisper model '{name}' is not on this PC{': ' + reason if reason else ''}.\n"
        f"This project runs offline and never downloads a model on its own. Do ONE of these, then run again:\n"
        f"  1. python scripts/download_model.py {name}\n"
        f"       downloads it once (base.en is about 145 MB) into {folder}\n"
        f"  2. download config.json, model.bin, tokenizer.json and vocabulary.txt from\n"
        f"       https://huggingface.co/{repo}/tree/main\n"
        f"     and put them in {folder}\n"
        f"  3. pass --model <folder> pointing at a faster-whisper model folder you already have")


def check_model_dir(p: Path, name: str) -> str:
    missing = [f for f in MODEL_FILES if not (p / f).is_file()]
    if not any(p.glob("vocabulary.*")):
        missing.append("vocabulary.txt")
    if missing:
        raise ModelNotFoundError(missing_model_message(
            name, f"the folder {p} is incomplete (missing {', '.join(missing)})"))
    if (p / "model.bin").stat().st_size < 1_000_000:
        raise ModelNotFoundError(missing_model_message(
            name, f"{p / 'model.bin'} is only {(p / 'model.bin').stat().st_size} bytes (an unfinished download?)"))
    return str(p.resolve())


def download_model_files(name: str, dest: Optional[Path] = None) -> str:
    """Download a model into the project's models folder. The only code path that uses the network."""
    import shutil
    from huggingface_hub import snapshot_download
    dest = dest or model_dir(name)
    # Download next to the final folder and rename at the end, so an interrupted
    # or failed download never leaves a half-filled model folder behind.
    partial = dest.with_name(dest.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_repo(name), local_dir=str(partial), allow_patterns=MODEL_PATTERNS)
    check_model_dir(partial, name)
    shutil.rmtree(partial / ".cache", ignore_errors=True)   # huggingface_hub bookkeeping
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    return check_model_dir(dest, name)


def resolve_model(name_or_path: str, allow_download: bool = False) -> str:
    """
    Return the local folder holding the CTranslate2 model. Looks, in order, at:
      1. --model as a folder path
      2. <project>/models/faster-whisper-<name>/  (where scripts/download_model.py puts it)
      3. the Hugging Face cache (a model fetched earlier by another faster-whisper program)
    and only downloads (into 2.) when allow_download is set. Otherwise it raises
    ModelNotFoundError with step-by-step instructions.
    """
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return check_model_dir(p, p.name)
    hub_id = re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name_or_path) and \
        name_or_path.split("/")[0] not in (".", "..", "models")
    if p.exists() or (("/" in name_or_path or "\\" in name_or_path) and not hub_id):
        raise ModelNotFoundError(missing_model_message(
            "base.en", f"--model {name_or_path} is not a model folder (use a model name such as base.en, "
            f"or a folder containing model.bin, config.json and tokenizer.json)"))
    for folder in (model_dir(name_or_path), MODELS_DIR / name_or_path):
        if folder.is_dir():
            return check_model_dir(folder, name_or_path)
    from faster_whisper.utils import _MODELS, download_model
    if "/" not in name_or_path and name_or_path not in _MODELS:
        raise ModelNotFoundError(f"unknown model name '{name_or_path}'. English models: tiny.en, base.en, "
                                 f"small.en, medium.en (or pass a model folder with --model).")
    try:
        cached = download_model(name_or_path, local_files_only=True)
        return check_model_dir(Path(cached), name_or_path)
    except ModelNotFoundError:
        raise
    except Exception:
        pass
    if not allow_download:
        raise ModelNotFoundError(missing_model_message(name_or_path))
    try:
        return download_model_files(name_or_path)
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise ModelNotFoundError(missing_model_message(
            name_or_path, f"the download failed ({type(e).__name__}: {e})")) from e


class Transcriber:
    """Loads the model once and reuses it for every file in the queue."""

    def __init__(self, config: TranscriberConfig, model=None):
        self.config = config
        self._model = model
        self._batched = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            path = resolve_model(self.config.model, self.config.allow_download)
            log.info("loading model %s from %s (device=%s, compute_type=%s)",
                     self.config.model, path, self.config.device, self.config.compute_type)
            try:
                self._model = WhisperModel(path, device=self.config.device,
                                           compute_type=self.config.compute_type, local_files_only=True)
            except Exception as e:  # damaged model.bin, wrong format, unsupported compute type...
                raise ModelNotFoundError(missing_model_message(
                    self.config.model, f"the files in {path} could not be loaded ({type(e).__name__}: {e}); "
                    f"delete that folder and download the model again")) from e
        return self._model

    def transcribe(self, path: Path,
                   on_segment: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Long files: Whisper only ever sees 30-second windows. faster-whisper
        decodes the file, slides that window across it, and yields segments
        lazily as a generator, so we consume them one by one (and report
        progress) instead of waiting for the whole file. VAD drops long
        silences first, which saves time and avoids hallucinated text in quiet
        stretches. Audio is held in memory as 16 kHz float32, about 230 MB per
        hour, which is fine on any normal machine.
        """
        c = self.config
        kwargs = dict(language=c.language, beam_size=c.beam_size,
                      vad_filter=c.vad, word_timestamps=c.word_timestamps)
        if c.batch_size > 0:
            if self._batched is None:
                from faster_whisper import BatchedInferencePipeline
                self._batched = BatchedInferencePipeline(model=self.model)
            seg_iter, info = self._batched.transcribe(str(path), batch_size=c.batch_size, **kwargs)
        else:
            seg_iter, info = self.model.transcribe(str(path), **kwargs)

        segments = []
        for seg in seg_iter:
            item = segment_to_dict(len(segments), seg)
            if not item["text"]:
                continue
            segments.append(item)
            if on_segment:
                on_segment(item)

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 3),
            "duration_after_vad": round(getattr(info, "duration_after_vad", info.duration) or 0, 3),
            "segments": segments,
        }


def segment_to_dict(index: int, seg) -> dict:
    item = {
        "id": index,
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": clean_text(seg.text),
        "avg_logprob": round(seg.avg_logprob, 4),
        "no_speech_prob": round(seg.no_speech_prob, 4),
    }
    if getattr(seg, "words", None):
        item["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3),
                          "word": w.word.strip(), "probability": round(w.probability, 3)}
                         for w in seg.words]
    return item


# ---------------------------------------------------------------------------
# Post-processing for downstream use
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip whitespace and Whisper's non-speech markers like [BLANK_AUDIO]."""
    text = re.sub(r"\[(BLANK_AUDIO|MUSIC|NOISE|SILENCE)\]|\((music|silence)\)", "", text, flags=re.I)
    return " ".join(text.split())


def fmt_ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    blocks = [f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n"
              for i, s in enumerate(segments, 1)]
    return "\n".join(blocks)


def to_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip() + "\n"


def write_outputs(result: dict, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{stem}.json",
        "srt": out_dir / f"{stem}.srt",
        "txt": out_dir / f"{stem}.txt",
    }
    _atomic_write(paths["json"], json.dumps(result, indent=2, ensure_ascii=False))
    _atomic_write(paths["srt"], to_srt(result["segments"]))
    _atomic_write(paths["txt"], to_text(result["segments"]))
    return {k: str(v) for k, v in paths.items()}


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(data, encoding="utf-8", newline="")  # same bytes on every OS (no CRLF on Windows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

@dataclass
class Job:
    path: str
    status: str = PENDING
    attempts: int = 0
    error: Optional[str] = None
    outputs: dict = field(default_factory=dict)
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    processing_seconds: Optional[float] = None


class JobQueue:
    """
    FIFO queue of audio files, processed one at a time by a single worker
    thread that owns the model (a Whisper model is too heavy to load per
    file, and one model per thread would multiply memory use).

    State is saved to queue.json after every change, so:
      - re-running the same command skips files already done,
      - a crash mid-file leaves that job "processing" and it is retried,
      - failed files are retried up to max_attempts, then left as "failed"
        with the error message, without stopping the rest of the queue.
    """

    def __init__(self, transcriber: Transcriber, out_dir: Path, max_attempts: int = 2):
        self.transcriber = transcriber
        self.out_dir = out_dir
        self.max_attempts = max_attempts
        self.state_file = out_dir / "queue.json"
        self.jobs: dict[str, Job] = {}
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._in_queue: set[str] = set()
        out_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.jobs = {k: Job(**v) for k, v in data.items()}

    def _save(self) -> None:
        with self._save_lock:  # worker and producer threads both save
            with self._lock:
                data = {k: asdict(v) for k, v in self.jobs.items()}
            _atomic_write(self.state_file, json.dumps(data, indent=2))

    # -- queue operations ---------------------------------------------------
    def add(self, path: Path, force: bool = False) -> bool:
        """Queue a file. Returns False if it was already done (and not forced)."""
        key = str(path.resolve())
        with self._lock:
            if key in self._in_queue:
                return False  # same file given twice in this run
            job = self.jobs.get(key)
            if job and job.status == DONE and not force:
                return False
            # New, failed, or left "processing" by a crashed run: (re)queue it.
            self.jobs[key] = Job(path=key)
            self._in_queue.add(key)
        self._save()
        self._q.put(key)
        return True

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="transcribe-worker", daemon=True)
        self._worker.start()

    def join(self) -> None:
        """Wait until every queued file is finished, then stop the worker."""
        self._q.join()
        self._q.put(None)
        if self._worker:
            self._worker.join()

    def _run(self) -> None:
        while True:
            key = self._q.get()
            try:
                if key is None:
                    return
                try:
                    self._process(key)
                except Exception as e:  # never let one job kill the worker (the queue would hang)
                    log.exception("[%s] unexpected error", key)
                    job = self.jobs.get(key)
                    if job:
                        job.status, job.error = FAILED, f"{type(e).__name__}: {e}"
                        try:
                            self._save()
                        except Exception:
                            log.exception("could not save queue state")
                with self._lock:
                    self._in_queue.discard(key)
            finally:
                self._q.task_done()

    def _process(self, key: str) -> None:
        job = self.jobs[key]
        path = Path(job.path)
        while job.attempts < self.max_attempts:
            job.attempts += 1
            job.status, job.error, job.started_at = PROCESSING, None, time.time()
            self._save()
            try:
                meta = probe_audio(path)
                log.info("[%s] start (%s/%s, %ss)", path.name, meta["container"],
                         meta["codec"], meta["duration"])
                progress = _Progress(path.name, meta["duration"])
                result = self.transcriber.transcribe(path, on_segment=progress)
                result = {"file": str(path), "input": meta, "model": self.transcriber.config.model,
                          **result}
                elapsed = time.time() - job.started_at
                result["processing_seconds"] = round(elapsed, 2)
                if result["duration"]:
                    result["real_time_factor"] = round(elapsed / result["duration"], 3)
                job.outputs = write_outputs(result, self.out_dir, _output_stem(path))
                job.status, job.finished_at = DONE, time.time()
                job.processing_seconds = round(elapsed, 2)
                log.info("[%s] done: %d segments in %.1fs", path.name,
                         len(result["segments"]), elapsed)
                break
            except AudioError as e:
                # Bad input won't get better with a retry.
                job.status, job.error = FAILED, str(e)
                log.error("[%s] failed: %s", path.name, e)
                break
            except Exception as e:
                job.status, job.error = FAILED, f"{type(e).__name__}: {e}"
                log.error("[%s] attempt %d failed: %s", path.name, job.attempts, job.error)
            finally:
                self._save()
        self._save()

    def summary(self) -> dict:
        counts = {s: 0 for s in (PENDING, PROCESSING, DONE, FAILED)}
        with self._lock:  # other threads may be adding jobs
            for j in self.jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
        return counts

    def get(self, path: Path) -> Optional[Job]:
        """A snapshot of the job for this file, safe to read from any thread."""
        with self._lock:
            job = self.jobs.get(str(path.resolve()))
            return replace(job, outputs=dict(job.outputs)) if job else None

    def remove(self, path: Path) -> bool:
        """Forget a finished or failed job. Refuses (False) while it is queued or running."""
        key = str(path.resolve())
        with self._lock:
            job = self.jobs.get(key)
            if not job or key in self._in_queue or job.status in (PENDING, PROCESSING):
                return False
            del self.jobs[key]
        self._save()
        return True


def _output_stem(path: Path) -> str:
    # Keep the extension in the name so song.mp3 and song.wav don't overwrite each other.
    return f"{path.stem}{path.suffix.replace('.', '_')}"


class _Progress:
    def __init__(self, name: str, duration: Optional[float]):
        self.name, self.duration, self.last = name, duration, 0.0

    def __call__(self, seg: dict) -> None:
        if self.duration and seg["end"] - self.last >= 60:  # log once a minute of audio
            self.last = seg["end"]
            log.info("[%s] %.0f%% (%s)", self.name, 100 * seg["end"] / self.duration,
                     fmt_ts(seg["end"], "."))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def watch(jq: JobQueue, folders: list[Path], interval: float) -> None:
    """Poll folders and queue new audio files as they appear (Ctrl+C to stop)."""
    seen: set[str] = set()
    log.info("watching %s (Ctrl+C to stop)", ", ".join(map(str, folders)))
    try:
        while True:
            for f in collect_files(map(str, folders)):
                key = str(f.resolve())
                if key in seen or not _is_stable(f):
                    continue
                seen.add(key)
                if jq.add(f):
                    log.info("queued %s", f.name)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("stopping; finishing files already queued")


def _is_stable(f: Path, wait: float = 0.5) -> bool:
    """Skip files that are still being copied in (size still changing)."""
    try:
        size = f.stat().st_size
        time.sleep(wait)
        return size > 0 and f.stat().st_size == size
    except FileNotFoundError:
        return False


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Queue audio files and transcribe them with faster-whisper.")
    ap.add_argument("inputs", nargs="*", help="audio/video files or folders")
    ap.add_argument("--out", default="transcripts", help="output folder (default: transcripts)")
    ap.add_argument("--model", default="base.en",
                    help="model name already in the local cache (tiny.en, base.en, small.en, medium.en) "
                         "or a local model folder (default: base.en, English-only)")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow fetching the model from the internet if it is not on this PC")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default="default", help="int8, int8_float16, float16, float32")
    ap.add_argument("--language", default="en", help="spoken language (default: en)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="don't skip silence")
    ap.add_argument("--word-timestamps", action="store_true", help="also save per-word timestamps")
    ap.add_argument("--batch-size", type=int, default=0, help="use batched inference (good on GPU)")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts for a failed file")
    ap.add_argument("--force", action="store_true", help="re-transcribe files already done")
    ap.add_argument("--watch", action="store_true", help="keep watching input folders for new files")
    ap.add_argument("--interval", type=float, default=5.0, help="watch poll interval in seconds")
    ap.add_argument("--status", action="store_true", help="print the queue state and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out_dir = Path(args.out)

    config = TranscriberConfig(model=args.model, device=args.device, compute_type=args.compute_type,
                               language=args.language, beam_size=args.beam_size, vad=not args.no_vad,
                               word_timestamps=args.word_timestamps, batch_size=args.batch_size,
                               allow_download=args.allow_download)
    if args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "0"  # huggingface_hub is not imported yet at this point
    transcriber = Transcriber(config)
    jq = JobQueue(transcriber, out_dir, max_attempts=1 + args.retries)

    if args.status:
        for j in jq.jobs.values():
            print(f"{j.status:<10} {Path(j.path).name}" + (f"  ({j.error})" if j.error else ""))
        print(json.dumps(jq.summary()))
        return 0

    if not args.inputs:
        ap.error("give at least one file or folder")

    try:  # load the model up front, so a missing model fails once with a clear message
        transcriber.model
    except ModelNotFoundError as e:
        log.error("%s", e)
        return 2

    files = collect_files(args.inputs)
    jq.start()
    queued = sum(jq.add(f, force=args.force) for f in files)
    log.info("queued %d file(s), skipped %d already done", queued, len(files) - queued)

    if args.watch:
        watch(jq, [Path(p) for p in args.inputs if Path(p).is_dir()], args.interval)
    jq.join()

    s = jq.summary()
    log.info("finished: %d done, %d failed. Outputs in %s", s[DONE], s[FAILED], out_dir.resolve())
    return 1 if s[FAILED] else 0


if __name__ == "__main__":
    sys.exit(main())
