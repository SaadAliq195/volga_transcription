# Transcription pipeline (faster-whisper)

Turns English audio files into timestamped transcripts. Files go into a job
queue and one Whisper model (loaded once) processes them one at a time. It
runs as a command-line tool (`transcribe.py`) or as an HTTP API (`server.py`),
and it is **fully offline** once the model is downloaded.

## Quick start

```bash
# 1. Python 3.9+ and the dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. The Whisper model, once (about 145 MB, into models/faster-whisper-base.en/)
python scripts/download_model.py

# 3. Transcribe
python transcribe.py test_audio/real/vctk_p225_002.flac --out transcripts
```

**Model not downloaded?** Every entry point (`transcribe.py`, `server.py`, the
tests) stops with exit code 2 and says exactly what to do: run
`python scripts/download_model.py`, or download `config.json`, `model.bin`,
`tokenizer.json` and `vocabulary.txt` from
<https://huggingface.co/Systran/faster-whisper-base.en/tree/main> into
`models/faster-whisper-base.en/`. An incomplete or damaged model folder is
reported the same way. Nothing is ever downloaded on its own. See
[models/README.md](models/README.md).

## Usage

```bash
python transcribe.py meeting.mp3                       # one file
python transcribe.py a.wav b.m4a podcasts/ --out out   # many files and folders, queued
python transcribe.py incoming/ --watch                 # keep queuing new files as they land
python transcribe.py --status --out out                # what's done / failed
python transcribe.py talk.mp4 --model small.en --word-timestamps
```

For each input it writes `<name>_<ext>.json`, `.srt` and `.txt` to the output
folder, plus `queue.json` with the state of every job.

```json
{
  "file": "meeting.mp3",
  "input": {"container": "mp3", "codec": "mp3float", "sample_rate": 44100, "channels": 2, "duration": 312.4},
  "model": "base.en", "language": "en", "duration": 312.4, "duration_after_vad": 251.0,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "Welcome everyone.", "avg_logprob": -0.21, "no_speech_prob": 0.01}
  ],
  "processing_seconds": 41.3, "real_time_factor": 0.132
}
```

## Part 1: how the requirements are met

**Accepts an audio file (WAV, MP3, ...).** Any file FFmpeg can decode is
accepted, including the audio track of video files. Each file is probed first
(`probe_audio`), so a missing, empty, corrupt or audio-less file fails fast
with a clear error instead of crashing the run.

**Transcribes speech to text.** faster-whisper is a CTranslate2
re-implementation of OpenAI Whisper, about 4x faster than the reference code
at the same accuracy, with int8 quantisation on CPU. The model is loaded once
and reused for every file. It is English-only (`base.en`, `--language en`),
which is faster and more accurate on English than the multilingual model of
the same size.

**Timestamps per segment.** Every segment has `start`/`end` in seconds plus
confidence (`avg_logprob`, `no_speech_prob`). `--word-timestamps` adds
per-word times. The same data is written as SRT subtitles.

**Different audio formats.** faster-whisper decodes with PyAV (bundled FFmpeg
libraries) and resamples in memory to 16 kHz mono float32, the format Whisper
was trained on. No external ffmpeg binary and no temporary WAV files are
needed. Tested: WAV (8 to 96 kHz, 16/24/32-bit, mono/stereo), MP3, M4A/AAC,
raw AAC, FLAC, OGG Vorbis, Opus, WebM, AIFF, and MP4/MKV/MOV video.

**Long audio files.** Whisper only sees 30-second windows. faster-whisper
slides the window across the file and yields segments lazily, so results are
streamed and progress is logged per minute of audio. Silero VAD (on by
default) removes silence first, which saves time and prevents made-up text in
quiet parts. Memory use is about 230 MB per hour of audio. On a GPU,
`--batch-size 16` switches to batched inference for a large speed-up.

**Queue for multiple files.** `JobQueue` is first in, first out, with a single
worker thread that owns the model. State is saved to `queue.json` after every
change, so re-running the command skips finished files, resumes after a crash
and retries failed ones. A bad file is marked `failed` with its error and the
rest of the queue keeps going. Transient errors are retried (`--retries`).
`--watch` polls folders and queues new files once they finish copying.

**Downstream use.** JSON for programs, SRT for players and editors, and cleaned
plain text (non-speech markers like `[BLANK_AUDIO]` removed, whitespace
normalised) for search, NLP or LLM steps. Output files are written atomically
and are byte-identical on every OS. The CLI exits non-zero if any file failed,
so it can sit in a script or CI job.

## Part 2: HTTP API and system design

`server.py` puts the same queue behind a small REST API (Python standard
library only, offline). Uploads are accepted concurrently and answered with
`202` straight away. One worker owns the model and transcribes the uploads in
order. Bad audio is rejected with `422`, a full queue answers `503` with
`Retry-After`, duplicate uploads are stored once, and unfinished jobs resume
after a crash or restart.

```bash
python server.py --port 8000 --data data
curl --data-binary @meeting.mp3 "http://127.0.0.1:8000/v1/transcriptions?filename=meeting.mp3"
curl http://127.0.0.1:8000/v1/transcriptions/<id>          # status + segments when done
curl http://127.0.0.1:8000/v1/transcriptions/<id>/srt      # or /json, /txt
curl -X POST http://127.0.0.1:8000/v1/transcriptions/<id>/retry
curl -X DELETE http://127.0.0.1:8000/v1/transcriptions/<id>
curl http://127.0.0.1:8000/v1/transcriptions?status=failed
```

[DESIGN.md](DESIGN.md) answers the system-design questions: concurrent
uploads, storage, retry and recovery, and the API. It includes the full
endpoint table.

## Verification

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1   # Windows: everything, offline
bash scripts/run_checks.sh                                         # macOS / Linux
```

That runs, with `HF_HUB_OFFLINE=1`:

- `tests/verify.py --variant 1|2|3`: Part 1, end to end through the queue.
- `tests/verify_api.py`: Part 2, the HTTP API, including concurrent clients, a
  kill-and-restart crash test and every malformed request we could think of.
  Add `--quick` / `-Quick` to skip the crash test.

The test speech in `test_audio/real/` is 40 real English recordings from 32
speakers, each with its reference transcript:

- LibriSpeech test-clean and test-other (CC BY 4.0).
- VCTK, with English, Scottish, Welsh, Irish, Northern Irish, American, New
  Zealand, South African and Indian accents (CC BY 4.0).
- EdAcc, conversational and accented (CC BY-SA 4.0).

See [test_audio/real/ATTRIBUTION.md](test_audio/real/ATTRIBUTION.md) and
`manifest.json`. `scripts/fetch_test_audio.py` re-creates the set. Each
variant uses its own group of speakers (no speaker appears in two), plus one
clip in the local Windows voice:

| Variant | Audio | Broken / edge-case inputs |
|---|---|---|
| 1 | clean; WAV 44.1k stereo, MP3, M4A, FLAC 22k, OGG, Opus, MP4 video; long file with 12 s gaps | corrupt bytes, empty file, missing file |
| 2 | 20 dB SNR noise; 8 kHz phone WAV, 96 kHz float WAV, 22k MP3, raw AAC, WebM, MKV video; 25 s gaps | WAV header with no samples, text renamed .wav, random bytes .flac, missing folder, truncated MP3 (should give a partial transcript) |
| 3 | 15 dB SNR noise; 24-bit WAV and Unicode MP3 with upper-case extensions, AIFF, Ogg mislabelled .flac, MOV video; 40 s gaps | video with no audio track, zero-filled MP3, a folder named .wav, a 0.3 s clip |

Each run checks formats, timestamp validity and order, SRT/TXT output, word
error rate (WER) per clip (at most 40%) and per variant (median at most 15%)
with a per-speaker table, and a long file stitched from several speakers with
silent gaps. It also checks silence, sequential first-in-first-out processing,
duplicate handling, failure isolation, resume and retry, and the CLI exit
code. Outputs go to `test_output/` (not committed).

### Latest results (base.en, CPU, Windows 11, offline)

| Run | Checks | Real-speech WER (median / worst) |
|---|---|---|
| Part 1, variant 1 (clean) | 153 / 154 | 4.7% / 22% |
| Part 1, variant 2 (20 dB noise, 8 kHz phone audio) | 158 / 159 | 2.5% / 42% |
| Part 1, variant 3 (15 dB noise) | 138 / 139 | 9.5% / 50% |
| Part 2, HTTP API | 67 / 67 | median 0% over 30 concurrent uploads |

Every functional check passes: formats, timestamps, queue, retries, broken
files, API and crash recovery. The three misses are accuracy limits of the
small `base.en` model, kept visible on purpose rather than hidden by loosening
the thresholds:

- **Variant 1:** in the stitched long file, base.en skips one short
  conversational EdAcc utterance that sits between long read passages. The
  same clip is transcribed correctly on its own, and VAD does detect it.
- **Variant 2:** an Indian-English conversational clip at 8 kHz with noise
  scores 42% WER, just over the 40% per-clip limit.
- **Variant 3:** a 2.8 s Scottish clip at 15 dB noise ("regional government"
  heard as "original government") scores 50% WER.

A larger model (`python scripts/download_model.py small.en`, then
`--model small.en`) is the usual fix for accents and noise, at about 3x the
CPU time.

## Project layout

```
transcribe.py              pipeline + CLI: probing, transcription, job queue
server.py                  HTTP API on top of the same queue
DESIGN.md                  Part 2 system-design answers
requirements.txt
models/                    the Whisper model goes here (not committed; see models/README.md)
scripts/
  download_model.py        one-time model download (network)
  fetch_test_audio.py      one-time test-speech download (network; already done, clips are committed)
  run_checks.ps1 / .sh     run every verification offline
tests/
  verify.py                Part 1 end-to-end checks (3 variants)
  verify_api.py            Part 2 API checks
test_audio/real/           40 real English test clips + manifest.json + ATTRIBUTION.md
```
