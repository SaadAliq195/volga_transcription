#!/usr/bin/env python3
"""
End-to-end verification of transcribe.py against the assessment requirements.

  python tests/verify.py                    # real faster-whisper model (default: base.en)
  python tests/verify.py --variant 2        # a different group of speakers and formats
  python tests/verify.py --mock             # no model needed: fake model, real decoding
  python tests/verify.py --variant 3 --data-dir checks/v3

The test speech is real recordings of many different people (LibriSpeech,
VCTK, EdAcc; see test_audio/real/manifest.json). Each variant uses its own,
non-overlapping group of speakers, converts their clips to that variant's
formats and sample rates, adds that variant's background noise, stitches a
long multi-speaker file with silent gaps, and adds broken files. Everything
goes through the job queue and every output is checked; transcripts are
scored by word error rate (WER) against the reference text. A clip made with
the local Windows SAPI voice is included as one extra case.

Everything runs on this PC: the model comes from the local cache
(transcribe.py is offline by default) and all generated audio and outputs are
kept in --data-dir (default: test_output/variant_N in the project folder).
Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "test_audio" / "real"

# Per variant: formats, background noise (signal-to-noise ratio in dB, None =
# clean), silence between the speakers of the long file, and the SAPI script.
VARIANTS = {
    1: dict(snr=None, gap=12, rate=160, voice=0,
            script=("The quick brown fox jumps over the lazy dog. "
                    "Speech recognition converts spoken language into written text."),
            specs=[(".wav", "pcm_s16le", None, 44100, "stereo", False),
                   (".mp3", "libmp3lame", None, 44100, "stereo", False),
                   (".m4a", "aac", "ipod", 48000, "mono", False),
                   (".flac", "flac", None, 22050, "mono", False),
                   (".ogg", "libvorbis", None, 44100, "stereo", False),
                   (".opus", "libopus", "ogg", 48000, "mono", False),
                   ("_video.mp4", "aac", "mp4", 44100, "stereo", True)]),
    2: dict(snr=20, gap=25, rate=190, voice=1,
            script=("Please schedule the quarterly budget review for next Thursday afternoon. "
                    "Send the slides to everyone before the meeting starts."),
            specs=[("_8k.wav", "pcm_s16le", None, 8000, "mono", False),       # phone quality
                   ("_float.wav", "pcm_f32le", None, 96000, "stereo", False),
                   ("_22k.mp3", "libmp3lame", None, 22050, "mono", False),
                   (".aac", "aac", "adts", 44100, "stereo", False),
                   (".webm", "libopus", "webm", 48000, "stereo", False),
                   ("_video.mkv", "libvorbis", "matroska", 44100, "mono", True)]),
    3: dict(snr=15, gap=40, rate=135, voice=0,
            script=("Our weather station recorded heavy rain across the northern valley overnight. "
                    "Temperatures will drop sharply by the weekend."),
            specs=[(" With Spaces.WAV", "pcm_s24le", "wav", 48000, "stereo", False),
                   ("_ünïcode.MP3", "libmp3lame", "mp3", 48000, "mono", False),
                   (".aiff", "pcm_s16be", "aiff", 32000, "mono", False),
                   ("_ogg_named.flac", "libvorbis", "ogg", 44100, "mono", False),  # wrong extension
                   ("_video.mov", "aac", "mov", 48000, "stereo", True)]),
}
V = dict(VARIANTS[1], id=1)
SCRIPT = V["script"]

# Honest targets for base.en on real, varied speech.
MEDIAN_WER_MAX = 0.15   # across this variant's real clips
CLIP_WER_MAX = 0.40     # any single real clip
TTS_WER_MAX = 0.25      # the synthetic SAPI clip
LONG_WER_MAX = 0.25     # the stitched multi-speaker file

SR = 16000
RESULTS: list[tuple[str, bool, str]] = []
BAD_INPUTS: dict[int, list[str]] = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    path = REAL / "manifest.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run `python scripts/fetch_test_audio.py` once to download the test speech.")
    return json.loads(path.read_text(encoding="utf-8"))


def split_speakers(manifest: list[dict]) -> dict[int, list[str]]:
    """Deal each dataset's speakers round-robin over the variants, so every
    variant gets a mix of sources and no speaker is in two variants."""
    groups: dict[int, list[str]] = {1: [], 2: [], 3: []}
    by_ds: dict[str, list[str]] = {}
    for m in manifest:
        if m["language"] == "en" and m["speaker"] not in by_ds.setdefault(m["dataset"] + m["split"], []):
            by_ds[m["dataset"] + m["split"]].append(m["speaker"])
    i = 0
    for ds in sorted(by_ds):
        for spk in sorted(by_ds[ds]):
            groups[i % 3 + 1].append(spk)
            i += 1
    return groups


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def make_speech(path: Path) -> bool:
    """Synthesize SCRIPT with the local OS voice (SAPI on Windows)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", V["rate"])
        voices = engine.getProperty("voices") or []
        if voices:
            engine.setProperty("voice", voices[V["voice"] % len(voices)].id)
        engine.save_to_file(SCRIPT, str(path))
        engine.runAndWait()
        return path.exists() and path.stat().st_size > 1000
    except Exception as e:
        print(f"  text-to-speech unavailable ({e}); skipping the SAPI clip")
        return False


def write_audio(path: Path, samples: np.ndarray, codec: str, fmt: str | None = None,
                rate: int = SR, layout: str = "mono", video: bool = False) -> None:
    """Encode float32 mono samples into any container/codec with PyAV (bundled FFmpeg)."""
    import av
    with av.open(str(path), "w", format=fmt) as out:
        if video:  # tiny black video track, to prove audio is pulled out of video files
            v = out.add_stream("mpeg4", rate=10)
            v.width, v.height, v.pix_fmt = 64, 64, "yuv420p"
        a = out.add_stream(codec, rate=rate)
        a.layout = layout
        data = samples
        if rate != SR:
            data = np.interp(np.linspace(0, len(samples), int(len(samples) * rate / SR), endpoint=False),
                             np.arange(len(samples)), samples).astype(np.float32)
        chans = 2 if layout == "stereo" else 1
        frame_size = 1024
        for i in range(0, len(data), frame_size):
            chunk = data[i:i + frame_size]
            arr = np.tile(chunk, (chans, 1)) if chans > 1 else chunk[None, :]
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(arr), format="fltp", layout=layout)
            frame.sample_rate = rate
            frame.pts = i  # in samples; without it some muxers (Ogg) write a wrong duration
            frame.time_base = Fraction(1, rate)
            for pkt in a.encode(frame):
                out.mux(pkt)
            if video and i % (rate // 10) < frame_size:
                vf = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
                for pkt in v.encode(vf):
                    out.mux(pkt)
        for pkt in a.encode(None):
            out.mux(pkt)
        if video:
            for pkt in v.encode(None):
                out.mux(pkt)


def add_noise(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if V["snr"] is None:
        return x, 0.0
    std = float(np.sqrt(np.mean(x ** 2))) / 10 ** (V["snr"] / 20)
    return (x + rng.normal(0, std, x.shape)).astype(np.float32), std


def build_dataset(d: Path, manifest: list[dict], speakers: list[str]) -> dict:
    from faster_whisper import decode_audio
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(V["id"])
    specs = V["specs"]
    clips: dict[str, dict] = {}   # output file name -> manifest entry (+ samples)
    files: dict[str, Path] = {}
    noise_stds = []

    mine = [m for m in manifest if m["speaker"] in speakers]
    for i, m in enumerate(mine):
        audio = decode_audio(str(REAL / m["file"]), sampling_rate=SR)
        audio, std = add_noise(audio, rng)
        noise_stds.append(std)
        suffix, codec, fmt, rate, layout, video = specs[i % len(specs)]
        name = f"{m['clip_id']}{suffix}"
        try:
            write_audio(d / name, audio, codec, fmt, rate, layout, video)
        except Exception as e:
            print(f"  could not create {name}: {e}")
            continue
        files[name] = d / name
        clips[name] = dict(m, samples=audio)
    print(f"  {len(clips)} real clips from {len(speakers)} speakers, "
          f"noise {'none' if V['snr'] is None else str(V['snr']) + ' dB SNR'}")

    # One extra synthetic clip from the local SAPI voice.
    raw = d / "_tts.wav"
    if make_speech(raw):
        audio, _ = add_noise(decode_audio(str(raw), sampling_rate=SR), rng)
        suffix, codec, fmt, rate, layout, video = specs[len(clips) % len(specs)]
        name = f"sapi_voice{suffix}"
        write_audio(d / name, audio, codec, fmt, rate, layout, video)
        files[name] = d / name
        clips[name] = dict(clip_id="sapi_voice", speaker="SAPI-voice", dataset="local Windows SAPI",
                           accent=None, gender=None, text=SCRIPT, samples=audio, synthetic=True)
        raw.unlink(missing_ok=True)

    # Long file: different speakers one after another with silent gaps, so it
    # spans many 30-second windows and VAD has long quiet stretches to skip.
    gap_std = statistics.median(noise_stds) if noise_stds else 0.0
    gap = (rng.normal(0, gap_std, int(V["gap"] * SR)) if gap_std else np.zeros(int(V["gap"] * SR))).astype(np.float32)
    pieces, used, t, parts = [], set(), 0.0, []
    for c in sorted((c for c in clips.values() if not c.get("synthetic")), key=lambda c: -len(c["samples"])):
        if c["speaker"] in used:
            continue
        if parts:
            parts.append(gap)
            t += len(gap) / SR
        pieces.append(dict(speaker=c["speaker"], text=c["text"], start=t, end=t + len(c["samples"]) / SR))
        parts.append(c["samples"])
        t += len(c["samples"]) / SR
        used.add(c["speaker"])
        if t >= 150 and len(pieces) >= 4:
            break
    parts.append(np.zeros(2 * SR, np.float32))
    long_audio = np.concatenate(parts)
    p = d / "long_recording.mp3"
    write_audio(p, long_audio, "libmp3lame", None, 44100, "mono")
    files["long_recording.mp3"] = p

    # Silence only: should give zero segments, not hallucinated text.
    p = d / "silence.wav"
    write_audio(p, np.zeros(10 * SR, np.float32), "pcm_s16le")
    files["silence.wav"] = p

    # Broken inputs: must fail cleanly without stopping the queue.
    bad = BAD_INPUTS[V["id"]] = []
    def broken(name, data=None):
        p = d / name
        if data is not None:
            p.write_bytes(data)
        files[p.name] = p
        bad.append(p.name)
    first = lambda ext: next(files[n] for n in clips if n.lower().endswith(ext))
    if V["id"] == 1:
        broken("corrupt.mp3", b"this is not audio" * 200)
        broken("empty.wav", b"")
        broken("missing.wav")
    elif V["id"] == 2:
        good = first(".mp3").read_bytes()
        broken("header_only.wav", first(".wav").read_bytes()[:44])   # WAV header, no samples
        broken("text_renamed.wav", "plain text, not audio\n".encode() * 100)
        broken("random_bytes.flac", np.random.default_rng(2).bytes(50_000))
        broken("missing_folder/missing.m4a")
        # A truncated download: first 40% of a real MP3. Decoders recover, so this
        # should succeed with a partial transcript, not crash.
        p = d / "truncated.mp3"
        p.write_bytes(good[: int(len(good) * 0.4)])
        files["truncated.mp3"] = p
    else:
        p = d / "video_no_audio.mp4"
        import av
        with av.open(str(p), "w") as out:
            v = out.add_stream("mpeg4", rate=10)
            v.width, v.height, v.pix_fmt = 64, 64, "yuv420p"
            for _ in range(20):
                for pkt in v.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")):
                    out.mux(pkt)
            for pkt in v.encode(None):
                out.mux(pkt)
        files["video_no_audio.mp4"] = p
        bad.append("video_no_audio.mp4")
        broken("zero_padded.mp3", bytes(20_000))
        (d / "a_folder.wav").mkdir()
        broken("a_folder.wav")
        # Very short clip (0.3 s): valid audio, must not crash.
        some = next(iter(clips.values()))["samples"]
        p = d / "tiny_clip.wav"
        write_audio(p, some[: int(0.3 * SR)], "pcm_s16le")
        files["tiny_clip.wav"] = p
    return {"files": files, "clips": clips, "pieces": pieces, "long_seconds": len(long_audio) / SR}


# ---------------------------------------------------------------------------
# Mock model: real audio decoding, fake text
# ---------------------------------------------------------------------------

class MockModel:
    """Decodes the file exactly like faster-whisper does, then emits a segment per 5s of sound."""

    def transcribe(self, audio, **kw):
        from types import SimpleNamespace as NS
        from faster_whisper import decode_audio
        samples = decode_audio(audio, sampling_rate=SR)
        dur = len(samples) / SR

        def gen():
            words = SCRIPT.split()
            for i, start in enumerate(np.arange(0, dur, 5.0)):
                end = min(start + 5.0, dur)
                chunk = samples[int(start * SR):int(end * SR)]
                if chunk.size == 0 or np.abs(chunk).max() < 1e-3:
                    continue
                text = " ".join(words[(i * 4) % len(words):][:4])
                yield NS(start=float(start), end=float(end), text=" " + text, avg_logprob=-0.2,
                         no_speech_prob=0.01, words=None)
        return gen(), NS(language="en", language_probability=1.0, duration=dur, duration_after_vad=dur)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen " \
        "fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ABBR = {"mr": "mister", "mrs": "missus", "dr": "doctor", "ok": "okay", "st": "saint"}
_FILLERS = {"um", "uh", "mm", "mmm", "hmm", "er", "erm", "ah", "mhm", "uhm"}


def _num(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num(n % 100))
    if n < 1_000_000:
        return _num(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num(n % 1000))
    return " ".join(_ONES[int(c)] for c in str(n))


def words(s: str) -> list[str]:
    """Lowercase, drop punctuation and [tags], spell out numbers, drop fillers (um, uh)."""
    s = re.sub(r"[\[<][^\]>]*[\]>]", " ", s.lower()).replace("-", " ").replace("%", " percent")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    out = []
    for w in re.findall(r"[a-z0-9']+", s):
        w = w.strip("'")
        if not w or w in _FILLERS:
            continue
        if w.isdigit():
            out += _num(int(w)).split()
        else:
            out += _ABBR.get(w, w).split()
    return out


def edit_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference words)."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def wer(ref: str, hyp: str) -> float:
    e, n = edit_counts(ref, hyp)
    return e / max(n, 1)


SRT_BLOCK = re.compile(r"^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+$", re.M)


def check_output(name: str, job: T.Job) -> dict:
    data = json.loads(Path(job.outputs["json"]).read_text(encoding="utf-8"))
    segs = data["segments"]
    dur = data["duration"]

    check(f"{name}: json/srt/txt written", all(Path(p).exists() for p in job.outputs.values()))
    check(f"{name}: has segments", len(segs) > 0, f"{len(segs)} segments")
    ok_keys = all({"start", "end", "text"} <= s.keys() for s in segs)
    ok_order = all(0 <= s["start"] < s["end"] <= dur + 0.5 for s in segs) and \
        all(a["start"] <= b["start"] for a, b in zip(segs, segs[1:]))
    check(f"{name}: timestamps valid and in order", ok_keys and ok_order,
          f"first {segs[0]['start']:.2f}-{segs[0]['end']:.2f}s, last ends {segs[-1]['end']:.2f}s of {dur:.1f}s" if segs else "")

    srt = Path(job.outputs["srt"]).read_text(encoding="utf-8").strip()
    check(f"{name}: SRT well-formed", len(SRT_BLOCK.findall(srt)) == len(segs))
    txt = Path(job.outputs["txt"]).read_text(encoding="utf-8")
    check(f"{name}: plain text matches segments", txt.strip() == " ".join(s["text"] for s in segs))
    return data


def main() -> int:
    global V, SCRIPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--variant", type=int, choices=sorted(VARIANTS), default=1,
                    help="which speaker group / formats / noise / broken files to use")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="where test audio and outputs are written and kept "
                         "(default: test_output/variant_N)")
    args = ap.parse_args()
    if not args.mock:  # fail fast, with instructions, if the model isn't on this PC
        try:
            T.resolve_model(args.model)
        except T.ModelNotFoundError as e:
            print(f"\n{e}", file=sys.stderr)
            return 2

    V = dict(VARIANTS[args.variant], id=args.variant)
    SCRIPT = V["script"]
    manifest = load_manifest()
    groups = split_speakers(manifest)
    speakers = groups[args.variant]
    print(f"Variant {args.variant}: {len(speakers)} speakers: {', '.join(speakers)}")

    work = (args.data_dir or ROOT / "test_output" / f"variant_{args.variant}").resolve()
    data_dir, out_dir = work / "audio", work / "out"
    # Start from a clean slate: a queue.json left by an earlier run would make
    # every file count as already done. Only the folders this script creates are removed.
    for sub in ("audio", "out", "out_retry", "cli"):
        shutil.rmtree(work / sub, ignore_errors=True)
    print(f"\nTest folder: {work}\n\n1. Building test audio")
    ds = build_dataset(data_dir, manifest, speakers)
    files, clips = ds["files"], ds["clips"]
    all_groups = [set(g) for g in groups.values()]
    check("speaker groups of the 3 variants don't overlap",
          not (all_groups[0] & all_groups[1] or all_groups[0] & all_groups[2] or all_groups[1] & all_groups[2]),
          " / ".join(str(len(g)) for g in all_groups) + " speakers")

    cfg = T.TranscriberConfig(model=args.model, device="cpu", compute_type="int8",
                              language="en", word_timestamps=True)
    transcriber = T.Transcriber(cfg, model=MockModel() if args.mock else None)
    real = not args.mock

    print("\n2. Checking input validation")
    speech_files = list(clips)
    bad = BAD_INPUTS[V["id"]]
    for name in speech_files:
        m = T.probe_audio(files[name])
        check(f"probe {name}", m["duration"] and m["duration"] > 1, f"{m['container']}/{m['codec']} {m['sample_rate']}Hz {m['channels']}ch")
    for name in bad:
        try:
            T.probe_audio(files[name])
            check(f"reject {name}", False)
        except T.AudioError as e:
            check(f"reject {name}", True, str(e)[:60])

    print("\n3. Running every file through the queue (one at a time)")
    jq = T.JobQueue(transcriber, out_dir, max_attempts=2)
    jq.start()
    order = [files[k] for k in files]
    t0 = time.time()
    for f in order:
        jq.add(f)
    jq.add(order[0])  # duplicate in the same run must not be processed twice
    jq.join()
    total = time.time() - t0
    print(f"  queue finished in {total:.1f}s")

    jobs = {Path(j.path).name: j for j in jq.jobs.values()}
    check("every file got a job", len(jobs) == len(order), f"{len(jobs)} jobs")
    done = sorted((j for j in jobs.values() if j.status == T.DONE), key=lambda j: j.started_at)
    seq = all(a.finished_at <= b.started_at for a, b in zip(done, done[1:]))
    check("files processed one by one, never overlapping", seq)
    fifo = [Path(j.path).name for j in done] == [p.name for p in order if p.name in {Path(j.path).name for j in done}]
    check("files processed in the order queued", fifo)
    check("duplicate not processed twice", all(j.attempts == 1 for j in done))

    print("\n4. Checking outputs per clip and format")
    scores = []  # (clip, speaker, edits, ref words, wer)
    for name in speech_files:
        j = jobs.get(name)
        c = clips[name]
        if not check(f"{name}: transcribed", j is not None and j.status == T.DONE, j.error if j else "no job"):
            continue
        data = check_output(name, j)
        if real:
            hyp = " ".join(s["text"] for s in data["segments"])
            e, n = edit_counts(c["text"], hyp)
            limit = TTS_WER_MAX if c.get("synthetic") else CLIP_WER_MAX
            check(f"{name}: transcript accurate (WER <= {limit:.0%})", e / max(n, 1) <= limit,
                  f"WER {e / max(n, 1):.0%}" + ("" if e / max(n, 1) <= limit else f"; ref: {c['text'][:70]!r} hyp: {hyp[:70]!r}"))
            if not c.get("synthetic"):
                scores.append((name, c, e, n, e / max(n, 1)))

    if real and scores:
        med = statistics.median(s[4] for s in scores)
        worst = max(scores, key=lambda s: s[4])
        check(f"median WER over {len(scores)} real clips <= {MEDIAN_WER_MAX:.0%}", med <= MEDIAN_WER_MAX,
              f"median {med:.1%}, worst {worst[4]:.0%} ({worst[0]})")
        print("\n  Per-speaker WER (pooled over the speaker's clips)")
        print(f"  {'speaker':<14} {'source':<26} {'gender':<7} {'accent':<30} clips   WER")
        table = {}
        for name, c, e, n, _ in scores:
            row = table.setdefault(c["speaker"], dict(source=c["dataset"].split("/")[-1] + " " + c.get("split", ""),
                                                      gender=c.get("gender") or "?", accent=c.get("accent") or "?",
                                                      clips=0, edits=0, words=0))
            row["clips"] += 1; row["edits"] += e; row["words"] += n
        for spk, r in sorted(table.items(), key=lambda kv: -kv[1]["edits"] / max(kv[1]["words"], 1)):
            r["wer"] = round(r["edits"] / max(r["words"], 1), 4)
            print(f"  {spk:<14} {r['source']:<26} {r['gender']:<7} {r['accent'][:30]:<30} {r['clips']:>5}  {r['wer']:5.1%}")
    else:
        table, med = {}, None

    print("\n5. Long multi-speaker audio")
    j = jobs["long_recording.mp3"]
    pieces = ds["pieces"]
    print(f"  {len(pieces)} speakers, {ds['long_seconds']:.0f}s: " +
          ", ".join(f"{p['speaker']} @{p['start']:.0f}s" for p in pieces))
    if check("long file transcribed", j.status == T.DONE, j.error or ""):
        data = check_output("long_recording", j)
        segs = data["segments"]
        check("long file: segments span the whole recording",
              segs[-1]["end"] > 0.8 * ds["long_seconds"] and segs[0]["start"] < 30,
              f"{segs[0]['start']:.0f}s to {segs[-1]['end']:.0f}s of {ds['long_seconds']:.0f}s")
        check("long file: covers many 30s windows", data["duration"] > 120, f"{data['duration']:.0f}s")
        if real:
            check("long file: VAD skipped the silent gaps", data["duration_after_vad"] < data["duration"] * 0.9,
                  f"{data['duration_after_vad']:.0f}s of speech in {data['duration']:.0f}s")
            per = []
            for p in pieces:
                hyp = " ".join(s["text"] for s in segs if p["start"] - 1 <= (s["start"] + s["end"]) / 2 <= p["end"] + 1)
                per.append(wer(p["text"], hyp))
            check(f"long file: every speaker's part transcribed in place (WER <= {CLIP_WER_MAX:.0%} each)",
                  all(w <= CLIP_WER_MAX for w in per), ", ".join(f"{p['speaker']} {w:.0%}" for p, w in zip(pieces, per)))
            e = wer(" ".join(p["text"] for p in pieces), " ".join(s["text"] for s in segs))
            check(f"long file: transcript accurate (WER <= {LONG_WER_MAX:.0%})", e <= LONG_WER_MAX, f"WER {e:.1%}")
            check("word timestamps present", "words" in segs[0] and segs[0]["words"][0]["end"] >= segs[0]["words"][0]["start"])

    print("\n6. Silence and broken files")
    j = jobs["silence.wav"]
    if check("silence.wav handled", j.status == T.DONE, j.error or ""):
        n = len(json.loads(Path(j.outputs["json"]).read_text())["segments"])
        check("silence gives no hallucinated text", n == 0, f"{n} segments")
    for name in ("truncated.mp3", "tiny_clip.wav"):
        if name in jobs:
            j = jobs[name]
            check(f"{name} handled without crashing", j.status == T.DONE, j.error or "")
            if name == "truncated.mp3" and j.status == T.DONE:
                n = len(json.loads(Path(j.outputs["json"]).read_text(encoding="utf-8"))["segments"])
                check("truncated.mp3 gives a partial transcript", n > 0, f"{n} segments")
    for name in bad:
        j = jobs[name]
        check(f"{name} marked failed with a reason", j.status == T.FAILED and bool(j.error), (j.error or "")[:60])
        check(f"{name} not retried (bad input)", j.attempts == 1)

    print("\n7. Resume and retry")
    jq2 = T.JobQueue(transcriber, out_dir, max_attempts=2)
    check("queue state saved to disk", (out_dir / "queue.json").exists() and len(jq2.jobs) == len(jobs))
    jq2.start()
    added = [jq2.add(f) for f in order]
    jq2.join()
    skipped = sum(not a for a in added)
    check("re-run skips files already done", skipped == len(done), f"{skipped} skipped, {sum(added)} re-queued (failed ones)")

    class Flaky:
        """Fails the first call, then works: checks retries for transient errors."""
        def __init__(self, inner): self.inner, self.calls = inner, 0
        def transcribe(self, audio, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated out-of-memory")
            return self.inner.transcribe(audio, **kw)
    inner = transcriber.model
    flaky = T.Transcriber(cfg, model=Flaky(inner))
    jq3 = T.JobQueue(flaky, work / "out_retry", max_attempts=2)
    jq3.start(); jq3.add(files[speech_files[0]]); jq3.join()
    j = next(iter(jq3.jobs.values()))
    check("transient error is retried and succeeds", j.status == T.DONE and j.attempts == 2, f"attempts={j.attempts}")

    print("\n8. Command line")
    rc = T.main([str(files[speech_files[-1]]), str(files[bad[0]]), "--out", str(work / "cli"),
                 "--model", args.model, "--language", "en", "--compute-type", "int8"]) if real else None
    if real:
        good = work / "cli" / f"{T._output_stem(files[speech_files[-1]])}.json"
        check("CLI exits 1 when a file failed, still writes the good one", rc == 1 and good.exists())
    found = T.collect_files([str(data_dir)])
    check("folder input finds every audio file, skips non-audio",
          {f.name for f in found} >= set(speech_files) and not any(f.is_dir() for f in found),
          f"{len(found)} files")

    passed = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{'=' * 60}\n{passed}/{len(RESULTS)} checks passed"
          f" (variant {args.variant}, {'mock model' if args.mock else 'model ' + args.model}, queue time {total:.1f}s)")
    if real and scores:
        print(f"Real-speech WER: median {med:.1%}, worst {max(s[4] for s in scores):.0%} over {len(scores)} clips")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {n} {d}")
    report = work / "verification_report.json"
    report.write_text(json.dumps({
        "variant": args.variant, "model": "mock" if args.mock else args.model, "speakers": speakers,
        "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in RESULTS],
        "clip_wer": [{"file": n, "speaker": c["speaker"], "wer": round(w, 4)} for n, c, _, _, w in scores] if real else [],
        "speaker_wer": table,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {report}\nTest audio: {data_dir}\nOutputs: {out_dir}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
