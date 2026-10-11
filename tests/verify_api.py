#!/usr/bin/env python3
"""
End-to-end checks for server.py (the HTTP API), fully offline.

  python tests/verify_api.py                 # real base.en model + fake models for timing-sensitive cases
  python tests/verify_api.py --skip-crash    # leave out the kill-and-restart test

Covers concurrent uploads, duplicate uploads, every malformed request we could
think of, bad audio, back-pressure (503), automatic and manual retry, delete,
and recovery after the server process is killed mid-transcription. Uses the
real English speech in test_audio/real. Everything it writes is kept in
test_output/api_check/.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe as T  # noqa: E402
import server as S  # noqa: E402
import verify as VF  # noqa: E402  (audio writer and WER scoring)

REAL = ROOT / "test_audio" / "real"
WORK = ROOT / "test_output" / "api_check"
SR = 16000
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

class Api:
    def __init__(self, port: int):
        self.port = port

    def call(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
             conn: http.client.HTTPConnection | None = None):
        c = conn or http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        if conn is None:
            c.close()
        try:
            parsed = json.loads(data) if data and "json" in (r.getheader("Content-Type") or "") else data
        except json.JSONDecodeError:
            parsed = data
        return r.status, dict(r.getheaders()), parsed

    def upload(self, data: bytes, filename: str | None):
        q = "" if filename is None else "?filename=" + urllib.parse.quote(filename)
        return self.call("POST", "/v1/transcriptions" + q, data)

    def raw(self, request: bytes, body_then_close: bytes | None = None) -> str:
        """Send hand-made bytes; returns the status line (for requests http.client won't make)."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        s.sendall(request)
        if body_then_close is not None:
            s.sendall(body_then_close)
            s.shutdown(socket.SHUT_WR)
        buf = b""
        try:
            while b"\r\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        return buf.split(b"\r\n", 1)[0].decode(errors="replace")

    def wait(self, ids, timeout: float = 600, until=(T.DONE, T.FAILED)) -> dict:
        deadline, out = time.time() + timeout, {}
        pending = set(ids)
        while pending and time.time() < deadline:
            for i in list(pending):
                st, _, d = self.call("GET", f"/v1/transcriptions/{i}")
                if st == 200 and d["status"] in until:
                    out[i] = d
                    pending.discard(i)
            if pending:
                time.sleep(0.5)
        for i in pending:
            out[i] = self.call("GET", f"/v1/transcriptions/{i}")[2]
        return out


class FakeModel:
    """Stand-in model for timing-sensitive tests: optional delay, optional first failures."""

    def __init__(self, delay: float = 0.0, fail_first: int = 0):
        self.delay, self.fail_first, self.calls = delay, fail_first, 0

    def transcribe(self, audio, **kw):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("simulated GPU out of memory")
        time.sleep(self.delay)
        seg = NS(start=0.0, end=1.0, text=" hello world", avg_logprob=-0.1, no_speech_prob=0.0, words=None)
        return [seg], NS(language="en", language_probability=1.0, duration=1.0, duration_after_vad=1.0)


def start(data: Path, transcriber: T.Transcriber, **kw):
    svc = S.Service(data, transcriber, **kw)
    svc.start()
    httpd = S.make_server(svc, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return svc, httpd, Api(httpd.server_port)


def stop(httpd) -> None:
    httpd.shutdown()
    httpd.server_close()


def fake(**kw) -> T.Transcriber:
    return T.Transcriber(T.TranscriberConfig(model="fake", language="en"), model=FakeModel(**kw))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def build_inputs(src: Path, manifest: list[dict]) -> dict:
    from faster_whisper import decode_audio
    src.mkdir(parents=True, exist_ok=True)
    clips = sorted(manifest, key=lambda m: m["duration"] or 0)
    plain = clips[:24]                       # uploaded as-is (flac / wav)
    formats = [(".mp3", "libmp3lame", None, 44100, "stereo", False),
               (".m4a", "aac", "ipod", 48000, "mono", False),
               (".ogg", "libvorbis", None, 44100, "stereo", False),
               (".webm", "libopus", "webm", 48000, "mono", False),
               ("_video.mp4", "aac", "mp4", 44100, "stereo", True),
               ("_8k.wav", "pcm_s16le", None, 8000, "mono", False)]
    converted = []
    for m, spec in zip(clips[24:30], formats):
        suffix, codec, fmt, rate, layout, video = spec
        p = src / f"{m['clip_id']}{suffix}"
        VF.write_audio(p, decode_audio(str(REAL / m["file"]), sampling_rate=SR), codec, fmt, rate, layout, video)
        converted.append((p, m))
    good = [(REAL / m["file"], m) for m in plain] + converted

    bad = {}
    bad["corrupt.mp3"] = b"this is not audio" * 200
    bad["notes_renamed.wav"] = "plain text, not audio\n".encode() * 100
    bad["random_bytes.flac"] = np.random.default_rng(7).bytes(50_000)
    wav = (REAL / next(m["file"] for m in clips if m["file"].endswith(".wav"))).read_bytes()
    bad["header_only.wav"] = wav[:44]
    p = src / "video_no_audio.mp4"
    import av
    with av.open(str(p), "w") as out:
        v = out.add_stream("mpeg4", rate=10)
        v.width, v.height, v.pix_fmt = 64, 64, "yuv420p"
        for _ in range(20):
            for pkt in v.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")):
                out.mux(pkt)
        for pkt in v.encode(None):
            out.mux(pkt)
    bad["video_no_audio.mp4"] = p.read_bytes()

    VF.write_audio(src / "silence.wav", np.zeros(10 * SR, np.float32), "pcm_s16le")
    first = decode_audio(str(REAL / clips[-1]["file"]), sampling_rate=SR)
    VF.write_audio(src / "tiny_clip.wav", first[: int(0.3 * SR)], "pcm_s16le")

    # A ~2.5 minute recording of several speakers, for the crash test.
    gap = np.zeros(15 * SR, np.float32)
    parts = []
    for m in clips[-6:]:
        parts += [decode_audio(str(REAL / m["file"]), sampling_rate=SR), gap]
    VF.write_audio(src / "long_meeting.mp3", np.concatenate(parts), "libmp3lame", None, 44100, "mono")
    return {"good": good, "bad": bad, "silence": src / "silence.wav", "tiny": src / "tiny_clip.wav",
            "long": src / "long_meeting.mp3", "long_parts": clips[-6:], "plain": plain,
            "spare": clips[30:34]}   # never uploaded to the main server before section 6


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--skip-crash", action="store_true")
    args = ap.parse_args()

    real = T.Transcriber(T.TranscriberConfig(model=args.model, device="cpu", compute_type="int8", language="en"))
    try:
        real.model  # load once; fail fast, with instructions, if it isn't on this PC
    except T.ModelNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    mpath = REAL / "manifest.json"
    if not mpath.exists():
        print(f"{mpath} not found. Run `python scripts/fetch_test_audio.py` once.", file=sys.stderr)
        return 2
    manifest = [m for m in json.loads(mpath.read_text(encoding="utf-8")) if m["language"] == "en"]
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"Work folder: {WORK}\n\n1. Building inputs")
    inp = build_inputs(WORK / "inputs", manifest)
    print(f"  {len(inp['good'])} speech files, {len(inp['bad'])} broken files")

    svc, httpd, api = start(WORK / "main", real)

    print("\n2. Basics")
    st, h, d = api.call("GET", "/healthz")
    check("GET /healthz", st == 200 and d["status"] == "ok" and d["model"] == args.model, str(d))
    st, h, d = api.call("HEAD", "/healthz")
    check("HEAD /healthz has no body", st == 200 and d == b"")
    st, h, d = api.call("PUT", "/v1/transcriptions", b"x")
    check("wrong method -> 405 with Allow header", st == 405 and h.get("Allow") == "GET, POST", f"{st} Allow={h.get('Allow')}")
    for path in ("/", "/v2/transcriptions", "/v1/transcriptions/abc/def/ghi", "/v1/transcriptions/0123456789abcdef/foo"):
        st, _, d = api.call("GET", path)
        check(f"unknown endpoint {path} -> 404 JSON", st == 404 and "error" in d, f"{st}")

    print("\n3. Concurrent uploads (16 clients at once, real speech, mixed formats)")
    t0 = time.time()
    latencies = {}

    def up(item):
        path, m = item
        t = time.time()
        r = api.upload(path.read_bytes(), path.name)
        latencies[path.name] = time.time() - t
        return path, m, r
    with ThreadPoolExecutor(16) as pool:
        ups = list(pool.map(up, inp["good"]))
    t_uploaded = time.time() - t0
    statuses = [r[0] for _, _, r in ups]
    ids = [r[2]["id"] for _, _, r in ups if r[0] == 202]
    check("every upload accepted with 202", statuses.count(202) == len(ups), f"{statuses.count(202)}/{len(ups)}")
    check("every upload got its own id", len(set(ids)) == len(ups), f"{len(set(ids))} ids")
    summary = api.call("GET", "/healthz")[2]["queue"]
    check("uploads answered without waiting for transcription", summary[T.DONE] < len(ups),
          f"all {len(ups)} uploads in {t_uploaded:.1f}s (slowest {max(latencies.values()):.2f}s); "
          f"only {summary[T.DONE]} transcribed by then")
    done = api.wait(ids)
    t_all = time.time() - t0
    check("all concurrent uploads transcribed", all(d["status"] == T.DONE for d in done.values()),
          f"{sum(d['status'] == T.DONE for d in done.values())}/{len(ids)} in {t_all:.0f}s")
    runs = sorted((d["started_at"], d["finished_at"]) for d in done.values() if d["started_at"])
    check("one file at a time on the single model worker", all(a[1] <= b[0] for a, b in zip(runs, runs[1:])))
    wers = []
    for (path, m, r) in ups:
        d = done[r[2]["id"]]
        hyp = " ".join(s["text"] for s in d.get("result", {}).get("segments", []))
        wers.append((VF.wer(m["text"], hyp), path.name))
    med = statistics.median(w for w, _ in wers)
    worst = max(wers)
    check("transcripts accurate: median WER <= 15%", med <= 0.15, f"median {med:.1%}")
    check("transcripts accurate: no clip above 40% WER", worst[0] <= 0.40, f"worst {worst[0]:.0%} ({worst[1]})")
    i0 = ids[0]
    st_j, _, j = api.call("GET", f"/v1/transcriptions/{i0}/json")
    st_s, hs, srt = api.call("GET", f"/v1/transcriptions/{i0}/srt")
    st_t, ht, txt = api.call("GET", f"/v1/transcriptions/{i0}/txt")
    check("json/srt/txt downloads", st_j == st_s == st_t == 200 and bool(j["segments"])
          and srt.decode().startswith("1\n") and "subrip" in hs["Content-Type"] and "text/plain" in ht["Content-Type"],
          f"{st_j}/{st_s}/{st_t}, srt starts {srt[:12]!r}")
    check("txt download matches the segments", txt.decode().strip() == " ".join(s["text"] for s in j["segments"]))
    check("timestamps valid and in order",
          all(0 <= s["start"] < s["end"] for s in j["segments"]) and
          all(a["start"] <= b["start"] for a, b in zip(j["segments"], j["segments"][1:])))

    print("\n4. Duplicate uploads (same audio, 6 clients at once, different names)")
    dup_src = inp["plain"][0]
    data = (REAL / dup_src["file"]).read_bytes()
    with ThreadPoolExecutor(6) as pool:
        rs = list(pool.map(lambda k: api.upload(data, f"copy {k}.flac"), range(6)))
    dup_ids = {r[2]["id"] for r in rs}
    check("all copies map to one id", len(dup_ids) == 1, str(dup_ids))
    check("already-uploaded audio answered 200 duplicate=true",
          all(r[0] == 200 and r[2].get("duplicate") for r in rs), str([r[0] for r in rs]))
    lst = api.call("GET", "/v1/transcriptions")[2]
    check("listed once, transcribed once", sum(i["id"] in dup_ids for i in lst["items"]) == 1
          and all(i["attempts"] == 1 for i in lst["items"] if i["id"] in dup_ids))
    fresh = inp["tiny"].read_bytes()
    with ThreadPoolExecutor(6) as pool:
        rs = list(pool.map(lambda k: api.upload(fresh, f"race {k}.wav"), range(6)))
    check("racing uploads of new audio: one 202, the rest 200, one job",
          sorted(r[0] for r in rs) == [200] * 5 + [202] and len({r[2]["id"] for r in rs}) == 1,
          str(sorted(r[0] for r in rs)))

    print("\n5. Malformed requests")
    files_before = sorted(p.name for p in (WORK / "main" / "uploads").iterdir())
    st, _, d = api.upload(b"", "empty.wav")
    check("empty body -> 400", st == 400 and "empty" in d["error"], d.get("error", ""))
    line = api.raw(b"POST /v1/transcriptions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
    check("chunked / no Content-Length -> 411", " 411 " in line, line)
    line = api.raw(b"POST /v1/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n")
    check("Content-Length: abc -> 400", " 400 " in line, line)
    line = api.raw(b"POST /v1/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n")
    check("negative Content-Length -> 400", " 400 " in line, line)
    line = api.raw(b"POST /v1/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Length: 1000000000000\r\n\r\n")
    check("1 TB upload refused before reading it -> 413", " 413 " in line, line)
    line = api.raw(b"POST /v1/transcriptions?filename=cut.wav HTTP/1.1\r\nHost: x\r\nContent-Length: 5000\r\n\r\n",
                   body_then_close=b"RIFF" + b"\0" * 96)
    check("body shorter than Content-Length -> 400", " 400 " in line, line)
    for name, blob in inp["bad"].items():
        st, _, d = api.upload(blob, name)
        check(f"bad audio {name} -> 422 with reason", st == 422 and d.get("error"), (d.get("error") or "")[:70])
    time.sleep(0.2)
    files_after = sorted(p.name for p in (WORK / "main" / "uploads").iterdir())
    check("rejected uploads leave nothing behind (no file, no job)", files_after == files_before,
          f"{len(files_before)} -> {len(files_after)} files")
    c = http.client.HTTPConnection("127.0.0.1", api.port, timeout=30)
    st1 = api.call("POST", "/v1/transcriptions?filename=x.mp3", inp["bad"]["corrupt.mp3"], conn=c)[0]
    st2 = api.call("GET", "/healthz", conn=c)[0]
    c.close()
    check("connection still usable after a rejected upload (keep-alive)", (st1, st2) == (422, 200), f"{st1}, {st2}")
    st, _, d = api.call("GET", "/v1/transcriptions?status=bogus")
    check("unknown status filter -> 400", st == 400)
    bogus = "0123456789abcdef"
    codes = [api.call("GET", f"/v1/transcriptions/{bogus}")[0], api.call("GET", f"/v1/transcriptions/{bogus}/srt")[0],
             api.call("POST", f"/v1/transcriptions/{bogus}/retry")[0], api.call("DELETE", f"/v1/transcriptions/{bogus}")[0]]
    check("unknown id -> 404 on status/srt/retry/delete", codes == [404] * 4, str(codes))
    st, _, _ = api.call("GET", "/v1/transcriptions/" + urllib.parse.quote("../../server.py", safe=""))
    check("path traversal in id -> 404", st == 404)
    st, _, d = api.call("GET", f"/v1/transcriptions/{ids[0]}/retry")
    check("GET on retry -> 405", st == 405)

    print("\n6. Edge-case audio and file names")
    st, _, d = api.upload(inp["silence"].read_bytes(), "silence.wav")
    sil = d["id"]
    st2, _, d2 = api.upload(inp["long"].read_bytes()[: 1 << 16], "partial download.mp3")  # first 64 KB of an MP3
    spare = [(REAL / m["file"]).read_bytes() for m in inp["spare"]]
    uni_name = "réunion d'été – notes 📝.flac"
    st3, _, d3 = api.upload(spare[0], uni_name)
    st4, _, d4 = api.upload(spare[1], "../../outside.wav")
    st5, _, d5 = api.call("POST", "/v1/transcriptions", spare[2])
    check("uploads with unicode, traversal and no file name accepted",
          [st, st2, st3, st4, st5] == [202] * 5, str([st, st2, st3, st4, st5]))
    res = api.wait([sil, d2["id"], d3["id"], d4["id"], d5["id"]])
    check("silence -> done with 0 segments", res[sil]["status"] == T.DONE and res[sil]["result"]["segments"] == [],
          f"{len(res[sil].get('result', {}).get('segments', []))} segments")
    check("truncated MP3 (64 KB of a download) -> partial transcript", res[d2["id"]]["status"] == T.DONE
          and len(res[d2["id"]]["result"]["segments"]) > 0)
    check("unicode file name kept", res[d3["id"]]["filename"] == uni_name and res[d3["id"]]["status"] == T.DONE,
          res[d3["id"]]["filename"])
    check("client path stripped from file name, nothing written outside uploads/",
          res[d4["id"]]["filename"] == "outside.wav" and not (WORK / "outside.wav").exists()
          and not (WORK / "main" / "outside.wav").exists())
    check("upload without a file name works", res[d5["id"]]["status"] == T.DONE and res[d5["id"]]["filename"] == "upload")
    st, _, d = api.upload(inp["tiny"].read_bytes(), "tiny again.wav")
    check("0.3 s clip handled", api.wait([d["id"]])[d["id"]]["status"] == T.DONE)

    print("\n7. Delete")
    victim = ids[-1]
    meta = json.loads((WORK / "main" / "uploads" / f"{victim}.meta.json").read_text(encoding="utf-8"))
    outs = list((WORK / "main" / "transcripts").glob(f"{victim}_*"))
    st, _, _ = api.call("DELETE", f"/v1/transcriptions/{victim}")
    check("DELETE finished job -> 204", st == 204)
    check("audio, metadata and transcripts removed", not (WORK / "main" / "uploads" / meta["stored_as"]).exists()
          and not list((WORK / "main" / "transcripts").glob(f"{victim}_*")) and len(outs) == 3)
    check("deleted job -> 404", api.call("GET", f"/v1/transcriptions/{victim}")[0] == 404)
    lst = api.call("GET", "/v1/transcriptions?status=done")[2]
    check("list ?status=done consistent with queue", lst["count"] == lst["summary"][T.DONE], f"{lst['count']} done")
    stop(httpd)

    print("\n8. Back-pressure and conflicts (slow fake model, queue limit 3)")
    svc2, httpd2, api2 = start(WORK / "slow", fake(delay=1.5), max_pending=3, max_attempts=1)
    blobs = [(REAL / m["file"]).read_bytes() for m in inp["plain"][3:8]]
    codes, slow_ids, retry_after = [], [], None
    for b in blobs:
        st, h, d = api2.upload(b, "clip.flac")
        codes.append(st)
        if st == 202:
            slow_ids.append(d["id"])
        if st == 503:
            retry_after = h.get("Retry-After")
    check("uploads beyond the queue limit -> 503 with Retry-After", codes[:3] == [202] * 3 and 503 in codes[3:]
          and retry_after == "30", str(codes))
    st, _, d = api2.call("GET", f"/v1/transcriptions/{slow_ids[-1]}/srt")
    check("transcript of an unfinished job -> 409", st == 409 and d.get("status") in (T.PENDING, T.PROCESSING), str(d))
    st, _, _ = api2.call("DELETE", f"/v1/transcriptions/{slow_ids[-1]}")
    check("DELETE of an unfinished job -> 409", st == 409)
    st, _, _ = api2.call("POST", f"/v1/transcriptions/{slow_ids[-1]}/retry")
    check("retry of an unfinished job -> 409", st == 409)
    api2.wait(slow_ids)
    st, _, _ = api2.upload(blobs[4], "clip.flac")
    check("accepted again once the queue drains", st == 202)
    stop(httpd2)

    print("\n9. Retry and recovery of failed transcriptions")
    svc3, httpd3, api3 = start(WORK / "retry_manual", fake(fail_first=1), max_attempts=1)
    st, _, d = api3.upload(blobs[0], "flaky.flac")
    r = api3.wait([d["id"]])[d["id"]]
    check("transient error with no retries left -> failed with reason",
          r["status"] == T.FAILED and "simulated" in (r["error"] or ""), r["error"] or "")
    check("failed jobs listed by ?status=failed", api3.call("GET", "/v1/transcriptions?status=failed")[2]["count"] == 1)
    st, _, _ = api3.call("POST", f"/v1/transcriptions/{d['id']}/retry")
    r = api3.wait([d["id"]])[d["id"]]
    check("POST retry -> 202, then done", st == 202 and r["status"] == T.DONE, f"{st}, {r['status']}")
    st, _, _ = api3.call("POST", f"/v1/transcriptions/{d['id']}/retry")
    check("retry of a finished job -> 409", st == 409)
    st, _, d2 = api3.upload(blobs[0], "flaky again.flac")
    check("re-uploading a failed-then-fixed file reuses the result", st == 200 and d2["status"] == T.DONE)
    stop(httpd3)
    svc4, httpd4, api4 = start(WORK / "retry_auto", fake(fail_first=1), max_attempts=2)
    st, _, d = api4.upload(blobs[0], "flaky.flac")
    r = api4.wait([d["id"]])[d["id"]]
    check("transient error retried automatically", r["status"] == T.DONE and r["attempts"] == 2, f"attempts={r['attempts']}")
    stop(httpd4)

    class Exploding:
        def transcribe(self, audio, **kw):
            raise MemoryError("simulated")
    svc5, httpd5, api5 = start(WORK / "worker_survives", T.Transcriber(T.TranscriberConfig(model="fake"), model=Exploding()),
                               max_attempts=1)
    st, _, d = api5.upload(blobs[1], "a.flac")
    r1 = api5.wait([d["id"]])[d["id"]]
    svc5.queue.transcriber = fake()
    st, _, d2 = api5.upload(blobs[2], "b.flac")
    r2 = api5.wait([d2["id"]], timeout=30)[d2["id"]]
    check("worker keeps going after a job blows up (MemoryError)",
          r1["status"] == T.FAILED and r2["status"] == T.DONE, f"{r1['status']} / {r2['status']}")
    stop(httpd5)

    if not args.skip_crash:
        print("\n10. Crash recovery: kill the server process mid-transcription, start it again")
        data = WORK / "crash"
        port = free_port()
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, str(ROOT / "server.py"), "--port", str(port), "--data", str(data),
               "--model", args.model, "--compute-type", "int8"]
        log1 = open(WORK / "crash_run1.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log1, stderr=subprocess.STDOUT, env=env)
        capi = Api(port)
        for _ in range(120):
            try:
                if capi.call("GET", "/healthz")[0] == 200:
                    break
            except OSError:
                time.sleep(0.5)
        st, _, dl = capi.upload(inp["long"].read_bytes(), "long_meeting.mp3")
        shorts = [capi.upload((REAL / m["file"]).read_bytes(), m["file"])[2]["id"] for m in inp["plain"][8:11]]
        seen = None
        for _ in range(200):
            seen = capi.call("GET", f"/v1/transcriptions/{dl['id']}")[2]["status"]
            if seen == T.PROCESSING:
                break
            time.sleep(0.1)
        time.sleep(1.0)
        proc.kill()   # hard kill: no cleanup runs
        proc.wait()
        log1.close()
        state = json.loads((data / "transcripts" / "queue.json").read_text(encoding="utf-8"))
        statuses = sorted(j["status"] for j in state.values())
        check("killed mid-transcription; state on disk shows unfinished work",
              seen == T.PROCESSING and T.PROCESSING in statuses and T.PENDING in statuses, str(statuses))
        (data / "uploads" / ".upload-deadbeef.wav").write_bytes(b"half an upload")
        log2 = open(WORK / "crash_run2.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log2, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(120):
                try:
                    if capi.call("GET", "/healthz")[0] == 200:
                        break
                except OSError:
                    time.sleep(0.5)
            res = capi.wait([dl["id"], *shorts], timeout=600)
            check("after restart every job finishes",
                  all(r["status"] == T.DONE for r in res.values()), str([r["status"] for r in res.values()]))
            long_r = res[dl["id"]]
            hyp = " ".join(s["text"] for s in long_r["result"]["segments"])
            e = VF.wer(" ".join(m["text"] for m in inp["long_parts"]), hyp)
            check("the interrupted long file is complete after recovery (WER <= 25%)", e <= 0.25,
                  f"WER {e:.1%}, {len(long_r['result']['segments'])} segments, ends {long_r['result']['segments'][-1]['end']:.0f}s")
            check("half-written upload from the crash cleaned up", not (data / "uploads" / ".upload-deadbeef.wav").exists())
        finally:
            proc.kill()
            proc.wait()
            log2.close()
        log_text = (WORK / "crash_run2.log").read_text(encoding="utf-8")
        check("restart log reports the re-queued jobs", "re-queued" in log_text,
              next((l for l in log_text.splitlines() if "re-queued" in l), ""))

        r = subprocess.run([sys.executable, str(ROOT / "server.py"), "--port", str(free_port()), "--data",
                            str(WORK / "nomodel"), "--model", "small.en"], capture_output=True, text=True,
                           env=env, timeout=120)
        check("server refuses to start without the model (exit 2, clear message)",
              r.returncode == 2 and "scripts/download_model.py small.en" in r.stderr, f"exit {r.returncode}")

    passed = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{'=' * 60}\n{passed}/{len(RESULTS)} API checks passed (model {args.model})")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {n}  {d}")
    (WORK / "api_report.json").write_text(json.dumps([{"check": n, "passed": ok, "detail": d} for n, ok, d in RESULTS],
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
