#!/usr/bin/env python3
"""
HTTP API for the transcription pipeline. Standard library only, runs offline.

  python server.py --port 8000 --data data

  # upload (raw bytes in the body; the original name goes in ?filename=)
  curl --data-binary @meeting.mp3 "http://127.0.0.1:8000/v1/transcriptions?filename=meeting.mp3"
    -> 202 {"id": "3f2a9c...", "status": "pending", "links": {...}}
  curl http://127.0.0.1:8000/v1/transcriptions/3f2a9c...        # status, plus segments when done
  curl http://127.0.0.1:8000/v1/transcriptions/3f2a9c.../srt    # also /json and /txt
  curl -X POST   http://127.0.0.1:8000/v1/transcriptions/3f2a9c.../retry
  curl -X DELETE http://127.0.0.1:8000/v1/transcriptions/3f2a9c...
  curl http://127.0.0.1:8000/v1/transcriptions?status=failed    # list
  curl http://127.0.0.1:8000/healthz

Concurrency: every request gets its own thread, and uploads are streamed to
disk in 1 MB chunks while being hashed, so many clients can upload at once
without holding whole files in memory. Transcription itself runs on the one
queue worker that owns the model, one file at a time; uploads are answered
with 202 straight away and never wait for the model. When too many jobs are
waiting, new uploads get 503 with Retry-After instead of piling up.

Storage (under --data):
  uploads/<id><ext>          the audio, named by content hash: the same file
                             uploaded twice is stored and transcribed once
  uploads/<id>.meta.json     original file name, size, sha256, upload time, probe
  transcripts/<id>_<ext>.json/.srt/.txt   the results
  transcripts/queue.json     state of every job (status, attempts, error, times)

Recovery: bad audio is rejected at upload (422) and never queued. Transient
errors are retried automatically (--retries). A failed job can be retried with
POST .../retry. On start-up every stored upload that is not finished
(including one that was mid-transcription when the server died) is queued
again, so a crash or restart loses nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe as T  # noqa: E402  (also switches the Hugging Face hub to offline)

log = logging.getLogger("transcribe.server")

ID_RE = re.compile(r"^[0-9a-f]{16}$")
EXT_RE = re.compile(r"^\.[a-z0-9]{1,10}$")
CHUNK = 1 << 20
CONTENT_TYPES = {"json": "application/json; charset=utf-8",
                 "srt": "application/x-subrip; charset=utf-8",
                 "txt": "text/plain; charset=utf-8"}


class ApiError(Exception):
    def __init__(self, http_status: HTTPStatus, message: str, headers: Optional[dict] = None, **extra):
        super().__init__(message)
        self.status, self.message, self.headers, self.extra = http_status, message, headers or {}, extra


class Service:
    """Storage plus the job queue; the HTTP handler is a thin layer over this."""

    def __init__(self, data_dir: Path, transcriber: T.Transcriber, max_upload_bytes: int = 500 << 20,
                 max_pending: int = 100, max_attempts: int = 2):
        self.uploads = data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.queue = T.JobQueue(transcriber, data_dir / "transcripts", max_attempts=max_attempts)
        self.max_upload_bytes = max_upload_bytes
        self.max_pending = max_pending
        self._lock = threading.Lock()  # makes "already stored?" + store + queue one step

    def start(self) -> int:
        """Start the worker and re-queue every stored upload that isn't finished."""
        for tmp in self.uploads.glob(".upload-*"):  # half-written uploads from a crash
            tmp.unlink(missing_ok=True)
        self.queue.start()
        requeued = 0
        for meta_path in sorted(self.uploads.glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            path = self.uploads / meta["stored_as"]
            job = self.queue.get(path)
            if path.exists() and (job is None or job.status in (T.PENDING, T.PROCESSING)):
                requeued += self.queue.add(path)
        if requeued:
            log.info("re-queued %d unfinished upload(s) from a previous run", requeued)
        return requeued

    # -- helpers ----------------------------------------------------------
    def _meta(self, job_id: str) -> dict:
        if not ID_RE.match(job_id):
            raise ApiError(HTTPStatus.NOT_FOUND, f"no transcription with id {job_id!r}")
        p = self.uploads / f"{job_id}.meta.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no transcription with id {job_id!r}") from None

    def _waiting(self) -> int:
        s = self.queue.summary()
        return s[T.PENDING] + s[T.PROCESSING]

    def describe(self, job_id: str, with_result: bool = True) -> dict:
        meta = self._meta(job_id)
        job = self.queue.get(self.uploads / meta["stored_as"])
        base = f"/v1/transcriptions/{job_id}"
        out = {"id": job_id, "filename": meta["filename"], "bytes": meta["bytes"],
               "uploaded_at": meta["uploaded_at"], "input": meta["input"],
               "status": job.status if job else T.PENDING,
               "attempts": job.attempts if job else 0, "error": job.error if job else None,
               "started_at": job.started_at if job else None,
               "finished_at": job.finished_at if job else None,
               "processing_seconds": job.processing_seconds if job else None,
               "links": {"self": base, "json": f"{base}/json", "srt": f"{base}/srt",
                         "txt": f"{base}/txt", "retry": f"{base}/retry"}}
        if with_result and job and job.status == T.DONE:
            try:
                result = json.loads(Path(job.outputs["json"]).read_text(encoding="utf-8"))
            except FileNotFoundError:   # deleted by another request a moment ago
                raise ApiError(HTTPStatus.NOT_FOUND, f"no transcription with id {job_id!r}") from None
            out["result"] = {k: result.get(k) for k in ("language", "duration", "duration_after_vad",
                                                         "real_time_factor", "segments")}
        return out

    # -- operations --------------------------------------------------------
    def upload(self, body, length: int, filename: str) -> tuple[HTTPStatus, dict]:
        if self._waiting() >= self.max_pending:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                           f"queue is full ({self.max_pending} jobs waiting); try again later",
                           headers={"Retry-After": "30"})
        name = Path(filename.replace("\\", "/")).name or "upload"   # never trust client paths
        ext = Path(name).suffix.lower()
        ext = ext if EXT_RE.match(ext) else ""
        tmp = self.uploads / f".upload-{secrets.token_hex(8)}{ext}"
        sha, received = hashlib.sha256(), 0
        try:
            with open(tmp, "wb") as f:
                while received < length:
                    chunk = body.read(min(CHUNK, length - received))
                    if not chunk:
                        raise ApiError(HTTPStatus.BAD_REQUEST,
                                       f"upload ended early: got {received} of {length} bytes")
                    f.write(chunk)
                    sha.update(chunk)
                    received += len(chunk)
            try:  # reject bad audio now, so it never takes a queue slot
                probe = T.probe_audio(tmp)
            except T.AudioError as e:
                raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(e).replace(str(tmp), name)
                               .replace(tmp.name, name)) from None

            job_id = sha.hexdigest()[:16]
            stored = self.uploads / f"{job_id}{ext}"
            with self._lock:
                meta_path = self.uploads / f"{job_id}.meta.json"
                if meta_path.exists():   # same audio already uploaded: reuse that job
                    return HTTPStatus.OK, dict(self.describe(job_id, with_result=False), duplicate=True)
                os.replace(tmp, stored)
                meta = {"id": job_id, "filename": name, "stored_as": stored.name, "bytes": received,
                        "sha256": sha.hexdigest(), "uploaded_at": time.time(), "input": probe}
                T._atomic_write(meta_path, json.dumps(meta, indent=2, ensure_ascii=False))
                self.queue.add(stored)
            log.info("accepted %s as %s (%d bytes)", name, job_id, received)
            return HTTPStatus.ACCEPTED, self.describe(job_id, with_result=False)
        finally:
            tmp.unlink(missing_ok=True)

    def output(self, job_id: str, kind: str) -> bytes:
        meta = self._meta(job_id)
        job = self.queue.get(self.uploads / meta["stored_as"])
        if not job or job.status != T.DONE:
            raise ApiError(HTTPStatus.CONFLICT, "transcription is not finished",
                           status=job.status if job else T.PENDING)
        try:
            return Path(job.outputs[kind]).read_bytes()
        except FileNotFoundError:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no transcription with id {job_id!r}") from None

    def retry(self, job_id: str) -> dict:
        meta = self._meta(job_id)
        path = self.uploads / meta["stored_as"]
        job = self.queue.get(path)
        if job and job.status != T.FAILED:
            raise ApiError(HTTPStatus.CONFLICT, f"only failed jobs can be retried (status: {job.status})")
        self.queue.add(path)
        return self.describe(job_id, with_result=False)

    def delete(self, job_id: str) -> None:
        meta = self._meta(job_id)
        path = self.uploads / meta["stored_as"]
        with self._lock:
            job = self.queue.get(path)
            if job and not self.queue.remove(path):
                raise ApiError(HTTPStatus.CONFLICT, f"cannot delete while {job.status}; try again when it finishes")
            for p in [*(job.outputs.values() if job else []), path, self.uploads / f"{job_id}.meta.json"]:
                Path(p).unlink(missing_ok=True)

    def list(self, status: Optional[str]) -> dict:
        items = []
        for meta_path in sorted(self.uploads.glob("*.meta.json"), key=lambda p: p.stat().st_mtime):
            d = self.describe(meta_path.name[:16], with_result=False)
            if status is None or d["status"] == status:
                items.append({k: d[k] for k in ("id", "filename", "status", "attempts", "error")})
        return {"count": len(items), "summary": self.queue.summary(), "items": items}


class Handler(BaseHTTPRequestHandler):
    service: Service
    server_version = "transcribe/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 60   # a client that stops sending can't hold a thread forever

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status: HTTPStatus, body, content_type: str = CONTENT_TYPES["json"], headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body, indent=2, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _route(self, method: str) -> None:
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.strip("/").split("/") if p]
        query = urllib.parse.parse_qs(url.query)
        s = self.service
        try:
            if parts == ["healthz"]:
                self._allow(method, "GET")
                return self._send(HTTPStatus.OK, {"status": "ok", "model": s.queue.transcriber.config.model,
                                                  "queue": s.queue.summary()})
            if parts[:2] != ["v1", "transcriptions"] or len(parts) > 4:
                raise ApiError(HTTPStatus.NOT_FOUND, f"no such endpoint: {url.path}")
            if len(parts) == 2:
                self._allow(method, "GET", "POST")
                if method == "GET":
                    status = query.get("status", [None])[0]
                    if status not in (None, T.PENDING, T.PROCESSING, T.DONE, T.FAILED):
                        raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown status filter {status!r}")
                    return self._send(HTTPStatus.OK, s.list(status))
                return self._upload(query)
            job_id = parts[2]
            if len(parts) == 3:
                self._allow(method, "GET", "DELETE")
                if method == "DELETE":
                    s.delete(job_id)
                    return self._send(HTTPStatus.NO_CONTENT, b"")
                return self._send(HTTPStatus.OK, s.describe(job_id))
            action = parts[3]
            if action == "retry":
                self._allow(method, "POST")
                return self._send(HTTPStatus.ACCEPTED, s.retry(job_id))
            if action in CONTENT_TYPES:
                self._allow(method, "GET")
                return self._send(HTTPStatus.OK, s.output(job_id, action), CONTENT_TYPES[action])
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such endpoint: {url.path}")
        except ApiError as e:
            if e.status in (HTTPStatus.LENGTH_REQUIRED, HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            HTTPStatus.SERVICE_UNAVAILABLE) or (method == "POST" and e.status == HTTPStatus.BAD_REQUEST):
                self.close_connection = True   # the unread body would corrupt the next request
            self._send(e.status, {"error": e.message, **e.extra}, headers=e.headers)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except Exception as e:   # a bug must not take the server down
            log.exception("unhandled error on %s %s", method, self.path)
            self.close_connection = True
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"internal error: {type(e).__name__}"})

    def _allow(self, method: str, *allowed: str) -> None:
        if method not in allowed:
            raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} not allowed here",
                           headers={"Allow": ", ".join(allowed)})

    def _upload(self, query: dict) -> None:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked" or "Content-Length" not in self.headers:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(self.headers["Content-Length"])
            if length < 0:
                raise ValueError
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from None
        if length == 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty upload: send the audio file as the request body")
        if length > self.service.max_upload_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                           f"file is {length} bytes; the limit is {self.service.max_upload_bytes}")
        filename = query.get("filename", [""])[0] or self.headers.get("X-Filename", "") or "upload"
        status, body = self.service.upload(self.rfile, length, filename)
        self._send(status, body, headers={"Location": body["links"]["self"]})

    def do_GET(self): self._route("GET")
    def do_HEAD(self): self._route("GET")
    def do_POST(self): self._route("POST")
    def do_DELETE(self): self._route("DELETE")
    def do_PUT(self): self._route("PUT")
    def do_PATCH(self): self._route("PATCH")


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HTTP API for the faster-whisper transcription queue.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default="data", help="where uploads and transcripts are stored")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default="default")
    ap.add_argument("--word-timestamps", action="store_true")
    ap.add_argument("--retries", type=int, default=1, help="extra automatic attempts for a failed job")
    ap.add_argument("--max-upload-mb", type=float, default=500)
    ap.add_argument("--max-pending", type=int, default=100, help="queue depth before uploads get 503")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "0"

    cfg = T.TranscriberConfig(model=args.model, device=args.device, compute_type=args.compute_type,
                              language="en", word_timestamps=args.word_timestamps,
                              allow_download=args.allow_download)
    transcriber = T.Transcriber(cfg)
    try:
        transcriber.model   # load once, before accepting uploads
    except T.ModelNotFoundError as e:
        log.error("%s", e)
        return 2
    service = Service(Path(args.data), transcriber, int(args.max_upload_mb * (1 << 20)),
                      args.max_pending, 1 + args.retries)
    service.start()
    httpd = make_server(service, args.host, args.port)
    log.info("listening on http://%s:%d (data in %s)", args.host, httpd.server_port, Path(args.data).resolve())
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down; unfinished jobs resume on the next start")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
