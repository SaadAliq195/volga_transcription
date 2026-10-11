#!/usr/bin/env python3
"""
One-time download of real speech test clips (already done: the clips are in
the repo). Besides scripts/download_model.py, this is the only step that uses
the network; transcribe.py, server.py and the tests run fully offline.

  python scripts/fetch_test_audio.py            # fetch into test_audio/real/
  python scripts/fetch_test_audio.py --force    # re-download everything

Clips come from open datasets that ship reference transcripts, fetched one
file at a time through the Hugging Face dataset viewer API (no full dataset
download):

  LibriSpeech test-clean / test-other  openslr/librispeech_asr   CC BY 4.0
  VCTK (many English accents)          sanchit-gandhi/vctk       CC BY 4.0
  EdAcc (conversational, accented)     edinburghcstr/edacc       CC BY-SA 4.0

English only: the pipeline targets English audio.

Writes test_audio/real/<clip>.<ext> and test_audio/real/manifest.json with the
file, source, speaker, gender, accent, licence and reference text of each clip.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_audio" / "real"

LICENCES = {
    "openslr/librispeech_asr": "CC BY 4.0",
    "sanchit-gandhi/vctk": "CC BY 4.0",
    "edinburghcstr/edacc": "CC BY-SA 4.0",
}


_last_call = 0.0


def get_json(url: str, tries: int = 6) -> dict:
    """GET with pacing (the viewer API rate-limits bursts) and backoff on 429/5xx."""
    global _last_call
    for i in range(tries):
        time.sleep(max(0.0, 1.5 - (time.time() - _last_call)))
        _last_call = time.time()
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1:
                raise
            wait = 15 * 2 ** i if "429" in str(e) else 2 ** (i + 1)
            print(f"    retry in {wait}s ({e})")
            time.sleep(wait)
    raise AssertionError


def rows(ds: str, cfg: str, split: str, offset: int, length: int = 100) -> list[dict]:
    q = urllib.parse.urlencode(dict(dataset=ds, config=cfg, split=split, offset=offset, length=length))
    return [r["row"] for r in get_json(f"{API}/rows?{q}")["rows"]]


def num_rows(ds: str, cfg: str, split: str) -> int:
    q = urllib.parse.urlencode(dict(dataset=ds, config=cfg, split=split, offset=0, length=1))
    return get_json(f"{API}/rows?{q}")["num_rows_total"]


def download(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                tmp.write_bytes(r.read())
            tmp.replace(path)
            return
        except Exception as e:
            if i == 3:
                raise
            print(f"    retry download ({e})")
            time.sleep(2 ** (i + 1))


def audio_src(row: dict, key: str = "audio") -> tuple[str, str]:
    a = row[key][0]
    ext = {"audio/wav": ".wav", "audio/flac": ".flac", "audio/mpeg": ".mp3", "audio/ogg": ".ogg"}.get(a.get("type"), "")
    if not ext:
        ext = Path(urllib.parse.urlparse(a["src"]).path).suffix or ".wav"
    return a["src"], ext


def nwords(t: str) -> int:
    return len(t.split())


# ---------------------------------------------------------------------------
# Selection per source
# ---------------------------------------------------------------------------

def pick_librispeech(split: str, n_speakers: int, per_speaker: int) -> list[dict]:
    ds, total = "openslr/librispeech_asr", num_rows("openslr/librispeech_asr", "all", split)
    picked, speakers = [], []
    for k in range(n_speakers * 2):
        if len(speakers) >= n_speakers:
            break
        offset = int(total * k / (n_speakers * 2))
        page = rows(ds, "all", split, offset)
        by_spk: dict[str, list[dict]] = {}
        for r in page:
            by_spk.setdefault(str(r["speaker_id"]), []).append(r)
        for spk, rs in by_spk.items():
            if spk in speakers or len(speakers) >= n_speakers:
                continue
            rs = sorted(rs, key=lambda r: nwords(r["text"]))
            choice = [rs[-1]]                                   # longest: a long clip
            mid = [r for r in rs if 10 <= nwords(r["text"]) <= 30]
            if per_speaker > 1 and mid:
                choice.append(mid[len(mid) // 2])               # plus a medium one
            speakers.append(spk)
            for r in choice[:per_speaker]:
                picked.append(dict(row=r, dataset=ds, split=split, speaker=f"LS-{spk}",
                                   gender=None, accent=None, language="en", text=r["text"],
                                   condition="clean read speech" if split == "test.clean"
                                   else "harder read speech (test-other)",
                                   clip_id=f"ls{'c' if split == 'test.clean' else 'o'}_{spk}_{r['id'].split('-')[-1]}"))
            break  # one speaker per page keeps them spread over the split
    return picked


def pick_vctk(n_speakers: int) -> list[dict]:
    ds = "sanchit-gandhi/vctk"
    total = num_rows(ds, "default", "train")
    cands: dict[str, dict] = {}
    step = 3000
    for offset in range(0, total, step):
        for r in rows(ds, "default", "train", offset, length=8):
            if r["speaker_id"] not in cands and r.get("text") and nwords(r["text"]) >= 6:
                cands[r["speaker_id"]] = r
                break
    # Spread over accents first, then alternate gender.
    by_accent: dict[str, list[dict]] = {}
    for r in cands.values():
        by_accent.setdefault(r["accent"] or "unknown", []).append(r)
    chosen: list[dict] = []
    while len(chosen) < n_speakers and any(by_accent.values()):
        for acc in sorted(by_accent, key=lambda a: -len(by_accent[a])):
            if by_accent[acc] and len(chosen) < n_speakers:
                want = "F" if sum(c["gender"] == "F" for c in chosen) <= len(chosen) / 2 else "M"
                lst = by_accent[acc]
                r = next((x for x in lst if x["gender"] == want), lst[0])
                lst.remove(r)
                chosen.append(r)
    return [dict(row=r, dataset=ds, split="train", speaker=f"VCTK-{r['speaker_id']}",
                 gender={"F": "female", "M": "male"}.get(r["gender"], r["gender"]),
                 accent=f"{r['accent']}" + (f" ({r['region']})" if r.get("region") else ""),
                 language="en", text=r["text"], condition="studio read sentence (short)",
                 clip_id=f"vctk_{r['speaker_id']}_{r['text_id']}") for r in chosen]


def pick_edacc(n_speakers: int) -> list[dict]:
    ds = "edinburghcstr/edacc"
    total = num_rows(ds, "default", "test")
    chosen, accents = [], set()
    for k in range(40):
        if len(chosen) >= n_speakers:
            break
        offset = int(total * k / 40)
        for r in rows(ds, "default", "test", offset):
            t = r["text"]
            if (12 <= nwords(t) <= 40 and "<" not in t and r["speaker"] not in {c["speaker"] for c in chosen}
                    and r["accent"] not in accents):
                chosen.append(r)
                accents.add(r["accent"])
                break
    return [dict(row=r, dataset=ds, split="test", speaker=r["speaker"], gender=r["gender"],
                 accent=r["accent"], language="en", text=r["text"],
                 condition="conversational, accented, room audio",
                 clip_id=f"edacc_{r['speaker']}_{i}") for i, r in enumerate(chosen)]



# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download clips that already exist")
    ap.add_argument("--sources", default="ls-clean,ls-other,vctk,edacc",
                    help="comma list of sources to fetch; clips already in the manifest are kept")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    old = {} if args.force or not mpath.exists() else \
        {m["clip_id"]: m for m in json.loads(mpath.read_text(encoding="utf-8")) if m["language"] == "en"}

    sources = {"ls-clean": ("LibriSpeech test-clean", lambda: pick_librispeech("test.clean", 8, 2)),
               "ls-other": ("LibriSpeech test-other", lambda: pick_librispeech("test.other", 6, 1)),
               "vctk": ("VCTK", lambda: pick_vctk(12)),
               "edacc": ("EdAcc", lambda: pick_edacc(6))}
    print("Choosing clips")
    plan: list[dict] = []
    for key in args.sources.split(","):
        name, fn = sources[key.strip()]
        try:
            got = fn()
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            continue
        print(f"  {name}: {len(got)} clips, {len({g['speaker'] for g in got})} speakers")
        plan += got

    print("\nDownloading")
    manifest = [m for k, m in old.items() if k not in {c["clip_id"] for c in plan}]
    for c in plan:
        key = "audio" if "audio" in c["row"] else "flac"
        src, ext = audio_src(c["row"], key)
        path = OUT / f"{c['clip_id']}{ext}"
        if args.force or not path.exists():
            download(src, path)
        entry = {k: c[k] for k in ("clip_id", "dataset", "split", "speaker", "gender", "accent",
                                   "language", "condition", "text")}
        entry["file"] = path.name
        entry["licence"] = LICENCES.get(c["dataset"], "see dataset card")
        entry["bytes"] = path.stat().st_size
        try:
            sys.path.insert(0, str(ROOT))
            import transcribe as T
            entry["duration"] = T.probe_audio(path)["duration"]
        except Exception as e:
            entry["duration"] = None
            print(f"  warning: cannot probe {path.name}: {e}")
        manifest.append(entry)
        print(f"  {path.name:<40} {entry['duration'] or 0:6.1f}s  {entry['speaker']}")

    manifest.sort(key=lambda m: m["clip_id"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    en = [m for m in manifest if m["language"] == "en"]
    size = sum(m["bytes"] for m in manifest)
    print(f"\n{len(manifest)} clips from {len({m['speaker'] for m in en})} speakers, {size / 1e6:.1f} MB, "
          f"{sum(m['duration'] or 0 for m in manifest) / 60:.1f} min of audio")
    print(f"Saved to {OUT}")
    return 0 if en else 1


if __name__ == "__main__":
    sys.exit(main())
