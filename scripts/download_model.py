#!/usr/bin/env python3
"""
Download a faster-whisper model once, into <project>/models/faster-whisper-<name>/.

  python scripts/download_model.py            # base.en (English, ~145 MB), the default model
  python scripts/download_model.py small.en   # more accurate, ~480 MB, slower

This and scripts/fetch_test_audio.py are the only steps that use the network.
After this, transcribe.py, server.py and the tests run fully offline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"   # must be set before huggingface_hub is imported
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import transcribe as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("model", nargs="?", default="base.en", help="tiny.en, base.en, small.en, medium.en")
    ap.add_argument("--force", action="store_true", help="download again even if the folder looks complete")
    args = ap.parse_args()

    dest = T.model_dir(args.model)
    if not args.force:
        try:
            T.check_model_dir(dest, args.model)
            print(f"'{args.model}' is already in {dest}. Nothing to do (use --force to download again).")
            return 0
        except T.ModelNotFoundError:
            pass
    print(f"Downloading {T.model_repo(args.model)} into {dest} ...")
    try:
        path = T.download_model_files(args.model, dest)
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {e}\n", file=sys.stderr)
        print("Check the internet connection (and proxy settings), then run this again. Or download the files\n"
              f"by hand from https://huggingface.co/{T.model_repo(args.model)}/tree/main into {dest}",
              file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in Path(path).iterdir() if f.is_file())
    print(f"Done: {size / 1e6:.0f} MB in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
