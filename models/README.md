# models/

The Whisper model goes here. It is not in the repository (`model.bin` is about
145 MB, over GitHub's 100 MB file limit), so download it once:

```bash
python scripts/download_model.py          # base.en, the default (about 145 MB)
```

That creates:

```
models/
  faster-whisper-base.en/
    config.json
    model.bin
    tokenizer.json
    vocabulary.txt
```

Without internet access on the machine, download those four files by hand from
<https://huggingface.co/Systran/faster-whisper-base.en/tree/main> and put them
in `models/faster-whisper-base.en/`.

Other English models work the same way (`tiny.en`, `small.en`, `medium.en`):
`python scripts/download_model.py small.en`, then run with `--model small.en`.
A model folder somewhere else can be used with `--model <folder>`, and
`TRANSCRIBE_MODELS_DIR` changes where this folder is.

If the model is missing or incomplete, `transcribe.py`, `server.py` and the
tests stop with exit code 2 and print these instructions. They never download
anything on their own.
