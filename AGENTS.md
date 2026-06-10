# FireRedVAD Agent Guide

## Install

```bash
pip install -r requirements.txt          # pinned deps incl. torch (CUDA cu118) + kaldi native fbank
pip install -e .                         # editable install of the `fireredvad` package
```

No tests, CI, Makefile, or tox config exist in this repo.

## Lint

Ruff only, config is inline in `pyproject.toml`:

```bash
ruff check fireredvad
```

Target: Python 3.10, line-length 120, rules E/F/W/I.

## Entry Points

| Usage | Location |
|---|---|
| Quick API | `fireredvad.non_stream_vad()`, `fireredvad.stream_vad_full()`, `fireredvad.non_stream_aed()` |
| Single CLI | `fireredvad.bin.fireredvad_cli:main` (task: `vad` / `stream_vad` / `aed`) |
| Shell examples | `examples/inference_vad.sh`, `examples/inference_stream_vad.sh`, `examples/inference_aed.sh` |

## Model & Audio Requirements (not obvious)

- Model weights must be downloaded separately from HuggingFace or ModelScope into `pretrained_models/FireRedVAD/` (expected subdirs: `VAD`, `Stream-VAD`, `AED`).
- Audio must be **16 kHz, 16-bit mono PCM**. `detect()` takes a wav file path, not raw audio arrays.
- Fixed feature params in `fireredvad/core/constants.py`: 25 ms frames, 10 ms shift, 80-dim fbank.

## Repo Layout

- `fireredvad/` — main package (VAD, StreamVAD, AED classes, core in `fireredvad/core/`).
- `fireredvad/bin/` — standalone scripts (not auto-discovered by setuptools `packages.find`).
- `runtime/` — exports/serving code (ONNX, NCNN) and README.md only.
- `examples/` — shell inference scripts and sample WAVs.
