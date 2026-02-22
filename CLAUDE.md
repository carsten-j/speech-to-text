# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Danish speech-to-text transcription using the `syvai/hviske-v2` Whisper model via Hugging Face Transformers and PyTorch. Targets Apple Silicon (MPS) for inference.

## Commands

- **Install dependencies:** `uv sync`
- **Run transcription (chunked, MPS):** `uv run python transcribe.py <path-to-wav>`
- **Run pipeline-based transcription:** `uv run python hviske.py <path-to-wav>`
- **Convert MP3 to WAV:** `uv run python convert_audio.py`

## Architecture

- `transcribe.py` — Main transcription script. Loads WAV audio, chunks it (25s segments), and transcribes on MPS using WhisperProcessor/WhisperForConditionalGeneration directly. Hardcoded to Danish (`language="da"`).
- `hviske.py` — Alternative approach using HF `pipeline()` for simpler transcription (auto device selection, no manual chunking).
- `convert_audio.py` — Simple MP3-to-WAV converter using pydub.

## Key Details

- Python 3.12, managed with `uv`
- Model: `syvai/hviske-v2` (Danish Whisper variant)
- `transcribe.py` forces MPS device and float32; `hviske.py` auto-selects CUDA/CPU with float16/float32
- Audio is resampled to 16kHz mono before processing
