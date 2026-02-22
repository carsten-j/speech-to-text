#!/usr/bin/env python3
"""Transcribe an audio file using syvai/hviske-v2 via HF pipeline."""

import sys
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-audio>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "syvai/hviske-v2"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    t0 = time.perf_counter()
    result = pipe(audio_path, return_timestamps=True)
    elapsed = time.perf_counter() - t0

    print(result["text"])
    print(f"\n--- Transcription completed in {elapsed:.1f}s ---", file=sys.stderr)


if __name__ == "__main__":
    main()
