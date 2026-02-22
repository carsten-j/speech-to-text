#!/usr/bin/env python3
"""Transcribe a Danish WAV file using syvai/hviske-v2 on Apple Silicon (MPS)."""

import sys
import time

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16000
CHUNK_SECONDS = 25  # stay under Whisper's 30s limit


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-wav>", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    audio = waveform.squeeze().numpy()

    # Load model directly on MPS (Apple Silicon GPU)
    model_name = "syvai/hviske-v2"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to("mps")

    # Transcribe in chunks, printing each as it completes
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    audio_duration = len(audio) / SAMPLE_RATE
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(device="mps")
            predicted_ids = model.generate(
                input_features, language="da", task="transcribe"
            )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
                0
            ].strip()
            if text:
                print(text, flush=True)
    elapsed = time.perf_counter() - t0
    print(
        f"\n--- {elapsed:.1f}s to transcribe {audio_duration:.1f}s of audio "
        f"({audio_duration / elapsed:.1f}x realtime) ---",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
