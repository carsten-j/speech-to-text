import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Define device and data type
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_id = "syvai/hviske-v2"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# --- Example using a sample from a Hugging Face dataset ---
# dataset = load_dataset("alexandrainst/coral", split="test")
# sample_audio = dataset[0]["audio"]

# --- Example using a local audio file ---
# Ensure your audio file is in a supported format (e.g., .wav, .mp3)
sample_audio = "sample.mp3"

# Perform transcription (replace with your audio source)
result = pipe(sample_audio, return_timestamps=True)
print(result["text"])

# print("Hviske v2 pipeline klar. Erstat \"sample_audio\" med din lydkilde.")
