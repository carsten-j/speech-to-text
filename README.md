# Speech-to-Text

Danish speech-to-text transcription using the [syvai/hviske-v2](https://huggingface.co/syvai/hviske-v2) Whisper model via Hugging Face Transformers and PyTorch.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Local (Apple Silicon / CPU)

```bash
# Pipeline-based transcription (auto-selects CUDA/CPU)
uv run python hviske.py <path-to-audio>

# Chunked transcription on MPS (Apple Silicon GPU)
uv run python transcribe.py <path-to-wav>

# Convert MP3 to WAV
uv run python convert_audio.py
```

### Azure ML with GPU

Prerequisites: [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) with the `ml` extension.

```bash
az extension add -n ml
```

#### 1. Create a workspace (one-time)

```bash
az ml workspace create -n speech-ws -g my-resource-group -l westeurope
```

#### 2. Create a GPU compute cluster (one-time)

```bash
az ml compute create --name gpu-cluster --type amlcompute \
  --size Standard_NC4as_T4_v3 --min-instances 0 --max-instances 1 \
  -w speech-ws -g my-resource-group
```

Setting `--min-instances 0` scales the cluster to zero when idle to avoid charges.

#### 3. Upload audio as a data asset

```bash
az ml data create --name my-audio --path ./sample.mp3 --type uri_file \
  -w speech-ws -g my-resource-group
```

#### 4. Submit the transcription job

```bash
az ml job create --file job.yml -w speech-ws -g my-resource-group
```

This uses the configuration in `job.yml`, which runs `hviske.py` on the GPU cluster with the uploaded audio file as input.

#### 5. View job output

```bash
# List recent jobs
az ml job list -w speech-ws -g my-resource-group \
  --query "[].{name:name, status:status}" -o table

# Stream logs from a running or completed job
az ml job stream --name <job-name> -w speech-ws -g my-resource-group

# Download all outputs and logs locally
az ml job download --name <job-name> -w speech-ws -g my-resource-group
```

The job name is printed when you run `az ml job create`. You can also find it via `az ml job list` or in the [Azure ML Studio](https://ml.azure.com) portal under **Jobs**.

#### Cleanup

To delete all Azure resources and stop incurring charges, delete the resource group:

```bash
az group delete -n my-resource-group --yes --no-wait
```

This removes the workspace, compute cluster, storage, and all associated resources.
