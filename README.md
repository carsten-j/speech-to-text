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

### Hugging Face Inference Endpoints

Deploy `syvai/hviske-v2` as a serverless API on Hugging Face.

Prerequisites: Install the [HF CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and log in.

```bash
uv tool install 'huggingface-hub[cli]'
hf login
```

You also need a payment method on your HF account ([billing settings](https://huggingface.co/settings/billing)) and a token with `inference.endpoints.write` permission ([token settings](https://huggingface.co/settings/tokens)).

#### 1. Deploy the endpoint

```bash
hf endpoints deploy hviske-v2 \
  --repo syvai/hviske-v2 \
  --framework pytorch \
  --task automatic-speech-recognition \
  --accelerator gpu \
  --instance-size x1 \
  --instance-type nvidia-t4 \
  --region eu-west-1 \
  --vendor aws \
  --min-replica 0 \
  --max-replica 1 \
  --scale-to-zero-timeout 15
```

The endpoint scales to zero after 15 minutes of inactivity, so you only pay while it's in use.

#### 2. Check status

```bash
hf endpoints describe hviske-v2

# Show only the status and URL
hf endpoints describe hviske-v2 | jq '{state: .status.state, url: .status.url}'
```

Wait until the state shows `running` (usually 2-5 minutes on first deploy).

#### 3. Transcribe audio

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

curl https://<your-endpoint-url>.endpoints.huggingface.cloud \
  -X POST \
  --data-binary @your-audio.wav \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: audio/wav"
```

The endpoint URL is shown in the output of `hf endpoints describe hviske-v2`.

#### 4. Pause / resume

```bash
# Pause to stop charges
hf endpoints pause hviske-v2

# Resume when needed again
hf endpoints resume hviske-v2
```

#### Delete endpoint

To permanently delete the endpoint:

```bash
hf endpoints delete hviske-v2
```
