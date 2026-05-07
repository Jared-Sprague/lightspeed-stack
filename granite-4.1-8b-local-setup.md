# Granite 4.1 8B Local Setup Guide

Step-by-step instructions for running Granite 4.1 8B locally with llama.cpp on a Framework 16 (RTX 5070 Laptop GPU, 8GB GDDR7) and connecting it to lightspeed-stack.

---

## 1. Prerequisites

- **OS:** Fedora (or any Linux with recent kernel)
- **GPU:** NVIDIA RTX 5070 Laptop GPU (8GB GDDR7)
- **NVIDIA driver:** 570+ (verify with `nvidia-smi`)
- **CUDA toolkit:** 12.8+ (verify with `nvcc --version`)
- **System RAM:** 16GB+ (only the model runs on GPU; OS and lightspeed-stack use system RAM)
- **Disk space:** ~6 GB for the GGUF file, ~2 GB for the llama.cpp build
- **Tools:** `git`, `cmake`, `make`, `python3`, `huggingface-cli` (`pip install huggingface-hub`)

Confirm your GPU is visible:

```bash
nvidia-smi
```

### Install CUDA toolkit on Fedora 43

RPM Fusion provides the NVIDIA driver but not the CUDA toolkit. You need to add NVIDIA's repo for the toolkit, while excluding their driver packages so they don't conflict with RPM Fusion's driver.

**1. Add NVIDIA's CUDA repo:**

```bash
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/$(uname -m)/cuda-fedora43.repo
```

**2. Exclude driver packages** (keep using RPM Fusion's driver):

```bash
sudo dnf config-manager setopt cuda-fedora43-$(uname -m).exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig
```

**3. Install the toolkit:**

```bash
sudo dnf clean all
sudo dnf install cuda-toolkit
```

**4. Add `nvcc` to your PATH** (add to `~/.bashrc` for persistence):

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**5. Verify:**

```bash
nvcc --version
```

---

## 2. Build llama.cpp with CUDA

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

The server binary will be at `build/bin/llama-server`. Verify it built correctly:

```bash
./build/bin/llama-server --version
```

---

## 3. Download Granite 4.1 8B Q4_K_M

The recommended quantization is **Q4_K_M** (~5 GB). This fits entirely in 8GB VRAM with headroom for the KV cache. bartowski's imatrix quants are the standard community source for GGUFs.

```bash
huggingface-cli download bartowski/ibm-granite_granite-4.1-8b-GGUF \
  --include "ibm-granite_granite-4.1-8b-Q4_K_M.gguf" \
  --local-dir ~/models/granite-4.1-8b
```

---

## 4. Start llama-server

Run llama-server with the Granite model. This serves an **OpenAI-compatible API** on port 8082:

```bash
./build/bin/llama-server \
  -m ~/models/granite-4.1-8b/ibm-granite_granite-4.1-8b-Q4_K_M.gguf \
  -ngl 99 \
  -fa on \
  -np 1 \
  -c 16384 \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  --jinja \
  --temp 0.3 \
  --repeat-penalty 1.1 \
  --port 8082
```

### What each flag does

| Flag | Long name | Purpose |
|---|---|---|
| `-ngl 99` | `--n-gpu-layers` | Offload all layers to GPU (the model fits entirely in 8GB) |
| `-fa on` | `--flash-attn` | Flash attention — reduces VRAM and improves throughput |
| `-np 1` | `--parallel` | Single request slot — prevents KV cache duplication |
| `-c 16384` | `--ctx-size` | 16K token context window (max that fits on 8GB with this model) |
| `-ctk q8_0` | `--cache-type-k` | Quantize KV cache keys to 8-bit — halves KV cache VRAM |
| `-ctv q8_0` | `--cache-type-v` | Quantize KV cache values to 8-bit — halves KV cache VRAM |
| `-b 2048` | `--batch-size` | Batch size for prompt processing — higher = faster prefill |
| `-ub 2048` | `--ubatch-size` | Micro-batch size — controls GPU kernel granularity |
| `--jinja` | | Enable Jinja chat templates (required for tool calling) |
| `--temp 0.3` | `--temperature` | Sampling temperature (default: 0.8). Lowered for grounded, deterministic answers from RAG context |
| `--repeat-penalty 1.1` | | Penalizes repeated token sequences (default: 1.0 = disabled). Prevents small models from looping |
| `--port 8082` | | Listen port for the OpenAI-compatible API |

### Verify it's running

```bash
curl http://localhost:8082/v1/models
```

You should see the Granite model listed. Test a completion:

```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite_granite-4.1-8b-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 100
  }'
```

### VRAM budget

The RTX 5070 Laptop GPU advertises 8151 MiB, but ~600 MiB is reserved by the driver/firmware and CUDA runtime, leaving ~7500 MiB usable.

| Component | VRAM |
|---|---|
| Model weights (Q4_K_M) | 5,059 MiB |
| KV cache (16K ctx, q8_0) | 1,360 MiB |
| Compute buffers | 848 MiB |
| **Total** | **~7,267 MiB** |

This leaves ~250 MiB of headroom. 32K context requires ~2720 MiB for the KV cache alone, which OOMs on this card. 16K is the practical maximum.

---

## 5. Connect lightspeed-stack to llama-server

llama-server exposes an OpenAI-compatible API. Llama Stack connects to it using the `remote::vllm` provider type (works with any OpenAI-compatible server, not just vllm).

> **Important:** Do NOT use `remote::openai` — that provider always hits the real OpenAI API regardless of any URL you set. Use `remote::vllm` for local servers.

### Step 5a — Create a Llama Stack run config

Copy the existing `run.yaml` and modify the inference provider:

```bash
cp run.yaml run-local.yaml
```

In `run-local.yaml`, make these changes:

**1. Replace the inference provider** (change `remote::openai` to `remote::vllm`):

```yaml
providers:
  inference:
  - provider_id: local-granite
    provider_type: remote::vllm
    config:
      base_url: http://localhost:8082/v1
      api_token: "not-needed"
      max_tokens: 4096
```

> **Note:** The config field must be `base_url` (not `url`). Some example files in the repo use `url` but the current vllm provider requires `base_url`.

**2. Disable safety** (no safety model available offline):

```yaml
  safety: []
```

And in `registered_resources`, clear the shields:

```yaml
registered_resources:
  shields: []
```

And at the bottom of the file, replace the safety section:

```yaml
safety: {}
```

**3. Register the model** in `registered_resources.models`:

```yaml
registered_resources:
  models:
  - model_id: ibm-granite_granite-4.1-8b-Q4_K_M.gguf
    provider_id: local-granite
    model_type: llm
    provider_model_id: ibm-granite_granite-4.1-8b-Q4_K_M.gguf
```

> The `model_id` must match the name llama-server reports at `/v1/models`. Verify with: `curl http://localhost:8082/v1/models`

**4. Clear any stale registry** from previous runs:

```bash
rm -f ~/.llama/storage/rag/kv_store.db
```

### Step 5b — Create the lightspeed-stack config

Copy the existing config and point it at your local run config:

```bash
cp lightspeed-stack.yaml lightspeed-stack-local.yaml
```

In `lightspeed-stack-local.yaml`, change the library client config path:

```yaml
llama_stack:
  use_as_library_client: true
  library_client_config_path: run-local.yaml
```

### Step 5c — Run lightspeed-stack

```bash
uv run make run CONFIG=lightspeed-stack-local.yaml
```

Or directly:

```bash
uv run src/lightspeed_stack.py -c lightspeed-stack-local.yaml
```

---

## 6. Troubleshooting

### OOM on launch

The RTX 5070 8GB has ~7500 MiB usable VRAM (not the advertised 8151 MiB — ~600 MiB is reserved by the driver/firmware/CUDA runtime). Other GPU processes (Steam, games, desktop compositors) further reduce available VRAM.

**Check what's using your GPU:**

```bash
nvidia-smi
```

Close GPU-heavy applications (e.g., `steam -shutdown`) before launching llama-server.

**If it still OOMs**, reduce the context window:

```bash
-c 16384   # instead of 32768
```

Or drop to a smaller KV cache quantization:

```bash
-ctk q4_0 -ctv q4_0   # instead of q8_0
```

### Slow first response

The first prompt processes the full system prompt and any RAG context through prefill. This is normal — subsequent turns in the same conversation are faster because the KV cache is warm. Increase `-b` and `-ub` to `4096` if prefill is a bottleneck.

### Model name mismatch

llama-server derives the model name from the GGUF filename. If lightspeed-stack reports "model not found," check the exact name:

```bash
curl -s http://localhost:8082/v1/models | python3 -m json.tool
```

Update the `model_id` / `provider_model_id` / `default_model` values in your config files to match.

### Stale Llama Stack registry

Llama Stack persists model and shield registrations in a sqlite KV store at `~/.llama/storage/rag/kv_store.db`. If you change providers, models, or safety config in `run-local.yaml`, the old registrations can conflict with the new ones, causing errors like:

```
ValueError: Object of type 'shield' and identifier 'llama-guard' already exists with conflicting field values
```

Fix by deleting the stale registry — it gets recreated on next startup:

```bash
rm -f ~/.llama/storage/rag/kv_store.db
```

Do this any time you change inference providers, model IDs, or safety configuration in your Llama Stack run config.

### "No LLM model found" from lightspeed-stack

If lightspeed-stack starts but queries return `"Model not found"` / `"No Model is configured"`, the model wasn't registered with Llama Stack. Check:

1. **Model must be in `registered_resources.models`** in `run-local.yaml`:

```yaml
registered_resources:
  models:
  - model_id: ibm-granite_granite-4.1-8b-Q4_K_M.gguf
    provider_id: local-granite
    model_type: llm
    provider_model_id: ibm-granite_granite-4.1-8b-Q4_K_M.gguf
```

2. **The `model_id` must match** what llama-server reports. Verify with:

```bash
curl -s http://localhost:8082/v1/models | jq '.data[].id'
```

### "Incorrect API key" / hitting real OpenAI API

If you see errors like `Incorrect API key provided: not-needed` with a reference to `platform.openai.com`, you're using the wrong provider type. `remote::openai` always hits the real OpenAI API regardless of any URL config.

**Fix:** Use `remote::vllm` instead — it works with any OpenAI-compatible server (llama.cpp, vllm, TGI, etc.):

```yaml
  - provider_id: local-granite
    provider_type: remote::vllm    # NOT remote::openai
    config:
      base_url: http://localhost:8082/v1   # must be base_url, not url
```

### "You must provide a URL" from vllm provider

If you see `ValueError: You must provide a URL in config.yaml (or via the VLLM_URL environment variable) to use vLLM` even though you set a URL, the config field name is wrong. The vllm provider requires `base_url`, not `url`:

```yaml
    config:
      base_url: http://localhost:8082/v1   # correct
      # url: http://localhost:8082/v1      # wrong — silently ignored
```

### CUDA version issues

Blackwell GPUs (RTX 50-series) need CUDA 12.8+. If you see garbled output or crashes:

```bash
nvcc --version   # should be 12.8+
nvidia-smi       # driver should be 570+
```

### Verify GPU is being used

While llama-server is running:

```bash
nvidia-smi
```

You should see `llama-server` in the process list using ~7.5 GB of VRAM. If VRAM usage is near zero, the model is running on CPU — rebuild llama.cpp with `-DGGML_CUDA=ON`.

---

## 7. Quick Reference

| Item | Value |
|---|---|
| **Model** | Granite 4.1 8B |
| **Quantization** | Q4_K_M (~5 GB) |
| **Context window** | 16,384 tokens |
| **VRAM usage** | ~7.3 GB (with 16K context) |
| **Inference speed** | ~80+ tok/s (full GPU offload) |
| **API endpoint** | `http://localhost:8082/v1` |
| **API format** | OpenAI-compatible (chat completions) |
| **lightspeed-stack port** | `http://localhost:8080` |

---

## 8. Starting Everything (Cheat Sheet)

**Terminal 1 — llama-server:**

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/granite-4.1-8b/ibm-granite_granite-4.1-8b-Q4_K_M.gguf \
  -ngl 99 -fa on -np 1 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  --jinja --temp 0.3 --repeat-penalty 1.1 \
  --port 8082
```

**Terminal 2 — lightspeed-stack:**

```bash
cd ~/projects/lightspeed-stack
uv run make run CONFIG=lightspeed-stack-local.yaml
```

**Terminal 3 — verify:**

**1. Check llama-server is serving the model:**

```bash
curl http://localhost:8082/v1/models | jq
```

**2. Check lightspeed-stack sees the model:**

```bash
curl http://localhost:8080/v1/models | jq
```

You should see `ibm-granite_granite-4.1-8b-Q4_K_M.gguf` in the response.

**3. Run a test query through the full stack (RAG + inference):**

```bash
curl -sX POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "configure remote desktop using gnome"}' | jq .
```

**4. Verify GPU utilization during inference:**

In a separate terminal, watch GPU usage while a query is running:

```bash
watch -n 0.5 nvidia-smi
```

You should see GPU-Util spike during inference (from 0% idle to 50-90% under load). If GPU-Util stays at 0% while VRAM is allocated, the model loaded onto the GPU but isn't being used for compute — check that lightspeed-stack is routing to the correct model.
