# Local LLM on Framework 16 (RTX 5070): Model Recommendations for Technical Documentation Chat

## Hardware

The Framework 16's NVIDIA option is the **RTX 5070 Laptop GPU** (Blackwell, 100W TGP):

- **8GB GDDR7** — shipping since Nov 2025, $699 module
- **12GB GDDR7** — pre-order now, ships June 2026, $1,199 module

Run `nvidia-smi` to confirm which variant you have. If you bought before May 2026, you almost certainly have the 8GB version.

---

## Use Case

Technical documentation chat assistant — RAG-based Q&A over known content. Priorities: accurate retrieval-grounded answers, long context for large docs, low hallucination, good instruction following. Frontier-tier reasoning and coding benchmarks are secondary.

---

## Models Under Consideration

### Models that fit in 8GB VRAM (no CPU offloading)

| Model | Architecture | Total / Active Params | Q4_K_M GGUF Size | Context Window |
|---|---|---|---|---|
| **Granite 4.1 8B** | Dense | 8B / 8B | ~5 GB | 512K |
| **Phi-4 Mini** | Dense | 3.8B / 3.8B | ~3 GB | 128K |
| **Gemma 4 E4B** | Dense | ~4.5B / ~4.5B | ~3 GB | 128K |
| **Qwen 3.5 9B** | Dense | 9B / 9B | ~7 GB | 32K |

### Models requiring CPU offloading on 8GB VRAM

| Model | Architecture | Total / Active Params | Q4_K_M GGUF Size | Context Window |
|---|---|---|---|---|
| Qwen 3.6 35B-A3B | MoE | 35B / 3.6B | ~22 GB | 256K |
| Qwen 3.6 27B | Dense | 27B / 27B | ~17 GB | 128K |
| Gemma 4 26B-A4B | MoE | 26B / 3.8B | ~16 GB | 256K |
| Gemma 4 31B | Dense | 31B / 31B | ~20 GB | 256K |

The MoE models have fast compute (small active params) but still need their full weights in memory. On 8GB VRAM, weights spill to system RAM via PCIe, cutting throughput to 15-25 tok/s vs 80+ tok/s for models that fit entirely on-GPU.

---

## Recommended Models for Technical Documentation Chat

### Primary Recommendation: Granite 4.1 8B

IBM's Granite family is explicitly trained on enterprise documentation, RAG-grounded data, and tool-calling workflows. The 4.1 8B (released April 29, 2026) is a dense decoder-only transformer under Apache 2.0.

**Why it's the top pick for this use case:**

- **RAG-native training.** IBM penalized hallucinations against retrieved context during RL alignment. Granite 3.0 8B already led [RAGBench](https://research.ibm.com/blog/conversational-RAG-benchmark) evaluations (100K RAG tasks from user manuals). Granite 4.1 builds on that pipeline with a 5-stage pre-training process and 4-stage RL alignment focused on grounded answers.
- **512K context window** — 2-4x larger than all other candidates. Ingest entire technical manuals without chunking.
- **Tool calling.** BFCL V3 score of 68.3 — best in its size class. Useful if the doc assistant needs structured extraction or API calls.
- **Fits in 8GB VRAM.** ~5GB at Q4_K_M. Full GPU offload with headroom for KV cache. No CPU offloading penalty.
- **IFEval 87.1** — strong instruction following means it stays on-task with doc Q&A prompts.

**Benchmarks vs other 8GB-class models:**

| Benchmark | Granite 4.1 8B | Phi-4 Mini (3.8B) | Qwen 3.5 9B | Gemma 4 E4B (~4.5B) |
|---|---|---|---|---|
| MMLU | 73.8 | 73.0 | ~73 | 69.4 |
| HumanEval | 87.2 | 74.4 | — | — |
| BFCL V3 (tool call) | 68.3 | — | — | — |
| IFEval (instruction) | 87.1 | — | — | — |
| Context window | 512K | 128K | 32K | 128K |
| VRAM (Q4_K_M) | ~5 GB | ~3 GB | ~7 GB | ~3 GB |

**Tradeoff:** Weaker than Qwen/Gemma on open-ended creative generation and complex multi-step reasoning. For grounded doc Q&A, that's irrelevant.

**Recommended quant:** `Q4_K_M`. Official GGUFs available from [ibm-granite on HuggingFace](https://huggingface.co/ibm-granite) and via [IBM's GGUF repo](https://github.com/IBM/gguf).

### Secondary: Phi-4 Mini (3.8B)

Fastest option. ~3GB at Q4, 300+ tok/s on your GPU. 128K context. Matches Granite on MMLU but lacks its RAG-specific training and tool-calling strength. Worth testing as a low-latency alternative — if answer quality is sufficient on your docs, the speed is hard to beat.

### Secondary: Qwen 3.5 9B

Strongest raw quality that fits in 8GB (barely, at ~7GB Q4_K_M). Only 32K context — a real limitation for large doc ingestion. Best option if your docs are multilingual.

### Tertiary: Gemma 4 E4B (~4.5B)

Similar niche to Phi-4 Mini. 128K context, ~3GB. Weaker on reasoning (MMLU 69.4%) but supports vision if your docs include diagrams you want to query.

---

## Larger Models (require CPU offloading on 8GB VRAM)

If the 8B-class models prove insufficient for your documentation complexity, these are worth benchmarking despite the speed penalty.

### Gemma 4: Use 26B-A4B (MoE)

**Why:** The 26B-A4B delivers ~97% of the dense 31B's quality (MMLU Pro 82.6% vs 85.2%, LMArena Elo 1441 vs 1452) while activating only 3.8B params per token. On your RTX 5070, this means the compute is fast — the bottleneck is weight loading, not math. The 31B dense model is both larger on disk and demands full 31B of compute per token, making it strictly worse on your hardware for marginal quality gain.

The E4B (~4.5B dense) fits entirely in 8GB VRAM and runs at 80+ tok/s, but drops hard on reasoning and coding (MMLU Pro 69.4%, LiveCodeBench 52.0% vs 80.0% for 31B). It's a fallback, not a recommendation.

**Recommended quant:** `Q4_K_M` or Unsloth `UD-Q4_K_XL` (dynamic 4-bit, slightly better quality at same size).

**Benchmark highlights:**

| Benchmark | 31B Dense | 26B-A4B MoE | E4B |
|---|---|---|---|
| MMLU Pro | 85.2% | 82.6% | 69.4% |
| AIME 2026 | 89.2% | ~87% | — |
| LiveCodeBench v6 | 80.0% | — | 52.0% |
| LMArena Elo | 1452 | 1441 | — |

### Qwen 3.6: Use 35B-A3B (MoE)

**Why:** The dense 27B is higher quality across the board (SWE-bench 77.2% vs 73.4%, TerminalBench 59.3 vs 51.5), but it activates all 27B parameters per token — on 8GB VRAM with heavy CPU offloading, this translates to painfully slow inference. The 35B-A3B is 3-4x faster at generation because only 3.6B params are active. On VRAM-constrained hardware, that speed difference dominates the experience. The quality gap (aggregate 67 vs 74) is real but acceptable for interactive use where waiting 30+ seconds per response kills the workflow.

If you upgrade to 12GB VRAM or have 64GB+ system RAM with fast DDR5, the 27B dense becomes more viable.

**Recommended quant:** `Q4_K_M` or Unsloth `UD-Q4_K_XL`.

**Benchmark highlights:**

| Benchmark | 27B Dense | 35B-A3B MoE |
|---|---|---|
| SWE-bench Verified | 77.2% | 73.4% |
| Terminal-Bench 2.0 | 59.3 | 51.5 |
| SkillsBench | 48.2% | ~32.7% |
| Generation speed (RTX 3090) | ~28 tok/s | ~101 tok/s |

---

## Head-to-Head: Qwen 3.6 35B-A3B vs Gemma 4 26B-A4B

For your hardware, these are the two recommended models. Comparing them:

| Factor | Qwen 3.6 35B-A3B | Gemma 4 26B-A4B |
|---|---|---|
| GGUF size (Q4_K_M) | ~22 GB | ~16 GB |
| Active params | 3.6B | 3.8B |
| Context window | 256K | 256K |
| VRAM pressure | Higher — more to offload | Lower — better fit for 8GB |
| Coding (SWE-bench) | 73.4% | Not directly comparable |
| Reasoning (MMLU Pro) | — | 82.6% |
| Multimodal | Text only | Text + image (no audio at 26B) |
| RTX 50-series stability | No known issues | Known CUDA issues with some quants on Blackwell |

**For 8GB VRAM, Gemma 4 26B-A4B is the better fit.** Its GGUF is ~6GB smaller, meaning less weight shuttling through PCIe from system RAM, which is the primary bottleneck on your setup.

**For quality-sensitive coding tasks, Qwen 3.6 35B-A3B edges ahead** — Qwen 3.6 family benchmarks on coding are exceptionally strong (the dense 27B matches Claude 4.5 Opus on Terminal-Bench).

---

## llama.cpp Launch Configurations for RTX 5070

All configurations assume the Framework 16 RTX 5070 Laptop GPU. Option matrices cover both the 8GB and 12GB variants. All commands use `llama-server` for an OpenAI-compatible API endpoint.

### Common flags explained

| Flag | Purpose |
|---|---|
| `-ngl N` | Number of layers offloaded to GPU. `99`/`999` = all layers. Reduce if OOM. |
| `-fa on` | Flash attention — reduces VRAM usage and improves throughput. Use on all models. |
| `-c N` | Context window in tokens. Larger = more VRAM for KV cache. |
| `-ctk q8_0 -ctv q8_0` | Quantize KV cache to 8-bit. Halves KV cache VRAM. Recommended at long contexts. |
| `-b N -ub N` | Batch / micro-batch size for prompt processing. Higher = faster prefill. |
| `-t N` | CPU threads. Set to physical core count (Framework 16: typically 8-12). |
| `--n-cpu-moe N` | Offload N MoE expert layers to CPU. Only for MoE models. |
| `--jinja` | Enable Jinja chat templates. Required for tool calling and some models. |
| `-np 1` | Single slot. Prevents KV cache multiplication from parallel requests. |

---

### Granite 4.1 8B (Primary recommendation)

Download (bartowski, imatrix quants):
```bash
huggingface-cli download bartowski/ibm-granite_granite-4.1-8b-GGUF \
  --include "ibm-granite_granite-4.1-8b-Q4_K_M.gguf" --local-dir ./
```
Or (Unsloth, dynamic quants):
```bash
huggingface-cli download unsloth/granite-4.1-8b-GGUF \
  --include "*Q4_K_M*" --local-dir ./
```
No official IBM GGUF repo exists for 4.1 yet. Both bartowski and Unsloth are well-tested community sources.

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q4_K_M (~5 GB) | Q8_0 (~8.5 GB) |
| **Context** | 8192-32768 | 32768-65536 |
| **KV cache quant** | Yes (`-ctk q8_0 -ctv q8_0`) | Optional (only if >32K context) |
| **GPU layers** | 99 (all fit) | 99 (all fit) |
| **Est. VRAM usage** | ~6-7 GB @ 32K ctx | ~10-11 GB @ 64K ctx |

**8GB variant:**

```bash
llama-server \
  -m ibm-granite_granite-4.1-8b-Q4_K_M.gguf \
  -ngl 99 -fa on -np 1 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  --jinja --port 8080
```

**12GB variant (higher quant, larger context):**

```bash
llama-server \
  -m ibm-granite_granite-4.1-8b-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 4096 \
  --jinja --port 8080
```

**Sampling (doc Q&A — low creativity, grounded answers):**

```
--temp 0.3 --top-p 0.9 --top-k 40 --repeat-penalty 1.1
```

---

### Phi-4 Mini 3.8B

Download: `huggingface-cli download bartowski/microsoft_Phi-4-mini-instruct-GGUF --include "*Q8_0*"`

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q8_0 (~4 GB) | Q8_0 (~4 GB) |
| **Context** | 32768-65536 | 65536-131072 |
| **KV cache quant** | Optional | No (plenty of headroom) |
| **GPU layers** | 99 (all fit) | 99 (all fit) |
| **Est. VRAM usage** | ~5-6 GB @ 64K ctx | ~5-6 GB @ 64K ctx |

Even on 8GB, Phi-4 Mini at Q8_0 leaves substantial headroom. No reason to use Q4 — go near-lossless.

**8GB variant:**

```bash
llama-server \
  -m Phi-4-mini-instruct-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 65536 \
  -b 2048 -ub 2048 \
  --jinja --port 8080
```

**12GB variant (maximize context):**

```bash
llama-server \
  -m Phi-4-mini-instruct-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 131072 \
  -b 4096 -ub 4096 \
  --jinja --port 8080
```

**Sampling:**

```
--temp 0.3 --top-p 0.9 --top-k 40 --repeat-penalty 1.1
```

---

### Qwen 3.5 9B

Download: `huggingface-cli download unsloth/Qwen3.5-9B-GGUF --include "*Q4_K_M*"`

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q4_K_M (~7 GB) | Q8_0 (~10 GB) |
| **Context** | 8192-16384 | 16384-32768 |
| **KV cache quant** | Yes (mandatory) | Yes (recommended) |
| **GPU layers** | 99 (tight fit) | 99 (all fit) |
| **Est. VRAM usage** | ~7.5 GB @ 16K ctx | ~11 GB @ 32K ctx |

On 8GB, this is a tight fit. Keep context at 16K or below. Monitor VRAM — if you OOM, drop to Q3_K_M or reduce `-c`.

**8GB variant:**

```bash
llama-server \
  -m Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -fa on -np 1 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  --jinja --port 8080
```

**12GB variant:**

```bash
llama-server \
  -m Qwen3.5-9B-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 4096 \
  --jinja --port 8080
```

**Sampling:**

```
--temp 0.3 --top-p 0.9 --top-k 20 --presence-penalty 1.5
```

---

### Gemma 4 E4B (~4.5B)

Download: `huggingface-cli download unsloth/gemma-4-e4b-it-GGUF --include "*Q8_0*"`

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q8_0 (~5 GB) | Q8_0 (~5 GB) |
| **Context** | 32768-65536 | 65536-131072 |
| **KV cache quant** | Optional | No |
| **GPU layers** | 99 (all fit) | 99 (all fit) |
| **Est. VRAM usage** | ~6 GB @ 64K ctx | ~6 GB @ 64K ctx |

Like Phi-4 Mini, E4B is small enough for near-lossless quant on either VRAM variant.

**8GB variant:**

```bash
llama-server \
  -m gemma-4-e4b-it-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 65536 \
  -b 2048 -ub 2048 \
  --chat-template gemma \
  --jinja --port 8080
```

**12GB variant:**

```bash
llama-server \
  -m gemma-4-e4b-it-Q8_0.gguf \
  -ngl 99 -fa on -np 1 \
  -c 131072 \
  -b 4096 -ub 4096 \
  --chat-template gemma \
  --jinja --port 8080
```

**Sampling:**

```
--temp 0.7 --top-p 0.95 --top-k 64
```

**Caution (Blackwell GPUs):** If output is garbled, try `--chat-template gemma` (included above). Avoid CUDA 13.2 runtime. If gibberish persists, try `FORCE_CUBLAS` or a different quant ([known issue](https://github.com/ggml-org/llama.cpp/issues/21371)). Use `-m` with a direct path rather than `-hf` to avoid the vision projector auto-download OOM.

---

### Gemma 4 26B-A4B (MoE — requires CPU offloading on 8GB)

Download: `huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF --include "*Q4_K_M*"`

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q4_K_M (~16 GB) | Q4_K_M (~16 GB) |
| **Context** | 8192-16384 | 16384-32768 |
| **KV cache quant** | Yes (mandatory) | Yes (recommended) |
| **GPU layers** | 20-28 (partial offload) | 35-45 (more on GPU) |
| **CPU MoE offload** | Not applicable (use `-ngl` for partial) | Not applicable |
| **Est. speed** | ~15-20 tok/s | ~25-35 tok/s |

**8GB variant:**

```bash
llama-server \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -ngl 24 -fa on -np 1 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -t 8 -b 2048 -ub 2048 \
  --chat-template gemma \
  --jinja --port 8080
```

**12GB variant:**

```bash
llama-server \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -ngl 40 -fa on -np 1 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -t 8 -b 4096 -ub 4096 \
  --chat-template gemma \
  --jinja --port 8080
```

**Sampling:**

```
--temp 0.7 --top-p 0.95 --top-k 64
```

Tune `-ngl` experimentally: start at 24 (8GB) or 40 (12GB), increase until you hit OOM, then back off by 2. Each layer on GPU instead of CPU meaningfully improves throughput.

**Same Blackwell cautions as E4B above.**

---

### Qwen 3.6 35B-A3B (MoE — requires CPU offloading on 8GB)

Download: `huggingface-cli download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF --include "*Q4_K_M*"`

#### Option matrix

| | RTX 5070 8GB | RTX 5070 12GB |
|---|---|---|
| **Quant** | Q4_K_M (~22 GB) | Q4_K_M (~22 GB) |
| **Context** | 8192-16384 | 16384-32768 |
| **KV cache quant** | Yes (mandatory) | Yes (mandatory) |
| **GPU layers** | 999 (all, using `--n-cpu-moe`) | 999 (all, using `--n-cpu-moe`) |
| **CPU MoE experts** | 30-40 | 20-30 |
| **Est. speed** | ~15-20 tok/s | ~20-30 tok/s |

The MoE offload strategy (`-ngl 999 --n-cpu-moe N`) keeps attention/KV cache on GPU while expert weights live in system RAM. This is faster than partial layer offload (`-ngl N`) for MoE models because the attention layers are the latency-sensitive part.

**8GB variant:**

```bash
llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 999 --n-cpu-moe 40 \
  -fa on -np 1 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -t 8 -b 2048 -ub 2048 \
  --jinja --port 8080
```

**12GB variant ([community-tested settings](https://github.com/vikivanov/llamacpp-qwen3.6-35b-windows-cuda)):**

```bash
llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 999 --n-cpu-moe 30 \
  -fa on -np 1 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -t 8 -b 4096 -ub 4096 \
  --jinja --port 8080
```

**Sampling (doc Q&A mode — thinking disabled):**

```
--temp 0.3 --top-p 0.9 --top-k 20 --presence-penalty 1.5 \
--chat-template-kwargs '{"enable_thinking": false}'
```

**Sampling (reasoning mode):**

```
--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 \
--reasoning-format deepseek
```

Tune `--n-cpu-moe`: lower values = more experts on GPU = faster but more VRAM. Start with 40 (8GB) or 30 (12GB) and decrease until OOM, then back off by 5.

---

## Build llama.cpp with CUDA

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

---

## Summary

### For technical documentation chat on 8GB VRAM

| Priority | Model | Why |
|---|---|---|
| **1st** | Granite 4.1 8B Q4_K_M | RAG-optimized, 512K context, fits with headroom, best tool calling |
| **2nd** | Phi-4 Mini Q8_0 | Fastest inference, 128K context, good quality floor |
| **3rd** | Qwen 3.5 9B Q4_K_M | Strongest raw quality at this size, multilingual, but only 32K context |

### If 8B-class quality is insufficient

| Priority | Model | Why |
|---|---|---|
| **1st** | Gemma 4 26B-A4B Q4_K_M | Smaller GGUF, less offloading pain, 256K context |
| **2nd** | Qwen 3.6 35B-A3B Q4_K_M | Stronger coding benchmarks, larger offload penalty |

### If you upgrade to 12GB+ VRAM

| Family | Best pick |
|---|---|
| **Granite** | 4.1 8B Q8_0 (near-lossless, still fits) |
| **Gemma 4** | 26B-A4B Q4_K_M (more layers on GPU) |
| **Qwen 3.6** | 27B dense Q4_K_M (better quality than MoE, viable with partial offload) |

Start with Granite 4.1 8B. If it handles your docs well, you're done. If not, try the larger MoE models and decide whether the quality gain justifies the latency cost.

## Sources

### Granite
- [Granite 4.1 IBM Research Announcement](https://research.ibm.com/blog/granite-4-1-ai-foundation-models)
- [Granite 4.1 HuggingFace Technical Blog](https://huggingface.co/blog/ibm-granite/granite-4-1)
- [Granite 4.1 8B on HuggingFace](https://huggingface.co/ibm-granite/granite-4.1-8b)
- [Granite 4.1 Complete Guide](https://www.aimadetools.com/blog/granite-4-1-complete-guide/)
- [Granite 4.1 vs Gemma 4](https://www.aimadetools.com/blog/granite-4-1-vs-gemma-4/)
- [Granite 4.1 vs Qwen 3.6-27B](https://www.aimadetools.com/blog/granite-4-1-vs-qwen-3-6-27b/)
- [IBM GGUF Repository](https://github.com/IBM/gguf)
- [IBM MTRAG Benchmark](https://research.ibm.com/blog/conversational-RAG-benchmark)

### Qwen
- [Unsloth Qwen 3.6 Docs](https://unsloth.ai/docs/models/qwen3.6)
- [Qwen Official llama.cpp Guide](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [RTX 5070 Ti Qwen 3.6 Benchmark Project](https://github.com/vikivanov/llamacpp-qwen3.6-35b-windows-cuda)
- [Qwen 3.6 on 24GB VRAM Benchmark](https://aminrj.com/posts/llamacpp-qwen36-35b/)
- [Qwen 3.6 CPU Offload on 12GB](https://www.xda-developers.com/i-replaced-chatgpt-and-claude-with-this-local-llm/)
- [Qwen 3.6 27B vs 35B-A3B](https://benchlm.ai/compare/qwen3-6-27b-vs-qwen3-6-35b-a3b)
- [Qwen 3.6 27B Announcement](https://qwen.ai/blog?id=qwen3.6-27b)

### Gemma
- [Unsloth Gemma 4 Docs](https://unsloth.ai/docs/models/gemma-4)
- [Google AI - Gemma + llama.cpp](https://ai.google.dev/gemma/docs/integrations/llamacpp)
- [HuggingFace Gemma 4 Blog](https://huggingface.co/blog/gemma4)
- [Gemma 4 Model Comparison](https://avenchat.com/blog/gemma-4-31b-vs-26b-vs-e4b)
- [Gemma 4 Blackwell CUDA Issue](https://github.com/ggml-org/llama.cpp/issues/21371)

### Phi-4
- [Phi-4 Mini Benchmarks & Guide](https://localaimaster.com/models/phi-4-mini)
- [Phi-4 Mini RAG Implementation](https://www.marktechpost.com/2026/04/20/a-coding-implementation-on-microsofts-phi-4-mini-for-quantized-inference-reasoning-tool-use-rag-and-lora-fine-tuning/)

### Hardware & General
- [Framework RTX 5070 12GB](https://frame.work/blog/framework-laptop-16-now-with-rtx-5070-12gb-and-launch-event-re-cap)
- [Best Local LLMs for 8GB VRAM (2026)](https://localllm.in/blog/best-local-llms-8gb-vram-2025)
- [Best Local LLM Models 2026](https://www.sitepoint.com/best-local-llm-models-2026/)
