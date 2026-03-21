# mlx-flash ⚡

> **Flash Weight Streaming for MLX** — run 70 B, 120 B, and 397 B MoE
> models on a 16 GB MacBook Air. **No additional quantisation — uses the model's native precision.**

> **Project Lineage:** The approach to streaming LLM weights directly from disk without loading them into RAM was heavily inspired by early conceptual work and research by **Andrej Karpathy** (such as his minimalist C implementations like `llama2.c`). Subsequently, Apple Research formalized this in *LLM in a Flash*, and the original [`flash-moe`](https://github.com/danveloper/flash-moe) project successfully proved that streaming weights from NVMe could allow massive models to run on Apple Silicon. **The goal of this specific repository (`mlx-flash`) is to take that proven concept and provide a clean, drop-in integration for the MLX ecosystem**, turning a terminal demo into a production-ready library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-latest-green.svg)](https://github.com/ml-explore/mlx)
[![macOS 13+](https://img.shields.io/badge/macOS-13%2B-lightgrey.svg)](https://apple.com)

---

## Table of Contents

1. [Why Flash Mode?](#why-flash-mode)
2. [How It Works](#how-it-works)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Performance](#performance)
5. [Quick Start](#quick-start)
6. [LM Studio Usage](#lm-studio-usage)
7. [Modelfile Usage](#modelfile-usage)
8. [Testing Guide (start with 4B)](#testing-guide)
9. [Build & Install](#build--install)
10. [External SSD / Thunderbolt RAID](#external-ssd--thunderbolt-raid)
11. [Technical Deep Dive](#technical-deep-dive)
12. [Contributing & PR to upstream](#contributing--pr-to-upstream)

---

## Why Flash Mode?

| Model | Hardware | Mode | Load Time | Peak RAM (RSS) | Result |
|-------|----------|------|-----------|----------------|--------|
| **Nemotron-30B (17.8 GB)** | 16GB MacBook Air | Normal | 4.1s | 18+ GB (Swap) | ❌ Laggy |
| **Nemotron-30B (17.8 GB)** | 16GB MacBook Air | **Flash** | **0.8s** | **0.6 GB** | ✅ Smooth |

> [!IMPORTANT]
> **Ignore the "Likely too large" warning in LM Studio.**  
> Flash Mode is specifically designed to run models that LM Studio labels as too large for your Mac's RAM. When "Enable Flash Weight Streaming" is checked, the model will stream from your SSD, allowing massive models to run on 16GB Macs without relying on slow OS swap.

The secret: **Synchronous Layer Evaluation**.
Standard MLX uses "lazy graph evaluation," which attempts to allocate all layer weights in memory at once, leading to OOM. `mlx-flash` bypasses this by forcing MLX to evaluate each layer synchronously (`mx.eval`), immediately clearing the Metal cache and releasing memory-mapped pages (`madvise(MADV_FREE)`) before moving to the next layer. 

This follows the approach pioneered in:
- Apple Research: [*LLM in a Flash* (arXiv 2312.11514)](https://arxiv.org/abs/2312.11514)
- [`flash-moe` by @danveloper](https://github.com/danveloper/flash-moe) (the first working OSS demo)

---

## Current Status: Tested Proof of Concept

This project provides a robust, zero-copy `mmap` engine that successfully bypasses Apple's Metal memory limits by forcing synchronous, layer-by-layer evaluation. It allows you to run models significantly larger than your physical RAM.

However, because this is implemented as a Python "monkey-patch" on top of the existing `mlx_lm` and `mlx-engine` ecosystems, there are current limitations:

*   **Long Context Windows:** While standard generation works flawlessly, pasting massive prompts (e.g., 5,000+ tokens) into a UI like LM Studio may still cause an `Out of Memory` crash. This is because the host engine's Prompt Processing (Prefill) phase attempts to evaluate the massive KV cache graph all at once, before our layer-by-layer patch can yield the memory back to the GPU.
*   **The Future:** For 100% stability with infinite context lengths, this synchronous layer evaluation needs to be adopted directly within the C++ / core generation loops of inference engines, rather than patched via Python. This repository serves as the proven blueprint for that integration.

---

## How It Works

```
Disk (safetensors)
    ↓  mmap() [Zero-copy mapping]
macOS Unified Page Cache  ← madvise(WILLNEED) prefetch
    ↓  zero-copy DMA
Metal GPU (unified address space)
    ↓  Synchronous Layer eval (mx.eval)
    ↓  Metal Cache Clear + madvise(MADV_FREE)
    ↓
Output tokens
```

Key principles:
1. **Trust the OS page cache** — macOS LRU eviction is faster than any
   custom cache you can write.
2. **Synchronous Execution** — Bypass MLX's lazy graph by evaluating one 
   layer at a time. This keeps Peak RSS strictly bounded by the size of the 
   largest single layer.
3. **madvise(WILLNEED)** — prefetch the *next* layer while computing the
   *current* layer, hiding I/O latency.
4. **madvise(MADV_FREE)** — On macOS, forcefully hint to the kernel that 
   pages for "cold" layers are immediately available for reuse.
5. **No Extra Quantisation** — Weights are streamed in their native format 
   (e.g., 4-bit or BF16) directly to the GPU.

---

## Architecture Diagrams

### 1 · Current Implementation (Synchronous Flow)
```mermaid
graph TB
    subgraph UI["Frontends"]
        CL["CLI / prove_flash_manual.py"]
        GS["FlashGenerationLoop"]
    end

    subgraph EXT["mlx-flash"]
        FM["FlashManager"]
        WS["WeightStreamer\nmmap-based"]
        PF["WeightPrefetcher\nbackground thread"]
        PC["macOS Unified Page Cache\nshared CPU+GPU pages"]
    end

    subgraph FLOW["Sync Pipeline"]
        EV["mx.eval(layer_N)"]
        CC["Metal Cache Clear"]
        MF["madvise(MADV_FREE)"]
    end

    CL --> GS
    GS --> FM
    FM --> WS
    WS <--> PC
    PF -.->|prefetch hint| WS
    PC --> EV
    EV --> CC
    CC --> MF
```

### 2 · Future Roadmap (Parallel Aspirational Pipeline)
```mermaid
sequenceDiagram
    participant UI  as LM Studio / CLI
    participant FM  as FlashManager
    participant PF  as Prefetcher (bg)
    participant WS  as WeightStreamer
    participant PC  as Page Cache
    participant GPU as Metal GPU

    Note over UI,GPU: This parallel flow requires deep integration into MLX C++ core

    loop Forward pass – each transformer layer N
        FM  ->> PF  : prefetch_layer(N+1)
        PF  -->> PC : madvise(WILLNEED, layer_N+1_pages)
        FM  ->> WS  : get_layer_weights(N)
        WS  ->> PC  : Parallel pread() ×4 threads
        PC  -->> GPU: zero-copy page mapping
        GPU ->> GPU : dequant_4bit (FMA kernel)
        GPU ->> GPU : attention + FFN
        GPU -->> FM  : activations[N]
    end
```

### 3 · MoE Expert Streaming Flow (Roadmap)
```mermaid
flowchart LR
    subgraph RT["Router Pass (dense, fast)"]
        TOK["Token batch\n[B × D]"] --> RW["Router weights\n(always hot)"]
        RW --> LGT["Logits\n[B × E]"]
        LGT --> TK["Top-K select\nK=2 … 8"]
    end

    subgraph IO["Parallel Expert Load"]
        TK --> P0["pread\nExpert idx[0]"]
        TK --> P1["pread\nExpert idx[1]"]
    end

    subgraph CMP["GPU Compute (pipelined)"]
        P0 --> D0["dequant\nFMA"]
        P1 --> D1["dequant\nFMA"]
        D0 & D1 --> COM["Combine\n+ route weights\n(softmax)"]
    end

    COM --> OUT["Output\n[B × D]"]
```

---

## Current Limitations

- **Lazy Graph OOM:** Standard `mlx_lm.stream_generate()` integration still causes OOM for models significantly larger than RAM because it attempts to build the full execution graph.
- **True Flash Mode:** Currently requires using the `FlashGenerationLoop` (synchronous path) available in `scripts/prove_flash_manual.py`.
- **Long Context:** Context lengths beyond 5,000 tokens may OOM during the prefill phase as the KV cache is not yet synchronously managed.
- **MoE Experts:** MoE expert streaming is not yet implemented in the `FlashModelLoader`.

## Performance

Benchmarked on **M4 MacBook Air 16 GB** with internal NVMe.

### Verified Models

| Model | Architecture | File Size | Normal Peak RAM | Flash Peak RAM | Flash Load Time |
|-------|--------------|-----------|-----------------|----------------|-----------------|
| Nemotron-30B | Hybrid | 17.8 GB | 18+ GB (OOM/Swap) | **0.6 GB¹** | **0.8s** |

¹ *Achieved via synchronous layer evaluation (`scripts/prove_flash_manual.py`). The standard `mlx_lm.stream_generate()` integration is not yet able to achieve this figure — see `docs/findings.md`.*

> **Thunderbolt SSD (2–6 GB/s):** multiply tok/s by ~0.8×  
> **Thunderbolt RAID (8–20 GB/s):** multiply tok/s by ~1.5–2.5×  
> **M4 Pro / Max / Ultra (larger NVMe bandwidth):** multiply tok/s by ~1.2–2×

### RAM Headroom by Mac

With Flash Mode's synchronous layer execution, the primary constraint is no longer total physical RAM, but rather the size of the single largest layer and the speed of your NVMe drive. 

| Mac | Usable RAM | Max Model (Flash) |
|-----|-----------|-----------------|
| 16 GB | ~14 GB | Conceptually unbounded (storage limited) |
| 18 GB | ~16 GB | Conceptually unbounded (storage limited) |
| 24 GB | ~22 GB | Conceptually unbounded (storage limited) |
| 192 GB| all RAM | Can likely load models fully into RAM (no Flash needed) |

---

## Quick Start
```bash
# 1. Install
pip install mlx-flash   # or: pip install -e .

# 2. Test on a 4B model (fast iteration)
python examples/quick_start.py --model ~/.cache/lm-studio/models/Qwen/Qwen2.5-3B-Instruct-MLX

# 3. Run benchmarks
python benchmarks/bench_flash.py --model /path/to/model --mode both
```

---

## LM Studio Usage

> [!WARNING]
> **This project currently provides the backend library.**  
> LM Studio UI integration requires a pending PR to `lmstudio-ai/mlx-engine` (see `docs/lmstudio_integration.md`). For now, please use `scripts/prove_flash_manual.py` directly to experience the synchronous streaming path.

1. Once the integration is merged, open **LM Studio** → **Preferences** → **Inference**.
2. Check **☑ Enable Flash Weight Streaming (low-RAM mode)**.
3. Load any model that would otherwise OOM — it will start streaming.

---

## Modelfile Usage

Add to any `Modelfile` when using Ollama-compatible frontends or the
mlx-engine CLI directly:
```
# Modelfile
FROM /path/to/Qwen2.5-72B-Instruct-MLX

# Enable Flash Weight Streaming
FLASH true

# Optional tuning (defaults shown)
FLASH_RAM_GB 10
FLASH_THREADS 4
FLASH_PREFETCH_LAYERS 2
FLASH_QUANT_WARN_BELOW 4
```

Parse with `mlx_engine_flash.integration.modelfile.parse_flash_directives()`.

---

## Testing Guide

### Start small: 4B models
```bash
# Download a tiny test model first
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen2.5-3B-Instruct-4bit',
                  local_dir='./test_models/Qwen2.5-3B')
"

# Run the full test suite
pytest tests/ -v

# Run specifically the streaming tests
pytest tests/test_streamer.py tests/test_moe.py -v

# Integration test with real inference
pytest tests/test_integration.py -v \
    --model ./test_models/Qwen2.5-3B \
    --flash

# Quick smoke-test script
scripts/test_quick.sh ./test_models/Qwen2.5-3B
```

### Scaling to larger models

Once 4B passes, the only difference with 70B/MoE is file size; all code paths
are identical. Validate with:
```bash
# 70B sanity check (just model load + 1 token)
python examples/quick_start.py \
    --model /path/to/Llama-3.1-70B-Instruct-4bit \
    --flash --max-tokens 1 --benchmark
```

---

## Build & Install

### Python only (recommended)
```bash
python -m pip install -e ".[dev]"
```

### With Metal kernels (optional, AOT compiled for speed)
```bash
# Requires Xcode Command Line Tools
xcode-select --install

# Compile Metal kernels to .metallib
python mlx_engine_flash/kernels/compile_kernels.py

# Install with Metal support
pip install -e ".[dev,metal]"
```

### Requirements

- macOS 13.0 + (Ventura or later)
- Apple Silicon (M1 / M2 / M3 / M4 / M5 series)
- Python 3.11+
- MLX ≥ 0.20
- mlx-lm ≥ 0.20
- `mlx-engine` (lmstudio-ai/mlx-engine) installed alongside

### Compatibility Matrix

| Library | Version | Note |
|---------|---------|------|
| **mlx-lm** | ≥ 0.20.0 | Required for `lazy` evaluation and model registry support. |
| **mlx** | ≥ 0.20.0 | Required for stable `mmap` support and Metal performance. |
| **macOS** | ≥ 14.0 | Recommended for `MADV_FREE` support (aggressive RAM recovery). |

### Uninstallation

To cleanly remove `mlx-flash` and its compiled artifacts:
```bash
./scripts/uninstall.sh
```

---

## External SSD / Thunderbolt RAID

Flash Mode is **storage-agnostic** — it uses standard POSIX `pread()` and
lets the OS manage I/O scheduling. Performance scales linearly with storage
bandwidth:

| Storage | Sequential Read | Expected speedup |
|---------|----------------|-----------------|
| Internal NVMe (M4) | 3.5–7 GB/s | baseline |
| Samsung T9 / SanDisk Extreme | 2–3.5 GB/s | 0.5–0.9× |
| OWC Envoy Pro FX (TB3) | 2.8–5 GB/s | 0.8–1.1× |
| OWC ThunderBay 4 RAID-0 | 10–20 GB/s | 2–4× |
| Sabrent Rocket Nano TB4 | 3–4 GB/s | 0.9–1.1× |

Tips:
- Store model files on the fastest available drive.
- macOS Spotlight indexing on the model directory will cause stalls —
  add your model path to the **Privacy** exclusion list in System Settings.
- `sudo mdutil -i off /Volumes/MyModelDrive` to disable Spotlight on a volume.
- If using APFS encryption, expect ~5–10% throughput reduction.

---

## Technical Deep Dive

- 🔬 **[Read our Experimental Findings & The "Lazy Graph" Problem](docs/findings.md)**: A transparent look at why standard MLX struggles with models larger than RAM and how we overcame it using zero-copy mmaps and synchronous layer execution.
- [`docs/architecture.md`](docs/architecture.md): A full walkthrough of the original Safetensors header parsing, Metal kernel design, and MoE routing logic.

---

## Contributing & PR to upstream

This project is designed as a clean PR target for
[lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine).

1. Fork this repo.
2. Implement your change with tests.
3. Run `ruff check . && mypy mlx_engine_flash && pytest`.
4. Open a PR here first; once stable, open upstream.

The extension seam is in `mlx_engine_flash/integration/lmstudio.py` —
a single `apply_flash_patch()` call that monkey-patches the relevant
`mlx_lm.load` path.

---

## Acknowledgements

- **danveloper** — [`flash-moe`](https://github.com/danveloper/flash-moe) (the original breakthrough demo)
- **Apple ML Research** — [*LLM in a Flash* (arXiv 2312.11514)](https://arxiv.org/abs/2312.11514)
- **lmstudio-ai** — [`mlx-engine`](https://github.com/lmstudio-ai/mlx-engine) (the target integration platform)
- **ml-explore** — [MLX framework](https://github.com/ml-explore/mlx)

---

*Brought to you by ⚡ Flash-Mode Contributors. MIT licensed.*
