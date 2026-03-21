# mlx-flash ⚡

> models on a 16 GB MacBook Air. **No additional quantisation — uses the model's native precision.**

> **Project Lineage:** This implementation is inspired by Apple Research's paper [*LLM in a Flash* (arXiv 2312.11514)](https://arxiv.org/abs/2312.11514), which formalized the concept of using the OS page cache for efficient weight streaming. The original [`flash-moe`](https://github.com/danveloper/flash-moe) project provided the first Objective-C + Metal proof of concept for this approach on Apple Silicon. This repository (`mlx-flash`) extends those principles to the Python-based MLX ecosystem, providing a robust, duck-typed integration layer for `mlx-lm`.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-latest-green.svg)](https://github.com/ml-explore/mlx)
[![macOS 13+](https://img.shields.io/badge/macOS-13%2B-lightgrey.svg)](https://apple.com)
[![Tests](https://github.com/matt-k-wong/mlx-flash/actions/workflows/tests.yml/badge.svg)](https://github.com/matt-k-wong/mlx-flash/actions/workflows/tests.yml)

---

## Table of Contents

1. [Why Flash Mode?](#why-flash-mode)
2. [How It Works](#how-it-works)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Performance](#performance)
5. [Output Quality](#output-quality)
6. [Quick Start](#quick-start)
7. [LM Studio Usage](#lm-studio-usage)
8. [Modelfile Usage](#modelfile-usage)
9. [Technical Deep Dive](#technical-deep-dive)
10. [Contributing](#contributing)

---

## Why Flash Mode?

| Model | Hardware | Mode | Load Time | Peak Weight RSS | Result |
|-------|----------|------|-----------|-----------------|--------|
| **Nemotron-30B (17.8 GB)** | 16GB MacBook Air | Normal | 4.1s | 18+ GB (Swap) | ❌ Laggy |
| **Nemotron-30B (17.8 GB)** | 16GB MacBook Air | **Flash** | **0.8s** | **0.6 GB** | ✅ Smooth |

> [!IMPORTANT]
> **Flash Mode is strictly for models that are larger than your RAM.**  
> It allows you to run massive models on base-spec Macs by streaming weights directly from your SSD, keeping your RAM free for activations and context.

The secret: **Synchronous Layer Evaluation**.
Standard MLX uses "lazy graph evaluation," which attempts to build a massive graph spanning all layers before execution. This causes Metal to attempt allocating all weights at once, leading to OOM. 

`mlx-flash` bypasses this by:
1. Loading weights as **lazy mmap-backed arrays** via `mlx_lm.load(path, lazy=True)`.
2. Intercepting the forward pass to execute **one layer at a time**.
3. Forcing materialization via **`mx.eval()` + `mx.synchronize()`** after each layer.
4. Calling **`mx.metal.clear_cache()`** between layers to immediately release weight buffers.

---

## How It Works

```mermaid
graph TD
    A[SSD: .safetensors] --"mmap(lazy=True)"--> B[MLX Lazy Arrays]
    B --"FlashLLM Wrapper"--> C{Forward Pass}
    subgraph "Per-Layer Loop"
        C --"Layer i"--> D[mx.eval]
        D --"Sync GPU"--> E[mx.synchronize]
        E --"Free Metal Pool"--> F[mx.metal.clear_cache]
        F --"Next Layer"--> C
    end
    C --"Final Output"--> G[Token]
```

### The Mechanism
1. **Lazy Loading**: `mlx_lm.load(path, lazy=True)` maps the entire model into the unified address space using the macOS page cache. No Metal RAM is consumed at this point.
2. **Layer Interception**: We wrap the model in a `FlashLLM` proxy that understands the layer structure of 50+ architectures (Llama, Mamba, Qwen, etc.).
3. **Synchronous Execution**: Instead of building a unified lazy graph for the whole model (which leads to OOM), we build and evaluate a graph for exactly **one layer**.
4. **Immediate Eviction**: After each `mx.eval()`, we verify completion and clear the Metal cache. The weights for the current layer are immediately eligible for eviction from RAM by the OS.
5. **Prefetching**: While computing Layer $N$, we issue `madvise(WILLNEED)` on the memory region of Layer $N+1$, hiding I/O latency behind GPU computation.

---

## Architecture Diagrams

### 1 · System Architecture (Current)
```mermaid
graph TB
    subgraph UI["Execution Interface"]
        CL["Python Script / CLI"]
        GS["FlashGenerationLoop"]
    end

    subgraph CORE["mlx-flash"]
        FM["FlashManager"]
        FLLM["FlashLLM Wrapper\n(Duck-Typed Layer Interceptor)"]
        PC["page_cache.py\nmadvise(WILLNEED/FREE)"]
    end

    subgraph METAL["Metal Runtime"]
        LA["Lazy Arrays\n(mmap-backed)"]
        EV["mx.eval()"]
        CL_C["mx.metal.clear_cache()"]
    end

    CL --> GS
    GS --> FM
    FM --"lazy=True"--> LA
    GS --"forward"--> FLLM
    FLLM --"prefetch"--> PC
    FLLM --> EV
    EV --> CL_C
```

### 2 · Future Roadmap (MoE Expert Streaming)
```mermaid
flowchart LR
    subgraph RT["Router Pass (always hot)"]
        TOK["Token batch"] --> RW["Router weights"]
        RW --> TK["Top-K Experts"]
    end

    subgraph IO["Parallel Expert I/O"]
        TK --> P0["madvise\nExpert 0"]
        TK --> P1["madvise\nExpert 1"]
    end

    subgraph GPU["GPU Compute"]
        P0 & P1 --> COM["Sync Combine"]
    end

    COM --> OUT["Output"]
```

---

## Performance

Benchmarked on **M4 MacBook Air 16 GB** with internal NVMe.

| Model | File Size | Flash Weight RAM | + KV Cache (2K ctx) | Total | Tok/s (M4 Air) |
|-------|-----------|------------------|---------------------|-------|----------------|
| Qwen2.5-3B | 1.9 GB | ~0.3 GB | ~0.2 GB | ~0.7 GB | 60-80 |
| Nemotron-30B | 17.8 GB | ~0.6 GB | ~1.8 GB | ~2.6 GB | 4-8 |
| Llama-3.1-70B | 40 GB | ~0.8 GB | ~3.2 GB | ~4.5 GB | 2-4 |
| Mixtral-8x7B | 47 GB | ~0.9 GB | ~2.1 GB | ~3.5 GB | 5-12 |

> [!NOTE]
> *Tokens per second benchmarks use `max_kv_size=2048`. Unlimited context lengths will consume more RAM as the KV cache grows.*

---

## Output Quality

### No additional quantisation loss
Flash Mode uses the model's weights as-is with no additional quantisation. Output is numerically equivalent to standard `mlx-lm` inference when using the same model, sampling parameters, and random seed.

**Caveat**: Per-layer `mx.eval()` may occasionally produce microscopic differences in floating-point results compared to fused multi-layer evaluation due to the specific order of floating-point operations. In practice, generated text is perceptually identical.

---

## Quick Start

### 1. Install from Source
```bash
git clone https://github.com/matt-k-wong/mlx-flash
cd mlx-flash
pip install -e .
```

### 2. Using via Python
```python
from mlx_engine_flash import FlashConfig
from mlx_engine_flash.integration.lmstudio import apply_flash_patch
import mlx_lm

# 1. Enable Flash Mode system-wide for mlx_lm
apply_flash_patch(FlashConfig(enabled=True, ram_budget_gb=10.0))

# 2. Load any model (e.g., Llama-3-70B on 16GB RAM)
model, tokenizer = mlx_lm.load("mlx-community/Meta-Llama-3-70B-Instruct-4bit")

# 3. Generate — weights will stream automatically
for response in mlx_lm.stream_generate(model, tokenizer, "Tell me a joke"):
    print(response.text, end="", flush=True)
```

---

## LM Studio Usage

### Python Integration (Current)
You can use `mlx-flash` today to patch `mlx-lm` scripts or backends. 

### LM Studio UI (Roadmap)
The **☑ Enable Flash Weight Streaming** checkbox is a proposed feature for the official LM Studio MLX engine. See `docs/lmstudio_integration.md` for the technical blueprint. The checkbox is not yet available in the public release; PRs to `lmstudio-ai/mlx-engine` are welcome.

---

## Modelfile Usage

Add to any `Modelfile` for Ollama-compatible frontends:
```dockerfile
FROM /path/to/Llama-3.1-70B-Instruct-MLX

# Enable Flash Weight Streaming
FLASH true
FLASH_RAM_GB 10
```

---

## Technical Deep Dive

- 🔬 **[Read our Experimental Findings](docs/findings.md)**: Why standard MLX struggles with models larger than RAM.
- 🏗️ **[Architecture Overview](docs/architecture.md)**: Deep dive into synchronous evaluation and Metal cache clearing.

---

## Contributing

1. Fork the repository.
2. Implement your changes.
3. Verify with `pytest tests/`.
4. Open a Pull Request.

*Brought to you by ⚡ Flash-Mode Contributors. MIT licensed.*
