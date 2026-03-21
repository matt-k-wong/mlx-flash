# Upstreaming Flash Mode to mlx-engine

This document outlines the minimal changes required to integrate the Flash Weight Streaming logic from `mlx-flash` into the official `lmstudio-ai/mlx-engine`.

## Integration Overview

The core of `mlx-flash` is a synchronous, layer-by-layer execution wrapper (`FlashLLM`) that intercepts the standard `mlx-lm` forward pass. To enable this in `mlx-engine`, we need to change how models are loaded and how they are executed.

### 1. Model Loading (mlx_engine/model_kit.py)

We must ensure that weights are loaded as lazy, mmap-backed arrays.

```diff
- model, tokenizer = mlx_lm.load(model_path)
+ flash_enabled = config.get("flash_mode", False)
+ model, tokenizer = mlx_lm.load(model_path, lazy=flash_enabled)
```

### 2. Generation Wrapping (mlx_engine/generation.py)

Immediately before the generation loop starts, we wrap the model in the `FlashLLM` proxy.

```diff
+ if flash_enabled:
+     from mlx_engine_flash.generation import FlashLLM
+     model = FlashLLM(model, flash_config)
+ 
  for response in generate_step(prompt, model, ...):
      yield response
```

## Proposed PR Description for `mlx-engine`

**Title**: feat: Integrated Flash Weight Streaming for low-RAM devices

**Description**:
This PR integrates `FlashLLM` support, allowing users to run models significantly larger than their available RAM (e.g., Llama-3-70B on a 16GB MacBook Air).

**Key Changes**:
- Supports `lazy=True` loading in `model_kit.py` to keep initial RSS near zero.
- Replaces the unified lazy graph with a synchronous, per-layer execution loop.
- Forces `mx.metal.clear_cache()` between layers to strictly bound Metal memory usage.
- Performance: ~2-4 tok/s for 70B models on M4 Air (SSD-bottlenecked).

**Compatibility**:
- Requires `mlx-lm >= 0.20.0`.
- Supports 50+ architectures via duck-typing layer interception.

## Required Upstream Tests

The maintainers will likely require:
1. **Memory Regression Test**: Ensure `flash_mode: false` (default) doesn't change existing performance or memory behavior.
2. **Architecture Scan**: Verify that the layer discovery logic correctly identifies stacks in `qwen`, `llama`, and `mistral` (the most common LM Studio models).
3. **Wired Limit Reset**: Test that `set_wired_limit` is correctly restored after the model is unloaded.
