# Experimental Findings & The "Lazy Graph" Problem

This document outlines the engineering journey, the challenges encountered with MLX's lazy evaluation, and the exact steps taken to achieve true zero-copy weight streaming for models larger than physical RAM.

## 1. The Problem: MLX's Lazy Graph and Metal OOM

Apple's MLX framework is fundamentally **lazy**. When you call `model(inputs)`, it doesn't immediately compute the matrix multiplications. Instead, it builds a "computation graph" representing the operations for all layers (e.g., all 52 layers of a 30B model).

MLX only allocates Metal (GPU) memory when `mx.eval()` is finally called at the end of the generation step. If the model is 18 GB and you have 16 GB of RAM, MLX attempts to allocate all 18 GB at once to evaluate the graph, resulting in:

    [METAL] Command buffer execution failed: Insufficient Memory

This is **not fixable** by patching `mlx_lm.load()` alone — the generation loop still builds a unified graph that requires evaluating the entire model state simultaneously.

## 2. What Didn't Work

### Approach A: `pread()` + `numpy.copy()`

The project initially used `os.pread()` combined with a thread pool to read weights from disk, followed by `.copy()` to convert them into NumPy/MLX arrays.

**Problem:** Calling `.copy()` pulls data from the macOS Page Cache into the Python process's private memory (RSS). The OS cannot safely reclaim this memory until the Python garbage collector frees it, leading to inflated RAM usage and eventually OOM.

### Approach B: `memoryview` zero-copy from custom mmap

We migrated to `mmap` and `memoryview`, allowing MLX to map the safetensors directly from the SSD into the GPU's unified address space (zero-copy). When we're done with a layer, calling `madvise(MADV_FREE)` allows macOS to instantly reclaim the physical pages.

**Problem:** While loading works perfectly, the standard `mlx_lm.stream_generate()` generation loop still builds a full lazy graph across all layers. OOM on the first token.

### Approach C: Patching `mlx_lm.load()` only

Simply intercepting `mlx_lm.load()` to load weights lazily isn't enough for models exceeding physical RAM. The model loads fine, but the generation loop builds the same unified graph → OOM.

## 3. What Works: FlashLLM with Per-Layer Synchronous Evaluation

The key insight: intercept at the **forward pass** level, not at load time.

The solution is `FlashLLM`, a duck-typed `nn.Module` proxy that wraps any mlx-lm model and replaces the unified lazy graph with sequential per-layer evaluation:

1. **Lazy Loading**: `mlx_lm.load(path, lazy=True)` maps the entire model into the unified address space using the macOS page cache. No Metal RAM is consumed at this point.

2. **Perfect Proxy**: We wrap the model in a `FlashLLM` proxy that behaves exactly like the original (same mask protocol, same cache management), but intercepts the layer loop.

3. **Synchronous Execution**: Instead of building a unified lazy graph for the whole model, we build and evaluate a graph for exactly **one layer**:
    ```
    for each layer i in [0, N):
        h = layer_i(h, mask=mask, cache=cache[i])
        mx.eval(h, cache[i].keys, cache[i].values)
        mx.synchronize()
        mx.metal.clear_cache()
    ```

4. **Why `__call__` wrapping works here (but not in Approach B):**
   - We do NOT patch `model.__call__` on the original model object (that breaks KV cache)
   - We wrap the model in `FlashLLM` which presents the same `nn.Module` interface to `mlx_lm`
   - `mlx_lm.stream_generate()` calls `FlashLLM(x)` which IS our synchronous loop
   - KV cache is eval'd alongside `h` at each layer, so it never builds a lazy graph either

**Result:** An 18 GB model runs on a 16 GB Mac with a peak active footprint of `< 1.0 GB`.

## 4. Remaining Hard Problems

* **Prefill OOM on very long prompts:** Processing the initial prompt creates large intermediate activation matrices. Chunked prefill mitigates but doesn't fully solve this for 32k+ token contexts.
* **KV cache growth:** For long conversations, the KV cache still grows in RAM. `RotatingKVCache` with `max_kv_size` and `kv_bits` quantisation mitigate this.
* **I/O-bottlenecked throughput:** Token generation speed is limited by SSD read bandwidth. Async prefetch (v0.2.0) will overlap I/O with GPU compute to hide latency.
* **Architecture-specific `__call__` signatures:** Some models don't accept `mask=` or have non-standard layer structures. `FlashLLM` handles this via pre-computed layer signature introspection, but novel architectures may require updates.

### Future Directions

1. **Upstream MLX Support:** The ideal solution would be for `mlx` to natively support a "streaming module" primitive that automatically yields memory between layer evaluations.
2. **KV Cache Offloading:** Apply the mmap streaming strategy to the KV cache itself for very long conversations.
3. **Dynamic RAM Budgeting:** Read `psutil.virtual_memory().available` at runtime and adjust the prefetch window and eviction strategy on the fly.
