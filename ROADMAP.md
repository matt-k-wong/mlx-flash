# mlx-flash Roadmap ⚡

This document outlines the planned milestones for `mlx-flash` as it moves from beta to a stable, production-grade utility.

## v0.1.x: Stability & Polished Beta
*Focus: Fixing the "paper cuts" and ensuring 100% correctness.*

- [ ] **PyPI Release**: Official package distribution for easier installation.
- [ ] **CI Benchmarking**: Automatic performance regression tracking in GitHub Actions.
- [ ] **Sampler Parity**: Ensure 100% numerical parity with standard `mlx-lm` for all sampling parameters.
- [ ] **Improved Diagnostics**: More granular RAM profiling in `flash-monitor`.

## v0.2.0: Background I/O & Disk KV
*Focus: Eliminating I/O latency and handling massive contexts.*

- [x] **Background I/O Thread**: Initial implementation complete.
- [x] **Stable Disk KV Cache**: Production-ready offloading in v0.3.2.
- [x] **Adaptive Budgeting**: Dynamic `ram_budget_gb` logic implemented in v0.3.5.

## v0.3.5: True Weight Streaming & "Any Model" Support
*Focus: Running models larger than RAM with deterministic memory footprints.*

- [x] **True Weight Streaming**: Force materialization and eviction of layer weights.
- [x] **Token-Local Caching**: Optimized safetensors loading for 50x speedup.
- [x] **30B+ Verification**: Success on 16GB hardware with <0.5GB Metal RAM.

## v0.4.0: High-Performance MoE & Expert Streaming
*Focus: Making massive MoE models (Mixtral, DeepSeek) run at 10+ tok/s.*

- [ ] **Expert Prefetching**: Predictively load the next top-K experts while the current experts are executing.
- [ ] **Layer Skipping Support**: Improved logic for architectures that support dynamic layer execution.
- [ ] **Multi-GPU / Multi-Node (Experimental)**: Exploring streaming across multiple unified memory pools.

---

## v1.0.0: Native Integration
*Focus: Moving from a monkey-patch to a standard feature.*

- [ ] **Upstream PR to `mlx-lm`**: Propose native `FlashLLM` support to eliminate the need for monkey-patching.
- [ ] **Official LM Studio Integration**: Seamless "Flash" checkbox in the LM Studio UI.
- [ ] **Documentation**: Comprehensive API reference and integration guides for other frameworks.
