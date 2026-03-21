# Changelog

All notable changes to this project will be documented in this file.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1]

### Added
- **Spotlight Auto-Exclusion**: Drops a `.metadata_never_index` file into model directories to prevent macOS Spotlight from aggressively scanning 100GB+ files and crippling SSD throughput.
- **Battery & P-Core Warnings**: Added automatic diagnostics and warnings when running enormous IO workloads on battery power (`pmset -g batt`), alerting users to thermal limits and Battery Drain.
- **macOS Unified Page Cache Integration**: Full, mathematical integration of OS-level `madvise()` calls for `.safetensors`. Explicit `MADV_WILLNEED` and `MADV_FREE` hints now run smoothly in-pipeline, keeping Metal RAM bounds strict while maximizing SSD throughput.

### Changed
- **Pipelined GPU Synchronization**: Refactored the proxy wrapper's synchronization (`mx.synchronize()`). CPU and GPU are no longer artificially serialized on every layer boundary. They now pipeline naturally across `pipeline_depth=2` layers for radically improved performance.
- Simplified README installation instructions specifically for PyPI release.

## [0.1.0]

### Added
- Initial Flash Weight Streaming implementation
- Parallel `pread()` weight streamer with configurable thread pool
- macOS unified page cache management (`madvise` WILLNEED / DONTNEED)
- MoE top-K expert streaming (Mixtral, DeepSeek, Qwen2-MoE)
- FMA-optimised Metal kernels: `flash_dequant_4bit`, `swiglu_fused`, `moe_dispatch`
- LM Studio extension hook + Modelfile `FLASH true` directive
- Background prefetch thread for I/O/compute overlap
- `FlashConfig` with RAM budget, thread count, and quantisation level controls
- Comprehensive test suite (unit + integration) targeting 4B models
- Benchmark suite comparing Flash vs Normal loading
- Full README with Mermaid architecture diagrams and performance tables
