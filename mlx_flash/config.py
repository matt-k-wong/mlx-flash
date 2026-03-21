"""
FlashConfig — all tuneable parameters for Flash Weight Streaming.
"""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FlashConfig:
    """
    Configuration for Flash Weight Streaming.

    Parameters
    ----------
    enabled:
        Master switch.  When False every other field is ignored and the
        normal mlx-lm load path is used unchanged.
    ram_budget_gb:
        Soft cap on resident weight RAM (in GB).  The OS page cache naturally
        enforces this via LRU eviction. Default 2.0 GB suits 8/16-GB Macs well.
    eviction_strategy:
        "dontneed" — MADV_DONTNEED: tell OS pages are unneeded (advisory).
        "free"     — MADV_FREE: allow OS to reuse pages immediately (macOS ≥14).
        "none"     — do nothing after layer; trust OS LRU entirely.
    metal_kernels:
        Whether to use Metal compute kernels (when available).
    expert_cache_size:
        Number of experts to keep in LRU cache (for MoE models).
    strict_guardrails:
        If True, FlashManager.load raises on RAM-budget violations.
        Set False only for tiny models or testing.
    debug:
        Print per-layer timing and page-cache stats to stderr.
    max_kv_size:
        Maximum KV-cache size in tokens.  None = unlimited.
    kv_keep:
        Tokens to keep when the rotating KV cache evicts.
    prefill_chunk_size:
        Number of tokens per prefill chunk.  0 = no chunking.
    moe_top_k_override:
        If set, overrides the model's default top-K for MoE routing.  Useful
        to reduce RAM further (e.g. force K=1 for very low RAM at quality cost).
    monitor_queue:
        If set, FlashLLM emits per-layer telemetry dicts to this queue.
    """

    enabled: bool = False
    ram_budget_gb: float = 2.0
    eviction_strategy: Literal["dontneed", "free", "none"] = "free"  # Planned v0.2+
    metal_kernels: bool = True                                       # Planned v0.2+
    expert_cache_size: int = 8                                       # Planned v0.3+ (MoE)
    strict_guardrails: bool = True                                   # Planned v0.2+
    debug: bool = False

    # KV Cache & Prefill Memory Management
    max_kv_size: int | None = None           # None = unlimited; 4096 = safe for 16GB
    kv_keep: int = 250                          # tokens to keep during rotation
    prefill_chunk_size: int = 512              # Planned v0.2+ (chunked prefill)

    moe_top_k_override: int | None = None   # Planned v0.3+ (MoE weight streaming)
    
    # Telemetry
    monitor_queue: queue.Queue | None = None # If set, emit telemetry events

    # Derived / auto-detected — not set by user
    _n_cpu_cores: int = field(default_factory=lambda: os.cpu_count() or 4,
                              init=False, repr=False)

    def validate(self) -> None:
        if self.ram_budget_gb < 0.5:
            raise ValueError("ram_budget_gb must be >= 0.5 GB")
        if self.prefill_chunk_size < 0:
            raise ValueError("prefill_chunk_size must be >= 0")
        if self.kv_keep < 0:
            raise ValueError("kv_keep must be >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> FlashConfig:
        """Build a FlashConfig from a plain dict (e.g. from JSON / Modelfile)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()
                 if not f.name.startswith("_")}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def __post_init__(self) -> None:
        self.validate()
