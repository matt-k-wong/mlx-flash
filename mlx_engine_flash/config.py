"""
FlashConfig — all tuneable parameters for Flash Weight Streaming.
"""

from __future__ import annotations

import os
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
        enforces this via LRU eviction; this value is used only to scale
        prefetch aggressiveness.  Default 10 GB suits 16-GB Macs well.
    n_io_threads:
        Thread pool size for parallel pread().  4 saturates most NVMe drives;
        raise to 6–8 for Thunderbolt RAID.
    prefetch_layers:
        How many transformer layers ahead to prefetch (madvise WILLNEED).
        2 hides I/O latency on ~4 GB/s drives; raise to 4 on fast RAID.
    moe_top_k_override:
        If set, overrides the model's default top-K for MoE routing.  Useful
        to reduce RAM further (e.g. force K=1 for very low RAM at quality cost).
    min_quant_bits:
        Warn (or raise if strict=True) when a model uses fewer than this many
        bits.  Default 4 = no warning for Q4; set to 8 to flag 2-bit models.
    strict_quant:
        If True, raise ValueError instead of just warning for sub-min_quant_bits.
    eviction_strategy:
        "dontneed" — MADV_DONTNEED: tell OS pages are unneeded (advisory).
        "free"     — MADV_FREE: allow OS to reuse pages immediately (macOS ≥14).
        "none"     — do nothing after layer; trust OS LRU entirely.
    metal_kernels:
        Use custom Flash Metal kernels if compiled.  Falls back to MLX built-ins
        if the .metallib is not present.
    debug:
        Print per-layer timing and page-cache stats to stderr.
    """

    enabled: bool = False
    ram_budget_gb: float = 6.0
    n_io_threads: int = 4
    prefetch_layers: int = 2
    moe_top_k_override: int | None = None
    min_quant_bits: int = 4
    strict_quant: bool = False
    eviction_strategy: Literal["dontneed", "free", "none"] = "free"
    metal_kernels: bool = True
    expert_cache_size: int = 8  # Number of experts to keep in LRU cache
    max_safe_prefill_tokens: int = 64
    enable_prefill_guardrail: bool = True   # Set False only for tiny models / advanced testing
    debug: bool = False

    # Derived / auto-detected — not set by user
    _n_cpu_cores: int = field(default_factory=lambda: os.cpu_count() or 4,
                              init=False, repr=False)

    def validate(self) -> None:
        if self.ram_budget_gb < 2.0:
            raise ValueError("ram_budget_gb must be >= 2.0 GB")
        if not (1 <= self.n_io_threads <= 32):
            raise ValueError("n_io_threads must be between 1 and 32")
        if self.prefetch_layers < 0:
            raise ValueError("prefetch_layers must be >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> FlashConfig:
        """Build a FlashConfig from a plain dict (e.g. from JSON / Modelfile)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()
                 if not f.name.startswith("_")}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def __post_init__(self) -> None:
        self.validate()
