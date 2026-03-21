"""
mlx-flash
=====================
Flash Weight Streaming for LM Studio / mlx-engine.

Typical usage inside mlx-engine (or any mlx-lm wrapper):

    from mlx_engine_flash import FlashConfig, FlashManager
    from mlx_engine_flash.integration.lmstudio import apply_flash_patch

    config = FlashConfig(enabled=True, ram_budget_gb=10.0, n_io_threads=4)
    apply_flash_patch(config)          # patches mlx_lm.load globally
    # ... rest of mlx-engine startup unchanged

Or via Modelfile:

    from mlx_engine_flash.integration.modelfile import parse_flash_directives
    config = parse_flash_directives(open("Modelfile").read())
"""

from .config import FlashConfig
from .generation import FlashGenerationLoop, FlashLLM
from .manager import FlashManager
from .page_cache import prefetch_array, release_array

__all__ = [
    "FlashConfig",
    "FlashManager",
    "FlashLLM",
    "FlashGenerationLoop",
    "prefetch_array",
    "release_array",
]

__version__ = "0.1.0"
