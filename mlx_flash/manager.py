from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm

from .config import FlashConfig
from .generation import FlashLLM


class FlashManager:
    """
    Orchestrates the Flash Weight Streaming environment.
    """
    def __init__(self, config: FlashConfig | None = None):
        self.config = config or FlashConfig()
        self.model: Any = None
        self.tokenizer: Any = None

    def _apply_wired_limit(self):
        """Set Metal wired memory limit based on RAM budget."""
        limit_bytes = int(self.config.ram_budget_gb * 1024 * 1024 * 1024)
        try:
            mx.metal.set_wired_limit(limit_bytes)
            if self.config.debug:
                print(f"[flash] Metal wired limit set to {self.config.ram_budget_gb:.1f} GB")
        except AttributeError:
            # Older MLX versions might not have this
            pass

    def load(self, model_path: str | Path) -> tuple[FlashLLM, Any]:
        """
        Load a model in lazy mode and wrap it for Flash execution.
        """
        self.config.validate()
        path = Path(model_path)
        
        # 1. Set Metal wired limit BEFORE loading weights
        self._apply_wired_limit()
        
        # 1.5 Start Telemetry Bridge for flash-monitor (opt-in)
        self._telemetry_bridge = None
        if self.config.monitor_queue is not None or self.config.debug:
            from .monitor import start_telemetry
            self._telemetry_bridge = start_telemetry(self.config)
        
        # 2. Native lazy load: weights are lazy mmap-backed MLX arrays.
        # Avoid recursion if mlx_lm is monkey-patched
        try:
            from .integration.lmstudio import _ORIGINAL_LOAD
            loader = _ORIGINAL_LOAD or mlx_lm.load
        except (ImportError, AttributeError):
            loader = mlx_lm.load
            
        model, self.tokenizer = loader(str(path), lazy=True)[:2]  # type: ignore
        
        # 3. Wrap in Flash execution engine
        self.model = FlashLLM(model, self.config)
        
        if self.config.debug:
            import mlx.utils
            n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))  # type: ignore
            print(f"[flash] Loaded {path.name}: {n_params/1e9:.1f}B params, lazy (0 Metal RAM)")
            
        return self.model, self.tokenizer

    def shutdown(self):
        """
        Release Metal resources and stop background telemetry.
        """
        import contextlib
        if hasattr(self, "_telemetry_bridge") and self._telemetry_bridge:
            with contextlib.suppress(Exception):
                self._telemetry_bridge.stop()
            self._telemetry_bridge = None
        # 2. Restore Metal wired limit to 0 (default).
        # If the monkey-patch is active, mx.metal.set_wired_limit may be a
        # no-op lambda.  Use the saved original when available.
        with contextlib.suppress(AttributeError, Exception):
            try:
                from .integration.lmstudio import _ORIGINAL_SET_WIRED_LIMIT
                setter = _ORIGINAL_SET_WIRED_LIMIT or mx.metal.set_wired_limit
            except ImportError:
                setter = mx.metal.set_wired_limit
            setter(0)

        # 3. Clear model and tokenizer references to allow GC
        self.model = None
        self.tokenizer = None
        
        # 4. Clear Metal cache
        mx.metal.clear_cache()

__all__ = ["FlashManager"]
