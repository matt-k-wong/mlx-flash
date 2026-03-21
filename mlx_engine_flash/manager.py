import mlx_lm
import mlx.core as mx
from pathlib import Path
from typing import Any, Tuple
from .config import FlashConfig
from .generation import FlashLLM

class FlashManager:
    """
    Orchestrates the Flash Weight Streaming environment.
    """
    def __init__(self, config: FlashConfig = None):
        self.config = config or FlashConfig()
        self.model = None
        self.tokenizer = None

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

    def load(self, model_path: str | Path) -> Tuple[FlashLLM, Any]:
        """
        Load a model in lazy mode and wrap it for Flash execution.
        """
        self.config.validate()
        path = Path(model_path)
        
        # 1. Set Metal wired limit BEFORE loading weights
        self._apply_wired_limit()
        
        # 1.5 Start Telemetry Bridge for flash-monitor
        from .monitor import start_telemetry
        self._telemetry_bridge = start_telemetry(self.config)
        
        # 2. Native lazy load: weights are lazy mmap-backed MLX arrays.
        # Avoid recursion if mlx_lm is monkey-patched
        try:
            from .integration.lmstudio import _ORIGINAL_LOAD
            loader = _ORIGINAL_LOAD or mlx_lm.load
        except (ImportError, AttributeError):
            loader = mlx_lm.load
            
        model, self.tokenizer = loader(str(path), lazy=True)
        
        # 3. Wrap in Flash execution engine
        self.model = FlashLLM(model, self.config)
        
        if self.config.debug:
            import mlx.utils
            n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
            print(f"[flash] Loaded {path.name}: {n_params/1e9:.1f}B params, lazy (0 Metal RAM)")
            
        return self.model, self.tokenizer

    def shutdown(self):
        pass

__all__ = ["FlashManager"]
