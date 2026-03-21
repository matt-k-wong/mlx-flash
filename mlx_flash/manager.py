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

    def _check_spotlight_warning(self, model_path: Path):
        """Warn users if Spotlight indexing might degrade Flash performance, and auto-exclude."""
        import os
        # Auto-exclude model directory from Spotlight
        metadata_file = model_path / ".metadata_never_index"
        if not metadata_file.exists():
            try:
                metadata_file.touch()
                if getattr(self.config, 'debug', False):
                    print(f"[flash] Auto-excluded {model_path} from Spotlight indexing.")
            except Exception:
                pass
                
        flag_file = Path.home() / ".mlx_flash_spotlight_warned"
        if not flag_file.exists():
            print("\n[flash] ✨ Tip: macOS Spotlight indexing on large model files can cause severe lag and SSD contention.")
            print("[flash]        We've attempted to auto-exclude the model directory via a .metadata_never_index file.")
            print("[flash]        If you still experience lag, explicitly add your models directory to:")
            print("[flash]        System Settings -> Siri & Spotlight -> Spotlight Privacy\n")
            try:
                flag_file.touch()
            except Exception:
                pass

    def _check_battery_warning(self):
        """Warn users if they are running heavy IO workloads on battery power."""
        import subprocess
        import sys
        flag_file = Path.home() / ".mlx_flash_battery_warned"
        if not flag_file.exists() and sys.platform == "darwin":
            try:
                out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=1).stdout
                if "Now drawing from 'Battery Power'" in out or "Battery Power" in out:
                    print("\n[flash] ⚠️  Warning: You are currently running on Battery Power.")
                    print("[flash]     Flash Weight Streaming reads enormous amounts of data from the SSD.")
                    print("[flash]     This will drain your Macbook's battery very quickly and may cause thermal throttling.")
                    print("[flash]     For maximum performance, connect to AC power.\n")
                    flag_file.touch()
            except Exception:
                pass

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
        
        # User Experience Warnings
        self._check_spotlight_warning(path)
        self._check_battery_warning()
        
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
        self.model = FlashLLM(model, self.config, model_path=path)
        
        try:
            from .safetensors_mmap import SafetensorsMmapCache
            self.model.mmap_cache = SafetensorsMmapCache(path)
        except Exception as e:
            if self.config.debug:
                print(f"[flash] Warning: Failed to initialize SafetensorsMmapCache: {e}")
        
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
        if hasattr(self.model, 'mmap_cache') and self.model.mmap_cache:
            try:
                self.model.mmap_cache.shutdown()
            except Exception:
                pass
        self.model = None
        self.tokenizer = None
        
        # 4. Clear Metal cache
        mx.clear_cache()

__all__ = ["FlashManager"]
