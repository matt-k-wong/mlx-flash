from __future__ import annotations
import os
# Force MLX to use memory mapping for zero-copy BEFORE importing mlx.core
os.environ["MLX_MEMORY_MAPPING"] = "1"

import functools
import gc
import psutil
import time
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

try:
    import mlx_lm
    _HAS_MLX_LM = True
except ImportError:
    _HAS_MLX_LM = True

from .config import FlashConfig
from .loader import FlashModelLoader, _update_model_weights
from .prefetch import WeightPrefetcher
from .streamer import WeightStreamer

class FlashManager:
    """
    Synchronous Inference Engine for Flash Weight Streaming.
    """

    def __init__(self, config: FlashConfig) -> None:
        self.config = config
        self._loader: FlashModelLoader | None = None
        self._streamer: WeightStreamer | None = None
        self._prefetcher: WeightPrefetcher | None = None
        self._metrics: dict[str, int] = {"cache_hits": 0, "cache_misses": 0}
        self._shared_dummy = mx.array(0.0, dtype=mx.float16)

    def _log_rss(self, stage: str):
        process = psutil.Process()
        rss = process.memory_info().rss / (1024**3)
        print(f"[flash-RAM] {stage}: {rss:.2f} GB", flush=True)

    def load(self, model_path: str, load_fn: Any | None = None, **mlx_lm_kwargs: Any) -> tuple[Any, Any]:
        import mlx_lm
        if load_fn is None: load_fn = mlx_lm.load

        self.config.validate()
        model_dir = Path(model_path)
        
        # 1. SKELETON LOAD
        original_load_weights = nn.Module.load_weights
        nn.Module.load_weights = lambda self, weights, strict=False: self
        
        _log("Building zero-weight skeleton...")
        try:
            mlx_lm_kwargs["lazy"] = True
            model, tokenizer = load_fn(str(model_dir), **mlx_lm_kwargs)
        finally:
            nn.Module.load_weights = original_load_weights

        # 2. SETUP FLASH ASSETS
        self._loader = FlashModelLoader(model_dir, self.config).__enter__()
        self._streamer = self._loader._streamer
        self._prefetcher = WeightPrefetcher(self._streamer, self.config, self._loader.n_layers, loader=self._loader)
        if self.config.prefetch_layers > 0: self._prefetcher.start()

        # 3. POPULATE PERMANENT WEIGHTS
        _log("Loading permanent weights...")
        idx = self._loader._streamer.index
        perm_names = [n for n in idx.tensor_names() if "layers." not in n]
        perm_weights = self._loader.to_mlx(self._streamer.stream_tensors(perm_names))
        _update_model_weights(model, perm_weights)
        
        # 4. PATCH FOR SYNCHRONOUS FLOW
        self._patch_layers_for_sync(model)
        
        # 5. SET TIGHT LIMITS (Force allocator to be aggressive)
        limit_bytes = int(self.config.ram_budget_gb * 1024 * 1024 * 1024)
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(limit_bytes)
        else:
            mx.metal.set_cache_limit(limit_bytes)
        
        return model, tokenizer

    def _patch_layers_for_sync(self, model: Any) -> None:
        backbone = getattr(model, "model", getattr(model, "backbone", model))
        layers = backbone.layers
        manager = self

        def find_arrays(obj):
            if isinstance(obj, mx.array):
                return [obj]
            if isinstance(obj, (list, tuple)):
                res = []
                for x in obj: res.extend(find_arrays(x))
                return res
            if isinstance(obj, dict):
                res = []
                for x in obj.values(): res.extend(find_arrays(x))
                return res
            if hasattr(obj, "__dict__"):
                return find_arrays(obj.__dict__)
            if hasattr(obj, "state") and isinstance(obj.state, mx.array):
                return [obj.state]
            return []

        def _make_sync_call(original_call, layer_idx):
            @functools.wraps(original_call)
            def _sync_call(*args, **kwargs):
                # i. Load
                manager._log_rss(f"Layer {layer_idx} start")
                layer_weights = (manager._prefetcher.get_buffered_weights(layer_idx) if manager._prefetcher else None)
                if layer_weights is None:
                    layer_weights = manager._loader.get_layer_weights(layer_idx)
                
                prefix = manager._loader._streamer.index._layer_prefix.replace(".0.", f".{layer_idx}.")
                if "layers.0." in manager._loader._streamer.index._layer_prefix and not manager._loader._streamer.index._layer_prefix.startswith("."):
                    prefix = manager._loader._streamer.index._layer_prefix.replace("layers.0.", f"layers.{layer_idx}.")
                
                stripped = { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in layer_weights.items() }
                if any(not isinstance(v, mx.array) for v in stripped.values()):
                    stripped = manager._loader.to_mlx(stripped)
                
                _update_model_weights(layers[layer_idx], stripped)
                manager._log_rss(f"Layer {layer_idx} weights loaded")

                # ii. Execute
                output = original_call(*args, **kwargs)
                
                # iii. DEEP REALIZE & SYNC
                eval_targets = find_arrays([output, kwargs])
                if eval_targets:
                    mx.eval(*eval_targets)
                mx.synchronize()
                manager._log_rss(f"Layer {layer_idx} executed")
                
                # iv. HARD EVICTION
                dummy_map = { k: manager._shared_dummy for k in stripped }
                _update_model_weights(layers[layer_idx], dummy_map)
                
                # Clear all caches
                mx.clear_cache()
                if hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
                gc.collect()
                
                # Release page cache
                if manager._streamer:
                    manager._streamer.release_layer(layer_idx)
                
                # Give OS a moment to reclaim
                time.sleep(0.1)
                manager._log_rss(f"Layer {layer_idx} evicted")
                
                return output
            return _sync_call

        for i, layer in enumerate(layers):
            layer.__call__ = _make_sync_call(layer.__call__, i)

    def shutdown(self) -> None:
        if self._prefetcher: self._prefetcher.stop()
        if self._loader: self._loader.close()

def _log(msg: str) -> None:
    import sys
    print(f"[flash] {msg}", file=sys.stderr)
