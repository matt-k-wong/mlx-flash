import inspect
import os
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.base import create_attention_mask

from .config import FlashConfig
from . import page_cache


class FlashLLM(nn.Module):
    """
    Wraps any mlx-lm Model to execute layers synchronously.
    
    Strategy: intercept execution at the layer level to iterate layers one
    at a time with forced mx.eval() between each, rather than building
    one unified lazy graph.
    """
    
    def __init__(self, model: nn.Module, config: FlashConfig, model_path: str | Path | None = None):
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_model_path", Path(model_path) if model_path else None)
        # Identify the main sub-structure (Llama uses .model, Nemotron uses .backbone)
        object.__setattr__(self, "_inner", (getattr(model, "model", None) or 
                                             getattr(model, "backbone", None) or 
                                             model))
        
        # Identify the layers
        object.__setattr__(self, "_layers", (getattr(self._inner, "layers", None) or 
                                             getattr(self._inner, "h", None) or
                                             getattr(self._inner, "blocks", None)))
        
        if self._layers is None:
            raise AttributeError("Cannot find layer stack (tried layers, h, blocks)")
            
        object.__setattr__(self, "_n_layers", len(self._layers))
        object.__setattr__(self, "_pre_layer_fn", self._build_pre_layer_fn(model))
        object.__setattr__(self, "_post_layer_fn", self._build_post_layer_fn(model))
        object.__setattr__(self, "_layer_sigs", self._cache_layer_signatures())
        object.__setattr__(self, "disk_cache", None)
        object.__setattr__(self, "mmap_cache", None)
        
        # Build per-layer weight file index for true weight streaming.
        # This maps layer_idx -> [(safetensors_path, [tensor_keys_in_that_file])]
        # so we can re-load just one layer's weights after mx.eval() orphans them.
        weight_index = self._build_weight_index()
        object.__setattr__(self, "_layer_weight_index", weight_index)
        
        # Pre-calculate unique safetensors files to avoid globbing in the hot path
        all_files = set()
        for entries in weight_index:
            for sf_path, _ in entries:
                all_files.add(sf_path)
        object.__setattr__(self, "_weight_files", sorted(list(all_files)))
    
    def _build_weight_index(self) -> list:
        """Pre-compute per-layer weight file locations for true weight streaming.
        
        Returns a list where index i contains:
            [(safetensors_path, [tensor_keys_belonging_to_layer_i]), ...]
        """
        if self._model_path is None:
            return [[] for _ in range(self._n_layers)]
        
        import glob
        import json
        import struct
        
        # Detect layer key prefix (Llama: "model.layers", Nemotron: "backbone.layers")
        layer_attr_name = "layers"  # default
        for attr in ("layers", "h", "blocks"):
            if getattr(self._inner, attr, None) is self._layers:
                layer_attr_name = attr
                break
        
        # Detect parent prefix (e.g., "model." or "backbone.")
        parent_prefix = ""
        if hasattr(self._model, "model") and getattr(self._model, "model", None) is self._inner:
            parent_prefix = "model."
        elif hasattr(self._model, "backbone") and getattr(self._model, "backbone", None) is self._inner:
            parent_prefix = "backbone."
        
        index = [[] for _ in range(self._n_layers)]
        
        sf_files = sorted(self._model_path.glob("*.safetensors"))
        for sf in sf_files:
            try:
                with open(sf, "rb") as f:
                    header_len = struct.unpack('<Q', f.read(8))[0]
                    header = json.loads(f.read(header_len).decode('utf-8'))
                
                # Group tensor keys by layer
                layer_keys: dict[int, list] = {}
                for key in header:
                    if key == "__metadata__":
                        continue
                    prefix = f"{parent_prefix}{layer_attr_name}."
                    if prefix in key:
                        # Extract layer index from e.g. "backbone.layers.3.mixer.weight"
                        after_prefix = key.split(prefix)[1]
                        try:
                            layer_idx = int(after_prefix.split(".")[0])
                            if 0 <= layer_idx < self._n_layers:
                                layer_keys.setdefault(layer_idx, []).append(key)
                        except ValueError:
                            pass
                
                for layer_idx, keys in layer_keys.items():
                    index[layer_idx].append((str(sf), keys))
                    
            except Exception:
                continue
        
        return index
    
    def _reload_layer_weights(self, layer_idx: int, all_lazy: dict):
        """Re-load a layer's weights as fresh lazy arrays from the pre-loaded dict.
        
        This orphans the old materialized Metal-resident arrays, allowing
        Python's GC to release the Metal memory allocation.
        """
        entries = self._layer_weight_index[layer_idx]
        if not entries:
            return
        
        layer = self._layers[layer_idx]
        
        # Detect the full prefix for this layer (e.g., "backbone.layers.3.")
        layer_attr_name = "layers"
        for attr in ("layers", "h", "blocks"):
            if getattr(self._inner, attr, None) is self._layers:
                layer_attr_name = attr
                break
        parent_prefix = ""
        if hasattr(self._model, "model") and getattr(self._model, "model", None) is self._inner:
            parent_prefix = "model."
        elif hasattr(self._model, "backbone") and getattr(self._model, "backbone", None) is self._inner:
            parent_prefix = "backbone."
        full_prefix = f"{parent_prefix}{layer_attr_name}.{layer_idx}."
        
        # Collect fresh lazy weights from our token-local cache
        fresh_weights = []
        for sf_path, keys in entries:
            for key in keys:
                if key in all_lazy:
                    # Strip the prefix to get the local name (e.g., "mixer.weight")
                    local_name = key[len(full_prefix):]
                    fresh_weights.append((local_name, all_lazy[key]))
        
        if fresh_weights:
            layer.load_weights(fresh_weights, strict=False)
    
    def _build_pre_layer_fn(self, model):
        """Return a function that runs everything BEFORE the layer stack."""
        sub = self._inner
        embed = (getattr(sub, "embed_tokens", None) or
                 getattr(sub, "embeddings", None) or
                 getattr(sub, "wte", None) or
                 getattr(sub, "word_embeddings", None) or
                 getattr(sub, "token_embeddings", None))
        if embed is None:
            raise AttributeError("Cannot find embedding layer")
        
        def pre(x, mask=None):
            h = embed(x)
            return h
        return pre
    
    def _build_post_layer_fn(self, model):
        """Return a function that runs everything AFTER the layer stack."""
        sub = self._inner
        norm = (getattr(sub, "norm", None) or
                getattr(sub, "norm_f", None) or
                getattr(sub, "ln_f", None) or
                getattr(sub, "final_layer_norm", None))
        lm_head = (getattr(model, "lm_head", None) or
                   getattr(model, "head", None))
        
        def post(h):
            if norm is not None:
                h = norm(h)
            if lm_head is not None:
                h = lm_head(h)
            return h
        return post
    
    def _cache_layer_signatures(self) -> list[tuple[bool, bool, bool]]:
        """Pre-compute (is_mamba, has_mask, has_cache) per layer."""
        sigs = []
        for layer in self._layers:
            # Detect Mamba (Mamba2, SSM, mixer with ssm)
            mixer = getattr(layer, "mixer", None)
            is_mamba = (hasattr(layer, "ssm") or 
                        (mixer is not None and (hasattr(mixer, "ssm") or "Mamba" in str(type(mixer)))))
            
            # Detect if this layer uses a cache entry
            # In hybrid models (Nemotron-H), only Mamba (M) and Attention (*) blocks use cache.
            # MLP and other blocks do not.
            params = inspect.signature(layer.__call__).parameters
            has_cache_arg = "cache" in params
            
            # Heuristic for hybrid models: if a block has 'mixer', it's likely a container.
            # If the block has a 'block_type' (Nemotron), use it.
            block_type = getattr(layer, "block_type", None)
            if block_type is not None:
                has_cache = block_type in ("M", "*")
            else:
                # Fallback to checking if it's an Attention or Mamba layer
                has_cache = has_cache_arg and (
                    is_mamba or 
                    hasattr(layer, "attention") or 
                    hasattr(layer, "self_attn") or
                    (mixer is not None and ("Attention" in str(type(mixer)) or "Mixer" in str(type(mixer))))
                )
                # If it's a standard transformer (Llama), it has no mixer but all layers use cache
                if not has_cache and has_cache_arg and mixer is None:
                    has_cache = True

            sigs.append((is_mamba, "mask" in params, has_cache))
        return sigs
    
    def __call__(self, *args, **kwargs) -> mx.array:
        """Synchronous per-layer forward pass."""
        # Extract arguments robustly — mlx-lm models vary in (x, mask, cache) vs (x, cache, mask)
        x = args[0] if len(args) > 0 else kwargs.get("x")
        
        # Look for cache in kwargs or as the 2nd/3rd positional arg
        cache = kwargs.get("cache")
        if cache is None and len(args) > 1:
            for arg in args[1:]:
                # Check for KVCache list or object
                if isinstance(arg, list) or "KVCache" in str(type(arg)):
                    cache = arg
                    break
        
        # Look for mask in kwargs or as any non-cache positional arg
        mask = kwargs.get("mask")
        if mask is None and len(args) > 1:
            for arg in args[1:]:
                if arg is not cache and not isinstance(arg, mx.array) and not isinstance(arg, str):
                    continue # likely something else
                if arg is not cache:
                    mask = arg
                    break
                    
        if x is None:
            raise ValueError("FlashLLM.__call__ missing input 'x'")

        # TOKEN-LOCAL WEIGHT CACHE: Load fresh lazy arrays for the entire model
        # ONCE at the start of the token pass. This eliminates the 52x overhead
        # of repeatedly parsing safetensors headers in the layer loop.
        token_weights = {}
        if self._weight_files:
            for sf_path in self._weight_files:
                token_weights.update(mx.load(sf_path))

        # Pre-layer: embedding
        h = self._pre_layer_fn(x)
        
        # 1. Fix the Mask Problem
        # If mask is a string (e.g. "causal"), some hybrid models like Nemotron-H
        # will crash trying to index it. Convert to None or a real mask.
        if isinstance(mask, str):
            mask = None
            
        # Pipelined synchronization tracker
        pending_releases = []
        pipeline_depth = getattr(self._config, 'pipeline_depth', 2)
        
        # Prime the async prefetch pump
        if self.mmap_cache:
            for p in range(pipeline_depth):
                if p < self._n_layers:
                    self.mmap_cache.prefetch_layer_background(p)
            
        # Per-layer synchronous execution
        cache_ptr = 0
        for i in range(self._n_layers):
            layer = self._layers[i]
            is_mamba, has_mask, has_cache = self._layer_sigs[i]
            
            # Fetch cache entry if needed
            cache_entry = None
            if cache is not None and has_cache:
                if cache_ptr < len(cache):
                    cache_entry = cache[cache_ptr]
                    cache_ptr += 1
                else:
                    # Fallback for misaligned caches
                    cache_entry = None
            
            # Enqueue the next layer outside of our current pipeline window
            if self.mmap_cache and i + pipeline_depth < self._n_layers:
                self.mmap_cache.prefetch_layer_background(i + pipeline_depth)
                    
            # Run this layer (builds a small graph for ONE layer)
            # Use keyword arguments for robustness
            call_kwargs = {}
            if has_mask:
                call_kwargs["mask"] = mask
            if has_cache:
                call_kwargs["cache"] = cache_entry
            
            output = layer(h, **call_kwargs)
            
            # Handle hybrid return types: some return (h, cache), some just h
            if isinstance(output, (list, tuple)) and len(output) == 2:
                h, cache_entry = output
            else:
                h = output
            
            # CRITICAL: materialise NOW before the next layer's graph is built
            # We eval 'h' AND the cache state tensors to force the layer's 
            # weights to be consumed and the cache to be updated.
            if cache_entry is not None:
                if hasattr(cache_entry, "state"):
                    s = cache_entry.state
                    if isinstance(s, (list, tuple)):
                        mx.eval(h, *s)
                    else:
                        mx.eval(h, s)
                elif hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                    mx.eval(h, cache_entry.keys, cache_entry.values)
                elif isinstance(cache_entry, (list, tuple)):
                    mx.eval(h, *cache_entry)
                else:
                    mx.eval(h, cache_entry)
            else:
                mx.eval(h)
            
            # Queue this layer for release (OS Page Cache)
            if self.mmap_cache:
                current_ranges = self.mmap_cache.get_layer_ranges(i)
                pending_releases.append(current_ranges)
            else:
                pending_releases.append({})
            
            # If our pipeline queue is full, synchronize and release the oldest layer
            if len(pending_releases) >= pipeline_depth:
                mx.synchronize()
                oldest_ranges = pending_releases.pop(0)
                for mm, (start, end, _) in oldest_ranges.items():
                    if hasattr(page_cache, "release"):
                        page_cache.release(mm, start, end - start, strategy=self._config.eviction_strategy)
                    
            # Release intermediate Metal memory (activations, etc.) regardless of streaming
            mx.clear_cache()
            
            # MEMORY vs SPEED TRADEOFF
            # Only reload weights (streaming mode) if we are over our RAM budget.
            # If we have spare RAM, we keep the weights materialized for speed.
            try:
                mem_active = mx.metal.get_active_memory()
            except AttributeError:
                mem_active = mx.get_active_memory()
            
            budget_bytes = self._config.ram_budget_gb * 1024**3
            if mem_active > budget_bytes:
                # TRUE WEIGHT STREAMING: Re-load this layer's weights as fresh lazy
                # arrays from the token-local cache. This orphans the old materialized
                # Metal-resident arrays.
                self._reload_layer_weights(i, token_weights)
                mx.clear_cache()
                if self._config.debug and i % 10 == 0:
                    print(f"[flash] Budget exceeded ({mem_active/1e9:.1f}GB > {self._config.ram_budget_gb}GB). Streaming active.", file=sys.stderr)
            
            # Telemetry
            if self._config.monitor_queue is not None:
                try:
                    # Robust memory API check
                    try:
                        mem = mx.metal.get_active_memory()
                    except AttributeError:
                        mem = mx.get_active_memory()
                    
                    self._config.monitor_queue.put_nowait({
                        "type": "layer_complete",
                        "layer": i + 1,
                        "n_layers": self._n_layers,
                        "metal_active_mb": mem / 1e6,
                        "timestamp": time.monotonic(),
                    })
                except Exception:
                    pass
            
            if self._config.debug:
                try:
                    metal_mb = mx.metal.get_active_memory() / 1e6
                except AttributeError:
                    metal_mb = mx.get_active_memory() / 1e6
                print(f"[flash] layer {i:3d}/{self._n_layers}: "
                      f"Metal active {metal_mb:.0f} MB", file=sys.stderr)
        
        # Post-layer: norm + lm_head
        return self._post_layer_fn(h)
    
    def parameters(self):
        return self._model.parameters()
    
    def update(self, params):
        return self._model.update(params)

    def __getattr__(self, name: str) -> Any:
        # Don't delegate internal attributes (starting with _)
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            model = object.__getattribute__(self, "_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)

class FlashGenerationLoop:
    """
    High-level generator that uses FlashLLM and mlx_lm.generate_step.
    """
    def __init__(self, model_or_path: str | nn.Module, tokenizer: Any = None, config: FlashConfig = None):
        if config is None:
             from .config import FlashConfig
             config = FlashConfig()
        self.config = config
        
        if isinstance(model_or_path, (str, Path)):
            self.model, self.tokenizer = mlx_lm.load(str(model_or_path), lazy=True)[:2]  # type: ignore
            self.flash_model = FlashLLM(self.model, config)
            # Initialize Mmap cache here
            from .safetensors_mmap import SafetensorsMmapCache
            self.flash_model.mmap_cache = SafetensorsMmapCache(model_or_path)
            
        elif isinstance(model_or_path, FlashLLM):
            self.flash_model = model_or_path
            self.model = self.flash_model._model
            self.tokenizer = tokenizer
            if not hasattr(self.flash_model, 'mmap_cache'):
                self.flash_model.mmap_cache = None
        else:
            # Assume it is a base nn.Module from mlx_lm.load
            self.model = model_or_path
            self.tokenizer = tokenizer
            self.flash_model = FlashLLM(self.model, config)
            self.flash_model.mmap_cache = None
            
        n_layers = self.flash_model._n_layers
        if self.config.debug:
            print(f"[flash] FlashGenerationLoop ready: {n_layers} layers")


    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        """Generate tokens using the standard mlx_lm pipeline with FlashLLM.

        Delegates to mlx_lm.stream_generate so behavior is identical to the
        monkey-patch path.  FlashLLM is a transparent nn.Module proxy, so the
        standard pipeline handles prefill, caching, and sampling correctly.
        """
        # Extract sampling params that generate_step doesn't accept directly.
        # generate_step expects a `sampler` callable, not raw temp/top_p/top_k.
        from mlx_lm.sample_utils import make_sampler
        sampler_args = {
            "temp": kwargs.pop("temp", kwargs.pop("temperature", 0.0)),
            "top_p": kwargs.pop("top_p", 1.0),
            "top_k": kwargs.pop("top_k", 0),
        }
        kwargs["sampler"] = make_sampler(**sampler_args)

        # Inject DiskKVCache if enabled
        if getattr(self.config, "disk_kv_enabled", False):
            if "prompt_cache" not in kwargs:
                from mlx_flash.disk_kv_cache import DiskKVCache
                import shutil
                import tempfile
                import uuid

                kv_dir_cfg = getattr(self.config, "disk_kv_dir", "")
                if kv_dir_cfg:
                    kv_dir = Path(kv_dir_cfg)
                else:
                    kv_dir = Path(tempfile.gettempdir()) / f"mlx_flash_kv_{os.getpid()}_{uuid.uuid4().hex[:8]}"

                # Wipe old cache to ensure clean context
                if kv_dir.exists():
                    shutil.rmtree(kv_dir, ignore_errors=True)
                kv_dir.mkdir(parents=True, exist_ok=True)

                max_tokens = getattr(self.config, "disk_kv_max_tokens", None)

                # Build the array of DiskKVCaches
                prompt_cache = [DiskKVCache(layer_idx=i, cache_dir=str(kv_dir), max_tokens=max_tokens)
                              for i in range(self.flash_model._n_layers)]
                kwargs["prompt_cache"] = prompt_cache
                self._disk_kv_caches = prompt_cache

                if self.config.debug:
                    print(f"[flash] Injected {len(prompt_cache)} DiskKVCache layers at {kv_dir}", file=sys.stderr)

        for result in mlx_lm.stream_generate(
            self.flash_model, self.tokenizer, prompt,
            max_tokens=max_tokens, **kwargs,
        ):
            yield result.text

    def shutdown(self):
        """Clean up resources."""
        import contextlib
        # Close DiskKVCache layers if they exist
        for cache in getattr(self, "_disk_kv_caches", []):
            with contextlib.suppress(Exception):
                cache.close()
        self._disk_kv_caches = []
        with contextlib.suppress(AttributeError, Exception):
            mx.metal.clear_cache()
        self.flash_model = None
        self.model = None
        self.tokenizer = None
