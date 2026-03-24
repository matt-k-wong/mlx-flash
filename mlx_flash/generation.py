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

from . import page_cache
from .config import FlashConfig


class FlashLLM(nn.Module):
    """
    Wraps any mlx-lm Model to execute layers synchronously and stream weights.
    """
    
    def __init__(self, model: nn.Module, config: FlashConfig, model_path: str | Path | None = None):
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_model_path", Path(model_path) if model_path else None)
        
        # Identify structure (Llama uses .model, Nemotron uses .backbone)
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
        object.__setattr__(self, "mmap_cache", None)
        object.__setattr__(self, "_materialized_layers", [])
        
        # Build weight index
        weight_index, other_index = self._build_weight_index()
        object.__setattr__(self, "_layer_weight_index", weight_index)
        object.__setattr__(self, "_other_weight_index", other_index)
        
        # Track unique safetensors files
        all_files = set()
        for entries in weight_index:
            for sf_path, _ in entries:
                all_files.add(sf_path)
        for sf_path, _ in other_index:
            all_files.add(sf_path)
        object.__setattr__(self, "_weight_files", sorted(list(all_files)))

        # SESSION CACHE: Load all lazy handles once at startup.
        # These are truly lazy (0 bytes Metal) and allow us to bypass
        # mx.load() in the hot token loop.
        handles = {}
        for sf_path in self._weight_files:
            handles.update(mx.load(sf_path))
        object.__setattr__(self, "_weight_handles", handles)

    @property
    def layers(self):
        return self._layers

    def make_cache(self):
        return self._model.make_cache()

    def parameters(self):
        return self._model.parameters()

    def _build_weight_index(self) -> tuple[list, list]:
        """Pre-compute per-layer weight locations."""
        if self._model_path is None:
            return [[] for _ in range(self._n_layers)], []
        
        import json
        import struct
        import re
        
        layer_regex = re.compile(r'\b(?:layers|h|blocks)\.(\d+)\.')
        index: list[list[tuple[str, list[str]]]] = [[] for _ in range(self._n_layers)]
        other_index_map: dict[str, list[str]] = {}
        
        sf_files = sorted(self._model_path.glob("*.safetensors"))
        for sf in sf_files:
            try:
                with open(sf, "rb") as f:
                    header_len = struct.unpack('<Q', f.read(8))[0]
                    header = json.loads(f.read(header_len).decode('utf-8'))
                
                layer_keys: dict[int, list] = {}
                other_keys: list[str] = []
                for key in header:
                    if key == "__metadata__": continue
                    m = layer_regex.search(key)
                    if m:
                        l_idx = int(m.group(1))
                        if 0 <= l_idx < self._n_layers:
                            layer_keys.setdefault(l_idx, []).append(key)
                        else: other_keys.append(key)
                    else: other_keys.append(key)
                
                for l_idx, keys in layer_keys.items():
                    index[l_idx].append((str(sf), keys))
                if other_keys:
                    other_index_map[str(sf)] = other_keys
            except Exception:
                continue
        
        return index, [(path, keys) for path, keys in other_index_map.items()]
    
    def _reload_layer_weights(self, layer_idx: int):
        """Re-load weights from the session handle cache."""
        entries = self._layer_weight_index[layer_idx]
        if not entries: return
        
        layer = self._layers[layer_idx]
        import re
        local_regex = re.compile(rf".*?\b(?:layers|h|blocks)\.{layer_idx}\.(.*)")
        
        fresh_weights = []
        for sf_path, keys in entries:
            for key in keys:
                if key in self._weight_handles:
                    # Create a fresh shared pointer to the underlying lazy data
                    val = mx.array(self._weight_handles[key])
                    m = local_regex.match(key)
                    if m:
                        fresh_weights.append((m.group(1), val))
        
        if fresh_weights:
            layer.load_weights(fresh_weights, strict=False)

    def _reload_other_weights(self):
        """Re-load non-layer weights using session handles."""
        if not hasattr(self, "_other_weight_index") or not self._other_weight_index:
            return
        fresh = []
        for sf_path, keys in self._other_weight_index:
            for k in keys:
                if k in self._weight_handles:
                    fresh.append((k, mx.array(self._weight_handles[k])))
        if fresh:
            self._model.load_weights(fresh, strict=False)
    
    def _build_pre_layer_fn(self, model):
        sub = self._inner
        embed = (getattr(sub, "embed_tokens", None) or getattr(sub, "embeddings", None) or 
                 getattr(sub, "wte", None) or getattr(sub, "word_embeddings", None))
        def pre(x, mask=None): return embed(x)
        return pre
    
    def _build_post_layer_fn(self, model):
        sub = self._inner
        norm = (getattr(sub, "norm", None) or getattr(sub, "norm_f", None) or 
                getattr(sub, "ln_f", None) or getattr(sub, "final_layer_norm", None))
        lm_head = getattr(model, "lm_head", None) or getattr(model, "head", None)
        def post(h):
            if norm is not None: h = norm(h)
            if lm_head is not None: h = lm_head(h)
            return h
        return post
    
    def _cache_layer_signatures(self) -> list[tuple[bool, bool, bool]]:
        sigs = []
        for layer in self._layers:
            params = inspect.signature(layer.__call__).parameters
            block_type = getattr(layer, "block_type", None)
            if block_type is not None:
                has_cache = block_type in ("M", "*")
            else:
                mixer = getattr(layer, "mixer", None)
                has_cache = "cache" in params and (
                    hasattr(layer, "attention") or hasattr(layer, "self_attn") or
                    (mixer is not None and ("Attention" in str(type(mixer))))
                )
            sigs.append((False, "mask" in params, has_cache))
        return sigs
    
    def __call__(self, *args, **kwargs) -> mx.array:
        """Leak-proof synchronous per-layer execution."""
        # Clean environment
        import gc
        gc.collect()
        mx.clear_cache()

        x = args[0] if len(args) > 0 else kwargs.get("x")
        cache = kwargs.get("cache")
        if cache is None and len(args) > 1:
            for arg in args[1:]:
                if isinstance(arg, list) or "KVCache" in str(type(arg)):
                    cache = arg; break
        
        mask = kwargs.get("mask")
        if mask is None and len(args) > 1:
            for arg in args[1:]:
                if arg is not cache and isinstance(arg, mx.array):
                    mask = arg; break
                    
        if x is None: raise ValueError("Input x missing")

        # Phase-Aware Depth
        is_decode = x.shape[1] <= 1 if hasattr(x, "shape") else True
        pipeline_depth = getattr(self._config, 'pipeline_depth', 4) if is_decode else 1
        
        h = self._pre_layer_fn(x)
        if isinstance(mask, str): mask = None
        
        pending_releases = []
        if self.mmap_cache:
            k = self.mmap_cache.k_distance if hasattr(self.mmap_cache, 'k_distance') else pipeline_depth
            for p in range(k):
                if p < self._n_layers: self.mmap_cache.prefetch_layer_background(p)
            
        cache_ptr = 0
        budget_bytes = self._config.ram_budget_gb * 1024**3

        # Initialize PipelinedExecutor if pipelining is enabled
        pipelined_executor = None
        if getattr(self._config, 'pipelined_execution', False):
            from .pipeline.executor import PipelinedExecutor
            pipelined_executor = PipelinedExecutor(self.mmap_cache)

        for i in range(self._n_layers):
            layer = self._layers[i]
            _, has_mask, has_cache = self._layer_sigs[i]
            
            cache_entry = None
            if cache is not None and has_cache:
                # Handle subscriptable cache list or object
                if hasattr(cache, "__getitem__"):
                    try:
                        if cache_ptr < len(cache):
                            cache_entry = cache[cache_ptr]
                            cache_ptr += 1
                    except Exception:
                        cache_entry = cache
                else:
                    cache_entry = cache
            
            if pipelined_executor is not None:
                t0_compute = time.perf_counter()
                h = pipelined_executor.execute_dense_layer(
                    h, layer, i, mask=mask if has_mask else None, cache=cache_entry if has_cache else None
                )
                compute_time = time.perf_counter() - t0_compute
                if self.mmap_cache and hasattr(self.mmap_cache, 'record_compute_time'):
                    self.mmap_cache.record_compute_time(compute_time)
            else:
                if self.mmap_cache:
                    k = self.mmap_cache.k_distance if hasattr(self.mmap_cache, 'k_distance') else pipeline_depth
                    for p in range(1, k + 1):
                        if i + p < self._n_layers:
                            self.mmap_cache.prefetch_layer_background(i + p)
                    if hasattr(self.mmap_cache, 'wait_for_layer'):
                        self.mmap_cache.wait_for_layer(i)
                        
                call_kwargs = {}
                if has_mask: call_kwargs["mask"] = mask
                if has_cache: call_kwargs["cache"] = cache_entry
                
                t0_compute = time.perf_counter()
                output = layer(h, **call_kwargs)
                h = output[0] if (isinstance(output, (list, tuple)) and len(output) == 2) else output
                
                # Materialize
                if cache_entry is not None:
                    if hasattr(cache_entry, "state") and cache_entry.state is not None:
                        mx.eval(h, *[s for s in cache_entry.state if s is not None])
                    elif hasattr(cache_entry, "keys") and cache_entry.keys is not None:
                        mx.eval(h, cache_entry.keys, cache_entry.values)
                    else: mx.eval(h)
                else: mx.eval(h)
                
                mx.synchronize()
                compute_time = time.perf_counter() - t0_compute
                if self.mmap_cache and hasattr(self.mmap_cache, 'record_compute_time'):
                    self.mmap_cache.record_compute_time(compute_time)

            # Eviction
            if i not in self._materialized_layers:
                self._materialized_layers.append(i)
                
            try: mem_active = mx.metal.get_active_memory()
            except AttributeError: mem_active = mx.get_active_memory()

            while self._materialized_layers and mem_active > budget_bytes:
                oldest = self._materialized_layers.pop(0)
                self._reload_layer_weights(oldest)
                gc.collect()
                mx.clear_cache()
                try: 
                    mx.synchronize()
                    mem_active = mx.metal.get_active_memory()
                except AttributeError: mem_active = mx.get_active_memory()
            
            if self.mmap_cache:
                ranges = self.mmap_cache.get_layer_ranges(i)
                if ranges: pending_releases.append(ranges)
            
            if len(pending_releases) >= pipeline_depth:
                mx.synchronize()
                oldest_ranges = pending_releases.pop(0)
                for mm, (start, end, _) in oldest_ranges.items():
                    if hasattr(page_cache, "release"):
                        page_cache.release(mm, start, end - start, strategy=self._config.eviction_strategy)
                    
            mx.clear_cache()
            if self._config.debug:
                print(f"[flash] layer {i:2d}: Metal active {mem_active/1e6:.0f} MB", file=sys.stderr)
        
        result = self._post_layer_fn(h)
        mx.eval(result)
        mx.synchronize()
        self._reload_other_weights()
        mx.clear_cache()
        gc.collect()
        return result

    def parameters(self):
        return self._model.parameters()

class FlashGenerationLoop:
    def __init__(self, model_or_path: str | nn.Module, tokenizer: Any = None, config: FlashConfig = None):
        if config is None: config = FlashConfig()
        self.config = config
        
        if isinstance(model_or_path, (str, Path)):
            self.model, self.tokenizer = mlx_lm.load(str(model_or_path), lazy=True)[:2]
            
            if getattr(self.config, "tiled_execution", False):
                from .tiled import apply_tiling
                apply_tiling(self.model, tile_size=getattr(self.config, "tile_size", 1024))
                
            self.flash_model = FlashLLM(self.model, config, model_path=model_or_path)
            from .safetensors_mmap import SafetensorsMmapCache
            self.flash_model.mmap_cache = SafetensorsMmapCache(model_or_path)
        else:
            self.model = model_or_path
            
            if getattr(self.config, "tiled_execution", False):
                from .tiled import apply_tiling
                apply_tiling(self.model, tile_size=getattr(self.config, "tile_size", 1024))
                
            self.tokenizer = tokenizer
            self.flash_model = FlashLLM(self.model, config)
            
    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import generate_step
        
        temp = kwargs.pop("temperature", kwargs.pop("temp", 0.0))
        kwargs["sampler"] = make_sampler(temp=temp)
        if "prefill_step_size" not in kwargs:
            kwargs["prefill_step_size"] = getattr(self.config, "prefill_chunk_size", 32)

        import gc
        gc.collect()
        mx.clear_cache()

        prompt_arr = mx.array(self.tokenizer.encode(prompt)) if isinstance(prompt, str) else mx.array(prompt)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for token, _ in generate_step(prompt_arr, self.flash_model, **kwargs):
            tid = token.item() if hasattr(token, "item") else token
            detokenizer.add_token(tid)
            if detokenizer.last_segment: yield detokenizer.last_segment
            gc.collect()
            mx.clear_cache()
            if tid == self.tokenizer.eos_token_id: break
        
        detokenizer.finalize()
        if detokenizer.last_segment: yield detokenizer.last_segment

    def shutdown(self):
        import contextlib
        mx.clear_cache()
        if self.flash_model is not None and getattr(self.flash_model, "mmap_cache", None) is not None:
            with contextlib.suppress(Exception): self.flash_model.mmap_cache.shutdown()
        self.flash_model = None
        self.model = None
        self.tokenizer = None
