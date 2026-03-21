import sys
import time
from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

from .config import FlashConfig


class FlashLLM(nn.Module):
    """
    Wraps any mlx-lm Model to execute layers synchronously.
    
    Strategy: intercept execution at the layer level to iterate layers one
    at a time with forced mx.eval() between each, rather than building
    one unified lazy graph.
    """
    
    def __init__(self, model: nn.Module, config: FlashConfig):
        super().__init__()
        self._model = model
        self._config = config
        self._layers = self._find_layers(model)
        self._n_layers = len(self._layers)
        self._pre_layer_fn = self._build_pre_layer_fn(model)
        self._post_layer_fn = self._build_post_layer_fn(model)
        self.disk_cache = None  # optionally set by manager
    
    def _find_layers(self, model: nn.Module) -> list:
        """Find transformer layers via common attribute names."""
        for attr in ("layers", "transformer_layers", "blocks", "h"):
            # Check model.model.layers (most common mlx-lm pattern)
            sub = getattr(model, "model", model)
            layers = getattr(sub, attr, None)
            if layers is not None and len(layers) > 0:
                return list(layers)
        raise AttributeError(
            f"Cannot find transformer layers in {type(model).__name__}. "
            f"Attributes: {list(vars(model))}"
        )
    
    def _build_pre_layer_fn(self, model):
        """Return a function that runs everything BEFORE the layer stack."""
        sub = getattr(model, "model", model)
        embed = (getattr(sub, "embed_tokens", None) or
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
        sub = getattr(model, "model", model)
        norm = (getattr(sub, "norm", None) or
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
    
    def _is_mamba_layer(self, layer) -> bool:
        return hasattr(layer, "mixer") and hasattr(layer.mixer, "ssm")
    
    def __call__(
        self,
        x: mx.array,
        cache: list | None = None,
        mask: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        """Synchronous per-layer forward pass."""
        # Pre-layer: embedding
        h = self._pre_layer_fn(x)
        
        # Per-layer synchronous execution
        for i, layer in enumerate(self._layers):
            cache_entry = cache[i] if cache is not None else None
            
            # Prefetch next layer's mmap pages
            if i + 1 < self._n_layers:
                _prefetch_layer_params(self._layers[i + 1])
            
            # Run this layer (builds a small graph for ONE layer)
            if self._is_mamba_layer(layer):
                # Mamba layers have different cache structure (state)
                h, cache_entry = layer(h, cache_entry)
                if cache_entry is not None:
                    mx.eval(h, *cache_entry)
            else:
                try:
                    h = layer(h, mask=mask, cache=cache_entry)
                except TypeError:
                    # Some layers don't accept mask or cache
                    try:
                        h = layer(h, cache=cache_entry)
                    except TypeError:
                        h = layer(h)
                
                # CRITICAL: materialise NOW before the next layer's graph is built
                if cache_entry is not None:
                    # Modern MLX-LM cache objects have .keys and .values
                    mx.eval(h, cache_entry.keys, cache_entry.values)
                else:
                    mx.eval(h)
            
            # Disk KV Cache offloading (if enabled)
            if self.disk_cache and cache_entry is not None and not self._is_mamba_layer(layer):
                token_count = cache_entry.keys.shape[1]
                if token_count > self._config.max_in_memory_kv_tokens:
                    mid = token_count // 2
                    keys_to_evict = cache_entry.keys[:, :mid, :]
                    values_to_evict = cache_entry.values[:, :mid, :]
                    self.disk_cache.evict_to_disk(i, keys_to_evict, values_to_evict, (0, mid))
                    if self._config.debug:
                        print(f"[flash] layer {i:3d} evicted {mid} tokens to SSD", file=sys.stderr)
            
            # Synchronise: ensure GPU work is done before clearing cache
            mx.synchronize()
            
            # Release Metal pool memory
            mx.clear_cache()
            
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
                metal_mb = mx.get_active_memory() / 1e6
                print(f"[flash] layer {i:3d}/{self._n_layers}: "
                      f"Metal active {metal_mb:.0f} MB", file=sys.stderr)
        
        # Post-layer: norm + lm_head
        return self._post_layer_fn(h)
    
    def parameters(self):
        return self._model.parameters()
    
    def update(self, params):
        return self._model.update(params)

def _prefetch_layer_params(layer: nn.Module) -> None:
    """Issue madvise(WILLNEED) on the mmap pages of a layer's parameters."""
    import ctypes

    from .page_cache import MADV_WILLNEED, get_libc
    libc = get_libc()
    if libc is None:
        return
    for _, arr in tree_flatten(layer.parameters()):
        if not isinstance(arr, mx.array):
            continue
        try:
            # Access the memory pointer of the lazy array without evaluating it
            info = arr.__array_interface__
            if info is not None:
                ptr, _ = info["data"]
                libc.madvise(ctypes.c_void_p(ptr),
                             ctypes.c_size_t(arr.nbytes),
                             ctypes.c_int(MADV_WILLNEED))
        except Exception:
            pass

class DiskKVCache:
    """KV cache that evicts old entries to mmap'd files on SSD."""
    
    def __init__(self, n_layers: int, max_in_memory_tokens: int, cache_dir: str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(exist_ok=True, parents=True)
        self._max_in_mem = max_in_memory_tokens
        self._n_layers = n_layers
    
    def evict_to_disk(self, layer_idx: int, keys: mx.array, values: mx.array, token_range: tuple[int, int]):
        """Evict a range of KV entries to disk (safetensors format)."""
        path = self._dir / f"layer_{layer_idx}_tokens_{token_range[0]}_{token_range[1]}.safetensors"
        mx.save_safetensors(str(path), {"k": keys, "v": values})
        return path
    
    def load_from_disk(self, layer_idx: int, token_range: tuple[int, int]):
        """Load KV entries back from disk (lazy, mmap-backed)."""
        path = self._dir / f"layer_{layer_idx}_tokens_{token_range[0]}_{token_range[1]}.safetensors"
        data = mx.load(str(path))
        return data["k"], data["v"]

class FlashGenerationLoop:
    """
    High-level generator that uses FlashLLM and mlx_lm.generate_step.
    """
    def __init__(self, model_path: str, config: FlashConfig):
        self.config = config
        self.model, self.tokenizer = mlx_lm.load(model_path, lazy=True)
        self.flash_model = FlashLLM(self.model, config)
        
        n_layers = self.flash_model._n_layers
        
        # Initialize Cache
        if config.max_kv_size is not None:
            self._cache = [
                RotatingKVCache(max_size=config.max_kv_size, keep=config.kv_keep)
                for _ in range(n_layers)
            ]
        else:
            self._cache = make_prompt_cache(self.model)
            
        if config.kv_cache_dir:
            self.flash_model.disk_cache = DiskKVCache(
                n_layers, config.max_in_memory_kv_tokens, config.kv_cache_dir
            )

    def _chunked_prefill(self, prompt_tokens: list[int], **kwargs):
        """Process a long prompt in chunks to avoid attention OOM."""
        chunk_size = self.config.prefill_chunk_size
        num_tokens = len(prompt_tokens)
        
        if chunk_size <= 0 or num_tokens <= chunk_size:
            input_tokens = mx.array(prompt_tokens)[None]
            return self.flash_model(input_tokens, cache=self._cache, **kwargs)

        logits = None
        for i in range(0, num_tokens, chunk_size):
            chunk = prompt_tokens[i : i + chunk_size]
            chunk_input = mx.array(chunk)[None]
            logits = self.flash_model(chunk_input, cache=self._cache, **kwargs)
            mx.eval(logits)
            mx.clear_cache()
        return logits

    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        tokens = self.tokenizer.encode(prompt)
        
        # Extract sampling parameters
        from mlx_lm.sample_utils import make_sampler
        sampler_args = {
            "temp": kwargs.pop("temp", kwargs.pop("temperature", 0.0)),
            "top_p": kwargs.pop("top_p", 1.0),
            "top_k": kwargs.pop("top_k", 0),
        }
        sampler = make_sampler(**sampler_args)
        
        # mlx_lm expects the model to be callable and return logits
        logits = self._chunked_prefill(tokens, **kwargs)
        
        # The first token is already computed in prefill logits
        y = mx.argmax(logits[:, -1, :], axis=-1)
        yield self.tokenizer.decode([y.item()])
        
        from mlx_lm.generate import generate_step
        
        for count, (token_id, _) in enumerate(generate_step(y, self.flash_model, prompt_cache=self._cache, sampler=sampler, **kwargs)):
            if count >= max_tokens - 1:
                break
            if token_id == self.tokenizer.eos_token_id:
                break
            yield self.tokenizer.decode([token_id])

    def shutdown(self):
        pass
