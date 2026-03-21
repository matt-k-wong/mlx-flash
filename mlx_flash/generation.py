import inspect
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


class FlashLLM(nn.Module):
    """
    Wraps any mlx-lm Model to execute layers synchronously.
    
    Strategy: intercept execution at the layer level to iterate layers one
    at a time with forced mx.eval() between each, rather than building
    one unified lazy graph.
    """
    
    def __init__(self, model: nn.Module, config: FlashConfig):
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_layers", self._find_layers(model))
        object.__setattr__(self, "_n_layers", len(self._layers))
        object.__setattr__(self, "_pre_layer_fn", self._build_pre_layer_fn(model))
        object.__setattr__(self, "_post_layer_fn", self._build_post_layer_fn(model))
        object.__setattr__(self, "_layer_sigs", self._cache_layer_signatures())
        object.__setattr__(self, "disk_cache", None)
    
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
            f"Attributes: {dir(model)}"
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
    
    def _cache_layer_signatures(self) -> list[tuple[bool, bool, bool]]:
        """Pre-compute (is_mamba, has_mask, has_cache) per layer."""
        sigs = []
        for layer in self._layers:
            is_mamba = hasattr(layer, "mixer") and hasattr(layer.mixer, "ssm")
            if is_mamba:
                sigs.append((True, False, False))
            else:
                params = inspect.signature(layer.__call__).parameters
                sigs.append((False, "mask" in params, "cache" in params))
        return sigs
    
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
        
        # 1. Fix the Mask Problem (Correctness for prefill T > 1)
        if mask is None and cache is not None:
            mask = create_attention_mask(h, cache)
            
        # Per-layer synchronous execution
        for i, layer in enumerate(self._layers):
            cache_entry = cache[i] if cache is not None else None
            is_mamba, has_mask, has_cache = self._layer_sigs[i]
            
            # Run this layer (builds a small graph for ONE layer)
            if is_mamba:
                # Mamba layers have different cache structure (state)
                h, cache_entry = layer(h, cache_entry)
                if cache_entry is not None:
                    mx.eval(h, *cache_entry)
            else:
                if has_mask and has_cache:
                    h = layer(h, mask=mask, cache=cache_entry)
                elif has_cache:
                    h = layer(h, cache=cache_entry)
                else:
                    h = layer(h)
                
                # CRITICAL: materialise NOW before the next layer's graph is built
                if cache_entry is not None:
                    # Modern MLX-LM cache objects have .keys and .values
                    mx.eval(h, cache_entry.keys, cache_entry.values)
                else:
                    mx.eval(h)
            
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
        elif isinstance(model_or_path, FlashLLM):
            self.flash_model = model_or_path
            self.model = self.flash_model._model
            self.tokenizer = tokenizer
        else:
            # Assume it is a base nn.Module from mlx_lm.load
            self.model = model_or_path
            self.tokenizer = tokenizer
            self.flash_model = FlashLLM(self.model, config)
            
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

        for result in mlx_lm.stream_generate(
            self.flash_model, self.tokenizer, prompt,
            max_tokens=max_tokens, **kwargs,
        ):
            yield result.text

    def shutdown(self):
        """Clean up resources."""
        import contextlib
        with contextlib.suppress(AttributeError, Exception):
            mx.metal.clear_cache()
        self.flash_model = None
        self.model = None
        self.tokenizer = None
