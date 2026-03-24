from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import gc

from ..config import FlashConfig
from .hooks import ExecutionContext, HookRegistry
from .strategies import LayerStrategy, StandardStrategy

class FlashEngine:
    """
    Replaces FlashGenerationLoop. 
    A modular, hook-based orchestration engine for executing MLX models.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, config: FlashConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.registry = HookRegistry()
        
        # 1. Structural Phase: Allow hooks to safely rewrite the model tree
        # (e.g. replacing nn.Linear with TiledLinear)
        self.model = self.registry.dispatch_reduce("on_model_load", model)
        
        self._n_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        
        # We store layer signatures to avoid introspection in the hot loop
        self._layer_sigs = self._inspect_layers()
        
        # The default strategy. Complex strategies (Pipelined, MoE) are assigned 
        # dynamically by the execution logic or via hooks.
        self.default_strategy = StandardStrategy()

    def _inspect_layers(self) -> list:
        sigs = []
        layers = self.model.layers if hasattr(self.model, "layers") else self.model.model.layers
        for layer in layers:
            import inspect
            sig = inspect.signature(layer.__call__)
            params = sig.parameters
            has_mask = "mask" in params or "attention_mask" in params
            has_cache = "cache" in params
            sigs.append((layer, has_mask, has_cache))
        return sigs

    def pre_layer_fn(self, x: mx.array) -> mx.array:
        """Embeddings and initial processing before the transformer stack."""
        # This mirrors the logic from FlashLLM
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            x = self.model.model.embed_tokens(x)
            
        return x

    def forward_loop(self, x: mx.array, mask: mx.array = None, cache=None) -> mx.array:
        """
        The core generation loop. Cleanly dispatches events to registered hooks
        and delegates mathematical execution to the Strategy.
        """
        import gc
        gc.collect()
        mx.clear_cache()

        ctx = ExecutionContext(self, x, mask, cache)
        
        ctx.x = self.pre_layer_fn(ctx.x)
        if isinstance(ctx.mask, str): ctx.mask = None
        
        # Determine the pipelining depth constraint based on phase
        is_decode = ctx.x.shape[1] <= 1 if hasattr(ctx.x, "shape") else True
        ctx.metadata['pipeline_depth'] = getattr(self.config, 'pipeline_depth', 4) if is_decode else 1
        
        self.registry.dispatch("on_generation_start", ctx)
        
        layers = self.model.layers if hasattr(self.model, "layers") else self.model.model.layers
        cache_ptr = 0

        for i, layer in enumerate(layers):
            ctx.layer_idx = i
            _, ctx.has_mask, ctx.has_cache = self._layer_sigs[i]
            
            # Resolve Cache Entry
            ctx.cache_entry = None
            if ctx.cache is not None and ctx.has_cache:
                if hasattr(ctx.cache, "__getitem__"):
                    try:
                        if cache_ptr < len(ctx.cache):
                            ctx.cache_entry = ctx.cache[cache_ptr]
                            cache_ptr += 1
                    except Exception:
                        ctx.cache_entry = ctx.cache
                else:
                    ctx.cache_entry = ctx.cache

            # 1. Trigger pre-layer hooks (e.g. Prefetch N layers ahead)
            self.registry.dispatch("on_layer_start", ctx, layer)
            
            t0 = time.perf_counter()
            
            # 2. Mathematical Execution
            # In a fully migrated setup, the Strategy might be swapped dynamically per layer here.
            # We use the default strategy (or a pipelined one if configured globally).
            strategy = self.metadata.get('strategy', self.default_strategy) if hasattr(self, 'metadata') else self.default_strategy
            
            # For this exact translation, we'll allow the context to override the strategy 
            # if a hook injected a specialized one (like MoE).
            active_strategy = ctx.metadata.get(f'strategy_{i}', strategy)
            
            ctx.x = active_strategy.execute(ctx, layer)
            
            t1 = time.perf_counter()
            ctx.metadata['compute_time'] = t1 - t0
            
            # 3. Trigger post-layer hooks (e.g. Memory Eviction, Profiler update)
            self.registry.dispatch("on_layer_end", ctx, layer)
            
        self.registry.dispatch("on_generation_end", ctx)
        
        # Post-transformer normalization and LM head
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            ctx.x = self.model.model.norm(ctx.x)
        if hasattr(self.model, "lm_head"):
            ctx.x = self.model.lm_head(ctx.x)
            
        return ctx.x

    def __call__(self, *args, **kwargs) -> mx.array:
        """Wrapper to make FlashEngine compatible with standard MLX `model(x)` calls."""
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
        
        return self.forward_loop(x, mask, cache)

    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        """Provides the exact same generation interface as mlx_lm."""
        temp = kwargs.pop("temperature", kwargs.pop("temp", 0.0))
        kwargs["sampler"] = make_sampler(temp=temp)
        if "prefill_step_size" not in kwargs:
            kwargs["prefill_step_size"] = getattr(self.config, "prefill_chunk_size", 32)

        gc.collect()
        mx.clear_cache()

        prompt_arr = mx.array(self.tokenizer.encode(prompt)) if isinstance(prompt, str) else mx.array(prompt)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for token, _ in generate_step(prompt_arr, self, **kwargs):
            self.registry.dispatch("on_token_generated")
            
            tid = token.item() if hasattr(token, "item") else token
            detokenizer.add_token(tid)
            if detokenizer.last_segment: yield detokenizer.last_segment
            gc.collect()
            mx.clear_cache()
            if tid == self.tokenizer.eos_token_id: break
        
        detokenizer.finalize()
        if detokenizer.last_segment: yield detokenizer.last_segment

    def shutdown(self):
        self.registry.dispatch("on_shutdown")
        mx.clear_cache()
