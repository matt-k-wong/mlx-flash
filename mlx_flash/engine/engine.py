from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import gc

from ..config import FlashConfig
from .hooks import ExecutionContext, ExecutionGraph
from .strategies import LayerStrategy, StandardStrategy

class FlashEngine:
    """
    Replaces FlashGenerationLoop. 
    A modular, hook-based orchestration engine for executing MLX models.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, config: FlashConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.registry = ExecutionGraph()
        
        # Register standard hooks based on configuration
        from .hooks import PipeliningHook, TilingHook, DiagnosticsHook
        # Order of addition no longer matters due to explicit topological sorting
        self.registry.add_node(DiagnosticsHook(self.config))
        self.registry.add_node(PipeliningHook(self.config))
        self.registry.add_node(TilingHook(self.config))
        
        # 1. Structural Phase: Allow hooks to safely rewrite the model tree
        # (e.g. replacing nn.Linear with TiledLinear)
        self.model = self.registry.dispatch_reduce("on_model_load", model)
        
        # Proxy standard MLX properties for compatibility with generation scripts
        self.layers = self.model.layers if hasattr(self.model, "layers") else getattr(self.model, "model", self.model).layers
        self._n_layers = len(self.layers)
        
        # We store layer signatures to avoid introspection in the hot loop
        self._layer_sigs = self._inspect_layers()
        
        # The default strategy. Complex strategies (Pipelined, MoE) are assigned 
        # dynamically by the execution logic or via hooks.
        self.default_strategy = StandardStrategy()
        
        # Reset profiler
        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().reset()
        except ImportError:
            pass

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
        # Find embedding layer robustly
        inner = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
        embed = getattr(inner, "embed_tokens", getattr(inner, "wte", getattr(inner, "embeddings", None)))
        
        if embed is not None:
            x = embed(x)
            
        return x

    def forward_loop(self, x: mx.array, mask: mx.array = None, cache=None) -> mx.array:
        """
        The core generation loop. Cleanly dispatches events to registered hooks
        and delegates mathematical execution to the Strategy.
        """
        import gc
        gc.collect()

        ctx = ExecutionContext(self, x, mask, cache)
        
        ctx.x = self.pre_layer_fn(ctx.x)
        if isinstance(ctx.mask, str): ctx.mask = None
        
        # Determine the pipelining depth constraint based on phase
        is_decode = ctx.x.shape[1] <= 1 if hasattr(ctx.x, "shape") else True
        ctx.metadata['pipeline_depth'] = getattr(self.config, 'pipeline_depth', 4) if is_decode else 1
        
        if not hasattr(self, '_warmup_done'):
            self._warmup_done = True
            self._is_warmup = True
        else:
            self._is_warmup = False
        
        self.registry.dispatch("on_generation_start", ctx)
        
        layers = self.layers
        cache_ptr = 0

        for i, layer in enumerate(layers):
            ctx.layer_idx = i
            _, ctx.has_mask, ctx.has_cache = self._layer_sigs[i]
            
            # Notify bandwidth controller about layer transition for MPC prediction
            mmap_cache = getattr(self, 'mmap_cache', None)
            if mmap_cache is None and hasattr(self.model, 'manager'):
                 mmap_cache = getattr(self.model.manager.model, 'mmap_cache', None)
            
            if mmap_cache and hasattr(mmap_cache, 'prefetch_worker'):
                controller = getattr(mmap_cache.prefetch_worker, 'bandwidth_controller', None)
                if controller and hasattr(controller, 'notify_layer_start'):
                    controller.notify_layer_start(i)

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
            strategy = ctx.metadata.get('strategy', self.default_strategy)
            
            # For this exact translation, we'll allow the context to override the strategy 
            # if a hook injected a specialized one (like MoE).
            active_strategy = ctx.metadata.get(f'strategy_{i}', strategy)
            
            ctx.x = active_strategy.execute(ctx, layer)
            
            t1 = time.perf_counter()
            compute_time = t1 - t0
            ctx.metadata['compute_time'] = compute_time
            
            mmap_cache = getattr(self, 'mmap_cache', None)
            if mmap_cache is None and hasattr(self.model, 'manager'):
                 mmap_cache = getattr(self.model.manager.model, 'mmap_cache', None)
            
            if mmap_cache and hasattr(mmap_cache, 'prefetch_worker'):
                controller = getattr(mmap_cache.prefetch_worker, 'bandwidth_controller', None)
                if controller and hasattr(controller, 'register_compute_time'):
                    controller.register_compute_time(i, compute_time)
            
            # 3. Trigger post-layer hooks (e.g. Memory Eviction, Profiler update)
            self.registry.dispatch("on_layer_end", ctx, layer)
            
        self.registry.dispatch("on_generation_end", ctx)
        
        # Post-transformer normalization and LM head discovery robustly
        inner = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
        norm = getattr(inner, "norm", getattr(inner, "ln_f", None))
        if norm is not None:
            ctx.x = norm(ctx.x)
            
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            ctx.x = lm_head(ctx.x)
            
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

        prompt_arr = mx.array(self.tokenizer.encode(prompt)) if isinstance(prompt, str) else mx.array(prompt)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for token, _ in generate_step(prompt_arr, self, **kwargs):
            self.registry.dispatch("on_token_generated")
            
            tid = token.item() if hasattr(token, "item") else token
            detokenizer.add_token(tid)
            if detokenizer.last_segment: yield detokenizer.last_segment
            gc.collect()
            if tid == self.tokenizer.eos_token_id: break
        
        detokenizer.finalize()
        if detokenizer.last_segment: yield detokenizer.last_segment

    def shutdown(self):
        self.registry.dispatch("on_shutdown")
