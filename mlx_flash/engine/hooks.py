from typing import Any, Dict, List
import mlx.core as mx
import mlx.nn as nn

class ExecutionContext:
    """Carries state through the generation loop without polluting method signatures."""
    def __init__(self, engine, x: mx.array, mask: mx.array = None, cache=None):
        self.engine = engine
        self.x = x
        self.mask = mask
        self.cache = cache
        self.layer_idx = 0
        self.metadata: Dict[str, Any] = {} # For hooks to share data
        
        # Extracted signatures from the engine for convenience
        self.has_mask = False
        self.has_cache = False
        self.cache_entry = None

class InferenceHook:
    """Base class for all lifecycle side-effects."""
    
    # 1. Structural Phase (Initialization)
    def on_model_load(self, model: nn.Module) -> nn.Module:
        """Replace monkey-patching. e.g., Tiling modifies the model here."""
        return model

    # 2. Generation Phase
    def on_generation_start(self, ctx: ExecutionContext): pass
    def on_generation_end(self, ctx: ExecutionContext): pass

    # 3. Layer Loop Phase
    def on_layer_start(self, ctx: ExecutionContext, layer: nn.Module):
        """Ideal for triggering IO prefetch N layers ahead."""
        pass
        
    def on_layer_end(self, ctx: ExecutionContext, layer: nn.Module):
        """Ideal for triggering MADV_DONTNEED cache eviction or profiling."""
        pass
        
    # 4. Sub-component Phase (for Pipelined execution)
    def on_router_decision(self, ctx: ExecutionContext, top_k_indices: list):
        """Ideal for speculative MoE prefetching."""
        pass

class HookRegistry:
    def __init__(self):
        self.hooks: List[InferenceHook] = []
        
    def register(self, hook: InferenceHook):
        self.hooks.append(hook)
        
    def dispatch(self, event_name: str, *args, **kwargs):
        """Dispatches an event to all registered hooks."""
        for hook in self.hooks:
            method = getattr(hook, event_name, None)
            if method:
                method(*args, **kwargs)
                
    def dispatch_reduce(self, event_name: str, initial_value: Any, *args, **kwargs) -> Any:
        """Dispatches an event, threading the return value through all hooks."""
        value = initial_value
        for hook in self.hooks:
            method = getattr(hook, event_name, None)
            if method:
                value = method(value, *args, **kwargs)
        return value

class PipeliningHook(InferenceHook):
    """
    Detects model type and configures the appropriate Pipelined strategy.
    """
    def __init__(self, config):
        self.config = config
        self._executor = None
        self._moe_prefetcher = None
        
    def on_generation_start(self, ctx: ExecutionContext):
        if not self.config.pipelined_execution:
            return
            
        # Initialize Executor and Prefetchers lazily
        if self._executor is None:
            from .strategies import PipelinedDenseStrategy, PipelinedMoEStrategy
            from ..pipeline.executor import PipelinedExecutor
            
            # Find the mmap_cache from the model or engine
            mmap_cache = getattr(ctx.engine, 'mmap_cache', None)
            if mmap_cache is None and hasattr(ctx.engine.model, 'manager'):
                 mmap_cache = getattr(ctx.engine.model.manager.model, 'mmap_cache', None)
            
            self._executor = PipelinedExecutor(mmap_cache)
            
            # Setup MoE Prefetcher if needed
            from ..moe.manager import MoEPrefetcher, ExpertCache
            # We need a cache for the prefetcher
            moe_cache = ExpertCache(max_experts=8) # Placeholder
            self._moe_prefetcher = MoEPrefetcher(mmap_cache.prefetch_worker if mmap_cache else None, moe_cache)
            
            self._dense_strategy = PipelinedDenseStrategy(self._executor)
            self._moe_strategy = PipelinedMoEStrategy(self._executor, self._moe_prefetcher)

        # Assign strategies per layer
        for i, layer in enumerate(ctx.engine.layers):
            # Detect MoE layer
            is_moe = False
            mlp = getattr(layer, "mlp", getattr(layer, "mixer", None))
            if mlp is not None and (hasattr(mlp, "gate") or hasattr(mlp, "router")):
                is_moe = True
            
            strategy = self._moe_strategy if is_moe else self._dense_strategy
            ctx.metadata[f'strategy_{i}'] = strategy

    def on_layer_start(self, ctx: ExecutionContext, layer: nn.Module):
        if self._executor:
            # Enqueue next layer's weights if we are pipelining
            # This is a simple lookahead. Real system might do N layers.
            next_idx = ctx.layer_idx
            self._executor._enqueue_tensor(next_idx, "all")

class TilingHook(InferenceHook):
    """
    Applies tiled execution to the model layers to bound peak memory usage.
    """
    def __init__(self, config):
        self.config = config
        
    def on_model_load(self, model: nn.Module) -> nn.Module:
        if not self.config.tiled_execution:
            return model
            
        from ..tiled import apply_tiling
        apply_tiling(model, tile_size=self.config.tile_size)
        return model

class DiagnosticsHook(InferenceHook):
    """
    Automates profiling and bottleneck analysis.
    """
    def __init__(self, config):
        self.config = config

    def on_generation_start(self, ctx: ExecutionContext):
        pass

    def on_generation_end(self, ctx: ExecutionContext):
        if getattr(self.config, 'debug', False):
            from benchmarks.profiler.profiler import StreamingProfiler
            prof = StreamingProfiler()
            print(f"[flash] End of generation. Intervals: IO={len(prof.io_intervals)}, Comp={len(prof.compute_intervals)}")
            print(prof.analyze_bottlenecks())
            prof.print_waterfall()

