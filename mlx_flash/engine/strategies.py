import mlx.core as mx
import mlx.nn as nn
from .hooks import InferenceHook, ExecutionContext
import time

class LayerStrategy:
    """Defines how a layer mathematically executes."""
    def execute(self, ctx: ExecutionContext, layer: nn.Module) -> mx.array:
        raise NotImplementedError

class StandardStrategy(LayerStrategy):
    """
    The baseline execution model. 
    It dispatches the entire layer mathematically to MLX and waits for completion.
    """
    def execute(self, ctx: ExecutionContext, layer: nn.Module) -> mx.array:
        t0 = time.perf_counter()

        # Standard MLX calling convention
        kwargs = {}
        if ctx.has_mask: kwargs["mask"] = ctx.mask
        if ctx.has_cache: kwargs["cache"] = ctx.cache_entry

        h = layer(ctx.x, **kwargs)

        # Handle potential tuple returns if cache is used (though TransformerBlock usually returns array)
        if isinstance(h, (list, tuple)):
            h = h[0]

        mx.eval(h)
        mx.synchronize()
        t1 = time.perf_counter()

        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().record_compute_interval(t0, t1, f"layer_{ctx.layer_idx}")
        except ImportError:
            pass

        return h
class PipelinedDenseStrategy(LayerStrategy):
    """
    Wraps the PipelinedExecutor to execute the layer in overlapping phases.
    """
    def __init__(self, pipelined_executor):
        self.executor = pipelined_executor
        
    def execute(self, ctx: ExecutionContext, layer: nn.Module) -> mx.array:
        if ctx.engine.config.debug:
            print(f"[flash] Executing PipelinedDenseStrategy for L{ctx.layer_idx}")
        # Note: We pass mask and cache straight into the executor's dense logic.
        return self.executor.execute_dense_layer(
            ctx.x, layer, ctx.layer_idx, 
            mask=ctx.mask if ctx.has_mask else None, 
            cache=ctx.cache_entry if ctx.has_cache else None
        )

class PipelinedMoEStrategy(LayerStrategy):
    """
    Wraps the PipelinedExecutor specifically for speculative MoE routing.
    """
    def __init__(self, pipelined_executor, moe_prefetcher):
        self.executor = pipelined_executor
        self.moe_prefetcher = moe_prefetcher
        
    def execute(self, ctx: ExecutionContext, layer: nn.Module) -> mx.array:
        return self.executor.execute_moe_layer(
            ctx.x, layer, ctx.layer_idx, self.moe_prefetcher,
            mask=ctx.mask if ctx.has_mask else None, 
            cache=ctx.cache_entry if ctx.has_cache else None
        )
