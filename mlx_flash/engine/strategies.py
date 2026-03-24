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
        call_kwargs = {}
        if ctx.has_mask: call_kwargs["mask"] = ctx.mask
        if ctx.has_cache: call_kwargs["cache"] = ctx.cache_entry
        
        output = layer(ctx.x, **call_kwargs)
        h = output[0] if (isinstance(output, (list, tuple)) and len(output) == 2) else output
        
        # Materialize
        if ctx.cache_entry is not None:
            if hasattr(ctx.cache_entry, "state") and ctx.cache_entry.state is not None:
                mx.eval(h, *[s for s in ctx.cache_entry.state if s is not None])
            elif hasattr(ctx.cache_entry, "keys") and ctx.cache_entry.keys is not None:
                mx.eval(h, ctx.cache_entry.keys, ctx.cache_entry.values)
            else: mx.eval(h)
        else: mx.eval(h)
        
        mx.synchronize()
        return h

class PipelinedDenseStrategy(LayerStrategy):
    """
    Wraps the PipelinedExecutor to execute the layer in overlapping phases.
    """
    def __init__(self, pipelined_executor):
        self.executor = pipelined_executor
        
    def execute(self, ctx: ExecutionContext, layer: nn.Module) -> mx.array:
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
