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
