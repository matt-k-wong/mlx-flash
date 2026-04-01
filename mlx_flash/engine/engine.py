from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import gc

from ..config import FlashConfig
from .hooks import ExecutionContext, ExecutionGraph, InferenceHook
from .strategies import LayerStrategy, StandardStrategy

class StreamingProxy(nn.Module):
    """
    Wraps a single transformer layer to intercept execution.
    Forces synchronous evaluation to enable out-of-core weight streaming.
    """
    def __init__(self, layer: nn.Module, layer_idx: int, engine: 'FlashEngine'):
        super().__init__()
        # Use object.__setattr__ to avoid nn.Module dependency tracking
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "layer_idx", layer_idx)
        object.__setattr__(self, "engine", engine)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        # Create a transient execution context for hooks
        # Llama passes mask as second positional arg
        mask = args[0] if len(args) > 0 else kwargs.get("mask")
        cache_entry = args[1] if len(args) > 1 else kwargs.get("cache")

        ctx = ExecutionContext(self.engine, x, mask, cache_entry)
        ctx.layer_idx = self.layer_idx
        ctx.has_mask = mask is not None
        ctx.has_cache = cache_entry is not None
        ctx.cache_entry = cache_entry

        # 1. Trigger pre-layer hooks (e.g. Predictive Prefetch)
        self.engine.registry.dispatch("on_layer_start", ctx, self.layer)
        
        # 2. Execute layer logic via the assigned strategy
        # The PipeliningHook injects the correct strategy into the context
        strategy = ctx.metadata.get(f'strategy_{self.layer_idx}', self.engine.default_strategy)
        h = strategy.execute(ctx, self.layer)
        
        # 3. Trigger post-layer hooks
        self.engine.registry.dispatch("on_layer_end", ctx, self.layer)
        
        return h

    def __getattr__(self, name):
        return getattr(self.layer, name)

class FlashEngine(nn.Module):
    """
    A modular, hook-based orchestration engine for executing MLX models.
    Achieves quality by patching the original model's layers rather than 
    re-implementing the transformer loop.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, config: FlashConfig, model_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.registry = ExecutionGraph()
        self.model_path = Path(model_path) if model_path else None
        
        # Register standard hooks
        from .hooks import PipeliningHook, TilingHook, DiagnosticsHook
        self.registry.add_node(DiagnosticsHook(self.config))
        self.registry.add_node(PipeliningHook(self.config))
        self.registry.add_node(TilingHook(self.config))
        
        # 1. Structural Phase (Tiling, etc.)
        self.model = self.registry.dispatch_reduce("on_model_load", model)
        
        # 2. Patching Phase: Wrap layers in StreamingProxies
        inner = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
        layers_list = getattr(inner, "layers", getattr(inner, "h", getattr(inner, "blocks", [])))
        
        for i, layer in enumerate(layers_list):
            layers_list[i] = StreamingProxy(layer, i, self)
            
        self.layers = layers_list
        self._n_layers = len(self.layers)
        
        # The default strategy for math execution (handles sync + profiler intervals)
        self.default_strategy = StandardStrategy()
        
        # Reset profiler
        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().reset()
        except ImportError: pass

    def parameters(self): return self.model.parameters()
    def make_cache(self): return self.model.make_cache()

    def __call__(self, *args, **kwargs) -> mx.array:
        """
        Delegates back to the original model. 
        The patched layers will handle the synchronous streaming.
        """
        # We need to inform the registry about the generation lifecycle
        x = args[0] if len(args) > 0 else kwargs.get("x")
        ctx = ExecutionContext(self, x)
        
        if not hasattr(self, '_warmup_done'):
            self._warmup_done = True
            self._is_warmup = True
        else: self._is_warmup = False

        self.registry.dispatch("on_generation_start", ctx)
        
        # Execute original model logic (Embedding -> Patched Layers -> Norm -> Head)
        # Since we've patched self.model.layers IN PLACE, this call will
        # automatically use our StreamingProxies.
        logits = self.model(*args, **kwargs)
        
        self.registry.dispatch("on_generation_end", ctx)
        return logits

    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        """Provides the exact same generation interface as mlx_lm."""
        temp = kwargs.pop("temperature", kwargs.pop("temp", 0.0))
        kwargs["sampler"] = make_sampler(temp=temp)
        kwargs["max_tokens"] = max_tokens
        
        prompt_arr = mx.array(self.tokenizer.encode(prompt)) if isinstance(prompt, str) else mx.array(prompt)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        # generate_step will call our patched model's __call__
        for token, _ in generate_step(prompt_arr, self, **kwargs):
            tid = token.item() if hasattr(token, "item") else token
            detokenizer.add_token(tid)
            yield detokenizer.last_segment
            if tid == self.tokenizer.eos_token_id: break
        
        detokenizer.finalize()
        yield detokenizer.last_segment

    def shutdown(self):
        self.registry.dispatch("on_shutdown")
