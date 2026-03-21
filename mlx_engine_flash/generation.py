import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Generator, Dict
import mlx_lm
from .config import FlashConfig

class FlashLLM(nn.Module):
    """
    Wrapper for MLX models that forces synchronous per-layer evaluation.
    This bypasses the 'lazy graph accumulation' problem that causes OOM.
    """
    def __init__(self, base_model: nn.Module, config: FlashConfig):
        super().__init__()
        self.model = base_model
        self.config = config

    def __call__(
        self,
        x: mx.array,
        cache: Optional[list[Any]] = None,
        **kwargs
    ) -> mx.array:
        # Access the backbone (most mlx_lm models wrap the actual transformer in .model)
        backbone = getattr(self.model, "model", self.model)
        
        # 1. Embedding
        x = backbone.embed_tokens(x)
        
        # 2. Sequential Layers
        mask = kwargs.get("mask")
        num_layers = len(backbone.layers)
        
        from .page_cache import prefetch_array
        
        for i, layer in enumerate(backbone.layers):
            # Prefetch the NEXT layer while THIS one is still logic (lazy)
            if i + 1 < num_layers:
                next_layer = backbone.layers[i + 1]
                # Each parameter is an mx.array
                for p in next_layer.parameters().values():
                    if isinstance(p, mx.array):
                        prefetch_array(p)
                    elif isinstance(p, dict): # handle nested params (MoE)
                        for sp in p.values():
                            if isinstance(sp, mx.array):
                                prefetch_array(sp)
            
            l_cache = cache[i] if cache else None
            
            # Forward pass for one layer
            x = layer(x, mask=mask, cache=l_cache)
            
            # Force materialisation of this layer's output NOW
            mx.eval(x)
            
            # If we have a cache, evaluate it too to ensure keys/values are computed
            if l_cache is not None:
                # Modern MLX-LM cache objects have .keys and .values arrays
                mx.eval(l_cache.keys, l_cache.values)
            
            # Clear Metal pool memory to release weight buffers for this layer
            mx.clear_cache()
            
            if self.config.debug and i % 10 == 0:
                print(f"[flash] Layer {i} evaluated. Metal active: {mx.get_active_memory()/1e6:.1f} MB")

        # 3. Final Norm + Head
        x = backbone.norm(x)
        return self.model.lm_head(x) if hasattr(self.model, "lm_head") else backbone.lm_head(x)

class FlashGenerationLoop:
    """
    High-level generator for flashweight streaming using the FlashLLM wrapper.
    """
    def __init__(self, model_path: str, config: FlashConfig):
        self.config = config
        # Native MLX-LM load with luxury lazy=True
        self.model, self.tokenizer = mlx_lm.load(model_path, {"lazy": True})
        self.flash_model = FlashLLM(self.model, config)

    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        # Simple generation loop using mlx_lm.generate_step
        # Note: We must ensure generate_step uses our wrapped model
        
        # We can't easily replace the model inside mlx_lm.generate_step without 
        # monkeypadding or rewriting it. For now, we use a simple loop.
        
        tokens = mx.array(self.tokenizer.encode(prompt))
        # Use newer mlx_lm cache structure
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in range(len(self.model.layers))]
        
        count = 0
        input_tokens = tokens[None]
        
        while count < max_tokens:
            logits = self.flash_model(input_tokens, cache=cache, **kwargs)
            # Take last logit for next token
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            
            yield self.tokenizer.decode(next_token.tolist())
            
            input_tokens = next_token[None]
            count += 1
            
            if next_token.tolist()[0] == self.tokenizer.eos_token_id:
                break

    def shutdown(self):
        # Nothing to manually close; OS handles mmap cleanup
        pass
