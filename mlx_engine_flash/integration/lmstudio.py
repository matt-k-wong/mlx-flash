
import pathlib

from ..config import FlashConfig

_ORIGINAL_LOAD = None
_ORIGINAL_STREAM_GENERATE = None
_ORIGINAL_GENERATE = None
_LAST_LOOP = None

def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError as e:
        raise ImportError("mlx_lm not installed") from e

    if _ORIGINAL_LOAD is not None:
        return

    # LOCK DOWN METAL LIMITS
    mx.metal.set_cache_limit = lambda *args, **kwargs: None
    mx.metal.set_wired_limit = lambda *args, **kwargs: None

    _ORIGINAL_LOAD = mlx_lm.load
    _ORIGINAL_STREAM_GENERATE = mlx_lm.stream_generate
    _ORIGINAL_GENERATE = getattr(mlx_lm, "generate", None)

    # 1. Patch LOAD
    def _flash_load(path, *args, **kwargs):
        if not _should_use_flash(str(path), config):
            return _ORIGINAL_LOAD(path, *args, **kwargs)
        
        from ..manager import FlashManager
        
        # We use FlashManager to handle the lazy load and wrapping
        mgr = FlashManager(config)
        model, tokenizer = mgr.load(path)
        return model, tokenizer

    mlx_lm.load = _flash_load

    # 2. Patch stream_generate
    def _flash_stream_generate(model, tokenizer, prompt, **kwargs):
        from ..generation import FlashLLM
        if isinstance(model, FlashLLM):
            # We can't use mlx_lm.stream_generate directly because it expects 
            # the base model and builds a full graph.
            # We must use our custom FlashGenerationLoop or a simplified version.
            # Note: This is slightly suboptimal as it re-loads or wraps.
            # In a real integration, we'd use the FlashLLM directly.
            # For now, we manually iterate to keep it drop-in.
            from mlx_lm.models.cache import KVCache
            cache = [KVCache() for _ in range(len(model.model.layers))]
            
            import mlx.core as mx
            tokens = mx.array(tokenizer.encode(prompt))
            input_tokens = tokens[None]
            max_tokens = kwargs.get("max_tokens", 100)
            count = 0
            while count < max_tokens:
                logits = model(input_tokens, cache=cache, **kwargs)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
                token_id = next_token.tolist()[0]
                yield token_id
                input_tokens = next_token[None]
                count += 1
                if token_id == tokenizer.eos_token_id:
                    break
        else:
            yield from _ORIGINAL_STREAM_GENERATE(model, tokenizer, prompt, **kwargs)

    mlx_lm.stream_generate = _flash_stream_generate

    # 3. Patch generate
    if _ORIGINAL_GENERATE:
        def _flash_generate(model, tokenizer, prompt, **kwargs):
            from ..generation import FlashLLM
            if isinstance(model, FlashLLM):
                tokens = []
                for token in _flash_stream_generate(model, tokenizer, prompt, **kwargs):
                    tokens.append(token)
                return tokenizer.decode(tokens)
            else:
                return _ORIGINAL_GENERATE(model, tokenizer, prompt, **kwargs)
        mlx_lm.generate = _flash_generate


def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE, _LAST_LOOP
    if _ORIGINAL_LOAD is None:
        return
    
    import mlx_lm
    mlx_lm.load = _ORIGINAL_LOAD
    mlx_lm.stream_generate = _ORIGINAL_STREAM_GENERATE
    if _ORIGINAL_GENERATE:
        mlx_lm.generate = _ORIGINAL_GENERATE
        
    if _LAST_LOOP and hasattr(_LAST_LOOP, "manager"):
        _LAST_LOOP.manager.shutdown()
    
    _ORIGINAL_LOAD = None
    _ORIGINAL_STREAM_GENERATE = None
    _ORIGINAL_GENERATE = None
    _LAST_LOOP = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    if config.enabled:
        return True
    p = pathlib.Path(model_path)
    for mf_name in ("Modelfile", "modelfile"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled
    return config.enabled
