
import pathlib

from ..config import FlashConfig

_ORIGINAL_LOAD = None
_ORIGINAL_STREAM_GENERATE = None
_ORIGINAL_GENERATE = None
_ORIGINAL_SET_CACHE_LIMIT = None
_ORIGINAL_SET_WIRED_LIMIT = None
_LAST_LOOP = None

def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE
    global _ORIGINAL_SET_CACHE_LIMIT, _ORIGINAL_SET_WIRED_LIMIT

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError as e:
        raise ImportError("mlx_lm not installed") from e

    if _ORIGINAL_LOAD is not None:
        return

    # LOCK DOWN METAL LIMITS (and save originals)
    _ORIGINAL_SET_CACHE_LIMIT = mx.metal.set_cache_limit
    _ORIGINAL_SET_WIRED_LIMIT = mx.metal.set_wired_limit
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
        mgr = FlashManager(config)
        model, tokenizer = mgr.load(path)
        # Store manager on model for later shutdown
        model.manager = mgr
        return model, tokenizer

    mlx_lm.load = _flash_load

    # 2. Patch stream_generate
    def _flash_stream_generate(model, tokenizer, prompt, **kwargs):
        from ..generation import FlashGenerationLoop, FlashLLM
        global _LAST_LOOP
        
        if isinstance(model, FlashLLM):
            # Use the production generation loop for proper decoding and sampling
            loop = FlashGenerationLoop(model, tokenizer, config)
            _LAST_LOOP = loop
            yield from loop.stream_generate(prompt, **kwargs)
        else:
            yield from _ORIGINAL_STREAM_GENERATE(model, tokenizer, prompt, **kwargs)

    mlx_lm.stream_generate = _flash_stream_generate

    # 3. Patch generate
    if _ORIGINAL_GENERATE:
        def _flash_generate(model, tokenizer, prompt, **kwargs):
            from ..generation import FlashLLM
            if isinstance(model, FlashLLM):
                # stream_generate now returns text
                full_text = ""
                for chunk in _flash_stream_generate(model, tokenizer, prompt, **kwargs):
                    full_text += chunk
                return full_text
            else:
                return _ORIGINAL_GENERATE(model, tokenizer, prompt, **kwargs)
        mlx_lm.generate = _flash_generate


def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE
    global _ORIGINAL_SET_CACHE_LIMIT, _ORIGINAL_SET_WIRED_LIMIT, _LAST_LOOP
    
    if _ORIGINAL_LOAD is None:
        return
    
    import mlx.core as mx
    import mlx_lm
    
    # Restore original functions
    mlx_lm.load = _ORIGINAL_LOAD
    mlx_lm.stream_generate = _ORIGINAL_STREAM_GENERATE
    if _ORIGINAL_GENERATE:
        mlx_lm.generate = _ORIGINAL_GENERATE
        
    if _ORIGINAL_SET_CACHE_LIMIT:
        mx.metal.set_cache_limit = _ORIGINAL_SET_CACHE_LIMIT
    if _ORIGINAL_SET_WIRED_LIMIT:
        mx.metal.set_wired_limit = _ORIGINAL_SET_WIRED_LIMIT
        
    if _LAST_LOOP and hasattr(_LAST_LOOP, "model") and hasattr(_LAST_LOOP.model, "manager"):
        _LAST_LOOP.model.manager.shutdown()
    
    _ORIGINAL_LOAD = None
    _ORIGINAL_STREAM_GENERATE = None
    _ORIGINAL_GENERATE = None
    _ORIGINAL_SET_CACHE_LIMIT = None
    _ORIGINAL_SET_WIRED_LIMIT = None
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
