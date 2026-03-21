
import pathlib

from ..config import FlashConfig

_ORIGINAL_LOAD = None
_ORIGINAL_SET_CACHE_LIMIT = None
_ORIGINAL_SET_WIRED_LIMIT = None

def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD
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
    
    if config.enabled:
        mx.metal.set_cache_limit = lambda *args, **kwargs: None
        mx.metal.set_wired_limit = lambda *args, **kwargs: None

    _ORIGINAL_LOAD = mlx_lm.load

    # Patch LOAD — the only patch needed.
    # stream_generate / generate work unchanged because FlashLLM is a
    # transparent nn.Module proxy.
    def _flash_load(path, *args, **kwargs):
        # NOTE: When Flash is active, FlashManager.load always uses lazy=True
        # regardless of the caller's kwargs — this is intentional, since Flash
        # mode requires lazy loading for mmap-based weight streaming.
        if not _should_use_flash(str(path), config):
            return _ORIGINAL_LOAD(path, *args, **kwargs)
        
        from ..manager import FlashManager
        mgr = FlashManager(config)
        model, tokenizer = mgr.load(path)
        # Use object.__setattr__ to bypass nn.Module.__setattr__,
        # which would try to register mgr as a submodule/parameter.
        object.__setattr__(model, "manager", mgr)
        return model, tokenizer

    mlx_lm.load = _flash_load  # type: ignore


def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD
    global _ORIGINAL_SET_CACHE_LIMIT, _ORIGINAL_SET_WIRED_LIMIT
    
    if _ORIGINAL_LOAD is None:
        return
    
    import mlx.core as mx
    import mlx_lm
    
    # Restore original functions
    mlx_lm.load = _ORIGINAL_LOAD  # type: ignore
        
    if _ORIGINAL_SET_CACHE_LIMIT:
        mx.metal.set_cache_limit = _ORIGINAL_SET_CACHE_LIMIT
    if _ORIGINAL_SET_WIRED_LIMIT:
        mx.metal.set_wired_limit = _ORIGINAL_SET_WIRED_LIMIT
    
    _ORIGINAL_LOAD = None
    _ORIGINAL_SET_CACHE_LIMIT = None
    _ORIGINAL_SET_WIRED_LIMIT = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    """Check if Flash Mode should be used for this model."""
    if config.enabled:
        return True
    # Check for Modelfile directives in the model directory
    p = pathlib.Path(model_path)
    for mf_name in ("Modelfile", "modelfile"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled
    return False
