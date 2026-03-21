
import functools
import pathlib
from typing import Any

from ..config import FlashConfig
from ..manager import FlashManager

_ORIGINAL_LOAD = None
_MANAGER: FlashManager | None = None


def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD, _MANAGER

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx_lm
    except ImportError as e:
        raise ImportError("mlx_lm not installed") from e

    if _ORIGINAL_LOAD is not None:
        if _MANAGER is not None: _MANAGER.config = config
        return

    _ORIGINAL_LOAD = mlx_lm.load
    _MANAGER = FlashManager(config)

    @functools.wraps(_ORIGINAL_LOAD)
    def _patched_load(model: str, *args: Any, **kwargs: Any) -> Any:
        should_flash = _should_use_flash(model, config)
        if not should_flash:
            return _ORIGINAL_LOAD(model, *args, **kwargs)
        return _MANAGER.load(model, load_fn=_ORIGINAL_LOAD, **kwargs)

    mlx_lm.load = _patched_load

    # === PRODUCTION GUARDRAILS (Option 1 - stable & honest) ===
    def _flash_stream_generate(model, tokenizer, prompt, *args, **kwargs):
        """Stable entry point that respects the documented prefill limitation."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        # Basic safety (keeps your existing low-RAM behavior)
        # We MUST use prefill_step_size=1 for large models on base Macs
        # to prevent 'GPU Timeout Error' during the prefill phase.
        kwargs["prefill_step_size"] = 1
        kwargs.setdefault("kv_bits", 4)

        # 1. Encode prompt to tokens
        if isinstance(prompt, str):
            token_list = tokenizer.encode(prompt)
        else:
            token_list = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
        
        original_len = len(token_list)

        # 2. APPLY GUARDRAIL
        current_config = _MANAGER.config if _MANAGER else config
        if original_len > current_config.max_safe_prefill_tokens and current_config.enable_prefill_guardrail:
            safe_len = current_config.max_safe_prefill_tokens
            print(f"\n[mlx-flash] ⚠️  PRE-FILL GUARDRAIL ACTIVATED")
            print(f"   Prompt: {original_len} tokens → using last {safe_len} tokens")
            print(f"   Reason: MLX builds full KV cache graph during prefill (see README 'Known Limitations')")
            print(f"   Recommendation: For true 2k+ context use llama.cpp GGUF backend instead")
            print(f"   (This is expected behavior for 30B-class models on 16 GB Macs)\n")
            token_list = token_list[-safe_len:]

        # Create a Response wrapper
        class Response:
            def __init__(self, text, token):
                self.text = text
                self.token = token

        # Convert to MLX array
        tokens_mx = mx.array(token_list)

        # Filter kwargs
        valid_args = {
            "max_tokens", "sampler", "logits_processors", "max_kv_size",
            "prompt_cache", "prefill_step_size", "kv_bits", "kv_group_size"
        }
        gen_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

        for token, _ in generate_step(tokens_mx, model, **gen_kwargs):
            text = tokenizer.decode([token.item()])
            yield Response(text, token.item())
            # Synchronous engine will handle layer sync, but we sync token too
            mx.synchronize()
            mx.clear_cache()

    mlx_lm.stream_generate = _flash_stream_generate

def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD, _MANAGER
    if _ORIGINAL_LOAD is None: return
    import mlx_lm
    mlx_lm.load = _ORIGINAL_LOAD
    if _MANAGER: _MANAGER.shutdown()
    _ORIGINAL_LOAD = None
    _MANAGER = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    if config.enabled: return True
    p = pathlib.Path(model_path)
    for mf_name in ("Modelfile", "modelfile"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled
    return config.enabled
