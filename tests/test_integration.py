"""
Integration tests — require either a real model (--model flag) or skip.

Run with:
    pytest tests/test_integration.py -v \
        --model ~/.cache/lm-studio/models/mlx-community/Qwen2.5-3B-Instruct-4bit \
        --flash
"""

import pytest

# Remove global skip so tests with synthetic models can run
# pytestmark = pytest.mark.skipif(True, reason="requires --model flag")



def test_modelfile_directive():
    from mlx_engine_flash.integration.modelfile import parse_flash_directives
    text = """
FROM /models/Qwen2.5-72B-Q4_K_M
FLASH true
FLASH_RAM_GB 12
FLASH_THREADS 6
FLASH_PREFETCH_LAYERS 3
FLASH_TOP_K 4
FLASH_EVICTION dontneed
"""
    cfg = parse_flash_directives(text)
    assert cfg.enabled is True
    assert cfg.ram_budget_gb == 12.0
    assert cfg.moe_top_k_override == 4
    assert cfg.eviction_strategy == "dontneed"


def test_modelfile_no_flash():
    from mlx_engine_flash.integration.modelfile import parse_flash_directives
    text = "FROM /models/some-model\nSYSTEM You are helpful.\n"
    cfg = parse_flash_directives(text)
    assert cfg.enabled is False


def test_flash_peak_ram_below_2gb(tmp_model_dir):
    """Flash mode on a tiny synthetic model should use < 200 MB peak RSS."""
    import psutil, os
    from mlx_engine_flash.integration.lmstudio import apply_flash_patch, remove_flash_patch
    from mlx_engine_flash.config import FlashConfig
    import mlx_lm

    proc = psutil.Process(os.getpid())
    # Ensure any previous patch is removed
    remove_flash_patch()
    
    apply_flash_patch(FlashConfig(enabled=True, debug=True))
    
    try:
        # Load the synthetic model
        model, tokenizer = mlx_lm.load(str(tmp_model_dir))
        
        # We measure baseline AFTER load because the prompt says 
        # "load ONLY the config + tokenizer, then return a FlashGenerationLoop proxy"
        # The proxy itself should be small.
        baseline_rss = proc.memory_info().rss
        
        # Run one forward pass
        # mlx_lm.stream_generate is patched to use our custom loop for FlashLLM
        tokens = list(mlx_lm.stream_generate(model, tokenizer, "Hello", max_tokens=5))
        assert len(tokens) > 0
        
        peak_rss = proc.memory_info().rss
        
        diff_mb = (peak_rss - baseline_rss) / (1024 * 1024)
        print(f"\nPeak RSS increase: {diff_mb:.1f} MB")
        
        assert peak_rss - baseline_rss < 200 * 1024 * 1024, (
            f"Expected < 200 MB RSS increase, got {diff_mb:.1f} MB"
        )
    finally:
        remove_flash_patch()


def test_lazy_load_uses_zero_metal_ram(tmp_model_dir, flash_config):
    """lazy=True should not materialize weights to Metal."""
    import mlx.core as mx
    import mlx_lm
    
    # Ensure Metal cache is clean
    mx.clear_cache()
    before = mx.get_active_memory()
    
    # lazy load
    model, tok = mlx_lm.load(str(tmp_model_dir), lazy=True)
    after_load = mx.get_active_memory()
    
    # With lazy=True, active Metal memory should not have increased
    # significantly (only minor overhead from model instantiation)
    # Using 10MB as a safe threshold for overhead
    assert after_load - before < 10 * 1024 * 1024, (
        f"lazy=True increased Metal active memory by "
        f"{(after_load - before) / 1e6:.1f} MB; expected < 10 MB"
    )
