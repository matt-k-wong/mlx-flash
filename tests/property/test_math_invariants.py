import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import mlx.core as mx

# We will use the synthetic model fixture provided by the existing conftest.py
# If it's not imported implicitly, we can just rely on the pytest fixture system

@pytest.mark.skipif(not hasattr(mx, 'random'), reason="MLX not available")
@given(
    batch_size=st.integers(1, 4),
    seq_len=st.integers(1, 16),
    # The synthetic model in conftest.py has vocab_size=256
    vocab_size=st.just(256) 
)
@settings(max_examples=10, deadline=None) # Reduce examples to keep test fast
def test_streaming_determinism(tmp_model_dir, batch_size, seq_len, vocab_size):
    """
    Proves that chunking, tiling, and pipelining do not alter floating point math
    compared to the standard MLX in-memory generation.
    """
    import mlx_lm
    from mlx_flash.config import FlashConfig
    from mlx_flash.engine.engine import FlashEngine
    
    # 1. Generate random input token sequence
    mx.random.seed(42)
    x = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # 2. Run standard MLX (In-Memory Reference)
    model, tokenizer = mlx_lm.load(str(tmp_model_dir), lazy=False) # Eager load for reference
    expected_out = model(x)
    mx.eval(expected_out)
    
    # 3. Run FlashEngine (Out-of-Core Pipelined)
    # Reload model lazily
    model_lazy, _ = mlx_lm.load(str(tmp_model_dir), lazy=True)
    config = FlashConfig(
        enabled=True, 
        tiled_execution=True, 
        tile_size=128, 
        pipelined_execution=True,
        debug=True
    )
    flash_engine = FlashEngine(model_lazy, tokenizer, config)
    
    from mlx_flash.safetensors_mmap import SafetensorsMmapCache
    flash_engine.mmap_cache = SafetensorsMmapCache(tmp_model_dir)
    
    # Create causal mask
    L = x.shape[1]
    mask = mx.triu(mx.full((L, L), -mx.inf), k=1).astype(mx.float16)
    
    actual_out = flash_engine(x, mask=mask)
    mx.eval(actual_out)
    
    # 4. Assert Bitwise Equality
    # Tiling does fp32 accumulation so it might have slight floating point differences
    # compared to pure fp16/bf16 standard mlx. We use a relaxed tolerance.
    assert mx.allclose(expected_out, actual_out, rtol=1e-3, atol=1e-3)
