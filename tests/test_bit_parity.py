import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_flash.tiled import TiledColumnLinear, TiledRowLinear
from mlx_flash.kv_cache.quantized_disk_cache import QuantizedDiskKVCache

def test_tiled_column_parity():
    mx.random.seed(42)
    in_dim = 1024
    out_dim = 4096
    tile_size = 512
    
    # Standard Linear
    linear = nn.Linear(in_dim, out_dim, bias=True)
    linear.set_dtype(mx.float16)
    
    # Tiled Linear
    tiled = TiledColumnLinear(linear, tile_size=tile_size)
    
    # Input
    x = mx.random.normal((1, 16, in_dim)).astype(mx.float16)
    
    # Compare
    y_std = linear(x)
    y_tiled = tiled(x)
    
    mx.eval(y_std, y_tiled)
    
    # Even without special accumulation, concatenating might have tiny diffs
    # because of how MLX optimizes different matmul shapes.
    assert mx.allclose(y_std, y_tiled, atol=1e-3)

def test_tiled_row_parity():
    mx.random.seed(42)
    in_dim = 4096
    out_dim = 1024
    tile_size = 512
    
    # Standard Linear
    linear = nn.Linear(in_dim, out_dim, bias=True)
    linear.set_dtype(mx.float16)
    
    # Tiled Linear
    tiled = TiledRowLinear(linear, tile_size=tile_size)
    
    # Input
    x = mx.random.normal((1, 16, in_dim)).astype(mx.float16)
    
    # Compare
    y_std = linear(x)
    y_tiled = tiled(x)
    
    mx.eval(y_std, y_tiled)
    
    # TiledRowLinear uses FP32 accumulation, just like standard Linear on Metal
    assert mx.array_equal(y_std, y_tiled)

def test_quantized_kv_parity(tmp_path):
    mx.random.seed(42)
    batch, heads, seq, dim = 1, 8, 32, 64
    bits = 8
    
    k = mx.random.normal((batch, heads, seq, dim)).astype(mx.float16)
    v = mx.random.normal((batch, heads, seq, dim)).astype(mx.float16)
    
    # Standard MLX quantization path
    # We transpose to [Seq, Batch, Heads, Dim] for quantization parity
    k_t = k.transpose(2, 0, 1, 3)
    k_q, k_s, k_b = mx.quantize(k_t, bits=bits)
    k_std_deq = mx.dequantize(k_q, k_s, k_b, bits=bits).transpose(1, 2, 0, 3)
    
    # Flash Quantized Cache path
    cache_dir = tmp_path / "kv_parity"
    cache = QuantizedDiskKVCache(layer_idx=0, cache_dir=str(cache_dir), bits=bits, local_window_size=4)
    
    # Update and fetch (this will flush 28 tokens to disk and keep 4 in memory)
    out_k, out_v = cache.update_and_fetch(k, v)
    
    mx.eval(out_k, out_v, k_std_deq)
    
    # The first 28 tokens should match the standard dequantization exactly
    assert mx.array_equal(out_k[:, :, :28, :], k_std_deq[:, :, :28, :])
    # The last 4 tokens should be bit-perfect FP16
    assert mx.array_equal(out_k[:, :, 28:, :], k[:, :, 28:, :])
    
    cache.close()

def test_tiled_column_full_tile_parity():
    mx.random.seed(42)
    in_dim = 1024
    out_dim = 4096
    tile_size = 4096 # Full size
    
    linear = nn.Linear(in_dim, out_dim, bias=True)
    linear.set_dtype(mx.float16)
    tiled = TiledColumnLinear(linear, tile_size=tile_size)
    
    x = mx.random.normal((1, 16, in_dim)).astype(mx.float16)
    
    y_std = linear(x)
    y_tiled = tiled(x)
    
    mx.eval(y_std, y_tiled)
    
    # If this is not bit-perfect, check max diff
    max_diff = mx.max(mx.abs(y_std - y_tiled)).item()
    print(f"Full Tile Column Max Diff: {max_diff}")
    assert max_diff < 1e-5
