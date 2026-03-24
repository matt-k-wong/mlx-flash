import shutil
from pathlib import Path
import mlx.core as mx
import pytest
from mlx_flash.kv_cache.quantized_disk_cache import QuantizedDiskKVCache

@pytest.fixture
def kv_dir(tmp_path):
    d = tmp_path / "test_kv_quant"
    d.mkdir()
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)

def _make_kv(batch=1, heads=4, seq=1, dim=64, dtype=mx.float16):
    k = mx.random.normal((batch, heads, seq, dim)).astype(dtype)
    v = mx.random.normal((batch, heads, seq, dim)).astype(dtype)
    mx.eval(k, v)
    return k, v

class TestQuantizedUpdate:
    def test_single_token(self, kv_dir):
        # bits=8 for easier matching
        cache = QuantizedDiskKVCache(layer_idx=0, cache_dir=kv_dir, bits=8, local_window_size=2)
        k, v = _make_kv(seq=1)
        out_k, out_v = cache.update_and_fetch(k, v)
        assert out_k.shape == (1, 4, 1, 64)
        assert out_v.shape == (1, 4, 1, 64)
        cache.close()

    def test_hybrid_transition(self, kv_dir):
        # window size = 4
        cache = QuantizedDiskKVCache(layer_idx=0, cache_dir=kv_dir, bits=4, local_window_size=4)
        
        # Add 3 tokens (all in memory)
        for _ in range(3):
            k, v = _make_kv(seq=1)
            out_k, out_v = cache.update_and_fetch(k, v)
        
        assert cache.disk_offset == 0
        assert cache.local_k.shape[2] == 3
        
        # Add 2 more tokens (total 5, should flush 1 to disk)
        k, v = _make_kv(seq=2)
        out_k, out_v = cache.update_and_fetch(k, v)
        
        assert cache.disk_offset == 1
        assert cache.local_k.shape[2] == 4
        assert out_k.shape[2] == 5
        
        cache.close()

    def test_precision_8bit(self, kv_dir):
        # 8-bit should be very accurate
        cache = QuantizedDiskKVCache(layer_idx=0, cache_dir=kv_dir, bits=8, local_window_size=2)
        
        k, v = _make_kv(seq=10) # 8 tokens will go to disk
        out_k, out_v = cache.update_and_fetch(k, v)
        
        # Check that it's reasonably close to original
        # Note: we compare the last part (local window) and first part (disk)
        assert mx.allclose(out_k[:, :, -2:, :], k[:, :, -2:, :], atol=1e-5)
        # Disk part might have quantization error
        assert mx.allclose(out_k[:, :, :8, :], k[:, :, :8, :], atol=1e-1)
        
        cache.close()

    def test_large_prefill(self, kv_dir):
        cache = QuantizedDiskKVCache(layer_idx=0, cache_dir=kv_dir, bits=4, local_window_size=16)
        k, v = _make_kv(seq=128)
        out_k, out_v = cache.update_and_fetch(k, v)
        
        assert out_k.shape == (1, 4, 128, 64)
        assert cache.disk_offset == 128 - 16
        assert cache.local_k.shape[2] == 16
        cache.close()
