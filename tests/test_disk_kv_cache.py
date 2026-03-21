"""Unit tests for DiskKVCache — the disk-backed infinite context KV cache."""

import shutil
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_flash.disk_kv_cache import DiskKVCache


@pytest.fixture
def kv_dir(tmp_path):
    d = tmp_path / "test_kv"
    d.mkdir()
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_kv(batch=1, heads=4, seq=1, dim=64, dtype=mx.float16):
    """Create random key/value tensors in [B, Heads, Seq, Dim] format."""
    k = mx.random.normal((batch, heads, seq, dim)).astype(dtype)
    v = mx.random.normal((batch, heads, seq, dim)).astype(dtype)
    mx.eval(k, v)
    return k, v


class TestUpdateAndFetchShapes:
    def test_single_token(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=1)
        out_k, out_v = cache.update_and_fetch(k, v)
        assert out_k.shape == (1, 4, 1, 64)
        assert out_v.shape == (1, 4, 1, 64)
        cache.close()

    def test_multiple_tokens_accumulate(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        for i in range(5):
            k, v = _make_kv(seq=1)
            out_k, out_v = cache.update_and_fetch(k, v)
        assert out_k.shape == (1, 4, 5, 64)
        assert out_v.shape == (1, 4, 5, 64)
        cache.close()

    def test_prefill_multi_token(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=32)
        out_k, out_v = cache.update_and_fetch(k, v)
        assert out_k.shape == (1, 4, 32, 64)
        # Then add single tokens
        k2, v2 = _make_kv(seq=1)
        out_k2, out_v2 = cache.update_and_fetch(k2, v2)
        assert out_k2.shape == (1, 4, 33, 64)
        cache.close()


class TestKVCacheContract:
    def test_size_tracks_offset(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        assert cache.size() == 0
        k, v = _make_kv(seq=3)
        cache.update_and_fetch(k, v)
        assert cache.size() == 3
        cache.close()

    def test_empty_before_update(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        assert cache.empty()
        k, v = _make_kv(seq=1)
        cache.update_and_fetch(k, v)
        assert not cache.empty()
        cache.close()

    def test_nbytes_zero_when_empty(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        assert cache.nbytes == 0

    def test_nbytes_positive_after_update(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=10)
        cache.update_and_fetch(k, v)
        assert cache.nbytes > 0
        # For fp16: 10 tokens * 1 batch * 4 heads * 64 dim * 2 bytes * 2 (k+v)
        expected = 10 * 1 * 4 * 64 * 2 * 2
        assert cache.nbytes == expected
        cache.close()

    def test_is_trimmable(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        assert cache.is_trimmable()

    def test_to_quantized_raises(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        with pytest.raises(NotImplementedError):
            cache.to_quantized()


class TestTrim:
    def test_trim_reduces_offset(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=10)
        cache.update_and_fetch(k, v)
        assert cache.offset == 10
        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7
        assert cache.keys.shape[2] == 7
        cache.close()

    def test_trim_clamps_to_offset(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=5)
        cache.update_and_fetch(k, v)
        trimmed = cache.trim(100)
        assert trimmed == 5
        assert cache.offset == 0
        assert cache.keys is None
        cache.close()

    def test_trim_zero_is_noop(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=5)
        cache.update_and_fetch(k, v)
        trimmed = cache.trim(0)
        assert trimmed == 0
        assert cache.offset == 5
        cache.close()


class TestStateProp:
    def test_state_getter(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=3)
        cache.update_and_fetch(k, v)
        keys, values = cache.state
        assert keys.shape == (1, 4, 3, 64)
        assert values.shape == (1, 4, 3, 64)
        cache.close()

    def test_state_setter(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=5)
        cache.update_and_fetch(k, v)
        # Set state to new arrays
        new_k = mx.zeros((1, 4, 2, 64), dtype=mx.float16)
        new_v = mx.ones((1, 4, 2, 64), dtype=mx.float16)
        cache.state = (new_k, new_v)
        assert cache.offset == 2
        keys, values = cache.state
        assert keys.shape[2] == 2
        cache.close()


class TestResourceManagement:
    def test_close_releases_fds(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=1)
        cache.update_and_fetch(k, v)
        assert cache.fd_k is not None
        cache.close()
        assert cache.fd_k is None
        assert cache.fd_v is None

    def test_double_close_safe(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=1)
        cache.update_and_fetch(k, v)
        cache.close()
        cache.close()  # Should not raise

    def test_context_manager(self, kv_dir):
        with DiskKVCache(layer_idx=0, cache_dir=kv_dir) as cache:
            k, v = _make_kv(seq=1)
            cache.update_and_fetch(k, v)
            assert cache.fd_k is not None
        # After exiting context, fds should be closed
        assert cache.fd_k is None

    def test_safetensors_files_created(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=1)
        cache.update_and_fetch(k, v)
        kv_path = Path(kv_dir)
        assert (kv_path / "L0_k.safetensors").exists()
        assert (kv_path / "L0_v.safetensors").exists()
        cache.close()


class TestEviction:
    def test_eviction_bounds_offset(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir, max_tokens=20)
        # Fill past limit
        for _ in range(25):
            k, v = _make_kv(seq=1)
            cache.update_and_fetch(k, v)
        # After eviction, offset should be bounded
        assert cache.offset <= 20
        cache.close()

    def test_no_eviction_when_under_limit(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir, max_tokens=100)
        for _ in range(10):
            k, v = _make_kv(seq=1)
            cache.update_and_fetch(k, v)
        assert cache.offset == 10
        cache.close()

    def test_no_eviction_when_unlimited(self, kv_dir):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir, max_tokens=None)
        for _ in range(50):
            k, v = _make_kv(seq=1)
            cache.update_and_fetch(k, v)
        assert cache.offset == 50
        cache.close()


class TestDtypes:
    @pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
    def test_dtype_preserved(self, kv_dir, dtype):
        cache = DiskKVCache(layer_idx=0, cache_dir=kv_dir)
        k, v = _make_kv(seq=3, dtype=dtype)
        out_k, out_v = cache.update_and_fetch(k, v)
        assert out_k.dtype == dtype
        assert out_v.dtype == dtype
        cache.close()


class TestDebugOutput:
    def test_no_debug_output_by_default(self, tmp_model_dir, capsys):
        """Verify no unconditional print output from stream_generate."""
        from mlx_flash import FlashConfig, FlashGenerationLoop

        cfg = FlashConfig(
            enabled=True,
            disk_kv_enabled=True,
            disk_kv_dir=str(tmp_model_dir / "kv_test"),
            debug=False,
        )
        loop = FlashGenerationLoop(str(tmp_model_dir), config=cfg)
        list(loop.stream_generate("Hello", max_tokens=1))
        captured = capsys.readouterr()
        assert "[DEBUG]" not in captured.out
        assert "[DEBUG]" not in captured.err
        loop.shutdown()


class TestUniqueDirs:
    def test_default_dir_empty(self):
        from mlx_flash import FlashConfig
        cfg = FlashConfig()
        assert cfg.disk_kv_dir == ""
