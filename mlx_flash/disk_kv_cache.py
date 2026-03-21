import contextlib
import json
import struct
from pathlib import Path
from typing import IO, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


class DiskKVCache(KVCache):
    """
    Infinite Disk-Backed KV Cache that seamlessly bypasses Metal RAM limits.

    Architecture:
    We maintain two padded `.safetensors` files per layer (one for keys, one for values).
    As new tokens arrive, we transpose their tensors to:
        [SeqLen, Batch, Heads, HeadDim]
    This allows us to physically append raw bytes linearly to the end of the file
    and guarantee they represent a contiguous sequence block.

    We then update the `.safetensors` JSON header in-place.
    To evaluate, we `mx.load(..., lazy=True)` the file, providing perfect
    zero-copy infinite context via the macOS unified page cache.
    """

    def __init__(self, layer_idx: int, cache_dir: str = "/tmp/mlx_flash_kv",
                 max_tokens: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.k_path = self.cache_dir / f"L{self.layer_idx}_k.safetensors"
        self.v_path = self.cache_dir / f"L{self.layer_idx}_v.safetensors"

        # Clean up any old run
        for p in (self.k_path, self.v_path):
            if p.exists():
                p.unlink()

        self.offset = 0
        self.header_pad_size = 8192  # 8 KB padded header is plenty

        self.fd_k: IO[bytes] | None = None
        self.fd_v: IO[bytes] | None = None

        self.k_dtype_str: str | None = None
        self.k_shape: tuple[int, ...] | None = None
        self.v_shape: tuple[int, ...] | None = None
        self.bytes_per_elem = 2
        self._max_tokens = max_tokens
        self._closed = False
        self._exit_stack = contextlib.ExitStack()

        # Satisfy KVCache contract: keys/values are None until first update
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    # ── Resource Management ─────────────────────────────────────────────────

    def close(self):
        """Explicitly release all file descriptors."""
        if self._closed:
            return
        self._closed = True
        for fd in (self.fd_k, self.fd_v):
            if fd is not None:
                with contextlib.suppress(Exception):
                    fd.close()
        self._exit_stack.close()
        self.fd_k = None
        self.fd_v = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # ── File Setup ──────────────────────────────────────────────────────────

    def _init_files(self, k_shape, v_shape, dtype):
        self.fd_k = self._exit_stack.enter_context(open(self.k_path, "wb+"))  # noqa: SIM115
        self.fd_v = self._exit_stack.enter_context(open(self.v_path, "wb+"))  # noqa: SIM115

        # Force F32 for disk storage to avoid NumPy/MLX buffer protocol issues
        # with float16/bfloat16.
        self.k_dtype_str = "F32"
        self.bytes_per_elem = 4
        self.k_shape = k_shape
        self.v_shape = v_shape

        self._write_header(self.fd_k, "keys", 0, k_shape)
        self._write_header(self.fd_v, "values", 0, v_shape)

    def _write_header(self, fd: IO[bytes], name: str, seq_len: int, base_shape: tuple):
        """Write the padded safetensors header to the start of the file."""
        # disk shape is [SeqLen, Batch, Heads, HeadDim]
        disk_shape = [seq_len, base_shape[0], base_shape[1], base_shape[3]]
        n_bytes = seq_len * base_shape[0] * base_shape[1] * base_shape[3] * self.bytes_per_elem

        header = {
            name: {
                "dtype": self.k_dtype_str,
                "shape": disk_shape,
                "data_offsets": [0, n_bytes]
            },
            "__metadata__": {"format": "pt"}
        }

        header_json = json.dumps(header).encode("utf-8")
        assert len(header_json) <= self.header_pad_size, f"{name} Header exceeded pad size!"
        padded_json = header_json.ljust(self.header_pad_size, b" ")

        header_len_bytes = struct.pack("<Q", self.header_pad_size)

        fd.seek(0)
        fd.write(header_len_bytes)
        fd.write(padded_json)

    # ── Eviction ────────────────────────────────────────────────────────────

    def _maybe_evict(self, incoming_seq: int):
        """If adding incoming_seq would exceed max_tokens, keep the recent half."""
        if self._max_tokens is None:
            return
        if self.offset + incoming_seq <= self._max_tokens:
            return

        keep = self._max_tokens // 2
        if keep <= 0 or self.offset <= 0:
            # Reset entirely
            self.offset = 0
            self._rewrite_empty()
            return

        # Load current data, keep only the tail
        loaded_k = mx.load(str(self.k_path))
        loaded_v = mx.load(str(self.v_path))
        assert isinstance(loaded_k, dict) and isinstance(loaded_v, dict)
        old_k = cast(mx.array, loaded_k["keys"])  # [Seq, B, H, D]
        old_v = cast(mx.array, loaded_v["values"])
        tail_k = old_k[-keep:]
        tail_v = old_v[-keep:]
        mx.eval(tail_k, tail_v)

        # Rewrite files from scratch
        self.offset = 0
        self._rewrite_empty()
        k_buf = np.asarray(tail_k)
        v_buf = np.asarray(tail_v)
        assert self.fd_k is not None and self.fd_v is not None
        self.fd_k.seek(0, 2)
        self.fd_k.write(k_buf.tobytes())
        self.fd_v.seek(0, 2)
        self.fd_v.write(v_buf.tobytes())
        self.offset = keep
        self._write_header(self.fd_k, "keys", self.offset, self.k_shape)
        self._write_header(self.fd_v, "values", self.offset, self.v_shape)
        self.fd_k.flush()
        self.fd_v.flush()

    def _rewrite_empty(self):
        """Truncate files back to header-only state."""
        assert self.fd_k is not None and self.fd_v is not None
        assert self.k_shape is not None and self.v_shape is not None
        for fd, name, shape in [
            (self.fd_k, "keys", self.k_shape),
            (self.fd_v, "values", self.v_shape),
        ]:
            fd.seek(0)
            fd.truncate()
            self._write_header(fd, name, 0, shape)

    # ── Core: update_and_fetch ──────────────────────────────────────────────

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # inputs are [Batch, Heads, NewSeqLen, HeadDim]
        new_seq = keys.shape[2]
        if self.fd_k is None:
            self._init_files(keys.shape, values.shape, keys.dtype)

        # Evict if needed
        self._maybe_evict(new_seq)

        # 1. Transpose to [NewSeqLen, Batch, Heads, HeadDim]
        k_t = keys.transpose(2, 0, 1, 3)
        v_t = values.transpose(2, 0, 1, 3)

        # 2. Evaluate and write to disk
        # Use float32 for the temporary bytes conversion to avoid NumPy buffer alignment
        # issues often seen with float16/bfloat16 on some systems.
        k_t_f32 = k_t.astype(mx.float32)
        v_t_f32 = v_t.astype(mx.float32)
        mx.eval(k_t_f32, v_t_f32)
        
        k_bytes = np.asarray(k_t_f32).tobytes()
        v_bytes = np.asarray(v_t_f32).tobytes()

        # 3. Append physical bytes (data first — crash-safe ordering)
        assert self.fd_k is not None and self.fd_v is not None
        self.fd_k.seek(0, 2)
        self.fd_k.write(k_bytes)

        self.fd_v.seek(0, 2)
        self.fd_v.write(v_bytes)

        self.offset += new_seq

        # 4. Rewrite the JSON headers, then flush once
        assert self.k_shape is not None and self.v_shape is not None
        self._write_header(self.fd_k, "keys", self.offset, self.k_shape)
        self._write_header(self.fd_v, "values", self.offset, self.v_shape)
        self.fd_k.flush()
        self.fd_v.flush()
        
        # 5. Native MLX Lazy Load the entire growing cache
        # shape is [TotalSeq, Batch, Heads, HeadDim]
        loaded_k = mx.load(str(self.k_path))
        loaded_v = mx.load(str(self.v_path))
        assert isinstance(loaded_k, dict) and isinstance(loaded_v, dict)
        lazy_k_t = cast(mx.array, loaded_k["keys"])
        lazy_v_t = cast(mx.array, loaded_v["values"])

        # 6. Transpose back into expected MLX format [Batch, Heads, TotalSeq, HeadDim]
        # This is a purely metadata zero-copy operation in MLX
        # We also cast back to the input dtype!
        self.keys = lazy_k_t.transpose(1, 2, 0, 3).astype(keys.dtype)
        self.values = lazy_v_t.transpose(1, 2, 0, 3).astype(values.dtype)

        return self.keys, self.values

    # ── KVCache Contract ────────────────────────────────────────────────────

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        if self.keys is not None:
            self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        if n <= 0:
            return 0
        self.offset -= n
        # Rewrite headers to reflect the shorter sequence
        if self.fd_k is not None:
            self._write_header(self.fd_k, "keys", self.offset, self.k_shape)
            self._write_header(self.fd_v, "values", self.offset, self.v_shape)
            self.fd_k.flush()
            self.fd_v.flush()
            # Reload trimmed arrays
            if self.offset > 0:
                dtype = self.keys.dtype if self.keys is not None else mx.float16
                loaded_k = mx.load(str(self.k_path))
                loaded_v = mx.load(str(self.v_path))
                assert isinstance(loaded_k, dict) and isinstance(loaded_v, dict)
                lazy_k_t = cast(mx.array, loaded_k["keys"])
                lazy_v_t = cast(mx.array, loaded_v["values"])
                self.keys = lazy_k_t.transpose(1, 2, 0, 3).astype(dtype)
                self.values = lazy_v_t.transpose(1, 2, 0, 3).astype(dtype)
            else:
                self.keys = None
                self.values = None
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None or self.k_shape is None:
            return 0
        # Per token: B * H * D * bytes_per_elem * 2 (keys + values)
        # Use logical bytes per element from keys.dtype
        assert self.keys is not None and self.k_shape is not None
        logical_bytes = 4 if self.keys.dtype == mx.float32 else 2
        per_token = self.k_shape[0] * self.k_shape[1] * self.k_shape[3] * logical_bytes
        return self.offset * per_token * 2

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError("DiskKVCache does not support quantization")
