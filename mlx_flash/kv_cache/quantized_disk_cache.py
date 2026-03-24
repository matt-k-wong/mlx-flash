import contextlib
import json
import struct
from pathlib import Path
from typing import IO, cast, Optional, Tuple, Dict

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


class QuantizedDiskKVCache(KVCache):
    """
    Hybrid Quantized Disk-Backed KV Cache.
    
    Addresses attention instability by keeping a local window of tokens in full 
    precision while offloading older tokens to quantized disk storage with 
    proper scale and bias handling.
    """

    def __init__(self, layer_idx: int, cache_dir: str = "/tmp/mlx_flash_kv",
                 max_tokens: Optional[int] = None, bits: int = 4, group_size: int = 64,
                 local_window_size: int = 128):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bits = bits
        self.group_size = group_size
        self._max_tokens = max_tokens
        self.local_window_size = local_window_size
        
        # We need 6 files: Data, Scales, Biases for K and V
        self.paths = {
            "k_data": self.cache_dir / f"L{self.layer_idx}_k_data.safetensors",
            "k_scales": self.cache_dir / f"L{self.layer_idx}_k_scales.safetensors",
            "k_biases": self.cache_dir / f"L{self.layer_idx}_k_biases.safetensors",
            "v_data": self.cache_dir / f"L{self.layer_idx}_v_data.safetensors",
            "v_scales": self.cache_dir / f"L{self.layer_idx}_v_scales.safetensors",
            "v_biases": self.cache_dir / f"L{self.layer_idx}_v_biases.safetensors",
        }

        # Clean up any old run
        for p in self.paths.values():
            if p.exists():
                p.unlink()

        self.disk_offset = 0 
        self.header_pad_size = 8192  

        self.fds: Dict[str, IO[bytes]] = {}
        self.base_k_shape: Optional[Tuple[int, ...]] = None
        self.base_v_shape: Optional[Tuple[int, ...]] = None
        self.original_dtype = mx.float16

        # Local FP16 buffer
        self.local_k: Optional[mx.array] = None
        self.local_v: Optional[mx.array] = None

        self._closed = False
        self._exit_stack = contextlib.ExitStack()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for fd in self.fds.values():
            with contextlib.suppress(Exception):
                fd.close()
        self._exit_stack.close()
        self.fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def _init_files(self, k_shape, v_shape, dtype):
        self.original_dtype = dtype
        self.base_k_shape = k_shape
        self.base_v_shape = v_shape
        
        for name, path in self.paths.items():
            self.fds[name] = self._exit_stack.enter_context(open(path, "wb+"))

        self._write_headers(0)

    def _get_quantized_shapes(self, seq_len: int, base_shape: tuple):
        batch, heads, _, head_dim = base_shape
        
        # MLX packs quantized values into uint32
        pack_factor = 32 // self.bits
        packed_head_dim = head_dim // pack_factor
        data_shape = [seq_len, batch, heads, packed_head_dim]
        
        scales_head_dim = head_dim // self.group_size
        scales_shape = [seq_len, batch, heads, scales_head_dim]
        
        return data_shape, scales_shape

    def _write_header(self, fd: IO[bytes], name: str, shape: list, dtype_str: str, bytes_per_elem: int):
        import math
        n_bytes = math.prod(shape) * bytes_per_elem
        header = {
            name: {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [0, n_bytes]
            },
            "__metadata__": {"format": "pt"}
        }
        header_json = json.dumps(header).encode("utf-8")
        padded_json = header_json.ljust(self.header_pad_size, b" ")
        header_len_bytes = struct.pack("<Q", self.header_pad_size)
        fd.seek(0)
        fd.write(header_len_bytes)
        fd.write(padded_json)

    def _write_headers(self, seq_len: int):
        k_data_shape, k_scales_shape = self._get_quantized_shapes(seq_len, self.base_k_shape)
        v_data_shape, v_scales_shape = self._get_quantized_shapes(seq_len, self.base_v_shape)
        
        # Data is uint32 (4 bytes per elem)
        self._write_header(self.fds["k_data"], "data", k_data_shape, "U32", 4)
        self._write_header(self.fds["v_data"], "data", v_data_shape, "U32", 4)
        
        # Scales and Biases are float32 on disk
        self._write_header(self.fds["k_scales"], "scales", k_scales_shape, "F32", 4)
        self._write_header(self.fds["v_scales"], "scales", v_scales_shape, "F32", 4)
        self._write_header(self.fds["k_biases"], "biases", k_scales_shape, "F32", 4)
        self._write_header(self.fds["v_biases"], "biases", v_scales_shape, "F32", 4)

    def _flush_to_disk(self, k: mx.array, v: mx.array):
        if not self.fds: return

        k_t = k.transpose(2, 0, 1, 3)
        v_t = v.transpose(2, 0, 1, 3)

        k_q, k_s, k_b = mx.quantize(k_t, group_size=self.group_size, bits=self.bits)
        v_q, v_s, v_b = mx.quantize(v_t, group_size=self.group_size, bits=self.bits)
        
        k_s_f32, k_b_f32 = k_s.astype(mx.float32), k_b.astype(mx.float32)
        v_s_f32, v_b_f32 = v_s.astype(mx.float32), v_b.astype(mx.float32)
        
        mx.eval(k_q, k_s_f32, k_b_f32, v_q, v_s_f32, v_b_f32)

        self.fds["k_data"].seek(0, 2)
        self.fds["k_data"].write(np.asarray(k_q).tobytes())
        self.fds["k_scales"].seek(0, 2)
        self.fds["k_scales"].write(np.asarray(k_s_f32).tobytes())
        self.fds["k_biases"].seek(0, 2)
        self.fds["k_biases"].write(np.asarray(k_b_f32).tobytes())
        
        self.fds["v_data"].seek(0, 2)
        self.fds["v_data"].write(np.asarray(v_q).tobytes())
        self.fds["v_scales"].seek(0, 2)
        self.fds["v_scales"].write(np.asarray(v_s_f32).tobytes())
        self.fds["v_biases"].seek(0, 2)
        self.fds["v_biases"].write(np.asarray(v_b_f32).tobytes())

        self.disk_offset += k.shape[2]
        self._write_headers(self.disk_offset)
        
        for fd in self.fds.values():
            fd.flush()

    def _get_disk_kv(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if self.disk_offset == 0:
            return None, None

        # 1. Lazy Load
        loaded = {name: mx.load(str(path)) for name, path in self.paths.items()}
        
        lazy_k_data, lazy_k_scales, lazy_k_biases = loaded["k_data"]["data"], loaded["k_scales"]["scales"], loaded["k_biases"]["biases"]
        lazy_v_data, lazy_v_scales, lazy_v_biases = loaded["v_data"]["data"], loaded["v_scales"]["scales"], loaded["v_biases"]["biases"]

        # 2. Dequantize on the fly
        k_s, k_b = lazy_k_scales.astype(self.original_dtype), lazy_k_biases.astype(self.original_dtype)
        v_s, v_b = lazy_v_scales.astype(self.original_dtype), lazy_v_biases.astype(self.original_dtype)
        
        k_full = mx.dequantize(lazy_k_data, k_s, k_b, group_size=self.group_size, bits=self.bits)
        v_full = mx.dequantize(lazy_v_data, v_s, v_b, group_size=self.group_size, bits=self.bits)

        return k_full.transpose(1, 2, 0, 3), v_full.transpose(1, 2, 0, 3)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if not self.fds:
            self._init_files(keys.shape, values.shape, keys.dtype)

        if self.local_k is None:
            self.local_k, self.local_v = keys, values
        else:
            self.local_k = mx.concatenate([self.local_k, keys], axis=2)
            self.local_v = mx.concatenate([self.local_v, values], axis=2)

        if self.local_k.shape[2] > self.local_window_size:
            num_to_flush = self.local_k.shape[2] - self.local_window_size
            to_flush_k, to_flush_v = self.local_k[:, :, :num_to_flush, :], self.local_v[:, :, :num_to_flush, :]
            self.local_k, self.local_v = self.local_k[:, :, num_to_flush:, :], self.local_v[:, :, num_to_flush:, :]
            self._flush_to_disk(to_flush_k, to_flush_v)

        disk_k, disk_v = self._get_disk_kv()
        if disk_k is not None:
            return mx.concatenate([disk_k, self.local_k], axis=2), mx.concatenate([disk_v, self.local_v], axis=2)
        return self.local_k, self.local_v

    def size(self):
        return self.disk_offset + (self.local_k.shape[2] if self.local_k is not None else 0)

    @property
    def state(self):
        return self.local_k, self.local_v

    @state.setter
    def state(self, v):
        self.local_k, self.local_v = v

    def empty(self):
        return self.local_k is None and self.disk_offset == 0

    @property
    def nbytes(self):
        local_bytes = (self.local_k.nbytes + self.local_v.nbytes) if self.local_k is not None else 0
        disk_bytes = 0
        if self.disk_offset > 0 and self.base_k_shape is not None:
            batch, heads, _, head_dim = self.base_k_shape
            pack_factor = 32 // self.bits
            data_bytes = (batch * heads * (head_dim // pack_factor)) * 4
            meta_bytes = (batch * heads * (head_dim // self.group_size)) * 4 * 2
            disk_bytes = int(self.disk_offset * (data_bytes + meta_bytes) * 2)
        return local_bytes + disk_bytes
