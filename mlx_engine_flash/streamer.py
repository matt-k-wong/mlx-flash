"""
streamer.py — Zero-copy memory-mapped weight streamer.

This module provides zero-copy access to model weights by memory-mapping
safetensors files and returning NumPy arrays that alias the mmap'd memory.
This allows the OS to manage physical memory via the unified page cache,
reclaiming pages on-demand after computation.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FlashConfig
from .page_cache import prefetch, release, set_sequential


# ── dtype mapping: safetensors dtype string → numpy dtype ─────────────────
def _get_st_dtype_map() -> dict[str, Any]:
    m: dict[str, Any] = {
        "F64":  np.dtype("float64"),
        "F32":  np.dtype("float32"),
        "F16":  np.dtype("float16"),
        "I64":  np.dtype("int64"),
        "I32":  np.dtype("int32"),
        "I16":  np.dtype("int16"),
        "I8":   np.dtype("int8"),
        "U32":  np.dtype("uint32"),
        "U8":   np.dtype("uint8"),
        "BOOL": np.dtype("bool"),
    }
    # Try to add BF16 if available in numpy
    try:
        m["BF16"] = np.dtype("bfloat16")
    except TypeError:
        # Fallback to uint16; must be bitcast to BF16 when converted to MLX
        m["BF16"] = np.dtype("uint16")
    return m

_ST_DTYPE_MAP = _get_st_dtype_map()

# Quantised formats stored as raw uint8 blobs in safetensors
_QUANTISED_DTYPES = {"Q4_0", "Q4_1", "Q4_K", "Q6_K", "Q8_0", "Q2_K", "Q3_K"}


@dataclass(frozen=True)
class TensorEntry:
    """Metadata for one tensor in the safetensors index."""
    name: str
    file_path: Path
    data_offset: int     # absolute byte offset from start of file
    data_length: int     # byte length
    dtype: str           # safetensors dtype string
    shape: tuple[int, ...]
    n_bits: int          # 4 / 8 / 16 / 32 — used for quant validation


def _parse_safetensors_header(path: Path) -> tuple[dict, int]:
    """
    Read the safetensors header without loading the full file.
    Returns (header_dict, data_start_offset).
    """
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"File too small to be safetensors: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    data_start = 8 + header_len
    return header, data_start


def _quant_bits(dtype_str: str) -> int:
    """Best-effort extraction of effective bit-width from dtype string."""
    lower = dtype_str.lower()
    if "q2" in lower: return 2
    if "q3" in lower: return 3
    if "q4" in lower: return 4
    if "q5" in lower: return 5
    if "q6" in lower: return 6
    if "q8" in lower: return 8
    if lower in ("f16", "bf16"): return 16
    if lower == "f32": return 32
    return 8


class SafetensorsIndex:
    """
    Index of all tensors across one or more safetensors shards.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._entries: dict[str, TensorEntry] = {}
        self._mmaps: dict[Path, mmap.mmap] = {}
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self) -> None:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            self._load_sharded(index_path)
        else:
            shards = sorted(self.model_dir.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")
            for shard in shards:
                self._index_shard(shard)
        self._layer_prefix = self._detect_layer_prefix()

    def _detect_layer_prefix(self) -> str:
        for n in self._entries:
            if ".layers.0." in n:
                return n.split(".layers.0.")[0] + ".layers.0."
            if "layers.0." in n:
                return "layers.0."
        return "model.layers.0."

    def _load_sharded(self, index_path: Path) -> None:
        idx = json.loads(index_path.read_text())
        weight_map: dict[str, str] = idx.get("weight_map", {})
        shards_needed = set(weight_map.values())
        shard_headers: dict[str, tuple[dict, int]] = {}
        for shard_name in shards_needed:
            shard_path = self.model_dir / shard_name
            header, data_start = _parse_safetensors_header(shard_path)
            shard_headers[shard_name] = (header, data_start)

        for tensor_name, shard_name in weight_map.items():
            header, data_start = shard_headers[shard_name]
            if tensor_name not in header or tensor_name == "__metadata__":
                continue
            meta = header[tensor_name]
            start, end = meta["data_offsets"]
            self._entries[tensor_name] = TensorEntry(
                name=tensor_name,
                file_path=self.model_dir / shard_name,
                data_offset=data_start + start,
                data_length=end - start,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                n_bits=_quant_bits(meta["dtype"]),
            )

    def _index_shard(self, shard_path: Path) -> None:
        header, data_start = _parse_safetensors_header(shard_path)
        for tensor_name, meta in header.items():
            if tensor_name == "__metadata__": continue
            start, end = meta["data_offsets"]
            self._entries[tensor_name] = TensorEntry(
                name=tensor_name,
                file_path=shard_path,
                data_offset=data_start + start,
                data_length=end - start,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                n_bits=_quant_bits(meta["dtype"]),
            )

    def open_mmaps(self) -> None:
        files_needed = {e.file_path for e in self._entries.values()}
        for path in files_needed:
            if path not in self._fds:
                fd = os.open(str(path), os.O_RDONLY)
                self._fds[path] = fd
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self._mmaps[path] = mm
                set_sequential(mm, 0, mm.size())

    def close_mmaps(self) -> None:
        import contextlib
        for mm in self._mmaps.values():
            with contextlib.suppress(BufferError): mm.close()
        for fd in self._fds.values():
            with contextlib.suppress(OSError): os.close(fd)
        self._mmaps.clear()
        self._fds.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> TensorEntry:
        return self._entries[name]

    def get_mmap(self, path: Path) -> mmap.mmap:
        with self._lock:
            if path not in self._mmaps: self.open_mmaps()
        return self._mmaps[path]

    def tensor_names(self) -> list[str]: return list(self._entries.keys())

    def layer_tensor_names(self, layer_idx: int) -> list[str]:
        prefix = self._layer_prefix.replace(".layers.0.", f".layers.{layer_idx}.")
        if "layers.0." in self._layer_prefix and not self._layer_prefix.startswith("."):
             prefix = self._layer_prefix.replace("layers.0.", f"layers.{layer_idx}.")
        return [n for n in self._entries if n.startswith(prefix)]

    def expert_tensor_names(self, layer_idx: int, expert_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
        return [n for n in self._entries if n.startswith(prefix)]

    @property
    def n_layers(self) -> int:
        idxs = set()
        for name in self._entries:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try: idxs.add(int(parts[i + 1]))
                    except ValueError: pass
        return max(idxs) + 1 if idxs else 0

    @property
    def min_quant_bits(self) -> int:
        bits = [e.n_bits for e in self._entries.values()]
        return min(bits) if bits else 16


class WeightStreamer:
    """
    Zero-copy weight streamer.

    WARNING: Returned arrays alias mmap memory. Do not close the WeightStreamer
    while any returned array is still referenced.
    """

    def __init__(self, model_dir: Path, config: FlashConfig) -> None:
        self.config = config
        self.index = SafetensorsIndex(model_dir)
        self.index.open_mmaps()

    def stream_tensor(self, name: str) -> np.ndarray:
        entry = self.index._entries[name]
        mm = self.index.get_mmap(entry.file_path)
        return self._decode(entry, mm)

    def stream_tensors(self, names: list[str], prefetch_names: list[str] | None = None) -> dict[str, np.ndarray]:
        if prefetch_names:
            for n in prefetch_names:
                entry = self.index._entries.get(n)
                if entry:
                    mm = self.index.get_mmap(entry.file_path)
                    prefetch(mm, entry.data_offset, entry.data_length)

        return {name: self.stream_tensor(name) for name in names if name in self.index._entries}

    def prefetch_layer(self, layer_idx: int) -> None:
        for name in self.index.layer_tensor_names(layer_idx):
            entry = self.index._entries.get(name)
            if entry:
                mm = self.index.get_mmap(entry.file_path)
                prefetch(mm, entry.data_offset, entry.data_length)

    def release_layer(self, layer_idx: int) -> None:
        for name in self.index.layer_tensor_names(layer_idx):
            entry = self.index._entries.get(name)
            if entry:
                mm = self.index.get_mmap(entry.file_path)
                release(mm, entry.data_offset, entry.data_length, self.config.eviction_strategy)

    def _decode(self, entry: TensorEntry, mm: mmap.mmap) -> np.ndarray:
        """Returns a zero-copy NumPy array slicing into the mmap."""
        mm_view = memoryview(mm)[entry.data_offset : entry.data_offset + entry.data_length]
        
        if entry.dtype in _QUANTISED_DTYPES:
            return np.frombuffer(mm_view, dtype=np.uint8)
        
        np_dtype = _ST_DTYPE_MAP.get(entry.dtype)
        if np_dtype is None:
            raise ValueError(f"Unknown dtype {entry.dtype!r} for tensor {entry.name!r}")
            
        arr = np.frombuffer(mm_view, dtype=np_dtype)
        if entry.shape:
            arr = arr.reshape(entry.shape)
        return arr

    def close(self) -> None:
        self.index.close_mmaps()

    def __enter__(self) -> WeightStreamer: return self
    def __exit__(self, *_) -> None: self.close()
