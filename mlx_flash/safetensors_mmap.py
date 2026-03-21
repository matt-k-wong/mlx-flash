import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any

from .prefetch_worker import BackgroundPrefetcher

class SafetensorsMmapCache:
    """
    Parses `.safetensors` headers in a directory and maintains active mmap objects.
    This allows us to call OS-level madvise() on the exact byte ranges of tensors.
    """
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.file_mmaps: dict[str, mmap.mmap] = {}
        self.file_handles: dict[str, Any] = {}
        self.tensor_locations: dict[str, tuple[mmap.mmap, int, int]] = {}
        
        self._load_all()
        self.prefetch_worker = BackgroundPrefetcher(self.file_handles)

    def _load_all(self):
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if not safetensor_files:
            return

        for sf in safetensor_files:
            try:
                f = open(sf, "rb")
                self.file_handles[sf.name] = f
                
                # Safetensors header format: 8-byte little-endian uint64 length of JSON header
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    continue
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # Read JSON string
                header_json_str = f.read(header_len).decode('utf-8')
                metadata = json.loads(header_json_str)
                
                headers_end = 8 + header_len
                
                # Create mmap for the entire file
                # mmap(fileno, length, flags, prot, access, offset)
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.file_mmaps[sf.name] = mm
                
                # Map each tensor to its mmap and absolute byte range
                for tensor_name, info in metadata.items():
                    if tensor_name == "__metadata__":
                        continue
                    offsets = info.get("data_offsets")
                    if offsets and len(offsets) == 2:
                        abs_start = headers_end + offsets[0]
                        abs_end = headers_end + offsets[1]
                        self.tensor_locations[tensor_name] = (mm, abs_start, abs_end, sf.name)
                        
            except Exception as e:
                # If a file fails to load, we skip it.
                continue

    def get_tensor_range(self, tensor_name: str) -> tuple[mmap.mmap, int, int] | None:
        """Return the (mmap_obj, absolute_start, absolute_end) for a given tensor."""
        info = self.tensor_locations.get(tensor_name)
        if info:
            return (info[0], info[1], info[2])
        return None
    
    def get_layer_ranges(self, layer_idx: int) -> dict[mmap.mmap, tuple[int, int]]:
        """
        Groups all tensors belonging to `layer_idx` into contiguous or combined byte ranges 
        per physical mmap file to minimize madvise calls.
        """
        layer_prefix = f"model.layers.{layer_idx}."
        
        # Collect all intervals for this layer, grouped by mmap
        intervals_by_mmap = {}
        
        for t_name, info in self.tensor_locations.items():
            if t_name.startswith(layer_prefix):
                mm, start, end, filename = info
                if mm not in intervals_by_mmap:
                    intervals_by_mmap[mm] = []
                # Also store filename to pass to the BackgroundPrefetcher
                intervals_by_mmap[mm].append((start, end, filename))
                
        # Merge overlapping/adjacent intervals
        merged = {}
        for mm, intervals in intervals_by_mmap.items():
            intervals.sort(key=lambda x: x[0])
            min_start = intervals[0][0]
            max_end = intervals[-1][1]
            filename = intervals[0][2]
            merged[mm] = (min_start, max_end, filename)
            
        return merged
        
    def prefetch_layer_background(self, layer_idx: int):
        ranges = self.get_layer_ranges(layer_idx)
        for _, (start, end, filename) in ranges.items():
            self.prefetch_worker.enqueue(filename, start, end - start)

    def shutdown(self):
        if hasattr(self, 'prefetch_worker'):
            self.prefetch_worker.shutdown()
        for mm in self.file_mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.file_mmaps.clear()
        self.file_handles.clear()
        self.tensor_locations.clear()
