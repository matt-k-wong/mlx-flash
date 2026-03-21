import importlib.util
import subprocess
import time
from contextlib import contextmanager

import psutil

_HAS_MLX = importlib.util.find_spec("mlx.core") is not None

def get_metal_stats() -> dict:
    try:
        import mlx.core as mx
        return {
            "active_mb": mx.get_active_memory() / 1e6,
            "peak_mb": mx.get_peak_memory() / 1e6,
            "cache_mb": mx.get_cache_memory() / 1e6,
        }
    except (AttributeError, ImportError):
        return {"active_mb": -1, "peak_mb": -1, "cache_mb": -1}

class RAMProfiler:
    """
    Tracks RSS, Metal GPU allocation, and page cache usage during 
    Flash Mode inference. Use in debug mode to verify madvise is working.
    
    On macOS, uses:
    - psutil.Process().memory_info().rss for Python RSS
    - subprocess("vm_stat") for page cache stats  
    - mx.metal.get_active_memory() for Metal GPU allocation
    - mx.metal.get_peak_memory() for Metal peak
    """
    
    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()
        self._page_size = self._get_page_size()

    def _get_page_size(self) -> int:
        try:
            output = subprocess.check_output(["vm_stat"], encoding="utf-8")
            line = output.splitlines()[0]
            # Mach Virtual Memory Statistics: (page size of 16384 bytes)
            return int(line.split("page size of ")[1].split(" bytes")[0])
        except Exception:
            return 4096

    def _get_page_cache_mb(self) -> float:
        try:
            output = subprocess.check_output(["vm_stat"], encoding="utf-8")
            stats = {}
            for line in output.splitlines()[1:]:
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key, val = parts
                        stats[key.strip()] = int(val.strip().replace(".", ""))
            
            # Pages inactive + Pages speculative + Pages purgeable
            inactive = stats.get("Pages inactive", 0)
            speculative = stats.get("Pages speculative", 0)
            purgeable = stats.get("Pages purgeable", 0)
            
            return (inactive + speculative + purgeable) * self._page_size / 1e6
        except Exception:
            return -1.0

    def snapshot(self, label: str) -> dict:
        """Take a named memory snapshot. Returns dict with rss_mb, 
        metal_active_mb, metal_peak_mb, page_cache_mb."""
        rss_mb = self.process.memory_info().rss / 1e6
        metal = get_metal_stats()
        page_cache_mb = self._get_page_cache_mb()
        
        snap = {
            "label": label,
            "rss_mb": rss_mb,
            "metal_active_mb": metal["active_mb"],
            "metal_peak_mb": metal["peak_mb"],
            "page_cache_mb": page_cache_mb,
            "timestamp": time.time()
        }
        self.snapshots.append(snap)
        return snap
    
    @contextmanager
    def layer_context(self, layer_idx: int):
        """Context manager: snapshot before and after a layer, log delta."""
        self.snapshot(f"Layer {layer_idx} Before")
        yield
        self.snapshot(f"Layer {layer_idx} After")
    
    def report(self) -> str:
        """Print a table of all snapshots with deltas."""
        if not self.snapshots:
            return "No snapshots recorded."
            
        lines = []
        header = f"{'Label':<20} | {'RSS (MB)':>10} | {'Delta':>10} | {'Metal Active':>12} | {'Delta':>10} | {'Page Cache':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        
        # We start with the first snapshot as base for deltas
        base_rss = self.snapshots[0]["rss_mb"]
        base_metal = self.snapshots[0]["metal_active_mb"]
        
        prev_rss = base_rss
        prev_metal = base_metal
        
        for _i, snap in enumerate(self.snapshots):
            rss = snap["rss_mb"]
            metal = snap["metal_active_mb"]
            pc = snap["page_cache_mb"]
            
            rss_delta = rss - prev_rss
            metal_delta = metal - prev_metal
            
            line = f"{snap['label']:<20} | {rss:>10.2f} | {rss_delta:>10.2f} | {metal:>12.2f} | {metal_delta:>10.2f} | {pc:>10.2f}"
            lines.append(line)
            
            prev_rss = rss
            prev_metal = metal
            
        return "\n".join(lines)
