import time
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

class StreamingProfiler:
    """
    Passively collects timing metrics for IO and GPU compute across layers,
    and analyzes the data to identify true concurrent overlap.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StreamingProfiler, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.layer_stats = defaultdict(lambda: {'io_wait': 0.0, 'compute': 0.0, 'calls': 0})
        self.cache_stats = {'moe_hits': 0, 'moe_misses': 0, 'os_page_hits': 0, 'os_page_misses': 0}
        self.token_times = []
        self.start_time = time.perf_counter()
        
        # New: Timeline tracking for true overlap analysis
        self.io_intervals: List[tuple[float, float, int]] = [] # (start, end, bytes)
        self.compute_intervals: List[tuple[float, float, str]] = [] # (start, end, label)
        
    def record_layer_pass(self, layer_idx: int, io_wait: float, compute: float):
        self.layer_stats[layer_idx]['io_wait'] += io_wait
        self.layer_stats[layer_idx]['compute'] += compute
        self.layer_stats[layer_idx]['calls'] += 1

    def record_io_interval(self, start: float, end: float, bytes_read: int):
        self.io_intervals.append((start, end, bytes_read))
        duration = end - start
        if duration > 0:
            self.record_pread(duration, bytes_read)

    def record_compute_interval(self, start: float, end: float, label: str = "compute"):
        self.compute_intervals.append((start, end, label))

    def calculate_true_overlap(self) -> Dict[str, Any]:
        """
        Calculates the intersection of IO and Compute intervals to prove 
        concurrency.
        """
        if not self.io_intervals or not self.compute_intervals:
            return {
                "overlap_s": 0.0, 
                "compute_s": 0.0, 
                "io_s": 0.0, 
                "percent": 0.0, 
                "eff_bw_gb_s": 0.0
            }

        total_compute_s = sum(e - s for s, e, _ in self.compute_intervals)
        total_io_s = sum(e - s for s, e, _ in self.io_intervals)
        
        # Calculate intersection area
        overlap_s = 0.0
        # Sort intervals for efficient overlap calculation if needed, 
        # but for small batches simple nested loop is fine.
        for c_s, c_e, _ in self.compute_intervals:
            for i_s, i_e, _ in self.io_intervals:
                # Intersection of [c_s, c_e] and [i_s, i_e]
                inter_s = max(c_s, i_s)
                inter_e = min(c_e, i_e)
                if inter_e > inter_s:
                    overlap_s += (inter_e - inter_s)

        percent = (overlap_s / total_compute_s * 100) if total_compute_s > 0 else 0
        
        # Effective Bandwidth during overlap
        bytes_during_overlap = 0
        for i_s, i_e, b in self.io_intervals:
            # How much of this IO happened during ANY compute?
            io_dur = i_e - i_s
            if io_dur <= 0: continue
            
            overlap_with_compute = 0
            for c_s, c_e, _ in self.compute_intervals:
                inter_s = max(c_s, i_s)
                inter_e = min(c_e, i_e)
                if inter_e > inter_s:
                    overlap_with_compute += (inter_e - inter_s)
            
            # Attribute bytes proportionally to the overlap time
            bytes_during_overlap += b * (overlap_with_compute / io_dur)

        eff_bw = (bytes_during_overlap / overlap_s) if overlap_s > 0 else 0
        
        return {
            "overlap_s": overlap_s,
            "compute_s": total_compute_s,
            "io_s": total_io_s,
            "percent": percent,
            "eff_bw_gb_s": eff_bw / (1024**3)
        }

    def record_token(self):
        self.token_times.append(time.perf_counter())

    def record_moe_cache(self, hit: bool):
        if hit: self.cache_stats['moe_hits'] += 1
        else: self.cache_stats['moe_misses'] += 1

    def record_pread(self, duration: float, bytes_read: int):
        # Threshold: > 2GB/s indicates it was already in RAM.
        threshold_s = bytes_read / (2.0 * 1024 * 1024 * 1024) 
        if duration < threshold_s:
            self.cache_stats['os_page_hits'] += 1
        else:
            self.cache_stats['os_page_misses'] += 1

    def analyze_bottlenecks(self) -> str:
        """The 'Oracle' - identifies where the system is bottlenecked."""
        # print(f"DEBUG: analyze_bottlenecks called on {id(self)}. IO intervals: {len(self.io_intervals)}, Compute intervals: {len(self.compute_intervals)}")
        overlap_info = self.calculate_true_overlap()
        
        total_io = overlap_info['io_s']
        total_compute = overlap_info['compute_s']
        
        durations = [self.token_times[i] - self.token_times[i-1] for i in range(1, len(self.token_times))]
        avg_tps = (len(durations) / sum(durations)) if durations and sum(durations) > 0 else 0
        avg_latency = (sum(durations) / len(durations)) if durations else 0
        
        total_moe = self.cache_stats['moe_hits'] + self.cache_stats['moe_misses']
        moe_miss_rate = self.cache_stats['moe_misses'] / max(1, total_moe)
        
        report = []
        report.append("\n" + "="*50)
        report.append("🔍 MLX-FLASH PROFILER ORACLE")
        report.append("="*50)
        
        report.append(f"Tokens/Sec : {avg_tps:.2f} tok/s")
        report.append(f"Avg Latency: {avg_latency*1000:.1f} ms/tok")
        report.append(f"Total IO Wait  : {total_io:.2f} s")
        report.append(f"Total Compute  : {total_compute:.2f} s")
        report.append(f"True Overlap   : {overlap_info['overlap_s']:.2f} s ({overlap_info['percent']:.1f}% of compute)")
        report.append(f"Eff. Overlap BW: {overlap_info['eff_bw_gb_s']:.2f} GB/s")
        report.append(f"MoE Miss Rate  : {moe_miss_rate*100:.1f}%")
        
        report.append("\n--- OVERLAP PROOF ---")
        if overlap_info['percent'] > 50:
            report.append(f"✅ REAL OVERLAP: {overlap_info['percent']:.1f}% of GPU time was concurrent with SSD I/O.")
        elif overlap_info['percent'] > 5:
            report.append(f"⚠️  PARTIAL OVERLAP: Only {overlap_info['percent']:.1f}% overlap. Pipeline may be stalling.")
        else:
            report.append(f"❌ PIPELINED ILLUSION: Overlap is {overlap_info['percent']:.1f}%. System is behaving sequentially.")

        report.append("\n--- DIAGNOSIS ---")
        if moe_miss_rate > 0.8 and avg_latency > 3.0:
            report.append("⚠️  STATE: THRASHING")
            report.append("The working set size exceeds available RAM. The OS is evicting actively needed pages.")
        elif total_io > total_compute * 1.5:
            report.append("⚠️  STATE: IO-BOUND (GPU is Starving)")
        elif total_compute > total_io * 1.5:
            report.append("✅  STATE: COMPUTE-BOUND")
        else:
            report.append("✅  STATE: BALANCED")
            
        return "\n".join(report)

    def print_waterfall(self):
        """Prints a visual ASCII waterfall chart of Layer IO vs Compute."""
        if not self.layer_stats: return
        
        print("\n--- LAYER WATERFALL (Avg ms per call) ---")
        max_time = max((s['io_wait'] + s['compute']) / max(1, s['calls']) for s in self.layer_stats.values())
        
        # Terminal width budget for the bar chart
        bar_width = 40 
        
        for layer_idx in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_idx]
            calls = max(1, stats['calls'])
            avg_io_ms = (stats['io_wait'] / calls) * 1000
            avg_comp_ms = (stats['compute'] / calls) * 1000
            total_ms = avg_io_ms + avg_comp_ms
            
            io_chars = int((avg_io_ms / (max_time * 1000)) * bar_width) if max_time > 0 else 0
            comp_chars = int((avg_comp_ms / (max_time * 1000)) * bar_width) if max_time > 0 else 0
            
            bar = "\033[91m" + ("=" * io_chars) + "\033[0m" + "\033[92m" + ("=" * comp_chars) + "\033[0m"
            # Pad to keep alignment
            pad = " " * (bar_width - (io_chars + comp_chars))
            
            print(f"L{layer_idx:02d} | {bar}{pad} | {total_ms:5.1f}ms (IO: {avg_io_ms:4.1f}, Comp: {avg_comp_ms:4.1f})")

    def export(self, filepath="/tmp/mlx_flash_profile.json"):
        report = {
            "layer_stats": dict(self.layer_stats),
            "cache_stats": self.cache_stats,
            "token_times": self.token_times
        }
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
