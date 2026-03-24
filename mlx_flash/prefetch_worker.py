import contextlib
import os
import queue
import threading
import time
from typing import Any, Optional


class BackgroundPrefetcher:
    """
    Background worker that forces SSD data into the macOS unified page cache
    using an adaptive sliding window for optimal bandwidth utilization.
    """
    def __init__(self, file_handles: dict[str, Any]):
        self.file_handles = file_handles
        
        # Adaptive tuning state
        self.k_distance = 1
        self.max_k = 3
        self.compute_ema = 0.0
        self.io_ema = 0.0
        
        from mlx_flash.bandwidth.controller import UnifiedBandwidthController
        self.bandwidth_controller = UnifiedBandwidthController()
        
        self.queue: queue.Queue[tuple[Optional[int], str, int, int]] = queue.Queue(maxsize=16) 
        self.completed_prefetches = set()
        self._lock = threading.Lock()
        
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        # Base ~16MB chunking provides excellent sustained SSD queue depth
        base_chunk_size = 16 * 1024 * 1024 
        
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    continue
                    
                layer_idx, filename, offset, length, align_bytes = task
                f = self.file_handles.get(filename)
                if f is None:
                    # Try to open it if missing
                    try:
                        f = open(filename, "rb")
                        self.file_handles[filename] = f
                    except Exception:
                        pass
                
                t0 = time.perf_counter()
                
                if f is not None:
                    fd = f.fileno()
                    end = offset + length
                    curr = offset
                    
                    while curr < end and self.running:
                        try:
                            # 1. Ask Bandwidth Controller for throttle limits
                            requested_chunk = min(base_chunk_size, end - curr)
                            approved_chunk, sleep_sec = self.bandwidth_controller.calculate_throttle(requested_chunk)
                            
                            if sleep_sec > 0:
                                time.sleep(sleep_sec)
                                
                            # Ensure alignment
                            chunk_size = (approved_chunk // align_bytes) * align_bytes if align_bytes > 0 else approved_chunk
                            if chunk_size == 0: chunk_size = align_bytes # Prevent infinite loop
                            chunk_size = min(chunk_size, end - curr)
                            
                            t_read_0 = time.perf_counter()
                            os.pread(fd, chunk_size, curr)
                            t_read_1 = time.perf_counter()
                            
                            duration = t_read_1 - t_read_0
                            self.bandwidth_controller.update_stats(chunk_size, duration)
                            
                            try:
                                from benchmarks.profiler.profiler import StreamingProfiler
                                prof = StreamingProfiler()
                                prof.record_pread(duration, chunk_size)
                                prof.record_io_interval(t_read_0, t_read_1, chunk_size)
                            except Exception:
                                pass
                                
                            curr += chunk_size
                        except Exception:
                            break
                            
                io_time = time.perf_counter() - t0
                if layer_idx is not None:
                    self._update_io_ema(io_time)
                    with self._lock:
                        self.completed_prefetches.add(layer_idx)
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def _update_io_ema(self, new_val: float, alpha: float = 0.3):
        self.io_ema = (alpha * new_val) + ((1 - alpha) * self.io_ema) if self.io_ema > 0 else new_val

    def record_compute_time(self, compute_time: float):
        """Called by main thread to adjust prefetch distance."""
        alpha = 0.3
        self.compute_ema = (alpha * compute_time) + ((1 - alpha) * self.compute_ema) if self.compute_ema > 0 else compute_time
        
        if self.io_ema > self.compute_ema * 1.5 and self.k_distance < self.max_k:
            self.k_distance += 1
        elif self.compute_ema > self.io_ema * 1.5 and self.k_distance > 1:
            self.k_distance -= 1

    def wait_for_layer(self, layer_idx: int):
        """Stall if the layer isn't fully in page cache to prevent hard page fault."""
        while self.running:
            with self._lock:
                if layer_idx in self.completed_prefetches:
                    break
            if self.io_ema <= 0:
                break
            time.sleep(0.001)

    def enqueue(self, filename: str, offset: int, length: int, layer_idx: Optional[int] = None, align_bytes: int = 1):
        if not self.running:
            return
        if layer_idx is not None:
            with self._lock:
                self.completed_prefetches.discard(layer_idx)
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait((layer_idx, filename, offset, length, align_bytes))

    def shutdown(self):
        self.running = False
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass
        self.thread.join(timeout=1.0)
