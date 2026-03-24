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
        
        self.queue: queue.Queue[tuple[Optional[int], str, int, int]] = queue.Queue(maxsize=16) 
        self.completed_prefetches = set()
        
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        # 16MB chunking provides excellent sustained SSD queue depth
        chunk_size = 16 * 1024 * 1024 
        
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    continue
                    
                layer_idx, filename, offset, length = task
                f = self.file_handles.get(filename)
                
                t0 = time.perf_counter()
                
                if f is not None:
                    fd = f.fileno()
                    end = offset + length
                    curr = offset
                    while curr < end and self.running:
                        try:
                            chunk = min(chunk_size, end - curr)
                            os.pread(fd, chunk, curr)
                            curr += chunk
                        except Exception:
                            break
                            
                io_time = time.perf_counter() - t0
                if layer_idx is not None:
                    self._update_io_ema(io_time)
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
        while layer_idx not in self.completed_prefetches and self.io_ema > 0 and self.running:
            time.sleep(0.001)

    def enqueue(self, filename: str, offset: int, length: int, layer_idx: Optional[int] = None):
        if not self.running:
            return
        if layer_idx is not None:
            self.completed_prefetches.discard(layer_idx)
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait((layer_idx, filename, offset, length))

    def shutdown(self):
        self.running = False
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass
        self.thread.join(timeout=1.0)
