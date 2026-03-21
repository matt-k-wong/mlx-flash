import os
import queue
import threading
from typing import Any

class BackgroundPrefetcher:
    """
    Background worker that forces SSD data into the macOS unified page cache.
    
    Why not just madvise or mmap slicing?
    1. madvise(MADV_WILLNEED) is an asynchronous hint. The main thread will still
       block on evaluating the array if the SSD hasn't caught up.
    2. Slicing `mmap` in Python triggers a hard page fault. The OS halts the thread
       at the hardware level *while it holds the Python GIL*. This freezes
       the main GPU dispatch thread.
       
    By using standard file I/O (`os.read`), Python explicitly releases the GIL
    before issuing the blocking syscall. This allows the OS to physically pump 
    data from the SSD into the unified RAM pool while the main thread 
    continues dispatching kernels to the GPU completely unabated.
    """
    def __init__(self, file_handles: dict[str, Any]):
        self.file_handles = file_handles
        self.queue = queue.Queue(maxsize=16) # Don't get too far ahead
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        # 16MB chunking provides excellent sustained SSD queue depth
        # without starving other system processes
        CHUNK_SIZE = 16 * 1024 * 1024 
        
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    continue
                    
                filename, offset, length = task
                f = self.file_handles.get(filename)
                
                if f is not None:
                    # Get raw file descriptor for purely OS-level I/O bypassing Python buffers
                    fd = f.fileno()
                    
                    end = offset + length
                    curr = offset
                    while curr < end and self.running:
                        try:
                            # os.pread explicitly releases the GIL
                            chunk = min(CHUNK_SIZE, end - curr)
                            os.pread(fd, chunk, curr)
                            curr += chunk
                        except Exception:
                            # If read fails (e.g. file swapped), break
                            break
                            
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Catch any unexpected errors to prevent thread death
                pass

    def enqueue(self, filename: str, offset: int, length: int):
        if not self.running:
            return
        try:
            # Non-blocking put to avoid main thread getting stuck
            # if the SSD queue is backed up
            self.queue.put_nowait((filename, offset, length))
        except queue.Full:
            pass

    def shutdown(self):
        self.running = False
        try:
            # Drain queue
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass
        self.thread.join(timeout=1.0)
