import time
import collections

class UnifiedBandwidthController:
    """
    Model Predictive Control (MPC-lite) bandwidth scheduler for Apple Silicon.
    Predicts the next-step bandwidth demand based on layer compute times and 
    queued IO, preemptively adjusting the IO rate to prevent stalls.
    """
    def __init__(self, target_degradation=0.05):
        self.target_degradation = target_degradation
        self.base_times = {}
        
        self.B_limit = 1e9     # Start optimistic: 1 GB/s
        self.B_max = 5e9       # Physical ceiling (5 GB/s)
        self.B_min = 1e7       # 10 MB/s absolute minimum
        
        self.ema_alpha = 0.3
        self.current_ema = {}

        # Token Bucket State
        self.tokens = 2 * 1024 * 1024
        self.max_tokens = 2 * 1024 * 1024
        self.last_token_update = time.perf_counter()
        
        # Predictive State
        self.pending_io_bytes = 0
        self.current_layer = 0

    def update_stats(self, bytes_read: int, duration_sec: float):
        """Called by IO worker after a read completes."""
        self.pending_io_bytes = max(0, self.pending_io_bytes - bytes_read)

    def enqueue_io(self, bytes_to_read: int):
        """Called by the executor when prefetch is scheduled."""
        self.pending_io_bytes += bytes_to_read

    def notify_layer_start(self, layer_idx: int):
        """Predicts required bandwidth for the upcoming compute phase."""
        self.current_layer = layer_idx
        
        predicted_compute_time = self.current_ema.get(layer_idx, self.base_times.get(layer_idx, 0.0))
        
        if predicted_compute_time > 0 and self.pending_io_bytes > 0:
            # We want to finish reading pending bytes during this compute step
            required_bw = self.pending_io_bytes / predicted_compute_time
            
            # Add a 10% safety margin
            target_bw = required_bw * 1.1
            
            # Clamp to physical limits
            self.B_limit = max(self.B_min, min(self.B_max, target_bw))
        else:
            # If no IO pending or no prediction, fall back to a safe default
            self.B_limit = self.B_max if self.pending_io_bytes > 0 else self.B_min

    def register_compute_time(self, layer_idx: int, t_comp: float):
        """Updates the predictive model with actual compute times."""
        # 1. Warmup / Uncontended Baseline
        if layer_idx not in self.base_times:
            self.base_times[layer_idx] = t_comp
            self.current_ema[layer_idx] = t_comp
            return

        # Outlier rejection (e.g., OS context switch, SLC flush)
        if t_comp > self.base_times[layer_idx] * 3.0:
            return 

        # 2. Filter Update (Learns non-stationary workloads)
        self.current_ema[layer_idx] = (self.ema_alpha * t_comp) + \
                                      ((1 - self.ema_alpha) * self.current_ema[layer_idx])

    def consume_tokens(self, requested_bytes: int) -> float:
        """Returns sleep time required to satisfy requested bytes."""
        now = time.perf_counter()
        elapsed = now - self.last_token_update
        self.last_token_update = now
        
        # Add new tokens based on B_limit
        new_tokens = elapsed * self.B_limit
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        
        if self.tokens >= requested_bytes:
            self.tokens -= requested_bytes
            return 0.0
            
        # Deficit handling
        deficit = requested_bytes - self.tokens
        self.tokens = 0.0 # Consume whatever we have
        sleep_sec = deficit / self.B_limit
        return sleep_sec
