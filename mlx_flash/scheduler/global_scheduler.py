import mlx.core as mx
import time
import heapq
from typing import Dict, List, Any, Optional
from collections import deque
import threading

class ResourceTask:
    """Represents a discrete unit of work for the global scheduler."""
    def __init__(self, task_id: str, layer_idx: int, task_type: str, priority_score: float, 
                 dependencies: List[str] = None, callback=None):
        self.task_id = task_id
        self.layer_idx = layer_idx
        self.task_type = task_type  # 'io_read', 'gpu_compute', 'evict_cache'
        self.priority_score = priority_score # Higher is more urgent (e.g., blocking compute)
        self.dependencies = set(dependencies) if dependencies else set()
        self.callback = callback
        
    def __lt__(self, other):
        # Min-heap based on negative priority (so highest priority is popped first)
        return self.priority_score > other.priority_score

class UnifiedCostModel:
    def __init__(self, bandwidth_gb_s: float = 3.5, max_ram_gb: float = 8.0, total_layers: int = 32):
        self.bw_bytes_s = bandwidth_gb_s * 1024**3
        self.max_ram_bytes = max_ram_gb * 1024**3
        self.total_layers = total_layers
        
        # Track moving averages to adapt predictions
        self.ema_compute_ms_per_mb = 1.0
        self.ema_io_ms_per_mb = (1 / self.bw_bytes_s) * 1000 * 1024**2
        
    def predict_io_time(self, size_bytes: int) -> float:
        return (size_bytes / 1e6) * self.ema_io_ms_per_mb
        
    def predict_compute_time(self, size_bytes: int) -> float:
        return (size_bytes / 1e6) * self.ema_compute_ms_per_mb
        
    def get_urgency_score(self, current_layer: int, target_layer: int, 
                          is_blocking: bool, task_type: str, freq: float = 1.0) -> float:
        """
        Calculates how critical a task is to preventing a GPU stall.
        """
        # Distance to execution (wrap around for cyclic generation)
        distance = target_layer - current_layer
        if distance < 0: distance += self.total_layers
        if distance == 0 and task_type == 'io_read':
            # Highest possible urgency: we are waiting on this *right now*
            return float('inf')
            
        base_score = 10000.0 / max(1, distance)
        
        # MoE experts (stochastic distance) get multiplied by their predicted probability/frequency
        if task_type == 'io_read_expert':
            base_score *= freq
            
        return base_score

class GlobalScheduler:
    """
    Central brain for an out-of-core streaming inference engine.
    Instead of isolated components making blind decisions, the GlobalScheduler
    owns the 'Task Graph' and issues commands to the IO threads and GPU dispatch.
    """
    def __init__(self, cost_model: UnifiedCostModel, cache_manager, io_worker):
        self.cost_model = cost_model
        self.cache = cache_manager
        self.io = io_worker
        
        # Event loop structures
        self.pending_tasks: Dict[str, ResourceTask] = {}
        self.ready_queue = [] # Priority Queue (Heap)
        self.active_tasks: Dict[str, ResourceTask] = {}
        
        self.current_execution_layer = 0
        self.lock = threading.Lock()
        
    def submit_graph(self, layer_idx: int, graph_nodes: List[Dict[str, Any]]):
        """
        Receives a computational subgraph for a specific layer.
        E.g., [{id: 'L1_wqkv_io', type: 'io_read', deps: []}, {id: 'L1_attn_comp', type: 'gpu_compute', deps: ['L1_wqkv_io']}]
        """
        with self.lock:
            for node in graph_nodes:
                task = ResourceTask(
                    task_id=node['id'],
                    layer_idx=layer_idx,
                    task_type=node['type'],
                    priority_score=self.cost_model.get_urgency_score(self.current_execution_layer, layer_idx, False, node['type']),
                    dependencies=node.get('deps', []),
                    callback=node.get('callback')
                )
                
                self.pending_tasks[task.task_id] = task
                
                # If no dependencies, it's ready to schedule immediately
                if not task.dependencies:
                    heapq.heappush(self.ready_queue, task)

    def mark_completed(self, task_id: str, duration_ms: float, size_bytes: int = 0):
        """Called by workers (IO or GPU) when a task finishes."""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                
                # Update cost model
                if task.task_type.startswith('io'):
                    self.cost_model.ema_io_ms_per_mb = 0.8 * self.cost_model.ema_io_ms_per_mb + 0.2 * (duration_ms / max(1, size_bytes/1e6))
                elif task.task_type == 'gpu_compute':
                    self.cost_model.ema_compute_ms_per_mb = 0.8 * self.cost_model.ema_compute_ms_per_mb + 0.2 * (duration_ms / max(1, size_bytes/1e6))
                
                # Resolve dependencies
                ready_new = []
                for p_id, p_task in self.pending_tasks.items():
                    if task_id in p_task.dependencies:
                        p_task.dependencies.remove(task_id)
                        if not p_task.dependencies:
                            ready_new.append(p_id)
                            
                for p_id in ready_new:
                    ready_task = self.pending_tasks.pop(p_id)
                    # Re-evaluate priority before pushing
                    ready_task.priority_score = self.cost_model.get_urgency_score(self.current_execution_layer, ready_task.layer_idx, True, ready_task.task_type)
                    heapq.heappush(self.ready_queue, ready_task)

    def schedule_tick(self):
        """
        The core event loop step. Evaluates the ready queue and dispatches
        to the appropriate hardware resource (IO Thread or Metal GPU).
        """
        with self.lock:
            # 1. Proactive Memory Management (Eviction Phase)
            # Before scheduling IO, ensure we have RAM. If not, schedule eviction.
            if self.cache.current_hot_bytes > self.cache.hot_budget * 0.9:
                self.cache._evict_from(1) # Demote Hot to Warm
                
            if self.cache.current_warm_bytes > self.cache.warm_budget * 0.9:
                # We are critically low on RAM. Boost priority of any pending `MADV_DONTNEED` tasks.
                pass
                
            # 2. Dispatch Phase
            if not self.ready_queue:
                return
                
            # Pop the most critical task
            best_task = heapq.heappop(self.ready_queue)
            self.active_tasks[best_task.task_id] = best_task
            
        # Dispatch without holding the lock to prevent stalling the scheduler
        if best_task.task_type.startswith('io_read'):
            # Tell the prefetch worker to go
            self.io.enqueue_task(best_task)
            
        elif best_task.task_type == 'gpu_compute':
            # Tell the execution pipeline to dispatch MLX kernels
            if best_task.callback:
                best_task.callback()
                
    def on_router_decision(self, layer_idx: int, top_k_experts: List[int], probabilities: List[float]):
        """
        Integration point for MoE. The moment the router finishes, the scheduler dynamically 
        injects high-priority IO tasks for the required experts.
        """
        with self.lock:
            for i, exp_idx in enumerate(top_k_experts):
                task_id = f"L{layer_idx}_expert_{exp_idx}_io"
                
                # Only inject if it's not already cached
                if not self.cache.is_cached(layer_idx, exp_idx):
                    task = ResourceTask(
                        task_id=task_id,
                        layer_idx=layer_idx,
                        task_type='io_read_expert',
                        # Scale urgency by router probability
                        priority_score=self.cost_model.get_urgency_score(self.current_execution_layer, layer_idx, True, 'io_read_expert', freq=probabilities[i]),
                        dependencies=[]
                    )
                    heapq.heappush(self.ready_queue, task)
