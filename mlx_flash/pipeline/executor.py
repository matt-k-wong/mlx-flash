import mlx.core as mx
import mlx.nn as nn

class PipelinedExecutor:
    """
    Executes a transformer layer by breaking it into sub-components and explicitly 
    interleaving I/O prefetch requests with GPU compute dispatches.
    
    This replaces the monolithic `layer.forward(x)` with a state machine that
    understands the sub-tensors (wqkv, wo, w_gate_up, w_down).
    """
    def __init__(self, mmap_cache):
        self.mmap_cache = mmap_cache
        
    def _enqueue_tensor(self, layer_idx: int, tensor_name_hint: str):
        """Asks the background worker to load specific tensors based on name hints."""
        if not self.mmap_cache:
            return
            
        ranges = self.mmap_cache.get_layer_ranges(layer_idx)
        # In a fully realized system, we would filter `ranges` by tensor_name_hint
        # For now, we use the existing layer-level prefetch as a coarse fallback
        # if fine-grained isn't implemented in the cache yet.
        for _, (start, end, filename) in ranges.items():
             self.mmap_cache.prefetch_worker.enqueue(filename, start, end - start, layer_idx)

    def _wait_for_layer(self, layer_idx: int):
        if self.mmap_cache and hasattr(self.mmap_cache, 'wait_for_layer'):
            self.mmap_cache.wait_for_layer(layer_idx)

    def execute_dense_layer(self, x: mx.array, layer: nn.Module, layer_idx: int, mask=None, cache=None) -> mx.array:
        """
        Executes a standard dense transformer layer with internal pipelining.
        """
        # ==========================================
        # PHASE 1: PRE-ATTENTION PIPELINE
        # ==========================================
        # IO: Start fetching Attention weights
        self._enqueue_tensor(layer_idx, "attention")
        
        # GPU: Input Norm (fast, requires no SSD weights)
        if hasattr(layer, "input_layernorm"):
            h = layer.input_layernorm(x)
        elif hasattr(layer, "norm"):
            h = layer.norm(x)
        else:
            h = x
            
        # ==========================================
        # PHASE 2: ATTENTION OVERLAP
        # ==========================================
        # Wait for Attention weights
        self._wait_for_layer(layer_idx) 
        
        # IO: The moment attention begins computing, prefetch the MLP
        self._enqueue_tensor(layer_idx, "mlp")
        
        # GPU: Attention computation
        # (Assuming the layer has a standard self_attn or attention attribute)
        attn_module = getattr(layer, "self_attn", getattr(layer, "attention", None))
        
        if attn_module is not None:
            call_kwargs = {}
            if mask is not None: call_kwargs["mask"] = mask
            if cache is not None: call_kwargs["cache"] = cache
            
            attn_out = attn_module(h, **call_kwargs)
            # Handle tuple returns (some attention modules return cache states)
            attn_out = attn_out[0] if isinstance(attn_out, (list, tuple)) else attn_out
            
            x = x + attn_out
            
            # Force evaluation and memory clear before moving to MLP
            mx.eval(x)
            if cache is not None:
                 if hasattr(cache, "state") and cache.state is not None:
                     mx.eval(*[s for s in cache.state if s is not None])
                 elif hasattr(cache, "keys") and cache.keys is not None:
                     mx.eval(cache.keys, cache.values)
            mx.metal.clear_cache()
            
        # ==========================================
        # PHASE 3: MLP EXECUTION
        # ==========================================
        # GPU: Post-attention norm
        if hasattr(layer, "post_attention_layernorm"):
            h = layer.post_attention_layernorm(x)
        elif hasattr(layer, "norm_f"): # Specific to some architectures
             h = layer.norm_f(x)
        else:
            h = x
            
        # Wait for MLP weights (likely already loaded due to prefetch in Phase 2)
        self._wait_for_layer(layer_idx)
        
        # GPU: MLP computation
        mlp_module = getattr(layer, "mlp", getattr(layer, "mixer", None))
        if mlp_module is not None:
            mlp_out = mlp_module(h)
            x = x + mlp_out
            
        # Final layer sync
        mx.eval(x)
        mx.metal.clear_cache()
        
        return x
