import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import time


class TiledColumnLinear(nn.Module):
    """
    Expanding linear layer (e.g., MLP Up/Gate or Attention Q/K/V).
    Partitions the output features into tiles to reduce peak memory.
    """
    def __init__(self, original_linear: nn.Linear, tile_size: int = 1024):
        super().__init__()
        self.weight = getattr(original_linear, "weight")
        self.bias = getattr(original_linear, "bias", None)
        self.tile_size = tile_size
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        outputs = []
        
        for i in range(0, self.out_features, self.tile_size):
            t0 = time.perf_counter()
            # Slicing the lazy weight array. 
            # In MLX, this avoids materializing the whole weight if it's on disk.
            w_tile = self.weight[i:i+self.tile_size, :]
            
            # Matmul: x is [..., in_features], w_tile is [tile_size, in_features]
            # y_tile will be [..., tile_size]
            y_tile = mx.matmul(x, w_tile.T)
            
            if self.bias is not None:
                b_tile = self.bias[i:i+self.tile_size]
                y_tile = y_tile + b_tile
                
            mx.eval(y_tile) # Force evaluation to bound memory
            mx.synchronize()
            t1 = time.perf_counter()
            
            try:
                from benchmarks.profiler.profiler import StreamingProfiler
                StreamingProfiler().record_compute_interval(t0, t1, "tiled_column")
            except ImportError:
                pass
                
            outputs.append(y_tile)
            
            # Clear intermediate metal buffers
            del w_tile
            mx.metal.clear_cache()
            
        # Concatenate along the feature dimension
        return mx.concatenate(outputs, axis=-1)


class TiledRowLinear(nn.Module):
    """
    Contracting linear layer (e.g., MLP Down or Attention O).
    Partitions the input features into tiles and accumulates the result.
    Requires FP32 accumulation to prevent precision loss.
    """
    def __init__(self, original_linear: nn.Linear, tile_size: int = 1024):
        super().__init__()
        self.weight = getattr(original_linear, "weight")
        self.bias = getattr(original_linear, "bias", None)
        self.tile_size = tile_size
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        # CRITICAL: Accumulate in FP32 to prevent catastrophic cancellation
        # from many small additions
        original_dtype = x.dtype
        y_accum = mx.zeros((*x.shape[:-1], self.out_features), dtype=mx.float32)
        
        for i in range(0, self.in_features, self.tile_size):
            t0 = time.perf_counter()
            # Weight slice: [out_features, tile_size]
            w_tile = self.weight[:, i:i+self.tile_size]
            
            # Activation slice: [..., tile_size]
            x_tile = x[..., i:i+self.tile_size]
            
            # Compute partial dot product in FP32
            y_partial = mx.matmul(x_tile.astype(mx.float32), w_tile.T.astype(mx.float32))
            
            y_accum = y_accum + y_partial
            mx.eval(y_accum) # Force evaluation to bound memory
            mx.synchronize()
            t1 = time.perf_counter()
            
            try:
                from benchmarks.profiler.profiler import StreamingProfiler
                StreamingProfiler().record_compute_interval(t0, t1, "tiled_row")
            except ImportError:
                pass
            
            del w_tile, x_tile, y_partial
            mx.metal.clear_cache()
            
        if self.bias is not None:
            y_accum = y_accum + self.bias.astype(mx.float32)
            
        # Cast back to the original precision
        return y_accum.astype(original_dtype)


def apply_tiling(model: nn.Module, tile_size: int = 1024):
    """
    Recursively replaces target nn.Linear layers in the model with Tiled versions.
    Heuristics are used to guess which layers expand and which contract based on name.
    """
    for name, module in list(model.named_modules()):
        # Iterate over attributes to replace them in-place
        for child_name, child in list(module.items()) if hasattr(module, 'items') else []:
            if isinstance(child, nn.Linear):
                
                # Heuristic 1: Expanding Layers (Column-wise)
                # These take a smaller dim and project to a larger one, or keep same (Q/K/V)
                if any(x in child_name for x in ["up_proj", "gate_proj", "q_proj", "k_proj", "v_proj"]):
                    setattr(module, child_name, TiledColumnLinear(child, tile_size))
                    
                # Heuristic 2: Contracting Layers (Row-wise)
                # These take a large concatenated dim and project down
                elif any(x in child_name for x in ["down_proj", "o_proj"]):
                    setattr(module, child_name, TiledRowLinear(child, tile_size))
                    
                # If it's a fused QKV, it expands. Fused Gate+Up, it expands.
                elif child_name == "wqkv":
                    setattr(module, child_name, TiledColumnLinear(child, tile_size))
