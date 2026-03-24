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
            w_tile = self.weight[i:i+self.tile_size, :]
            y_tile = mx.matmul(x, w_tile.T)
            if self.bias is not None:
                b_tile = self.bias[i:i+self.tile_size]
                y_tile = y_tile + b_tile
            mx.eval(y_tile)
            mx.synchronize()
            t1 = time.perf_counter()
            try:
                from benchmarks.profiler.profiler import StreamingProfiler
                StreamingProfiler().record_compute_interval(t0, t1, "tiled_column")
            except ImportError:
                pass
            outputs.append(y_tile)
            del w_tile
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
        original_dtype = x.dtype
        y_accum = mx.zeros((*x.shape[:-1], self.out_features), dtype=mx.float32)
        for i in range(0, self.in_features, self.tile_size):
            t0 = time.perf_counter()
            w_tile = self.weight[:, i:i+self.tile_size]
            x_tile = x[..., i:i+self.tile_size]
            y_partial = mx.matmul(x_tile.astype(mx.float32), w_tile.T.astype(mx.float32))
            y_accum = y_accum + y_partial
            mx.eval(y_accum)
            mx.synchronize()
            t1 = time.perf_counter()
            try:
                from benchmarks.profiler.profiler import StreamingProfiler
                StreamingProfiler().record_compute_interval(t0, t1, "tiled_row")
            except ImportError:
                pass
            del w_tile, x_tile, y_partial
        if self.bias is not None:
            y_accum = y_accum + self.bias.astype(mx.float32)
        return y_accum.astype(original_dtype)


def apply_tiling(model: nn.Module, tile_size: int = 1024):
    """
    Recursively replaces target nn.Linear layers in the model with Tiled versions.
    Uses path-based lookup to safely replace modules in the tree.
    """
    # 1. Map all modules by path
    all_modules = dict(model.named_modules())
    
    # 2. Identify candidates for replacement
    to_replace = []
    for path, module in all_modules.items():
        if isinstance(module, nn.Linear):
            # Apply heuristics based on path name
            # e.g., "model.layers.0.self_attn.q_proj"
            name = path.split(".")[-1]
            
            # Heuristic 1: Expanding Layers (Column-wise)
            if any(x in name for x in ["up_proj", "gate_proj", "q_proj", "k_proj", "v_proj"]) or name == "wqkv":
                to_replace.append((path, TiledColumnLinear(module, tile_size)))
                
            # Heuristic 2: Contracting Layers (Row-wise)
            elif any(x in name for x in ["down_proj", "o_proj"]):
                to_replace.append((path, TiledRowLinear(module, tile_size)))

    # 3. Perform replacements
    for path, new_module in to_replace:
        if "." in path:
            parent_path, child_name = path.rsplit(".", 1)
            parent = all_modules[parent_path]
            setattr(parent, child_name, new_module)
        else:
            # Top-level replacement (unlikely for linear layers in LLMs)
            pass
