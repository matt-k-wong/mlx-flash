#!/usr/bin/env python3
import time
import os
import gc
import json
import struct
import shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
import psutil
from mlx_flash import FlashConfig, FlashManager, FlashGenerationLoop

def get_rss_mb():
    # Resident Set Size (Physical RAM)
    return psutil.Process().memory_info().rss / 1024 / 1024

def build_benchmark_model(model_path: Path):
    """Build a 1.5GB synthetic model for benchmarking."""
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 2 layers, 32 heads, 4096 dim
    # Each layer weight (weight matrix) is roughly 750MB
    weights = {
        "model.embed_tokens.weight": mx.zeros((100, 4096), dtype=mx.float16),
        "model.norm.weight": mx.ones((4096,), dtype=mx.float16),
        "lm_head.weight": mx.zeros((100, 4096), dtype=mx.float16),
    }
    
    for i in range(2):
        weights[f"model.layers.{i}.input_layernorm.weight"] = mx.ones((4096,), dtype=mx.float16)
        weights[f"model.layers.{i}.post_attention_layernorm.weight"] = mx.ones((4096,), dtype=mx.float16)
        weights[f"model.layers.{i}.self_attn.q_proj.weight"] = mx.random.uniform(shape=(4096, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.self_attn.k_proj.weight"] = mx.random.uniform(shape=(4096, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.self_attn.v_proj.weight"] = mx.random.uniform(shape=(4096, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = mx.random.uniform(shape=(4096, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.mlp.gate_proj.weight"] = mx.random.uniform(shape=(11008, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.mlp.up_proj.weight"] = mx.random.uniform(shape=(11008, 4096), dtype=mx.float16)
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = mx.random.uniform(shape=(4096, 11008), dtype=mx.float16)
    
    mx.save_safetensors(str(model_path / "model.safetensors"), weights)
    
    config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f)

def run_benchmark_iter(model_path: Path, context_len: int, disk_kv: bool):
    print(f"\n>>> Context: {context_len:5d} tokens | Disk KV: {str(disk_kv):5s}")
    
    # Tiny RAM budget to force Flash streaming
    cfg = FlashConfig(
        enabled=True,
        ram_budget_gb=0.1, 
        disk_kv_enabled=disk_kv,
        disk_kv_dir="/tmp/bench_kv"
    )
    
    loop = FlashGenerationLoop(str(model_path), config=cfg)
    prompt = [0] * context_len
    
    # Cleanup before run
    mx.metal.clear_cache()
    gc.collect()
    
    start_rss = get_rss_mb()
    t0 = time.monotonic()
    
    # 1. MEASURE PREFILL
    # We generate exactly 1 token to trigger the prefill of the context_len
    gen = loop.stream_generate(prompt, max_tokens=1)
    try:
        next(gen)
    except StopIteration:
        pass
    
    t_prefill = time.monotonic() - t0
    prefill_ts = context_len / t_prefill if t_prefill > 0 else 0
    
    # 2. MEASURE GENERATION (DECODE)
    # Generate 5 more tokens
    t1 = time.monotonic()
    tokens_count = 0
    for _ in range(5):
        try:
            next(gen)
            tokens_count += 1
        except StopIteration:
            break
    
    t_gen = time.monotonic() - t1
    gen_ts = tokens_count / t_gen if t_gen > 0 else 0
    
    peak_rss = get_rss_mb()
    ram_growth = peak_rss - start_rss
    
    return {
        "prefill_ts": prefill_ts,
        "gen_ts": gen_ts,
        "ram_growth_mb": ram_growth,
        "peak_rss_mb": peak_rss
    }

def main():
    bench_dir = Path("/tmp/bench_model")
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    
    # Requirement: MLX 0.16.0+ for metal memory stats if we wanted them
    print("Building benchmark model...")
    build_benchmark_model(bench_dir)
    
    contexts = [512, 1024, 2048, 4096, 8192]
    results = []
    
    # Warmup
    run_benchmark_iter(bench_dir, 128, False)
    
    for ctx in contexts:
        # Standard vs Disk KV
        res_std = run_benchmark_iter(bench_dir, ctx, False)
        res_flash = run_benchmark_iter(bench_dir, ctx, True)
        
        results.append({
            "context": ctx,
            "std": res_std,
            "flash": res_flash
        })
        
    # Pretty Print results
    print("\n" + "="*80)
    print(f"{'Context':<10} | {'Mode':<8} | {'Prefill T/s':<12} | {'Gen T/s':<8} | {'RAM Gain (MB)':<12}")
    print("-" * 80)
    for r in results:
        c = r['context']
        s = r['std']
        f = r['flash']
        print(f"{c:<10} | {'STD':<8} | {s['prefill_ts']:>11.1f} | {s['gen_ts']:>7.1f} | {s['ram_growth_mb']:>11.1f}")
        print(f"{'':<10} | {'FLASH':<8} | {f['prefill_ts']:>11.1f} | {f['gen_ts']:>7.1f} | {f['ram_growth_mb']:>11.1f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
