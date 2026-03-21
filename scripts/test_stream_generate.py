
import argparse
import time
import psutil
import os
import mlx.core as mx
from mlx_engine_flash import FlashConfig
from mlx_engine_flash.integration.lmstudio import apply_flash_patch

def log_rss(stage: str):
    rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    print(f"[flash-RAM] {stage}: {rss_gb:.2f} GB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-size", type=int, default=2000)
    args = parser.parse_args()

    log_rss("Initial State")

    # Configure mlx-flash
    config = FlashConfig(enabled=True, ram_budget_gb=8.0, debug=True, prefetch_layers=0)
    apply_flash_patch(config)
    
    from mlx_lm import load, stream_generate
    
    log_rss("Before patched load")
    model, tokenizer = load(args.model)
    log_rss("After load (skeleton should be ~0.4 GB)")
    
    prompt = "This is a test to simulate long context. " * (args.prompt_size // 8)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")

    print("\n--- Starting stream_generate ---")
    start_time = time.time()
    try:
        tokens_count = 0
        for response in stream_generate(model, tokenizer, prompt, max_tokens=5):
            tokens_count += 1
            log_rss(f"During generation token {tokens_count}")
        
        print(f"\nGeneration complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\n\nCaught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
