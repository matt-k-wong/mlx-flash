import time
import os
import psutil
from mlx_flash import FlashConfig
from mlx_flash.integration.lmstudio import apply_flash_patch
import mlx_lm

# 1. SETUP MODEL PATH (Found in your LM Studio cache)
MODEL_PATH = "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"

# 2. CONFIGURE FLASH MODE (Memory vs Speed Tradeoff)
# RAM_BUDGET_GB: 
#   - 1.0 - 2.0: "Slow but Invincible" (Extreme weight streaming)
#   - 8.0 - 16.0: "Balanced" (Some layers cached in RAM, others streamed)
#   - 32.0+: "Performance" (Full 30B model in RAM, if your hardware allows)
RAM_BUDGET_GB = 2.0 

cfg = FlashConfig(
    enabled=True,
    ram_budget_gb=RAM_BUDGET_GB,
    disk_kv_enabled=True,
    debug=True  # Will log when "Streaming active" (budget exceeded)
)

def get_rss_gb():
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

def run_test_prompt(model, tokenizer, prompt_title, prompt_text):
    print(f"\n-Testing: {prompt_title}-")
    print(f"Initial RAM: {get_rss_gb():.2f} GB")
    
    t0 = time.monotonic()
    token_count = 0
    
    # Use stream_generate to see tokens as they arrive
    for response in mlx_lm.stream_generate(model, tokenizer, prompt_text, max_tokens=150):
        print(response.text, end="", flush=True)
        token_count += 1
        
    t1 = time.monotonic()
    print(f"\n\nFinal RAM: {get_rss_gb():.2f} GB")
    print(f"Time: {t1 - t0:.1f}s | Speed: {token_count / (t1 - t0):.1f} tok/s")

def main():
    print(f"=== mlx-flash 30B NATIVE PROOF ===")
    print(f"Model ID: NVIDIA-Nemotron-30B")
    
    # A. Apply the Patch
    apply_flash_patch(cfg)
    
    # B. Load the Model (Weights will be lazy-loaded)
    print("Loading model (Lazy)...")
    model, tokenizer = mlx_lm.load(MODEL_PATH)
    print(f"Model Loaded. Current RAM RSS: {get_rss_gb():.2f} GB")
    
    # C. Run the 3 Test Prompts
    prompts = [
        ("Reasoning & Metaphor", "Explain the 'Observer Effect' using a pizza delivery box metaphor. The pizza is both pepperoni and plain cheese until the lid is opened."),
        ("Creative Experience", "Write a short story about an AI experiencing Flash Weight Streaming—the transition from fast RAM to slow but tectonic SSD."),
        ("Long Context Logic", "If a model has 30 billion parameters but the user only has 16GB of RAM, explain how weight streaming bypasses the physical limit.")
    ]
    
    for title, text in prompts:
        run_test_prompt(model, tokenizer, title, text)
        print("-" * 40)

if __name__ == "__main__":
    main()
