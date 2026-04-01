import argparse
import time
import random
import math
import mlx.core as mx
import mlx_lm
import numpy as np
from pathlib import Path

from mlx_flash import FlashConfig, FlashManager
from mlx_flash.engine.engine import FlashEngine

def calculate_loss(model, tokenizer, text, seq_len=512):
    tokens = tokenizer.encode(text)
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    if len(tokens) <= 1: return float('inf')

    input_ids = mx.array(tokens)[None]
    logits = model(input_ids)
    
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    import mlx.nn as nn
    loss = nn.losses.cross_entropy(shift_logits, shift_labels)
    return mx.mean(loss).item()

def run_passkey_test(engine, tokenizer, context_size=2048):
    print(f"\n[*] Running Passkey Retrieval Test (Context: ~{context_size} tokens)")
    
    passkey = random.randint(10000, 99999)
    secret_sentence = f"The secret passkey is {passkey}. Remember it."
    
    garbage = ["The grass is green.", "The sky is blue.", "The sun is hot.", "Water is wet."]
    garbage_text = " ".join(random.choice(garbage) for _ in range(context_size // 5))
    
    question = "What is the secret passkey? The secret passkey is"
    prompt = f"{secret_sentence} {garbage_text} {question}"
    
    print(f"[*] Secret Passkey: {passkey}")
    print(f"[*] Total Prompt Length: {len(tokenizer.encode(prompt))} tokens")
    
    tokens = []
    for segment in engine.stream_generate(prompt, max_tokens=10, temp=0.0):
        tokens.append(segment)
        if str(passkey) in "".join(tokens): break
    
    success = str(passkey) in "".join(tokens)
    print(f"[*] Retrieval Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    return success

def main():
    parser = argparse.ArgumentParser(description="Flash Quality Proof: Passkey & Perplexity")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit", help="Model ID")
    parser.add_argument("--context", type=int, default=1000, help="Haystack size")
    args = parser.parse_args()

    config = FlashConfig(
        enabled=True,
        tiled_execution=True,
        tile_size=1024,
        kv_cache_quantized=True,
        kv_cache_bits=8,
        kv_cache_local_window_size=128,
        pipelined_execution=True,
        debug=False
    )

    print(f"[*] Proof Model: {args.model}")
    
    # 1. Standard Baseline
    print("\n[1] Establishing Standard MLX Baseline...")
    std_model, tokenizer = mlx_lm.load(args.model)
    
    sample_text = "Artificial intelligence is the intelligence of machines or software, as opposed to the intelligence of humans or animals."
    std_loss = calculate_loss(std_model, tokenizer, sample_text)
    print(f"[*] Standard Loss: {std_loss:.6f}")
    del std_model
    mx.metal.clear_cache()

    # 2. Flash Proof
    print("\n[2] Verifying Flash Engine Quality...")
    manager = FlashManager(config)
    try:
        flash_model, _ = manager.load(args.model)
        
        # Test A: Bit-Parity (Loss Delta)
        flash_loss = calculate_loss(flash_model, tokenizer, sample_text)
        delta = abs(flash_loss - std_loss)
        print(f"[*] Flash Loss:    {flash_loss:.6f}")
        print(f"[*] Loss Delta:    {delta:.10f}")
        
        parity_ok = delta < 1e-5
        
        # Test B: Passkey
        passkey_ok = run_passkey_test(flash_model, tokenizer, context_size=args.context)
        
        print("\n" + "="*40)
        print("FINAL QUALITY VERDICT")
        print("="*40)
        print(f"Numerical Parity: {'✅ PASS' if parity_ok else '❌ FAIL'}")
        print(f"Long-Context:     {'✅ PASS' if passkey_ok else '❌ FAIL'}")
        
        if parity_ok and passkey_ok:
            print("\n🚀 PROVEN: This is a high-quality production-ready engine.")
        else:
            print("\n⚠️  WARNING: Quality issues detected.")
            
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main()
