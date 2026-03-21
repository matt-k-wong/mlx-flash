"""
conftest.py — pytest fixtures for mlx-flash tests.

Fixtures
--------
tmp_model_dir
    A temporary directory containing a minimal synthetic safetensors model
    (2-layer, 256 hidden, Q4_0 weights).  All streaming tests use this
    so no real model download is needed.

flash_config
    A FlashConfig with debug=True and minimal RAM budget for fast testing.

model_dir (session-scoped CLI option)
    Optional real model path passed via --model (for integration tests).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

# ── pytest CLI option ────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--model", action="store", default=None,
        help="Path to a real mlx model for integration tests",
    )
    parser.addoption(
        "--flash", action="store_true", default=False,
        help="Run integration tests in Flash Mode",
    )


@pytest.fixture(scope="session")
def model_dir_path(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def use_flash(request):
    return request.config.getoption("--flash")


# ── Synthetic safetensors helper ─────────────────────────────────────────────

def _make_q4_0_block(scale: float, values: list) -> bytes:
    """Create one Q4_0 block: 2B scale (f16) + 16B packed nibbles."""
    import struct
    # Pack scale as float16
    scale_bytes = struct.pack("<e", scale)  # little-endian float16
    # Pack 32 int values (-8..+7) as 16 bytes
    assert len(values) == 32
    data = bytearray(16)
    for i, v in enumerate(values):
        nibble = (int(v) + 8) & 0xF
        if i % 2 == 0:
            data[i // 2] = nibble
        else:
            data[i // 2] |= (nibble << 4)
    return scale_bytes + bytes(data)


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a minimal safetensors file."""
    header = {}
    data_parts = []
    offset = 0
    for name, (data_bytes, dtype_str, shape) in tensors.items():
        length = len(data_bytes)
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        data_parts.append(data_bytes)
        offset += length

    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        for d in data_parts:
            f.write(d)


@pytest.fixture(scope="session")
def tmp_model_dir(tmp_path_factory):
    """
    Synthetic 2-layer model with Q4_0 weights.
    hidden_dim=256, intermediate=512, n_experts=0 (dense).
    """
    mdir = tmp_path_factory.mktemp("test_model")
    hidden = 256
    inter = 512
    def rand_f16(shape: list[int]) -> bytes:
        rng = np.random.default_rng(42)
        return rng.standard_normal(shape).astype(np.float16).tobytes()

    tensors = {}
    # Embedding
    embed_data = rand_f16([256, 256])
    tensors["model.embed_tokens.weight"] = (embed_data, "F16", [256, 256])

    for layer in range(2):
        pfx = f"model.layers.{layer}"
        # Self-attention Q/K/V/O projections (F16)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            data = rand_f16([256, 256])
            tensors[f"{pfx}.self_attn.{proj}.weight"] = (data, "F16", [256, 256])
        # FFN gate + up + down
        for proj, shape in (
            ("gate_proj", [512, 256]),
            ("up_proj",   [512, 256]),
            ("down_proj", [256, 512]),
        ):
            data = rand_f16(shape)
            tensors[f"{pfx}.mlp.{proj}.weight"] = (data, "F16", shape)
        # Layer norm (f16, always hot)
        ln_data = np.ones(256, dtype=np.float16).tobytes()
        tensors[f"{pfx}.input_layernorm.weight"]         = (ln_data, "F16", [256])
        tensors[f"{pfx}.post_attention_layernorm.weight"] = (ln_data, "F16", [256])

    # Final norm + lm_head
    tensors["model.norm.weight"]  = (np.ones(256, dtype=np.float16).tobytes(), "F16", [256])
    tensors["lm_head.weight"]     = (np.random.randn(256, 256).astype(np.float16).tobytes(), "F16", [256, 256])

    _write_safetensors(mdir / "model.safetensors", tensors)

    # config.json
    cfg = {
        "model_type": "llama",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "tie_word_embeddings": False,
    }
    (mdir / "config.json").write_text(json.dumps(cfg))

    # Minimal valid tokenizer.json for Transformers
    tok_cfg = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
            "merges": []
        }
    }
    (mdir / "tokenizer.json").write_text(json.dumps(tok_cfg))
    (mdir / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>"}))

    return mdir


@pytest.fixture(autouse=True)
def reset_metal_state():
    """Clear Metal cache before and after each test."""
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        mx.metal.clear_cache()
        import gc; gc.collect()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def flash_config():
    from mlx_engine_flash.config import FlashConfig
    return FlashConfig(
        enabled=True,
        ram_budget_gb=1.0,   # Very tight budget to catch regressions
        debug=False,          # Avoid print noise in tests
        max_kv_size=64,       # Small KV cache for tests
        prefill_chunk_size=32,
    )
