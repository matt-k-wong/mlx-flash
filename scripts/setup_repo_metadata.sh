#!/usr/bin/env bash
# scripts/setup_repo_metadata.sh — Use gh CLI to set repository metadata.

set -euo pipefail

REPO="matt-k-wong/mlx-flash"

echo "Setting metadata for $REPO..."

gh repo edit "$REPO" \
    --description "Run 70B+ LLMs on 16GB MacBook Air via Flash Weight Streaming for LM Studio / mlx-engine" \
    --homepage "https://github.com/matt-k-wong/mlx-flash" \
    --add-topic "mlx" \
    --add-topic "apple-silicon" \
    --add-topic "llm-inference" \
    --add-topic "memory-optimization" \
    --add-topic "lm-studio" \
    --add-topic "flash-attention" \
    --add-topic "macos"

echo "✅ Repository metadata updated."
