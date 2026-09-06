#!/usr/bin/env bash
set -euo pipefail

# One checkpoint-free GPU evaluation.
# Pass --config-dir and --output-root; add --execute after inspecting the preview.
# Uses the active Python environment and works without Git or a fixed working directory.
exec python -m imindbench.launch \
  --dataset neuroprobev2 --experiment baseline \
  --preprocessor laplacian_multi_stft_2048Hz \
  --model mlp --task onset --target sub1_sess1 \
  --device cuda:0 --paths local \
  "$@"
