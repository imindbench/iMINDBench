#!/usr/bin/env bash
set -euo pipefail

# One checkpoint-free CPU evaluation.
# Pass --config-dir and --output-root; add --execute after inspecting the preview.
# Uses the active Python environment and works without Git or a fixed working directory.
exec python -m imindbench.launch \
  --recipe baselines --dataset neuroprobev2 \
  --model logistic --task onset --target sub1_sess1 \
  --device cpu --paths local \
  "$@"
