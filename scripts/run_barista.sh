#!/usr/bin/env bash
set -euo pipefail

# One GPU evaluation using the BaRISTA checkpoint in paths/local.yaml.
# Pass --config-dir and --output-root; add --execute after inspecting the preview.
# Uses the active Python environment and works without Git or a fixed working directory.
exec python -m imindbench.launch \
  --dataset neuroprobev2 --experiment barista \
  --preprocessor laplacian_wav_session_HPF_global_robust_scalar_2048Hz_zscore \
  --set dataset.brain_area_key=localization_Destrieux \
  --model barista --task onset --target sub1_sess1 \
  --device cuda:0 --paths local \
  "$@"
