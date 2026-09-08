#!/usr/bin/env bash
set -euo pipefail

# Copied from run_berezutskayapippi2022.sh for a two-fold historical-result check.
# Run with the benchmark environment active: bash scripts/run_berezutskayapippi2022_logistic_reproduce.sh
# Reference: 09_neurips/berezutskayapippi2022_tercile/high-cov/logistic_laplacian_multi_stft_2048Hz/
#            within-session/speech/sub1_sess1/population_btbank1_1_speech.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

# Use this checkout and the public TorchBrain dataset implementation.
TORCH_BRAIN_ROOT="${TORCH_BRAIN_ROOT:-${WORKSPACE_DIR}/torch_brain-public}"
export PYTHONPATH="${PROJECT_DIR}:${TORCH_BRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=logistic
PREPROCESSOR=multi_stft_2048Hz
EXPERIMENT=default

DATASET=berezutskayapippi2022
TASK=speech
SUBJECT=1
SESSION=1
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}/.tmp/berezutskayapippi2022_logistic_reproduce_config}"
# Choose a fresh output root after changing settings; completed JSONs are skipped.
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/berezutskayapippi2022_logistic_reproduction}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_DIR}/data/processed_1.1.0}"
RUN_DIR="$OUTPUT_ROOT/${MODEL}_${PREPROCESSOR}/within-session/$TASK/sub${SUBJECT}_sess${SESSION}"
RESULT_JSON="$RUN_DIR/population_btbank${SUBJECT}_${SESSION}_${TASK}.json"
if [[ -f "$RESULT_JSON" ]]; then
  echo "Skipping existing result: $RESULT_JSON"
  exit 0
fi

# This generated profile belongs to the reproduction run.
mkdir -p "$CONFIG_DIR/paths" "$RUN_DIR"
python - imindbench/conf/paths/example.yaml "$CONFIG_DIR/paths/local.yaml" "$DATASET_ROOT" <<'PYCONFIG'
import sys

from omegaconf import OmegaConf

template, path, dataset_root = sys.argv[1:]
cfg = OmegaConf.load(template)
cfg.dataset_root = dataset_root
OmegaConf.save(cfg, path)
PYCONFIG

# Select the historical window rule explicitly; cache keys and results record it.
# Extra arguments are ordinary Hydra overrides; --cfg job --resolve previews.
python -m imindbench.run_eval \
  --config-dir "$CONFIG_DIR" \
  paths=local dataset="$DATASET" model="$MODEL" \
  preprocessor="$PREPROCESSOR" experiment="$EXPERIMENT" \
  dataset.task="$TASK" dataset.test_subject="$SUBJECT" dataset.test_session="$SESSION" \
  dataset.regime=within-session dataset.window_slicing_policy=legacy_floor model.device=cpu \
  runtime.overwrite=false wandb.enabled=false \
  hydra.run.dir="$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/launcher.log"
