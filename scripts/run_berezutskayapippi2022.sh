#!/usr/bin/env bash
set -euo pipefail

# Edit this file, then run: bash scripts/run_berezutskayapippi2022.sh
# One model/input pairing across every task and subject/session listed below.
# Configure checkpoints and model settings in YAML; see README.md.

# ── Model and input settings ───────────────────────────────────────────────────
# Valid combinations (names refer to YAML configs under imindbench/conf/):
# MODEL             PREPROCESSOR                                                           EXPERIMENT
# logistic/mlp/cnn  laplacian_multi_stft_2048Hz                                            default
# logistic/mlp/cnn  laplacian_wav_HPF_global_robust_scalar_long_context_15s_2048Hzto500Hz  default
# htnet_500Hz       laplacian_wav_HPF_global_robust_scalar_long_context_15s_2048Hzto500Hz  default
# popt (PopT-v2)    laplacian_multi_stft_2048Hz                                            default
# linear_baseline   laplacian_stft_brainbert_2048Hz                                        default
# barista           laplacian_wav_session_HPF_global_robust_scalar_2048Hz_zscore           default
# diver             laplacian_wav_diverstyle_HPF_noSTD_long_context_15s_2048Hzto500Hz      default
MODEL=logistic
PREPROCESSOR=laplacian_multi_stft_2048Hz
EXPERIMENT=default                  # default or decodable cohort mode.
TRANSFER_EXPERIMENT=decodable      # Transfer preset: Main cohort and sample cap.

# ── Paths, tasks and subject/session pairs ─────────────────────────────────────
DATASET=berezutskayapippi2022
CONFIG_DIR="/path/to/config"          # Contains paths/local.yaml.
OUTPUT_ROOT="/path/to/runs/berezutskayapippi2022"

TASKS=(
  onset speech volume delta_volume pitch
  word_index word_gap gpt2_surprisal word_head_pos word_part_speech
  word_length global_flow local_flow frame_brightness face_num
)
TARGETS=(
  sub1_sess1 sub9_sess2 sub21_sess1 sub23_sess1
  sub42_sess1
)

# Uncomment optional commands below; comment out within-session to run only those.
# Transfer blocks require MODEL=popt and its multi-STFT PREPROCESSOR;
# they use TRANSFER_EXPERIMENT instead of EXPERIMENT.

# ── Within-session ────────────────────────────────────────────────────────────
# Runs the selected pairing for all listed tasks/targets, without a cohort filter.
python -m imindbench.launch \
  --dataset "$DATASET" \
  --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
  --task "${TASKS[@]}" --target "${TARGETS[@]}" \
  --model "$MODEL" --preprocessor "$PREPROCESSOR" \
  --experiment "$EXPERIMENT" \
  --output-group within_session

# ── Within-dataset ─────────────────────────────────────────────────────────────
# Optional PopT-v2 experiment on the validation-selected Main cohort.
# The preset applies an automatic per-subject/session training-sample cap.
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${TASKS[@]}" --target "${TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$TRANSFER_EXPERIMENT" \
#   --regime hold-in-session \
#   --output-group within_dataset

# ── Multi-dataset ──────────────────────────────────────────────────────────────
# Optional PopT-v2 experiment: Main cohort, training across all three datasets.
# The preset applies the same automatic training-sample cap across sources.
# python -m imindbench.launch \
#   --dataset "${DATASET}_multi_dataset_train" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${TASKS[@]}" --target "${TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$TRANSFER_EXPERIMENT" \
#   --output-group multi_dataset
