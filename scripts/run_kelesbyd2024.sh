#!/usr/bin/env bash
set -euo pipefail

# Edit this file, then run: bash scripts/run_kelesbyd2024.sh
# One model/input pairing across every task and subject/session listed below.
# Configure checkpoints and model settings in YAML; see README.md.

# ── Model and input settings ───────────────────────────────────────────────────
# Valid combinations (names refer to YAML configs under imindbench/conf/):
# MODEL              PREPROCESSOR                   EXPERIMENT
# logistic/mlp/cnn   multi_stft_1000Hz              default
# logistic/mlp/cnn   wav_hpf_robust_1000to500Hz     default
# htnet_500Hz        wav_hpf_robust_1000to500Hz     default
# popt (PopT-v2)     multi_stft_1000Hz              default
# linear_baseline    stft_brainbert_1000Hz          default
# barista            wav_barista_1000to2048Hz       default
# diver              wav_diver_1000to500Hz          default
MODEL=logistic
PREPROCESSOR=multi_stft_1000Hz
EXPERIMENT=default                  # default or decodable cohort mode.
TRANSFER_EXPERIMENT=decodable      # Transfer preset: Main cohort and sample cap.

# ── Paths, tasks and subject/session pairs ─────────────────────────────────────
DATASET=kelesbyd2024
CONFIG_DIR="/path/to/config"          # Contains paths/local.yaml.
OUTPUT_ROOT="/path/to/runs/kelesbyd2024"

TASKS=(
  onset speech volume delta_volume pitch
  word_index word_gap gpt2_surprisal word_head_pos word_part_speech
  word_length global_flow local_flow frame_brightness face_num
)
TARGETS=(
  sub41_sess1 sub41_sess2 sub42_sess1 sub42_sess2
  sub43_sess1 sub43_sess2 sub44_sess1 sub47_sess1
  sub47_sess2 sub48_sess1 sub48_sess2 sub49_sess1
  sub49_sess2 sub51_sess1 sub51_sess2 sub53_sess1
  sub53_sess2 sub54_sess1 sub54_sess2 sub55_sess1
  sub55_sess2 sub56_sess1 sub56_sess2 sub57_sess1
  sub57_sess2 sub58_sess1 sub60_sess1 sub62_sess1
  sub62_sess2
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
