#!/usr/bin/env bash
set -euo pipefail

# BaRISTA evaluation on 2048 Hz BerezutskayaPippi2022 high-coverage waveforms.
# The checkpoint and Destrieux region key are required model inputs.
LABEL_MODE="binary"
PREPROCESSOR="${PREPROCESSOR:-laplacian_wav_session_HPF_global_robust_scalar_2048Hz_zscore}"
BRAIN_AREA_KEY="label_destrieux"
BARISTA_CHECKPOINT="${BARISTA_CHECKPOINT:?set BARISTA_CHECKPOINT to the pretrained BaRISTA .ckpt}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_GROUP="${OUTPUT_GROUP:-berezutskayapippi2022_barista}"
PATHS_CFG="${PATHS_CFG:-default}"
DATASET_CFG="berezutskayapippi2022"
SUBSET_TIER="high-cov"
# Optional train-set controls.
TRAIN_SAME_SUBJECT_ONLY="false"
TRAIN_SAMPLE_FRACTION="1.0"
# Defaults are a smoke subset; coverage_contract.json defines the 75-cell sweep.
# Hyperparameters and preprocessing match the published 100/15-budget results.
MAX_ITER="${MAX_ITER:-100}"
PATIENCE="${PATIENCE:-15}"
# Worker settings affect RNG order; these values reproduce the recorded runs.
NUM_WORKERS="4"
PIN_MEMORY="true"
PERSISTENT_WORKERS="true"
# Pin the published update-based schedule instead of deriving it from epochs.
PIN_REFERENCE_SCHEDULE="${PIN_REFERENCE_SCHEDULE:-true}"

REGIMES=(
  "within-session"
  # "hold-in-session"
  # "hold-out-session"
  # "hold-out-subject"
)
TASKS="${TASKS:-onset word_index volume}" # speech delta_volume pitch word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num
SUBJECT_TRIALS=(
  "1 1"
  "9 2"
  # "21 1"
  # "23 1"
  # "42 1"
)
# Shard across GPUs by passing SUBJECT_TRIALS_STR="1 1;3 0" (";"-separated pairs).
if [[ -n "${SUBJECT_TRIALS_STR:-}" ]]; then
  IFS=";" read -r -a SUBJECT_TRIALS <<< "${SUBJECT_TRIALS_STR}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/runtime_paths.sh"

MODEL="barista"
FAILURES=0

# The published numbers were recorded at 1e-3 for both groups, not the paper's
# split rates (barista.yaml ships upstream 1e-4 / head 1e-3). At 1e-4 the
# upstream is undertrained on the smaller datasets and scores drop. Clear these
# to fall back to the barista.yaml defaults.
UPSTREAM_LR="${UPSTREAM_LR:-1e-3}"
HEAD_LR="${HEAD_LR:-1e-3}"
LR_ARGS=()
if [[ -n "${UPSTREAM_LR}" ]]; then
  LR_ARGS+=("model.upstream_lr=${UPSTREAM_LR}")
fi
if [[ -n "${HEAD_LR}" ]]; then
  LR_ARGS+=("model.head_lr=${HEAD_LR}")
fi

SCHEDULE_ARGS=()
if [[ "${PIN_REFERENCE_SCHEDULE}" == "true" ]]; then
  SCHEDULE_ARGS+=("++model.scheduler.warmup_steps=500" "++model.scheduler.step_size_updates=95")
fi

resolve_result_json_path() {
  local run_dir="$1"
  local task="$2"
  local subject="$3"
  local session="$4"
  echo "${run_dir}/population_btbank${subject}_${session}_${task}.json"
}

for REGIME in "${REGIMES[@]}"; do
  REGIME_TRAIN_SAME_SUBJECT_ONLY="false"
  if [[ "${REGIME}" == "hold-in-session" || "${REGIME}" == "hold-out-session" ]]; then
    REGIME_TRAIN_SAME_SUBJECT_ONLY="${TRAIN_SAME_SUBJECT_ONLY}"
  fi

  for TASK in ${TASKS}; do
    for SUBJECT_TRIAL in "${SUBJECT_TRIALS[@]}"; do
      read -r TEST_SUBJECT TEST_SESSION <<< "${SUBJECT_TRIAL}"
      RUN_DIR="${IMINDBENCH_OUTPUT_ROOT}/${OUTPUT_GROUP}/${MODEL}_${PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
      RESULT_JSON="$(resolve_result_json_path "${RUN_DIR}" "${TASK}" "${TEST_SUBJECT}" "${TEST_SESSION}")"

      mkdir -p "${RUN_DIR}"
      if [[ -f "${RESULT_JSON}" ]]; then
        echo "Skipping regime=${REGIME} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because ${RESULT_JSON} already exists"
        continue
      fi
      echo "Running regime=${REGIME} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} train_fraction=${TRAIN_SAMPLE_FRACTION}"

      if ! python -m imindbench.run_eval "${CONFIG_ARGS[@]}" \
        paths="${PATHS_CFG}" \
        dataset="${DATASET_CFG}" \
        dataset.regime="${REGIME}" \
        dataset.label_mode="${LABEL_MODE}" \
        dataset.task="${TASK}" \
        dataset.test_subject="${TEST_SUBJECT}" \
        dataset.test_session="${TEST_SESSION}" \
        dataset.train_same_subject_only="${REGIME_TRAIN_SAME_SUBJECT_ONLY}" \
        dataset.train_sample_fraction="${TRAIN_SAMPLE_FRACTION}" \
        dataset.subset_tier="${SUBSET_TIER}" \
        dataset.brain_area_key="${BRAIN_AREA_KEY}" \
        model="${MODEL}" \
        preprocessor="${PREPROCESSOR}" \
        model.device="${DEVICE}" \
        paths.barista_checkpoint="${BARISTA_CHECKPOINT}" \
        model.max_iter="${MAX_ITER}" \
        model.patience="${PATIENCE}" \
        runner.num_workers="${NUM_WORKERS}" \
        runner.pin_memory="${PIN_MEMORY}" \
        runner.persistent_workers="${PERSISTENT_WORKERS}" \
        ${LR_ARGS[@]+"${LR_ARGS[@]}"} \
        "${SCHEDULE_ARGS[@]}" \
        wandb.enabled=false \
        runtime.overwrite=false \
        runtime.verbose=true \
        hydra.run.dir="${RUN_DIR}" "$@"; then
        echo "Warning: failed for regime=${REGIME} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION}"
        FAILURES=$((FAILURES + 1))
      fi
    done
  done
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Completed with ${FAILURES} failed runs."
  exit 1
fi

echo "Completed all runs successfully."
