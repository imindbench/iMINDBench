#!/usr/bin/env bash
set -euo pipefail

# Edit these constants directly for your run.
LABEL_MODE="binary"
PREPROCESSOR="laplacian_stft_2048Hz"
DS_DM_PREPROCESSOR="laplacian_stft_region_pool_2048Hz"
DEVICE="cuda:0"
OUTPUT_GROUP="neuroprobe2025"
PATHS_CFG="server_ml7_popt_stft"
# Optional train-set controls:
# - true: for hold-in/hold-out-session, keep only sessions from test_subject in train split.
# - false: keep default regime-defined train recordings.
TRAIN_SAME_SUBJECT_ONLY="false"
# Random fraction of train windows to keep per fold in (0, 1].
TRAIN_SAMPLE_FRACTION="1.0"
MODELS=(
  "logistic"
  "mlp"
  "cnn"
  "popt"
)

REGIMES=(
  "SS-SM"
  "SS-DM"
  "DS-DM"
)
TASKS="onset" # speech volume delta_volume pitch word_index word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num"
SUBJECT_TRIALS=(
  ## No overlap subset
    # "1 1"
    # "3 0"
    # "4 0"
    # "10 0"
    # "7 0"
  ## Lite
  "1 1"
  "1 2"
  "2 0"
  "2 4"
  "4 1"
  "7 0"
  "7 1"
  "10 0"
  "10 1"
  "3 0"
  "3 1"
  "4 0"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
# Keep temp path short enough for AF_UNIX multiprocessing socket limits.
TMPDIR="${REPO_ROOT}/.tmp"
mkdir -p "${TMPDIR}"
export TMPDIR

FAILURES=0

cd "${EXAMPLES_DIR}"

resolve_result_json_path() {
  local run_dir="$1"
  local task="$2"
  local subject="$3"
  local session="$4"
  echo "${run_dir}/population_btbank${subject}_${session}_${task}.json"
}

for MODEL in "${MODELS[@]}"; do
  for REGIME in "${REGIMES[@]}"; do
    REGIME_TRAIN_SAME_SUBJECT_ONLY="false"
    if [[ "${REGIME}" == "hold-in-session" || "${REGIME}" == "hold-out-session" ]]; then
      REGIME_TRAIN_SAME_SUBJECT_ONLY="${TRAIN_SAME_SUBJECT_ONLY}"
    fi
    ACTIVE_PREPROCESSOR="${PREPROCESSOR}"
    if [[ "${REGIME}" == "DS-DM" && "${MODEL}" != "popt" ]]; then
      ACTIVE_PREPROCESSOR="${DS_DM_PREPROCESSOR}"
    fi

    for TASK in ${TASKS}; do
      for SUBJECT_TRIAL in "${SUBJECT_TRIALS[@]}"; do
        read -r TEST_SUBJECT TEST_SESSION <<< "${SUBJECT_TRIAL}"
        RUN_DIR="${PROJECT_DIR}/outputs/${OUTPUT_GROUP}/${MODEL}_${ACTIVE_PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
        RESULT_JSON="$(resolve_result_json_path "${RUN_DIR}" "${TASK}" "${TEST_SUBJECT}" "${TEST_SESSION}")"

        mkdir -p "${RUN_DIR}"
        if [[ -f "${RESULT_JSON}" ]]; then
          echo "Skipping regime=${REGIME} model=${MODEL} preprocessor=${ACTIVE_PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because ${RESULT_JSON} already exists"
          continue
        fi
        echo "Running regime=${REGIME} model=${MODEL} preprocessor=${ACTIVE_PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} same_subject_only=${REGIME_TRAIN_SAME_SUBJECT_ONLY} train_fraction=${TRAIN_SAMPLE_FRACTION}"

        MODEL_ARGS=()
        if [[ "${MODEL}" != "logistic" ]]; then
          MODEL_ARGS+=("model.device=${DEVICE}")
        fi

        if ! python -m neuroprobe_eval.run_eval \
          paths="${PATHS_CFG}" \
          dataset.provider=neuroprobe2025 \
          dataset.regime="${REGIME}" \
          dataset.label_mode="${LABEL_MODE}" \
          dataset.task="${TASK}" \
          dataset.test_subject="${TEST_SUBJECT}" \
          dataset.test_session="${TEST_SESSION}" \
          dataset.train_same_subject_only="${REGIME_TRAIN_SAME_SUBJECT_ONLY}" \
          dataset.train_sample_fraction="${TRAIN_SAMPLE_FRACTION}" \
          model="${MODEL}" \
          preprocessor="${ACTIVE_PREPROCESSOR}" \
          "${MODEL_ARGS[@]}" \
          wandb.enabled=false \
          runtime.overwrite=false \
          runtime.verbose=true \
          hydra.run.dir="${RUN_DIR}"; then
          echo "Warning: failed for regime=${REGIME} model=${MODEL} preprocessor=${ACTIVE_PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION}"
          FAILURES=$((FAILURES + 1))
        fi
      done
    done
  done
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Completed with ${FAILURES} failed runs."
  exit 1
fi

echo "Completed all runs successfully."
