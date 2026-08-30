#!/usr/bin/env bash
set -euo pipefail

# Edit these constants directly for your run.
LABEL_MODE="binary"
# Switch PREPROCESSOR to laplacian_wav_2048Hz before enabling HTNet.
PREPROCESSOR="laplacian_multi_stft_2048Hz"
DEVICE="cuda:0"
OUTPUT_GROUP="neuroprobev2"
PATHS_CFG="server_ml6_popt_multi_stft_time_pool"
DATASET_CFG="neuroprobev2"
# Optional train-set controls:
# - true: for hold-in/hold-out-session, keep only sessions from test_subject in train split.
# - false: keep default regime-defined train recordings.
TRAIN_SAME_SUBJECT_ONLY="false"
# Random fraction of train windows to keep per fold in (0, 1].
TRAIN_SAMPLE_FRACTION="1.0"
# Decodable subject filtering skips eval targets not listed for the task and filters eligible train recordings.
# Auto caps each subject to the smallest non-zero target train sample count.
DECODABLE_TRAIN_SUBJECT_SESSIONS_ONLY="false"
MAX_TRAIN_SAMPLES_PER_SUBJECT="auto"
MODELS=(
  "logistic"
  "mlp"
  "cnn"
  "popt"
  # "htnet_2048Hz"
)

REGIMES=(
  "within-session"
  "hold-in-session"
  "hold-out-session"
  "hold-out-subject"
)
TASKS="onset speech volume delta_volume pitch word_index word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num"
SUBJECT_TRIALS=(
  ## No overlap subset
  "1 1"
  "3 0"
  "4 0"
  "10 0"
  "7 0"
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
    # Same-subject train filtering only applies to session-based regimes: hold-in-session and hold-out-session.
    REGIME_TRAIN_SAME_SUBJECT_ONLY="false"
    if [[ "${REGIME}" == "hold-in-session" || "${REGIME}" == "hold-out-session" ]]; then
      REGIME_TRAIN_SAME_SUBJECT_ONLY="${TRAIN_SAME_SUBJECT_ONLY}"
    fi
    # When enabled, pass the same decodable manifest filter into run_eval train construction.
    REGIME_DECODABLE_DATASET_ARGS=()
    if [[ "${DECODABLE_TRAIN_SUBJECT_SESSIONS_ONLY}" == "true" ]]; then
      REGIME_DECODABLE_DATASET_ARGS+=(
        "dataset.train_decodable_subject_sessions_only=true"
      )
    fi
    for TASK in ${TASKS}; do
      for SUBJECT_TRIAL in "${SUBJECT_TRIALS[@]}"; do
        read -r TEST_SUBJECT TEST_SESSION <<< "${SUBJECT_TRIAL}"
        # Precheck the eval target before creating outputs; helper status 1 means skip.
        # Temporarily disable set -e so we can inspect the helper exit status.
        if [[ "${DECODABLE_TRAIN_SUBJECT_SESSIONS_ONLY}" == "true" ]]; then
          set +e
          "${PROJECT_DIR}/scripts/decodable_subject_sessions.sh" "${PATHS_CFG}" "${DATASET_CFG}" "${TASK}" "${TEST_SUBJECT}" "${TEST_SESSION}"
          DECODABLE_STATUS=$?
          set -e
          if [[ "${DECODABLE_STATUS}" -eq 1 ]]; then
            # Valid manifest lookup, but this subject/session is not decodable for the task.
            echo "Skipping task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because it is not listed as decodable for ${DATASET_CFG}"
            continue
          elif [[ "${DECODABLE_STATUS}" -ne 0 ]]; then
            # Missing configs/manifests or malformed data are setup errors.
            echo "Decodable subject/session check failed for paths=${PATHS_CFG} provider=${DATASET_CFG} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION}"
            exit "${DECODABLE_STATUS}"
          fi
        fi
        RUN_DIR="${PROJECT_DIR}/outputs/${OUTPUT_GROUP}/${MODEL}_${PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
        RESULT_JSON="$(resolve_result_json_path "${RUN_DIR}" "${TASK}" "${TEST_SUBJECT}" "${TEST_SESSION}")"

        mkdir -p "${RUN_DIR}"
        if [[ -f "${RESULT_JSON}" ]]; then
          echo "Skipping regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because ${RESULT_JSON} already exists"
          continue
        fi
        echo "Running regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} same_subject_only=${REGIME_TRAIN_SAME_SUBJECT_ONLY} train_fraction=${TRAIN_SAMPLE_FRACTION}"

        MODEL_ARGS=()
        if [[ "${MODEL}" != "logistic" ]]; then
          MODEL_ARGS+=("model.device=${DEVICE}")
        fi

        if ! python -m imindbench.run_eval \
          paths="${PATHS_CFG}" \
          dataset="${DATASET_CFG}" \
          dataset.regime="${REGIME}" \
          dataset.label_mode="${LABEL_MODE}" \
          dataset.task="${TASK}" \
          dataset.test_subject="${TEST_SUBJECT}" \
          dataset.test_session="${TEST_SESSION}" \
          dataset.train_same_subject_only="${REGIME_TRAIN_SAME_SUBJECT_ONLY}" \
          dataset.train_sample_fraction="${TRAIN_SAMPLE_FRACTION}" \
          dataset.max_train_samples_per_subject="${MAX_TRAIN_SAMPLES_PER_SUBJECT}" \
          "${REGIME_DECODABLE_DATASET_ARGS[@]}" \
          model="${MODEL}" \
          preprocessor="${PREPROCESSOR}" \
          "${MODEL_ARGS[@]}" \
          wandb.enabled=false \
          runtime.overwrite=false \
          runtime.verbose=true \
          hydra.run.dir="${RUN_DIR}"; then
          echo "Warning: failed for regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION}"
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
