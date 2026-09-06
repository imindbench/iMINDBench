#!/usr/bin/env bash
set -euo pipefail

# Edit these constants directly for your run.
LABEL_MODE="binary"
PREPROCESSOR="laplacian_stft_1000Hz"
DEVICE="cuda:1"
OUTPUT_GROUP="kelesbyd2024_stft_sweep_with_nonword_tercile"
PATHS_CFG="${PATHS_CFG:-default}"
DATASET_CFG="kelesbyd2024"
# Optional train-set controls:
# - true: for hold-in/hold-out-session, keep only sessions from test_subject in train split.
# - false: keep default regime-defined train recordings.
TRAIN_SAME_SUBJECT_ONLY="false"
# Random fraction of train windows to keep per fold in (0, 1].
TRAIN_SAMPLE_FRACTION="1.0"

# STFT sweep dimensions. These override preprocessor.chain.2 in
# conf/preprocessor/laplacian_stft_1000Hz.yaml.
NPERSEGS=(
  "125"
  "250"
  "500"
)
POVERLAPS=(
  "0.50"
  "0.75"
  "0.875"
)
MAX_FREQUENCIES=(
  "100"
  "150"
  "250"
  "400"
)

MODELS=(
  "logistic"
)

REGIMES=(
  "within-session"
)
TASKS="onset speech volume delta_volume pitch word_index word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num"
SUBJECT_TRIALS=(
  ## No overlap subset
  "41 1"
  "41 2"
  "42 1"
  "42 2"
  "43 1"
  "43 2"
  "44 1"
  "47 1"
  "47 2"
  "48 1"
  "48 2"
  "49 1"
  "49 2"
  "51 1"
  "51 2"
  "53 1"
  "53 2"
  "54 1"
  "54 2"
  "55 1"
  "55 2"
  "56 1"
  "56 2"
  "57 1"
  "57 2"
  "58 1"
  "60 1"
  "62 1"
  "62 2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/runtime_paths.sh"

OUTPUT_ROOT="${IMINDBENCH_OUTPUT_ROOT}/${OUTPUT_GROUP}"

FAILURES=0

format_sweep_value() {
  local value="$1"
  echo "${value//./p}"
}

resolve_result_json_path() {
  local run_dir="$1"
  local task="$2"
  local subject="$3"
  local session="$4"
  echo "${run_dir}/population_btbank${subject}_${session}_${task}.json"
}

mkdir -p "${OUTPUT_ROOT}"

for MODEL in "${MODELS[@]}"; do
  for REGIME in "${REGIMES[@]}"; do
    REGIME_TRAIN_SAME_SUBJECT_ONLY="false"
    if [[ "${REGIME}" == "hold-in-session" || "${REGIME}" == "hold-out-session" ]]; then
      REGIME_TRAIN_SAME_SUBJECT_ONLY="${TRAIN_SAME_SUBJECT_ONLY}"
    fi
    for NPERSEG in "${NPERSEGS[@]}"; do
      for POVERLAP in "${POVERLAPS[@]}"; do
        for MAX_FREQUENCY in "${MAX_FREQUENCIES[@]}"; do
          RUN_TAG="nperseg$(format_sweep_value "${NPERSEG}")_poverlap$(format_sweep_value "${POVERLAP}")_fmax$(format_sweep_value "${MAX_FREQUENCY}")"
          for TASK in ${TASKS}; do
            for SUBJECT_TRIAL in "${SUBJECT_TRIALS[@]}"; do
              read -r TEST_SUBJECT TEST_SESSION <<< "${SUBJECT_TRIAL}"
              RUN_DIR="${OUTPUT_ROOT}/${MODEL}_${PREPROCESSOR}/${RUN_TAG}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
              RESULT_JSON="$(resolve_result_json_path "${RUN_DIR}" "${TASK}" "${TEST_SUBJECT}" "${TEST_SESSION}")"

              mkdir -p "${RUN_DIR}"
              if [[ -f "${RESULT_JSON}" ]]; then
                echo "Skipping regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} ${RUN_TAG} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because ${RESULT_JSON} already exists"
                continue
              fi
              echo "Running regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} ${RUN_TAG} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} same_subject_only=${REGIME_TRAIN_SAME_SUBJECT_ONLY} train_fraction=${TRAIN_SAMPLE_FRACTION}"

              MODEL_ARGS=()
              if [[ "${MODEL}" != "logistic" ]]; then
                MODEL_ARGS+=("model.device=${DEVICE}")
              fi

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
                model="${MODEL}" \
                preprocessor="${PREPROCESSOR}" \
                preprocessor.chain.2.nperseg="${NPERSEG}" \
                preprocessor.chain.2.poverlap="${POVERLAP}" \
                preprocessor.chain.2.max_frequency="${MAX_FREQUENCY}" \
                "${MODEL_ARGS[@]}" \
                wandb.enabled=false \
                runtime.overwrite=false \
                runtime.verbose=true \
                hydra.run.dir="${RUN_DIR}" "$@"; then
                echo "Warning: failed for regime=${REGIME} model=${MODEL} preprocessor=${PREPROCESSOR} ${RUN_TAG} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION}"
                FAILURES=$((FAILURES + 1))
              fi
            done
          done
        done
      done
    done
  done
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Completed with ${FAILURES} failed runs."
  exit 1
fi

echo "Completed all runs successfully."
