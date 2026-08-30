#!/usr/bin/env bash
#SBATCH -A
#SBATCH -C gpu #&hbm80g #gpu&hbm80g
#SBATCH -q shared # (if regular, then gpus-per-node, cpus-per-task 제거하기! )
#SBATCH --job-name=sub10_neuroprove_eval
#SBATCH --output=./neuroprove_eval_log/sub10-%A_%a.out
#SBATCH --error=./neuroprove_eval_log/sub10-%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 1:50:00
#SBATCH --array=0-14

set -euo pipefail

module load conda
CONDA_ENV="${CONDA_ENV:-neuroprobe_eval}"
conda activate "${CONDA_ENV}"

# Edit these constants directly for your run.
LABEL_MODE="binary"
PREPROCESSOR="laplacian_wav_diverstyle_HPF_noSTD_long_context_15s_1000Hzto500Hz"
DEVICE="cuda:0"
OUTPUT_GROUP="kelesbyd2024"
PATHS_CFG="default"
DATASET_CFG="kelesbyd2024"

TRAIN_SAME_SUBJECT_ONLY="false"
TRAIN_SAMPLE_FRACTION="1.0"

MODELS=(
  "diver"
)

REGIMES=(
  "within-session"
)

TASKS=(
  "onset"
  "speech"
  "volume"
  "delta_volume"
  "pitch"
  "word_index"
  "word_gap"
  "gpt2_surprisal"
  "word_head_pos"
  "word_part_speech"
  "word_length"
  "global_flow"
  "local_flow"
  "frame_brightness"
  "face_num"
)

SUBJECT_TRIALS=(
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
EXAMPLES_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"

TMPDIR="${REPO_ROOT}/.tmp"
mkdir -p "${TMPDIR}"
export TMPDIR

FAILURES=0

cd "${EXAMPLES_DIR}"

DIVER_MODEL_DIR="${DIVER_MODEL_DIR:?Set DIVER_MODEL_DIR to a writable MuP shape directory.}"
DIVER_UPSTREAM_CKPT="${DIVER_UPSTREAM_CKPT:?Set DIVER_UPSTREAM_CKPT to the DIVER checkpoint path.}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the processed dataset root.}"

PATH_OVERRIDES=(
  "paths.dataset_root=${DATASET_ROOT}"
)
if [[ -n "${ELECTRODE_DIR_BYD:-}" ]]; then
  PATH_OVERRIDES+=("paths.electrode_dir_byd=${ELECTRODE_DIR_BYD}")
fi
if [[ -n "${TRAIN_SOURCE_CACHE_DIR:-}" ]]; then
  PATH_OVERRIDES+=("paths.train_source_cache_dir=${TRAIN_SOURCE_CACHE_DIR}")
fi

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

    for TASK in "${TASKS[@]}"; do
      echo "=============================="
      echo "Starting TASK=${TASK}"
      echo "=============================="

      for SUBJECT_TRIAL in "${SUBJECT_TRIALS[@]}"; do
        read -r TEST_SUBJECT TEST_SESSION <<< "${SUBJECT_TRIAL}"

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
        if [[ "${MODEL}" == "diver" ]]; then
          MODEL_ARGS+=(
            "model.model_dir=${DIVER_MODEL_DIR}"
            "model.upstream_ckpt=${DIVER_UPSTREAM_CKPT}"
          )
        fi

        if ! python -m neuroprobe_eval.run_eval \
          paths="${PATHS_CFG}" \
          "${PATH_OVERRIDES[@]}" \
          dataset="${DATASET_CFG}" \
          dataset.regime="${REGIME}" \
          dataset.label_mode="${LABEL_MODE}" \
          dataset.task="${TASK}" \
          dataset.coordinate_profile="diver_mni" \
          dataset.test_subject="${TEST_SUBJECT}" \
          dataset.test_session="${TEST_SESSION}" \
          dataset.train_same_subject_only="${REGIME_TRAIN_SAME_SUBJECT_ONLY}" \
          dataset.train_sample_fraction="${TRAIN_SAMPLE_FRACTION}" \
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

      echo "Finished TASK=${TASK}"
    done
  done
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Completed with ${FAILURES} failed runs."
  exit 1
fi

echo "Completed all tasks successfully."
