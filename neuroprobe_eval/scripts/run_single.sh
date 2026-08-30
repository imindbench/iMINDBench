#!/usr/bin/env bash
set -euo pipefail

# Edit these constants directly for your run.
REGIME="SS-SM"  # SS-SM | SS-DM | DS-DM
PATHS_CFG="default"
TEST_SUBJECT=1
TEST_SESSION=1
TASK="onset"

LABEL_MODE="binary"
MODEL="logistic"
PREPROCESSOR="laplacian_stft_2048Hz"
DS_DM_PREPROCESSOR="laplacian_stft_region_pool_2048Hz"
OUTPUT_GROUP="release_2025_public"
DEVICE="cuda:0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/outputs/${OUTPUT_GROUP}/${MODEL}_${PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
ACTIVE_PREPROCESSOR="${PREPROCESSOR}"
if [[ "${REGIME}" == "DS-DM" ]]; then
  ACTIVE_PREPROCESSOR="${DS_DM_PREPROCESSOR}"
fi
RUN_DIR="${PROJECT_DIR}/outputs/${OUTPUT_GROUP}/${MODEL}_${ACTIVE_PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"
RESULT_JSON="${RUN_DIR}/population_btbank${TEST_SUBJECT}_${TEST_SESSION}_${TASK}.json"

mkdir -p "${RUN_DIR}"
if [[ -f "${RESULT_JSON}" ]]; then
  echo "Skipping regime=${REGIME} task=${TASK} subject=${TEST_SUBJECT} session=${TEST_SESSION} because ${RESULT_JSON} already exists"
  exit 0
fi
cd "${EXAMPLES_DIR}"

python -m neuroprobe_eval.run_eval \
  paths="${PATHS_CFG}" \
  dataset.provider=neuroprobe2025 \
  dataset.regime="${REGIME}" \
  dataset.label_mode="${LABEL_MODE}" \
  dataset.task="${TASK}" \
  dataset.test_subject="${TEST_SUBJECT}" \
  dataset.test_session="${TEST_SESSION}" \
  model="${MODEL}" \
  preprocessor="${ACTIVE_PREPROCESSOR}" \
  model.device="${DEVICE}" \
  wandb.enabled=false \
  runtime.overwrite=false \
  runtime.verbose=true \
  hydra.run.dir="${RUN_DIR}"
