#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/runtime_paths.sh"
RULE_NAME="stft_or_htnet_500hz_val_mean0p60"
START_DELAY_SECONDS="${MULTISOURCE_START_DELAY_SECONDS:-120}"
LOCK_DIR="${IMINDBENCH_OUTPUT_ROOT}/.multisource_locks"
LOCK_NAME="${MULTISOURCE_LOCK_NAME:-pippi_${RULE_NAME}}"
mkdir -p "${LOCK_DIR}"
exec 9> "${LOCK_DIR}/${LOCK_NAME}.lock"
if ! flock -n 9; then
  echo "PIPPI multisource run is already active; skipping duplicate launch."
  exit 0
fi

export PATHS_CFG="${PATHS_CFG:-default}"
export OUTPUT_GROUP="${OUTPUT_GROUP:-berezutskayapippi2022_tercile_multisource_${RULE_NAME}}"
export DECODABLE_SUBJECT_SESSIONS_DIR="${DECODABLE_SUBJECT_SESSIONS_DIR:-${PROJECT_DIR}/decodable_subject_sessions/${RULE_NAME}}"

sleep "${START_DELAY_SECONDS}"
exec "${SCRIPT_DIR}/run_pippi_decodable_subjects_multisource.sh" "$@"
