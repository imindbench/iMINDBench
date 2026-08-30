#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RULE_NAME="stft_or_htnet_500hz_val_mean0p60"
START_DELAY_SECONDS="${MULTISOURCE_START_DELAY_SECONDS:-0}"
LOCK_DIR="${PROJECT_DIR}/outputs/.multisource_locks"
LOCK_NAME="${MULTISOURCE_LOCK_NAME:-byd_${RULE_NAME}}"
mkdir -p "${LOCK_DIR}"
exec 9> "${LOCK_DIR}/${LOCK_NAME}.lock"
if ! flock -n 9; then
  echo "BYD multisource run is already active; skipping duplicate launch."
  exit 0
fi

export PATHS_CFG="${PATHS_CFG:-processed_1_1_0}"
export OUTPUT_GROUP="${OUTPUT_GROUP:-kelesbyd2024_tercile_multisource_${RULE_NAME}}"
export DECODABLE_SUBJECT_SESSIONS_DIR="${DECODABLE_SUBJECT_SESSIONS_DIR:-${PROJECT_DIR}/decodable_subject_sessions/${RULE_NAME}}"

sleep "${START_DELAY_SECONDS}"
exec "${SCRIPT_DIR}/run_byd_decodable_subjects_multisource.sh" "$@"
