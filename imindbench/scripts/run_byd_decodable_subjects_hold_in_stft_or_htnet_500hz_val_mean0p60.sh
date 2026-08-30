#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RULE_NAME="stft_or_htnet_500hz_val_mean0p60"

export PATHS_CFG="${PATHS_CFG:-default}"
export OUTPUT_GROUP="${OUTPUT_GROUP:-kelesbyd2024_tercile_${RULE_NAME}}"
export DECODABLE_SUBJECT_SESSIONS_DIR="${DECODABLE_SUBJECT_SESSIONS_DIR:-${PROJECT_DIR}/decodable_subject_sessions/${RULE_NAME}}"

exec "${SCRIPT_DIR}/run_byd_decodable_subjects_hold_in.sh" "$@"
