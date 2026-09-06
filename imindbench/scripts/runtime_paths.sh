#!/usr/bin/env bash
# Shared launcher setup. Source this after defining SCRIPT_DIR.
# Runtime files belong to the caller, including in wheel/archive installations.
: "${IMINDBENCH_OUTPUT_ROOT:?set IMINDBENCH_OUTPUT_ROOT to an absolute directory outside the installation}"
if [[ "${IMINDBENCH_OUTPUT_ROOT}" != /* ]]; then
  echo "IMINDBENCH_OUTPUT_ROOT must be absolute" >&2
  exit 2
fi
mkdir -p "${IMINDBENCH_OUTPUT_ROOT}"
IMINDBENCH_OUTPUT_ROOT="$(cd "${IMINDBENCH_OUTPUT_ROOT}" && pwd -P)"
export IMINDBENCH_OUTPUT_ROOT

# Respect existing temporary-directory overrides; /tmp keeps socket paths short.
TMPDIR="${BARISTA_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ "${TMPDIR}" != /* ]]; then
  echo "TMPDIR must be absolute" >&2
  exit 2
fi
mkdir -p "${TMPDIR}"
export TMPDIR

CONFIG_ARGS=()
if [[ -n "${IMINDBENCH_CONFIG_DIR:-}" ]]; then
  if [[ "${IMINDBENCH_CONFIG_DIR}" != /* || ! -d "${IMINDBENCH_CONFIG_DIR}" ]]; then
    echo "IMINDBENCH_CONFIG_DIR must be an existing absolute config directory" >&2
    exit 2
  fi
  CONFIG_ARGS+=(--config-dir "${IMINDBENCH_CONFIG_DIR}")
fi
