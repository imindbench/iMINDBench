#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -ne 5 ]]; then
  echo "usage: decodable_subject_sessions.sh <paths_cfg> <provider_or_dataset_cfg> <task> <subject> <session>" >&2
  exit 2
fi

PATHS_CFG="$1"
PROVIDER="$2"
TASK="$3"
SUBJECT="$4"
SESSION="$5"

SUBJECT_SESSION="sub${SUBJECT}_sess${SESSION}"

# Exit-code contract for launchers:
#   0 = subject/session is task-decodable and may run
#   1 = valid manifest lookup, but subject/session is not task-decodable; skip it
#   2 = configuration/manifest error; fail fast
python - "$PROJECT_DIR" "$PATHS_CFG" "$PROVIDER" "$TASK" "$SUBJECT_SESSION" <<'PY'
import json
import os
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
paths_cfg = sys.argv[2]
provider_or_dataset_cfg = sys.argv[3]
task = sys.argv[4]
subject_session = sys.argv[5]

def resolve_config_path(config_group, value):
    if "/" in value or value.endswith(".yaml"):
        return Path(value)
    return project_dir / "conf" / config_group / f"{value}.yaml"

def read_simple_yaml_key(path, requested_key):
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip() == requested_key:
            parsed = value.split("#", 1)[0].strip().strip("'\"")
            if parsed in {"", "null", "None"}:
                return None
            return parsed
    return None

cfg_path = resolve_config_path("paths", paths_cfg)

if not cfg_path.exists():
    print(f"missing paths config: {cfg_path}", file=sys.stderr)
    raise SystemExit(2)

manifest_dir = os.environ.get("DECODABLE_SUBJECT_SESSIONS_DIR")
if manifest_dir is None or not manifest_dir.strip():
    manifest_dir = read_simple_yaml_key(cfg_path, "decodable_subject_sessions_dir")

if not isinstance(manifest_dir, str) or not manifest_dir.strip():
    print(
        f"paths config {cfg_path} must set decodable_subject_sessions_dir",
        file=sys.stderr,
    )
    raise SystemExit(2)

manifest_provider = provider_or_dataset_cfg
dataset_cfg_path = resolve_config_path("dataset", provider_or_dataset_cfg)
if dataset_cfg_path.exists():
    resolved_provider = read_simple_yaml_key(dataset_cfg_path, "provider")
    if resolved_provider is not None:
        manifest_provider = resolved_provider

manifest = Path(manifest_dir) / f"{manifest_provider}.json"
if not manifest.exists():
    print(f"missing manifest: {manifest}", file=sys.stderr)
    raise SystemExit(2)

try:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    print(f"invalid manifest JSON {manifest}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

task_payload = payload.get("tasks", {}).get(task)
if task_payload is None:
    print(f"manifest {manifest} does not contain task {task!r}", file=sys.stderr)
    raise SystemExit(2)

raise SystemExit(
    0 if subject_session in task_payload.get("subject_sessions", []) else 1
)
PY
