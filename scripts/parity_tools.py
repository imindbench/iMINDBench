#!/usr/bin/env python3
"""Build parity commands and compare result records without running experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_STATUSES = {"PASS", "FAIL", "MISSING", "NOT-COMPARABLE"}
PARITY_SCOPE = "metric-config-record"
_SAFE_CASE_ID = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_SAFE_MODULE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.]*\Z")
_IDENTITY_KEYS = {
    "model_name",
    "preprocess_name",
    "task",
    "seed",
    "subject",
    "session",
    "regime",
    "recording_key",
    "scope",
    "time_bin_key",
    "time_bin_start",
    "time_bin_end",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains a non-finite constant: {value}")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash relative file names and bytes for a deterministic resource fingerprint."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _validate_git_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def _validate_finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_case(case_id: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(case_id, str) or _SAFE_CASE_ID.fullmatch(case_id) is None:
        raise ValueError(f"Unsafe parity case ID: {case_id!r}")
    case = _require_mapping(value, f"case {case_id}")
    _require_keys(
        case,
        {
            "comparability",
            "reference_record_uri",
            "reference_record_sha256",
            "dataset_config",
            "model_config",
            "preprocessor_config",
            "overrides",
            "checkpoint",
        },
        f"case {case_id}",
    )
    if case["comparability"] not in {"RUNNABLE", "NOT-COMPARABLE"}:
        raise ValueError(f"case {case_id}.comparability is invalid")
    if case["comparability"] == "NOT-COMPARABLE" and not isinstance(
        case.get("not_comparable_reason"), str
    ):
        raise ValueError(f"case {case_id}.not_comparable_reason is required")
    for key in ("dataset_config", "model_config", "preprocessor_config"):
        if not isinstance(case[key], str) or not case[key]:
            raise ValueError(f"case {case_id}.{key} must be a non-empty string")
    _validate_sha256(
        case["reference_record_sha256"], f"case {case_id}.reference_record_sha256"
    )
    overrides = _require_mapping(case["overrides"], f"case {case_id}.overrides")
    for key, override in overrides.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"case {case_id} has an invalid override key")
        if isinstance(override, float) and not math.isfinite(override):
            raise ValueError(f"case {case_id}.overrides.{key} must be finite")
        if not isinstance(override, (str, int, float, bool)) and override is not None:
            raise ValueError(f"case {case_id}.overrides.{key} has an invalid value")
    checkpoint = _require_mapping(case["checkpoint"], f"case {case_id}.checkpoint")
    _require_keys(
        checkpoint,
        {"required", "mapping_key", "reference_sha256"},
        f"case {case_id}.checkpoint",
    )
    if not isinstance(checkpoint["required"], bool):
        raise ValueError(f"case {case_id}.checkpoint.required must be boolean")
    _validate_sha256(
        checkpoint["reference_sha256"],
        f"case {case_id}.checkpoint.reference_sha256",
        allow_none=True,
    )
    if checkpoint["required"]:
        _require_keys(checkpoint, {"path_config_key"}, f"case {case_id}.checkpoint")
        if (
            not isinstance(checkpoint["mapping_key"], str)
            or not checkpoint["mapping_key"]
        ):
            raise ValueError(f"case {case_id}.checkpoint.mapping_key is required")
    blocked_by_unknown_checkpoint = (
        checkpoint["required"] and checkpoint["reference_sha256"] is None
    )
    if (case["comparability"] == "NOT-COMPARABLE") != blocked_by_unknown_checkpoint:
        raise ValueError(
            f"case {case_id}.comparability must reflect only an unknown required checkpoint hash"
        )
    resources = case.get("resources", [])
    if not isinstance(resources, list):
        raise ValueError(f"case {case_id}.resources must be a list")
    resource_keys: set[str] = set()
    for index, resource_value in enumerate(resources):
        resource = _require_mapping(
            resource_value, f"case {case_id}.resources[{index}]"
        )
        _require_keys(
            resource,
            {"mapping_key", "path_config_key", "kind"},
            f"case {case_id}.resources[{index}]",
        )
        mapping_key = resource["mapping_key"]
        if (
            not isinstance(mapping_key, str)
            or not mapping_key
            or mapping_key in resource_keys
        ):
            raise ValueError(
                f"case {case_id} has an invalid or duplicate resource mapping key"
            )
        resource_keys.add(mapping_key)
        if resource["kind"] not in {"file", "directory"}:
            raise ValueError(f"case {case_id} has an invalid resource kind")
    return case


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the complete parity manifest boundary."""
    manifest_path = Path(path).resolve()
    manifest = _require_mapping(_read_json(manifest_path), "manifest")
    _require_keys(
        manifest,
        {"schema_version", "source_code", "runner_module", "tolerance", "cases"},
        "manifest",
    )
    if manifest["schema_version"] != 1:
        raise ValueError("manifest.schema_version must be 1")
    source_code = _require_mapping(manifest["source_code"], "manifest.source_code")
    _require_keys(source_code, {"repository_uri", "commit"}, "manifest.source_code")
    _validate_git_sha(source_code["commit"], "manifest.source_code.commit")
    if not isinstance(source_code["repository_uri"], str) or not source_code[
        "repository_uri"
    ].startswith("torch-brain-source://"):
        raise ValueError(
            "manifest.source_code.repository_uri must use torch-brain-source://"
        )
    if (
        not isinstance(manifest["runner_module"], str)
        or _SAFE_MODULE.fullmatch(manifest["runner_module"]) is None
    ):
        raise ValueError("manifest.runner_module is invalid")
    tolerance = _require_mapping(manifest["tolerance"], "manifest.tolerance")
    _require_keys(tolerance, {"absolute", "relative"}, "manifest.tolerance")
    for key in ("absolute", "relative"):
        if _validate_finite_number(tolerance[key], f"manifest.tolerance.{key}") < 0:
            raise ValueError(f"manifest.tolerance.{key} must be non-negative")
    cases = _require_mapping(manifest["cases"], "manifest.cases")
    if not cases:
        raise ValueError("manifest.cases must not be empty")
    for case_id, case in cases.items():
        _validate_case(case_id, case)
    result = dict(manifest)
    result["_manifest_path"] = str(manifest_path)
    return result


def _select_case(manifest: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    cases = _require_mapping(manifest.get("cases"), "manifest.cases")
    if case_id not in cases:
        raise ValueError(f"Unknown parity case: {case_id}")
    return _validate_case(case_id, cases[case_id])


def _resolve_reference_path(manifest: Mapping[str, Any], uri: Any) -> Path:
    prefix = "imindbench://"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError("reference_record_uri must use the imindbench:// scheme")
    manifest_path = Path(str(manifest.get("_manifest_path", ""))).resolve()
    if not manifest_path.is_file():
        raise ValueError("manifest must be loaded with load_manifest")
    repository_root = manifest_path.parents[2]
    candidate = (repository_root / uri.removeprefix(prefix)).resolve()
    allowed_root = (
        repository_root / "artifacts" / "parity_reference" / "cases"
    ).resolve()
    if allowed_root not in candidate.parents:
        raise ValueError(
            "reference_record_uri escapes artifacts/parity_reference/cases"
        )
    return candidate


def _validate_reference_folds(
    folds_value: Any, case_id: str
) -> list[Mapping[str, Any]]:
    if not isinstance(folds_value, list) or not folds_value:
        raise ValueError(f"Reference {case_id} must contain folds")
    folds: list[Mapping[str, Any]] = []
    indices: set[int] = set()
    metric_keys: set[str] | None = None
    for position, fold_value in enumerate(folds_value):
        fold = _require_mapping(fold_value, f"reference {case_id}.folds[{position}]")
        index = fold.get("fold_idx")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in indices
        ):
            raise ValueError(
                f"Reference {case_id} fold indices must be unique non-negative integers"
            )
        indices.add(index)
        current_metrics = set(fold) - {"fold_idx"}
        if not current_metrics or (
            metric_keys is not None and current_metrics != metric_keys
        ):
            raise ValueError(
                f"Reference {case_id} folds must share non-empty metric keys"
            )
        metric_keys = current_metrics
        for metric in current_metrics:
            _validate_finite_number(
                fold[metric], f"reference {case_id}.folds[{position}].{metric}"
            )
        folds.append(fold)
    return folds


def load_reference_record(manifest: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Load a frozen record after validating its byte hash and full boundary."""
    case = _select_case(manifest, case_id)
    record_path = _resolve_reference_path(manifest, case["reference_record_uri"])
    expected_hash = _validate_sha256(
        case["reference_record_sha256"], f"case {case_id}.reference_record_sha256"
    )
    actual_hash = _sha256_file(record_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Reference hash mismatch for {case_id}: expected {expected_hash}, got {actual_hash}"
        )
    record = _require_mapping(_read_json(record_path), f"reference {case_id}")
    _require_keys(
        record,
        {"schema_version", "case_id", "source", "identity", "folds", "counts"},
        f"reference {case_id}",
    )
    if record["schema_version"] != 1 or record["case_id"] != case_id:
        raise ValueError(f"Reference identity mismatch for {case_id}")
    source = _require_mapping(record["source"], f"reference {case_id}.source")
    _require_keys(
        source,
        {"uri", "sha256", "config_sha256", "preprocess_sha256"},
        f"reference {case_id}.source",
    )
    if not isinstance(source["uri"], str) or not source["uri"].startswith(
        "original-results://"
    ):
        raise ValueError(f"reference {case_id}.source.uri is invalid")
    for field in ("sha256", "config_sha256", "preprocess_sha256"):
        _validate_sha256(source[field], f"reference {case_id}.source.{field}")
    identity = _require_mapping(record["identity"], f"reference {case_id}.identity")
    _require_keys(identity, _IDENTITY_KEYS, f"reference {case_id}.identity")
    for field in _IDENTITY_KEYS - {
        "seed",
        "subject",
        "session",
        "time_bin_start",
        "time_bin_end",
    }:
        if not isinstance(identity[field], str) or not identity[field]:
            raise ValueError(
                f"reference {case_id}.identity.{field} must be a non-empty string"
            )
    for field in ("seed", "subject", "session"):
        if not isinstance(identity[field], int) or isinstance(identity[field], bool):
            raise ValueError(f"reference {case_id}.identity.{field} must be an integer")
    for field in ("time_bin_start", "time_bin_end"):
        _validate_finite_number(
            identity[field], f"reference {case_id}.identity.{field}"
        )
    _validate_reference_folds(record["folds"], case_id)
    _require_mapping(record["counts"], f"reference {case_id}.counts")
    return dict(record)


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Hydra override values must be finite")
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"Unsupported Hydra override value: {value!r}")
    return str(value)


def _validate_existing_path(path_value: Any, label: str, kind: str) -> Path:
    if kind not in {"file", "directory"}:
        raise ValueError(f"{label} has unsupported path kind: {kind}")
    if not isinstance(path_value, (str, Path)):
        raise ValueError(f"{label} must be a path")
    path = Path(path_value).expanduser().resolve()
    valid = path.is_file() if kind == "file" else path.is_dir()
    if not valid:
        raise ValueError(f"{label} must name an existing {kind}: {path}")
    return path


def _resource_fingerprint(path: Path, kind: str) -> str:
    return _sha256_file(path) if kind == "file" else _sha256_directory(path)


def build_case_command(
    manifest: Mapping[str, Any],
    case_id: str,
    *,
    data_root: str | Path,
    output_root: str | Path,
    checkpoint_map: Mapping[str, str | Path],
    resource_map: Mapping[str, str | Path],
    device: str | None = None,
) -> list[str]:
    """Return one validated argv vector. This function never executes it."""
    case = _select_case(manifest, case_id)
    _require_mapping(checkpoint_map, "checkpoint_map")
    _require_mapping(resource_map, "resource_map")
    data_path = _validate_existing_path(data_root, "data_root", "directory")
    output_path = Path(output_root).expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"output_root must be fresh and not exist: {output_path}")
    argv = [
        sys.executable,
        "-m",
        str(manifest["runner_module"]),
        "paths=example",
        f"dataset={case['dataset_config']}",
        f"model={case['model_config']}",
        f"preprocessor={case['preprocessor_config']}",
        f"paths.dataset_root={data_path}",
    ]
    for key, value in case["overrides"].items():
        hydra_key = f"+{key}" if key.startswith("+") else key
        argv.append(f"{hydra_key}={_format_override(value)}")
    checkpoint = case["checkpoint"]
    if checkpoint["required"]:
        mapping_key = checkpoint["mapping_key"]
        if mapping_key not in checkpoint_map:
            raise ValueError(f"checkpoint_map is missing required key: {mapping_key}")
        checkpoint_path = _validate_existing_path(
            checkpoint_map[mapping_key], f"checkpoint_map.{mapping_key}", "file"
        )
        reference_hash = checkpoint["reference_sha256"]
        if (
            reference_hash is not None
            and _sha256_file(checkpoint_path) != reference_hash
        ):
            raise ValueError(
                f"checkpoint_map.{mapping_key} does not match the reference SHA256"
            )
        argv.append(f"{checkpoint['path_config_key']}={checkpoint_path}")
    for resource in case.get("resources", []):
        mapping_key = resource["mapping_key"]
        if mapping_key not in resource_map:
            raise ValueError(f"resource_map is missing required key: {mapping_key}")
        path = _validate_existing_path(
            resource_map[mapping_key], f"resource_map.{mapping_key}", resource["kind"]
        )
        argv.append(f"{resource['path_config_key']}={path}")
    if device is not None and case["model_config"] != "logistic":
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        argv.append(f"model.device={device}")
    argv.extend(
        [
            "wandb.enabled=false",
            "runtime.overwrite=false",
            "runtime.verbose=true",
            f"hydra.run.dir={output_path}",
        ]
    )
    return argv


def _identity_actual(
    candidate: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    config = candidate.get("config")
    if not isinstance(config, dict):
        return {}
    try:
        time_bin = candidate["evaluation_results"][identity["recording_key"]][
            identity["scope"]
        ][identity["time_bin_key"]]
    except (KeyError, TypeError):
        time_bin = {}
    preprocess = config.get("preprocess", {})
    return {
        "model_name": candidate.get("model_name"),
        "preprocess_name": (
            preprocess.get("name") if isinstance(preprocess, dict) else None
        ),
        "task": config.get("eval_name"),
        "seed": config.get("seed"),
        "subject": config.get("subject_id"),
        "session": config.get("trial_id"),
        "regime": config.get("splits_type"),
        "time_bin_start": (
            time_bin.get("time_bin_start") if isinstance(time_bin, dict) else None
        ),
        "time_bin_end": (
            time_bin.get("time_bin_end") if isinstance(time_bin, dict) else None
        ),
    }


def _candidate_folds(
    candidate: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    try:
        folds_value = candidate["evaluation_results"][identity["recording_key"]][
            identity["scope"]
        ][identity["time_bin_key"]]["folds"]
    except (KeyError, TypeError) as exc:
        raise ValueError("candidate is missing the referenced result identity") from exc
    if not isinstance(folds_value, list):
        raise ValueError("candidate folds must be a list")
    folds: list[Mapping[str, Any]] = []
    indices: set[int] = set()
    for position, fold_value in enumerate(folds_value):
        fold = _require_mapping(fold_value, f"candidate fold {position}")
        index = fold.get("fold_idx")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in indices
        ):
            raise ValueError(
                "candidate fold indices must be unique non-negative integers"
            )
        indices.add(index)
        for metric, value in fold.items():
            if metric != "fold_idx":
                _validate_finite_number(value, f"candidate fold {index}.{metric}")
        folds.append(fold)
    return folds


def _supplied_artifacts(
    case: Mapping[str, Any],
    checkpoint_map: Mapping[str, str | Path],
    resource_map: Mapping[str, str | Path],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    checkpoint = case["checkpoint"]
    checkpoint_detail = None
    if checkpoint["required"]:
        mapping_key = checkpoint["mapping_key"]
        if mapping_key not in checkpoint_map:
            raise ValueError(f"checkpoint_map is missing required key: {mapping_key}")
        path = _validate_existing_path(
            checkpoint_map[mapping_key], f"checkpoint_map.{mapping_key}", "file"
        )
        supplied_hash = _sha256_file(path)
        reference_hash = checkpoint["reference_sha256"]
        checkpoint_detail = {
            "mapping_key": mapping_key,
            "reference_sha256": reference_hash,
            "supplied_sha256": supplied_hash,
            "matches_reference": (
                None if reference_hash is None else supplied_hash == reference_hash
            ),
        }
    resource_details = []
    for resource in case.get("resources", []):
        mapping_key = resource["mapping_key"]
        if mapping_key not in resource_map:
            raise ValueError(f"resource_map is missing required key: {mapping_key}")
        path = _validate_existing_path(
            resource_map[mapping_key], f"resource_map.{mapping_key}", resource["kind"]
        )
        resource_details.append(
            {
                "mapping_key": mapping_key,
                "kind": resource["kind"],
                "supplied_sha256": _resource_fingerprint(path, resource["kind"]),
            }
        )
    return checkpoint_detail, resource_details


def compare_case(
    manifest: Mapping[str, Any],
    case_id: str,
    candidate_path: str | Path,
    *,
    checkpoint_map: Mapping[str, str | Path],
    resource_map: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Compare one result's metric/config record; execution provenance is not proven."""
    case = _select_case(manifest, case_id)
    reference = load_reference_record(manifest, case_id)
    checkpoint_map = _require_mapping(checkpoint_map, "checkpoint_map")
    resource_map = _require_mapping(resource_map or {}, "resource_map")
    checkpoint_detail, resource_details = _supplied_artifacts(
        case, checkpoint_map, resource_map
    )
    candidate_file = Path(candidate_path).expanduser().resolve()
    report: dict[str, Any] = {
        "case_id": case_id,
        "status": "MISSING",
        "parity_scope": PARITY_SCOPE,
        "execution_provenance_verified": False,
        "reference_record_verified": True,
        "reference_record_sha256": case["reference_record_sha256"],
        "source_result_sha256": reference["source"]["sha256"],
        "candidate_sha256": None,
        "checkpoint": checkpoint_detail,
        "resources": resource_details,
        "findings": [],
    }
    if not candidate_file.is_file():
        report["findings"].append("Candidate result JSON is missing.")
        return report
    report["candidate_sha256"] = _sha256_file(candidate_file)
    candidate = _require_mapping(_read_json(candidate_file), "candidate result")
    identity = reference["identity"]
    actual_identity = _identity_actual(candidate, identity)
    identity_mismatches = [
        {"field": key, "expected": expected, "actual": actual_identity.get(key)}
        for key, expected in identity.items()
        if key not in {"recording_key", "scope", "time_bin_key"}
        and actual_identity.get(key) != expected
    ]
    hash_mismatches: list[dict[str, Any]] = []
    config = candidate.get("config")
    if not isinstance(config, dict):
        hash_mismatches.append({"field": "config", "error": "missing mapping"})
    else:
        for field, value in (
            ("config_sha256", config),
            ("preprocess_sha256", config.get("preprocess")),
        ):
            expected = reference["source"][field]
            try:
                actual = _canonical_sha256(value)
            except (TypeError, ValueError) as exc:
                hash_mismatches.append({"field": field, "error": str(exc)})
                continue
            if actual != expected:
                hash_mismatches.append(
                    {"field": field, "expected": expected, "actual": actual}
                )
    metric_mismatches: list[dict[str, Any]] = []
    try:
        candidate_folds = _candidate_folds(candidate, identity)
        candidate_by_index = {fold["fold_idx"]: fold for fold in candidate_folds}
        expected_indices = {fold["fold_idx"] for fold in reference["folds"]}
        actual_indices = set(candidate_by_index)
        if actual_indices != expected_indices:
            metric_mismatches.append(
                {
                    "field": "fold_indices",
                    "expected": sorted(expected_indices),
                    "actual": sorted(actual_indices),
                }
            )
        absolute = float(manifest["tolerance"]["absolute"])
        relative = float(manifest["tolerance"]["relative"])
        for expected_fold in reference["folds"]:
            index = expected_fold["fold_idx"]
            actual_fold = candidate_by_index.get(index, {})
            for metric, expected in expected_fold.items():
                if metric == "fold_idx":
                    continue
                actual = actual_fold.get(metric)
                if (
                    not isinstance(actual, (int, float))
                    or isinstance(actual, bool)
                    or not math.isfinite(float(actual))
                    or not math.isclose(
                        float(actual),
                        float(expected),
                        abs_tol=absolute,
                        rel_tol=relative,
                    )
                ):
                    metric_mismatches.append(
                        {
                            "fold_idx": index,
                            "metric": metric,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
    except ValueError as exc:
        metric_mismatches.append({"field": "folds", "error": str(exc)})
    checkpoint_mismatch = bool(
        checkpoint_detail and checkpoint_detail["matches_reference"] is False
    )
    report.update(
        {
            "identity_mismatches": identity_mismatches,
            "hash_mismatches": hash_mismatches,
            "metric_mismatches": metric_mismatches,
        }
    )
    if (
        identity_mismatches
        or hash_mismatches
        or metric_mismatches
        or checkpoint_mismatch
    ):
        report["status"] = "FAIL"
        report["findings"].append(
            "Candidate differs from the frozen metric/config record contract."
        )
    elif case["comparability"] == "NOT-COMPARABLE":
        report["status"] = "NOT-COMPARABLE"
        report["findings"].append(case["not_comparable_reason"])
    else:
        report["status"] = "PASS"
        report["findings"].append(
            "Metric/config record identity, folds, hashes, and metrics match; execution provenance was not verified."
        )
    return report


def _overall_status(case_reports: Sequence[Mapping[str, Any]]) -> str:
    statuses = {report.get("status") for report in case_reports}
    if not statuses or not statuses <= VALID_STATUSES:
        raise ValueError("report contains an invalid status")
    for status in ("FAIL", "MISSING", "NOT-COMPARABLE"):
        if status in statuses:
            return status
    return "PASS"


def write_report(
    case_reports: Sequence[Mapping[str, Any]], report_dir: str | Path
) -> tuple[Path, Path]:
    """Atomically finalize JSON and Markdown reports into a new directory."""
    if not case_reports:
        raise ValueError("case_reports must not be empty")
    output = Path(report_dir).expanduser().resolve()
    if output.exists():
        raise ValueError(f"report_dir must be fresh and not exist: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"report_dir parent must exist: {output.parent}")
    payload = {
        "schema_version": 1,
        "status": _overall_status(case_reports),
        "parity_scope": PARITY_SCOPE,
        "execution_provenance_verified": False,
        "cases": list(case_reports),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        rows = [
            "# Parity report",
            "",
            f"Overall: **{payload['status']}**",
            "",
            "Scope: metric/config record parity only; execution provenance was not verified.",
            "",
            "| Case | Status | Finding |",
            "|---|---|---|",
        ]
        for report in case_reports:
            finding = (
                "; ".join(str(item) for item in report.get("findings", []))
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            rows.append(f"| {report['case_id']} | {report['status']} | {finding} |")
        (temporary / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError(
                f"report_dir became occupied before finalization: {output}"
            )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "report.json", output / "report.md"


def _load_mapping(path: str | Path, label: str) -> Mapping[str, Any]:
    return _require_mapping(_read_json(Path(path).expanduser().resolve()), label)


def _case_ids(manifest: Mapping[str, Any], selected: Sequence[str] | None) -> list[str]:
    if not selected:
        return list(manifest["cases"])
    for case_id in selected:
        _select_case(manifest, case_id)
    if len(set(selected)) != len(selected):
        raise ValueError("case selections must be unique")
    return list(selected)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="artifacts/parity_reference/manifest.json"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser(
        "build-commands", help="validate and print dry-run commands"
    )
    build.add_argument("--case", action="append", dest="cases")
    build.add_argument("--data-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--checkpoint-map", required=True)
    build.add_argument("--resource-map", required=True)
    build.add_argument("--device")
    compare = subparsers.add_parser(
        "compare", help="compare result JSONs and write reports"
    )
    compare.add_argument("--case", action="append", dest="cases")
    compare.add_argument("--candidate-map", required=True)
    compare.add_argument("--checkpoint-map", required=True)
    compare.add_argument("--resource-map", required=True)
    compare.add_argument("--report-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Comparison returns zero only when every case passes."""
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    case_ids = _case_ids(manifest, args.cases)
    checkpoint_map = _load_mapping(args.checkpoint_map, "checkpoint_map")
    if args.action == "build-commands":
        output_root = Path(args.output_root).expanduser().resolve()
        if output_root.exists():
            raise ValueError(f"output_root must be fresh and not exist: {output_root}")
        resource_map = _load_mapping(args.resource_map, "resource_map")
        commands = {
            case_id: shlex.join(
                build_case_command(
                    manifest,
                    case_id,
                    data_root=args.data_root,
                    output_root=output_root / case_id,
                    checkpoint_map=checkpoint_map,
                    resource_map=resource_map,
                    device=args.device,
                )
            )
            for case_id in case_ids
        }
        print(json.dumps(commands, indent=2))
        return 0
    candidate_map = _load_mapping(args.candidate_map, "candidate_map")
    resource_map = _load_mapping(args.resource_map, "resource_map")
    reports = [
        compare_case(
            manifest,
            case_id,
            candidate_map.get(case_id, ""),
            checkpoint_map=checkpoint_map,
            resource_map=resource_map,
        )
        for case_id in case_ids
    ]
    json_path, markdown_path = write_report(reports, args.report_dir)
    status = _overall_status(reports)
    print(
        json.dumps(
            {"status": status, "json": str(json_path), "markdown": str(markdown_path)}
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
