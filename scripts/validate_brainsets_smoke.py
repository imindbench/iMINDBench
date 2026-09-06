#!/usr/bin/env python3
"""CPU-only inventory and smoke checks for prepared public Brainsets data."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "config" / "brainsets_smoke_manifest.json"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SCM_COMMIT_RE = re.compile(r"(?:^|[.+])g([0-9a-f]{7,40})(?:$|[.])")


def _scm_commit_prefix(version: str) -> str | None:
    match = SCM_COMMIT_RE.search(version)
    if match is None:
        return None
    commit_prefix = match.group(1)
    if f"g{commit_prefix}.d" in version:
        raise RuntimeError(f"installed torch_brain artifact is dirty: {version}")
    return commit_prefix


def _archive_sha256(direct_url: dict[str, Any]) -> str | None:
    archive_info = direct_url.get("archive_info", {})
    value = archive_info.get("hashes", {}).get("sha256")
    if value is None:
        legacy = archive_info.get("hash")
        if isinstance(legacy, str) and legacy.startswith("sha256="):
            value = legacy.removeprefix("sha256=")
    return value if isinstance(value, str) else None


def _validate_artifact_source(
    *,
    expected_commit: str,
    expected_artifact_sha256: str,
    module_path: Path,
    distribution_module_path: Path,
    version: str,
    direct_url: dict[str, Any],
) -> dict[str, str]:
    if module_path != distribution_module_path:
        raise RuntimeError(
            "imported torch_brain module does not match installed distribution: "
            f"{module_path} != {distribution_module_path}"
        )
    if direct_url.get("dir_info", {}).get("editable") is True:
        raise RuntimeError("installed torch_brain artifact must not be editable")
    commit_prefix = _scm_commit_prefix(version)
    if commit_prefix is None or not expected_commit.startswith(commit_prefix):
        raise RuntimeError(
            "imported torch_brain commit mismatch: "
            f"expected {expected_commit}, installed artifact identifies "
            f"{commit_prefix or 'no SCM commit'}"
        )
    actual_artifact_sha256 = _archive_sha256(direct_url)
    if actual_artifact_sha256 != expected_artifact_sha256:
        raise RuntimeError(
            "installed torch_brain artifact SHA256 mismatch: "
            f"expected {expected_artifact_sha256}, got {actual_artifact_sha256}"
        )
    return {
        "commit": expected_commit,
        "commit_prefix": commit_prefix,
        "source": direct_url.get("url", str(module_path)),
        "version": version,
        "artifact_sha256": actual_artifact_sha256,
    }


def _validate_distribution_files(distribution: metadata.Distribution) -> None:
    package_files = [
        item
        for item in distribution.files or []
        if item.parts and item.parts[0] == "torch_brain" and item.hash is not None
    ]
    if not package_files:
        raise RuntimeError("installed torch_brain distribution has no RECORD hashes")
    for item in package_files:
        if item.hash.mode != "sha256":
            raise RuntimeError(f"unsupported RECORD hash for torch_brain file: {item}")
        path = Path(distribution.locate_file(item))
        if not path.is_file():
            raise RuntimeError(f"installed torch_brain file is missing: {path}")
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        actual = digest.rstrip(b"=").decode("ascii")
        if actual != item.hash.value:
            raise RuntimeError(f"installed torch_brain file hash mismatch: {path}")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "torch_brain_commit",
        "torch_brain_artifact_sha256",
        "pipelines",
        "datasets",
    }:
        raise ValueError("smoke manifest has missing or unknown top-level fields")
    if value["schema_version"] != 2:
        raise ValueError("smoke manifest schema_version must be 2")
    if not isinstance(value["torch_brain_commit"], str) or not COMMIT_RE.fullmatch(
        value["torch_brain_commit"]
    ):
        raise ValueError("torch_brain_commit must be a lowercase 40-character SHA")
    if not isinstance(value["torch_brain_artifact_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["torch_brain_artifact_sha256"]
    ):
        raise ValueError("torch_brain_artifact_sha256 must be a lowercase SHA256")
    pipelines = value["pipelines"]
    if (
        not isinstance(pipelines, list)
        or not pipelines
        or any(not isinstance(item, str) or not item for item in pipelines)
        or len(pipelines) != len(set(pipelines))
    ):
        raise ValueError("pipelines must be a non-empty list of unique strings")
    datasets = value["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty object")
    for dataset_id, spec in datasets.items():
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset ids must be non-empty strings")
        if not isinstance(spec, dict) or set(spec) != {
            "class",
            "dirname",
            "recording_id",
        }:
            raise ValueError(
                f"dataset '{dataset_id}' must declare class, dirname, and recording_id"
            )
        if any(not isinstance(spec[key], str) or not spec[key] for key in spec):
            raise ValueError(f"dataset '{dataset_id}' values must be non-empty strings")
        dirname = Path(spec["dirname"])
        if (
            dirname.is_absolute()
            or len(dirname.parts) != 1
            or dirname.name != spec["dirname"]
        ):
            raise ValueError(
                f"dataset '{dataset_id}' dirname must be one safe path component"
            )
    return value


def validate_torch_brain_source(
    expected_commit: str, expected_artifact_sha256: str
) -> dict[str, str]:
    """Require a clean checkout or immutable artifact from the pinned source."""
    import torch_brain

    module_path = Path(torch_brain.__file__).resolve()
    for candidate in module_path.parents:
        if not (candidate / ".git").exists():
            continue
        if module_path != (candidate / "torch_brain" / "__init__.py").resolve():
            continue
        top_level = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            top_level.returncode != 0
            or Path(top_level.stdout.strip()).resolve() != candidate
        ):
            continue
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_commit = result.stdout.strip()
        if actual_commit != expected_commit:
            raise RuntimeError(
                "imported torch_brain commit mismatch: "
                f"expected {expected_commit}, got {actual_commit} from {candidate}"
            )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(candidate),
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(
                f"imported torch_brain checkout has tracked modifications: {candidate}"
            )
        return {"commit": actual_commit, "source": str(candidate)}

    distribution = metadata.distribution("torch_brain")
    distribution_module_path = Path(
        distribution.locate_file("torch_brain/__init__.py")
    ).resolve()
    _validate_distribution_files(distribution)
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    if direct_url_text:
        actual_commit = direct_url.get("vcs_info", {}).get("commit_id")
        if actual_commit is not None:
            raise RuntimeError(
                "installed torch_brain must be the pinned wheel artifact, not a VCS install"
            )

    return _validate_artifact_source(
        expected_commit=expected_commit,
        expected_artifact_sha256=expected_artifact_sha256,
        module_path=module_path,
        distribution_module_path=distribution_module_path,
        version=metadata.version("torch_brain"),
        direct_url=direct_url,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(
    root: Path,
    dirname: str,
    *,
    hash_recording_id: str | None = None,
    full_hash: bool = False,
) -> dict[str, Any]:
    """Return a stable, read-only inventory of prepared H5 artifacts."""
    dataset_dir = root / dirname
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"prepared dataset directory not found: {dataset_dir}")
    files = []
    for path in sorted(dataset_dir.glob("*.h5")):
        stat = path.stat()
        entry = {"path": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if full_hash or path.stem == hash_recording_id:
            entry["sha256"] = _sha256(path)
        files.append(entry)
    if not files:
        raise FileNotFoundError(f"no prepared H5 files found: {dataset_dir}")
    return {"dataset_dir": str(dataset_dir.resolve()), "files": files}


def validate_loader(
    root: Path, spec: dict[str, Any], recording_id: str
) -> dict[str, Any]:
    """Open one explicit recording through the public API without modifying it."""
    from torch_brain import datasets

    dataset_class = getattr(datasets, spec["class"])
    dataset = dataset_class(
        root=root,
        dirname=spec["dirname"],
        recording_ids=[recording_id],
        keep_files_open=False,
    )
    recording = dataset.get_recording(recording_id)
    try:
        return {
            "class": spec["class"],
            "recording_id": recording_id,
            "domain_start": float(recording.domain.start[0]),
            "domain_end": float(recording.domain.end[0]),
        }
    finally:
        file_handle = getattr(recording, "file", None)
        if file_handle is not None:
            file_handle.close()


def validate_neuroprobe_fixed_window_equivalence(
    root: Path, spec: dict[str, Any], recording_id: str
) -> dict[str, Any]:
    """Compare one fixed neural window through both public Neuroprobe views."""
    from torch_brain.datasets import Neuroprobe2025, NeuroprobeV2

    arrays = []
    windows = []
    for dataset_class in (Neuroprobe2025, NeuroprobeV2):
        dataset = dataset_class(
            root=root,
            dirname=spec["dirname"],
            recording_ids=[recording_id],
            keep_files_open=False,
        )
        recording = dataset.get_recording(recording_id)
        try:
            start = float(recording.domain.start[0])
            end = min(start + 1.0, float(recording.domain.end[-1]))
            if end <= start:
                raise ValueError(f"recording '{recording_id}' has an empty domain")
            window = recording.slice(start, end, reset_origin=False)
            arrays.append(np.asarray(window.seeg_data.data))
            windows.append((start, end))
        finally:
            file_handle = getattr(recording, "file", None)
            if file_handle is not None:
                file_handle.close()

    if windows[0] != windows[1]:
        raise ValueError(
            "Neuroprobe2025 and NeuroprobeV2 returned different recording domains"
        )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("fixed Neuroprobe window contains non-finite neural values")
    if arrays[0].dtype != arrays[1].dtype or not np.array_equal(arrays[0], arrays[1]):
        raise ValueError(
            "Neuroprobe2025 and NeuroprobeV2 returned different fixed-window bytes"
        )
    digest = hashlib.sha256()
    digest.update(arrays[0].dtype.str.encode("ascii"))
    digest.update(str(arrays[0].shape).encode("ascii"))
    digest.update(arrays[0].tobytes(order="C"))
    return {
        "recording_id": recording_id,
        "window_start": windows[0][0],
        "window_end": windows[0][1],
        "shape": list(arrays[0].shape),
        "sha256": digest.hexdigest(),
    }


def validate_neuroprobe_v2_regimes(
    root: Path, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve and open every public NeuroprobeV2 regime/split combination."""
    from torch_brain.datasets import NeuroprobeV2

    results = []
    for regime in (
        "within-session",
        "hold-in-session",
        "hold-out-session",
        "hold-out-subject",
    ):
        for split in ("train", "val", "test"):
            for fold in (0, 1):
                dataset = NeuroprobeV2(
                    root=root,
                    dirname=spec["dirname"],
                    subset_tier="full",
                    test_subject=1,
                    test_session=1,
                    split=split,
                    label_mode="binary",
                    task="onset",
                    regime=regime,
                    fold=fold,
                    keep_files_open=False,
                )
                selector_counts = {}
                for recording_id in dataset.recording_ids:
                    recording = dataset.get_recording(recording_id)
                    try:
                        starts = np.asarray(recording.splits.start)
                        ends = np.asarray(recording.splits.end)
                        labels = np.asarray(recording.splits.label)
                        included = np.asarray(recording.channels.included)
                        if starts.shape != ends.shape or starts.shape != labels.shape:
                            raise ValueError(
                                f"split selector shape mismatch for '{recording_id}'"
                            )
                        if starts.size == 0 or not included.any():
                            raise ValueError(
                                f"empty interval/channel selector for '{recording_id}'"
                            )
                        selector_counts[recording_id] = {
                            "interval_count": int(starts.shape[0]),
                            "included_channel_count": int(included.sum()),
                        }
                    finally:
                        file_handle = getattr(recording, "file", None)
                        if file_handle is not None:
                            file_handle.close()
                results.append(
                    {
                        "regime": regime,
                        "split": split,
                        "fold": fold,
                        "recording_count": len(dataset.recording_ids),
                        "opened_recording_count": len(selector_counts),
                        "selector_counts": selector_counts,
                    }
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--recording-id")
    parser.add_argument(
        "--neuroprobe-v2-regimes",
        action="store_true",
        help="Open every fold/recording for all NeuroprobeV2 regime/split selections.",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash every prepared H5 file instead of only the selected recording.",
    )
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    source = validate_torch_brain_source(
        manifest["torch_brain_commit"], manifest["torch_brain_artifact_sha256"]
    )
    if args.list_datasets:
        print(json.dumps(sorted(manifest["datasets"])))
        return 0
    if args.root is None or args.dataset is None:
        parser.error("--root and --dataset are required unless --list-datasets is used")
    try:
        spec = manifest["datasets"][args.dataset]
    except KeyError as exc:
        raise SystemExit(f"unknown dataset id: {args.dataset}") from exc
    recording_id = args.recording_id or spec["recording_id"]
    before = inventory(
        args.root,
        spec["dirname"],
        hash_recording_id=recording_id,
        full_hash=args.full_hash,
    )
    result: dict[str, Any] = {"source": source, "inventory": before}
    result["loader"] = validate_loader(args.root, spec, recording_id)
    if args.neuroprobe_v2_regimes:
        if args.dataset != "neuroprobev2":
            parser.error("--neuroprobe-v2-regimes requires --dataset neuroprobev2")
        result["regime_splits"] = validate_neuroprobe_v2_regimes(args.root, spec)
        result["fixed_window_equivalence"] = (
            validate_neuroprobe_fixed_window_equivalence(args.root, spec, recording_id)
        )
    after = inventory(
        args.root,
        spec["dirname"],
        hash_recording_id=recording_id,
        full_hash=args.full_hash,
    )
    if before != after:
        raise RuntimeError("loader smoke changed prepared artifacts")
    result["loader_inventory_unchanged"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
