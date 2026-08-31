#!/usr/bin/env python3
"""CPU-only inventory and smoke checks for prepared public Brainsets data."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
from typing import Any

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "config" / "brainsets_smoke_manifest.json"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "torch_brain_commit",
        "pipelines",
        "datasets",
    }:
        raise ValueError("smoke manifest has missing or unknown top-level fields")
    if value["schema_version"] != 1:
        raise ValueError("smoke manifest schema_version must be 1")
    if not isinstance(value["torch_brain_commit"], str) or not COMMIT_RE.fullmatch(
        value["torch_brain_commit"]
    ):
        raise ValueError("torch_brain_commit must be a lowercase 40-character SHA")
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


def validate_torch_brain_source(expected_commit: str) -> dict[str, str]:
    """Require the imported public package to identify the pinned source commit."""
    import torch_brain

    module_path = Path(torch_brain.__file__).resolve()
    for candidate in module_path.parents:
        if not (candidate / ".git").exists():
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
                "--untracked-files=no",
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

    direct_url_text = metadata.distribution("torch_brain").read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        actual_commit = direct_url.get("vcs_info", {}).get("commit_id")
        if actual_commit == expected_commit:
            return {
                "commit": actual_commit,
                "source": direct_url.get("url", str(module_path)),
            }
    raise RuntimeError(
        "cannot verify imported torch_brain source commit; use the pinned public "
        f"checkout at {expected_commit} (publication is blocked until that branch is pushed)"
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
    return {
        "class": spec["class"],
        "recording_id": recording_id,
        "domain_start": float(recording.domain.start[0]),
        "domain_end": float(recording.domain.end[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--recording-id")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash every prepared H5 file instead of only the selected recording.",
    )
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    source = validate_torch_brain_source(manifest["torch_brain_commit"])
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
    after = inventory(
        args.root,
        spec["dirname"],
        hash_recording_id=recording_id,
        full_hash=args.full_hash,
    )
    if before != after:
        raise RuntimeError("loader smoke changed prepared artifacts")
    result["idempotent"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
