from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_brainsets_smoke", ROOT / "scripts" / "validate_brainsets_smoke.py"
)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_validator_lists_manifest_datasets_from_unrelated_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_brainsets_smoke.py"),
            "--list-datasets",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == [
        "berezutskayapippi2022",
        "kelesbyd2024",
        "neuroprobe2025",
        "neuroprobev2",
    ]


def test_inventory_is_stable_and_read_only(tmp_path):
    artifact = tmp_path / "neuroprobe_2025" / "sub_1_trial001.h5"
    _write_selector_h5(artifact)

    first = smoke.inventory(
        tmp_path, "neuroprobe_2025", hash_recording_id="sub_1_trial001"
    )
    second = smoke.inventory(
        tmp_path, "neuroprobe_2025", hash_recording_id="sub_1_trial001"
    )

    assert first == second
    assert first["files"][0]["path"] == artifact.name
    assert len(first["files"][0]["sha256"]) == 64
    assert first["files"][0]["mtime_ns"] == artifact.stat().st_mtime_ns


def test_inventory_skips_hashing_unselected_recordings(tmp_path):
    artifact = tmp_path / "neuroprobe_2025" / "sub_1_trial001.h5"
    _write_selector_h5(artifact)

    inventory = smoke.inventory(tmp_path, "neuroprobe_2025")

    assert "sha256" not in inventory["files"][0]


def test_inventory_fails_clearly_without_prepared_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepared dataset directory"):
        smoke.inventory(tmp_path, "neuroprobe_2025")


def _write_selector_h5(path: Path, *, include_mask: bool = True) -> None:
    from torch_brain.data import ArrayDict, Data, Interval

    path.parent.mkdir(parents=True)
    key = "full$binary$within_session$speech$fold0$train"
    domain = Interval(start=np.array([0.0]), end=np.array([1.0]))
    splits = Data(domain=domain)
    setattr(
        splits,
        key,
        Interval(start=np.array([0.0]), end=np.array([1.0]), label=np.array([1])),
    )
    channel_splits = Data()
    if include_mask:
        setattr(channel_splits, key, np.array([True], dtype=bool))
    data = Data(
        subject=Data(id="1"),
        session=Data(id="sub_1_trial001"),
        channels=ArrayDict(id=np.array(["ch0"]), included=np.array([False])),
        seeg_data=Data(unit="uV", scale_to_uV=1.0),
        splits=splits,
        channel_splits=channel_splits,
        domain=domain,
    )
    with h5py.File(path, "w") as handle:
        data.to_hdf5(handle)


@pytest.mark.parametrize("class_name", ["Neuroprobe2025", "NeuroprobeV2"])
def test_shared_artifact_selector_contract_and_no_writes(tmp_path, class_name):
    from torch_brain import datasets

    artifact = tmp_path / "neuroprobe_2025" / "sub_1_trial001.h5"
    _write_selector_h5(artifact)
    before = smoke.inventory(tmp_path, "neuroprobe_2025")
    cls = getattr(datasets, class_name)
    regime = "SS-SM" if class_name == "Neuroprobe2025" else "within-session"
    dataset = cls(
        root=tmp_path,
        dirname="neuroprobe_2025",
        subset_tier="full",
        label_mode="binary",
        task="speech",
        regime=regime,
        test_subject=1,
        test_session=1,
        split="train",
        fold=0,
        keep_files_open=False,
    )
    assert dataset.recording_ids == ["sub_1_trial001"]
    assert dataset._split_key() == "full$binary$within_session$speech$fold0$train"
    recording = dataset.get_recording("sub_1_trial001")
    assert recording.channels.included[:].tolist() == [True]
    assert recording.splits.start[:].tolist() == [0.0]
    recording.file.close()
    assert smoke.inventory(tmp_path, "neuroprobe_2025") == before


def test_missing_selector_fails_when_recording_is_loaded(tmp_path):
    from torch_brain.datasets import NeuroprobeV2

    artifact = tmp_path / "neuroprobe_2025" / "sub_1_trial001.h5"
    _write_selector_h5(artifact, include_mask=False)
    dataset = NeuroprobeV2(
        root=tmp_path,
        dirname="neuroprobe_2025",
        subset_tier="full",
        label_mode="binary",
        task="speech",
        regime="within-session",
        test_subject=1,
        test_session=1,
        split="train",
        fold=0,
        keep_files_open=False,
    )

    with pytest.raises(KeyError, match="Missing required split-selection"):
        dataset.get_recording("sub_1_trial001")


def test_manifest_and_imported_source_match_pinned_commit():
    manifest = smoke._load_manifest(ROOT / "config" / "brainsets_smoke_manifest.json")

    source = smoke.validate_torch_brain_source(manifest["torch_brain_commit"])

    assert source["commit"] == manifest["torch_brain_commit"]


def test_imported_source_rejects_wrong_commit():
    with pytest.raises(RuntimeError, match="commit mismatch"):
        smoke.validate_torch_brain_source("0" * 40)


@pytest.mark.parametrize(
    "mutation",
    [
        {"torch_brain_commit": "not-a-commit"},
        {"pipelines": []},
        {"datasets": {"unsafe": {"class": "Dataset", "dirname": "../escape"}}},
    ],
)
def test_manifest_validation_rejects_invalid_contract(tmp_path, mutation):
    manifest = json.loads(
        (ROOT / "config" / "brainsets_smoke_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(mutation)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        smoke._load_manifest(path)
