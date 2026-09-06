"""Check prepared-recording selection through the public dataset API."""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_selector_h5(path: Path, *, include_mask: bool = True) -> None:
    from torch_brain.data import ArrayDict, Data, Interval, RegularTimeSeries

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
        seeg_data=RegularTimeSeries(
            data=np.array([[1.0], [2.0]], dtype=np.float32),
            sampling_rate=2.0,
        ),
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
    before = artifact.read_bytes()
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
    assert artifact.read_bytes() == before


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
