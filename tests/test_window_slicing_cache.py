"""Exercise slicing policy through real window loading and disk cache reuse."""

import json
import pickle

import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from torch_brain.data import Data, Interval, RegularTimeSeries

from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter
from imindbench.utils.logging_utils import format_and_save_results


class _WindowProvider:
    def __init__(self, provider, split, regime):
        self.provider = provider
        self.split = split
        self.regime = regime
        self.read_count = 0
        self.recording_ids = [
            "sub_1_trial1" if provider == "neuroprobev2" else "sub-CS1_ses-P1CSR1"
        ]
        self.recording = Data(
            domain="auto",
            seeg_data=RegularTimeSeries(
                sampling_rate=10.0,
                data=np.arange(20, dtype=np.float32).reshape(-1, 1),
            ),
        )

    def get_sampling_intervals(self):
        return {
            self.recording_ids[0]: Interval(
                start=np.array([0.15, 1.15]),
                end=np.array([0.45, 1.45]),
                label=np.array([0, 1]),
            )
        }

    def get_recording(self, recording_id):
        assert recording_id in self.recording_ids
        self.read_count += 1
        return self.recording

    def get_channel_metadata(self, recording_id):
        assert recording_id in self.recording_ids
        return {
            "indices": np.array([0]),
            "ids": np.array(["channel_0"]),
            "names": np.array(["channel_0"]),
            "included_mask": np.array([True]),
        }

    def describe_selection(self):
        return {"provider": self.provider, "recording_ids": self.recording_ids}


@pytest.mark.parametrize("multi_source", [False, True])
@pytest.mark.parametrize("sample_cap", [None, "auto"])
def test_window_slicing_cache_matches_uncached_execution(
    tmp_path, monkeypatch, multi_source, sample_cap
):
    providers = []

    def build_provider(*, dataset_provider, split, regime, **kwargs):
        provider = _WindowProvider(dataset_provider, split, regime)
        providers.append(provider)
        return provider

    monkeypatch.setattr(data_adapter, "build_processed_split_provider", build_provider)
    cfg = OmegaConf.create(
        {
            "provider": "neuroprobev2",
            "root": str(tmp_path),
            "dirname": "recordings",
            "subset_tier": "full",
            "label_mode": "binary",
            "task": "onset",
            "regime": "within-session",
            "test_subject": 1,
            "test_session": 1,
            "max_train_samples_per_subject": sample_cap,
            "uniquify_channel_ids_with_subject": True,
            "uniquify_channel_ids_with_session": False,
        }
    )
    if multi_source:
        cfg.train_sources = [
            {"provider": provider, "regime": "hold-in-session"}
            for provider in ("neuroprobev2", "kelesbyd2024")
        ]
    preprocessor_cfg = OmegaConf.create({"name": "raw"})

    def build_fold(cache_enabled):
        providers.clear()
        return data_adapter.build_neuroprobe_torch_fold(
            cfg,
            preprocessor=build_preprocessor(preprocessor_cfg),
            preprocessor_cfg=preprocessor_cfg,
            fold_idx=0,
            seed=7,
            require_coords=False,
            needs_pool=False,
            train_source_cache_enabled=cache_enabled and multi_source,
            train_source_cache_dir=str(tmp_path / "cache"),
            preprocessed_split_cache_enabled=cache_enabled and not multi_source,
            preprocessed_split_cache_dir=str(tmp_path / "cache"),
        )

    previous_paths = set()
    for policy, expected in (("ceil", [2, 3, 4]), ("legacy_floor", [1, 2, 3])):
        cfg.window_slicing_policy = policy
        uncached = build_fold(False)
        written = build_fold(True)
        pattern = "*.pkl" if multi_source else "manifest.json"
        paths = set((tmp_path / "cache").rglob(pattern))
        new_paths = paths - previous_paths
        assert len(new_paths) == (2 if multi_source else 1)
        timestamps = {path: path.stat().st_mtime_ns for path in paths}
        for path in new_paths:
            if multi_source:
                with path.open("rb") as handle:
                    identity = pickle.load(handle)["identity"]
            else:
                identity = json.loads(path.read_text())["identity"]
            assert identity["window_slicing_policy"] == policy

        cached = build_fold(True)
        assert {path: path.stat().st_mtime_ns for path in paths} == timestamps
        # A real cache hit avoids even reading raw recordings for cached splits.
        # Multi-source evaluation still reads the target provider to fit its own
        # preprocessing; the independent source must be served entirely by cache.
        cached_providers = (
            [provider for provider in providers if provider.provider == "kelesbyd2024"]
            if multi_source
            else providers
        )
        assert cached_providers
        assert all(provider.read_count == 0 for provider in cached_providers)
        predictions = []
        for fold in (uncached, written, cached):
            assert fold["metadata"]["window_slicing_policy"] == policy
            arrays = {}
            for split in ("train", "val", "test"):
                dataset = fold[f"{split}_split"]
                assert len(dataset) == (4 if multi_source and split == "train" else 2)
                for sample in dataset:
                    expected_window = np.asarray(expected) + 10 * sample["y"]
                    np.testing.assert_array_equal(sample["x"], [expected_window])
                arrays[split] = (
                    np.stack([sample["x"].ravel() for sample in dataset]),
                    np.array([sample["y"] for sample in dataset]),
                )
            model = LogisticRegression(random_state=7).fit(*arrays["train"])
            predictions.append(model.predict_proba(arrays["test"][0]))
        np.testing.assert_array_equal(predictions[0], predictions[1])
        np.testing.assert_array_equal(predictions[0], predictions[2])
        previous_paths = paths


@pytest.mark.parametrize("policy", [None, "legacy_floor"])
def test_saved_results_record_effective_window_slicing_policy(tmp_path, policy):
    cfg = OmegaConf.create(
        {"dataset": {}, "preprocessor": {"name": "raw"}, "runtime": {"seed": 7}}
    )
    if policy is not None:
        cfg.dataset.window_slicing_policy = policy
    output_path = tmp_path / "population.json"
    format_and_save_results(
        cfg=cfg,
        dataset_provider="neuroprobev2",
        model_name="logistic",
        preprocess_type="raw",
        subject_id=1,
        trial_id=1,
        eval_name="onset",
        results_splits_type="within-session",
        results_population={},
        data_load_time=0.0,
        regression_run_time=0.0,
        file_save_path=str(output_path),
    )
    result = json.loads(output_path.read_text())
    assert result["config"]["window_slicing_policy"] == (policy or "ceil")
