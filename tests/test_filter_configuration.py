"""Explicit filter metadata preserves historical processing and cache isolation."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import signal

from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter
from imindbench.utils.logging_utils import (
    build_internal_eval_result,
    build_public_export_result,
    save_results,
)

CONF = Path(__file__).resolve().parents[1] / "imindbench/conf/preprocessor"


@pytest.mark.parametrize("sampling_rate", [1000, 2048])
@pytest.mark.parametrize("high_pass_hz", [0.0, 0.5])
@pytest.mark.parametrize(
    "notch_zero_phase,high_pass_zero_phase",
    [(False, False), (False, True), (True, True)],
)
def test_standard_filter_matches_historical_scipy_processing(
    sampling_rate, high_pass_hz, notch_zero_phase, high_pass_zero_phase
):
    x = np.random.default_rng(0).normal(size=(2, sampling_rate * 2)).astype(np.float32)
    cfg = {
        "name": "time_domain_filter",
        "sampling_rate": sampling_rate,
        "notch_zero_phase": notch_zero_phase,
        "high_pass_zero_phase": high_pass_zero_phase,
    }
    if high_pass_hz:
        cfg["high_pass_hz"] = high_pass_hz
    legacy = build_preprocessor(OmegaConf.create(cfg))
    explicit = build_preprocessor(
        OmegaConf.create(
            {
                **cfg,
                "high_pass_hz": high_pass_hz,
                "notch_freqs": [60, 120, 180, 240, 300, 360],
            }
        )
    )
    expected = x.astype(np.float64)
    for frequency in [60, 120, 180, 240, 300, 360]:
        b, a = signal.iirnotch(frequency / (sampling_rate / 2), 30)
        filter_fn = signal.filtfilt if notch_zero_phase else signal.lfilter
        expected = filter_fn(b, a, expected, axis=-1)
    if high_pass_hz:
        sos = signal.butter(
            4, high_pass_hz, btype="highpass", fs=sampling_rate, output="sos"
        )
        filter_fn = signal.sosfiltfilt if high_pass_zero_phase else signal.sosfilt
        expected = filter_fn(sos, expected, axis=-1)
    for pre in [legacy, explicit]:
        actual = pre.transform_samples([{"x": x}])[0]["x"]
        np.testing.assert_array_equal(actual, expected.astype(np.float32))


@pytest.mark.parametrize("high_pass_hz", [None, 0.0, 0.5, 1.0])
def test_diver_filter_matches_mne_with_effective_cutoff(high_pass_hz):
    from mne.filter import filter_data, notch_filter

    sampling_rate = 1000
    x = np.random.default_rng(1).normal(size=(2, 15000)).astype(np.float32)
    cfg = {"name": "time_domain_filter_diver_style", "sampling_rate": sampling_rate}
    omitted = build_preprocessor(OmegaConf.create(cfg))
    cfg["high_pass_hz"] = high_pass_hz
    pre = build_preprocessor(OmegaConf.create(cfg))
    effective = 0.5 if high_pass_hz is None else high_pass_hz
    expected = filter_data(
        x.astype(np.float64),
        sfreq=sampling_rate,
        l_freq=effective or None,
        h_freq=None,
        verbose=False,
    )
    expected = notch_filter(
        expected, Fs=sampling_rate, freqs=[60, 120, 180], verbose=False
    )
    actual = pre.transform_samples([{"x": x}])[0]["x"]
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    if effective == 0.5:
        np.testing.assert_array_equal(
            omitted.transform_samples([{"x": x}])[0]["x"], actual
        )


def test_standard_notches_control_processing_and_cache(tmp_path):
    x = np.random.default_rng(2).normal(size=(2, 1000)).astype(np.float32)
    cfg = {
        "name": "time_domain_filter",
        "sampling_rate": 500,
        "filter_cache_enabled": True,
        "filter_cache_dir": str(tmp_path),
    }
    first = build_preprocessor(OmegaConf.create({**cfg, "notch_freqs": [60, 250, 300]}))
    second = build_preprocessor(OmegaConf.create({**cfg, "notch_freqs": [120]}))
    unfiltered = build_preprocessor(OmegaConf.create({**cfg, "notch_freqs": []}))
    result = first.transform_samples([{"x": x}])[0]["x"]
    # Notches at and above Nyquist are ignored, preserving the historical rule.
    b, a = signal.iirnotch(60 / 250, 30)
    expected = signal.lfilter(b, a, x.astype(np.float64), axis=-1).astype(np.float32)
    np.testing.assert_array_equal(result, expected)
    assert len(list(tmp_path.rglob("*.npy"))) == 1
    np.testing.assert_array_equal(first.transform_samples([{"x": x}])[0]["x"], result)
    changed = second.transform_samples([{"x": x}])[0]["x"]
    assert not np.array_equal(result, changed)
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    np.testing.assert_array_equal(unfiltered.transform_samples([{"x": x}])[0]["x"], x)
    assert len(list(tmp_path.rglob("*.npy"))) == 3


def test_explicit_defaults_reuse_standard_filter_cache(tmp_path):
    cfg = {
        "name": "time_domain_filter",
        "filter_cache_enabled": True,
        "filter_cache_dir": str(tmp_path),
    }
    sample = {"x": np.random.default_rng(2).normal(size=(2, 2048)).astype(np.float32)}
    legacy = build_preprocessor(OmegaConf.create(cfg))
    expected = legacy.transform_samples([sample])[0]["x"]
    explicit = build_preprocessor(
        OmegaConf.create(
            {
                **cfg,
                "high_pass_hz": 0,
                "notch_freqs": [60, 120, 180, 240, 300, 360],
                "filter_cache_mode": "read_only",
            }
        )
    )
    np.testing.assert_array_equal(
        explicit.transform_samples([sample])[0]["x"], expected
    )
    assert len(list(tmp_path.rglob("*.npy"))) == 1


@pytest.mark.parametrize(
    "cache_kind,old_version", [("train_source", 5), ("preprocessed_split", 2)]
)
def test_outer_caches_reject_results_from_ignored_filter_settings(
    tmp_path, cache_kind, old_version
):
    # A complete old payload must not be accepted merely because its config
    # already contained a field whose filtering semantics have now changed.
    identity = {
        "provider": "neuroprobev2",
        "preprocessor": {
            "chain": [{"name": "time_domain_filter", "notch_freqs": [50]}],
        },
    }
    if cache_kind == "train_source":
        path = tmp_path / "source.pkl"
        data_adapter._save_train_source_cache(
            path,
            identity=identity,
            preprocess_state=None,
            samples=[{"x": np.ones((1, 10))}],
        )
        assert data_adapter._load_train_source_cache(path) is not None
        payload = pickle.loads(path.read_bytes())
        payload["cache_version"] = old_version
        path.write_bytes(pickle.dumps(payload))
        assert data_adapter._load_train_source_cache(path) is None
    else:
        manifest = {
            "cache_version": data_adapter._PREPROCESSED_SPLIT_CACHE_VERSION,
            "identity": identity,
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        (tmp_path / "preprocess_state.pkl").write_bytes(pickle.dumps(None))
        for split in ("train", "val", "test"):
            (tmp_path / f"{split}.pkl").write_bytes(
                pickle.dumps([{"x": np.ones((1, 10))}])
            )
        assert data_adapter._load_preprocessed_split_cache(tmp_path) is not None
        manifest["cache_version"] = old_version
        path.write_text(json.dumps(manifest))
        assert data_adapter._load_preprocessed_split_cache(tmp_path) is None


@pytest.mark.parametrize(
    "stage", ["time_domain_filter", "time_domain_filter_diver_style"]
)
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        ("0.5", TypeError),
        (-0.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (500, ValueError),
    ],
)
def test_filter_rejects_invalid_cutoffs(stage, value, error):
    with pytest.raises(error, match="high_pass_hz"):
        build_preprocessor(
            OmegaConf.create(
                {
                    "name": stage,
                    "sampling_rate": 1000,
                    "high_pass_hz": value,
                }
            )
        )


def test_standard_filter_rejects_null_cutoff():
    with pytest.raises(TypeError, match="high_pass_hz"):
        build_preprocessor(
            OmegaConf.create({"name": "time_domain_filter", "high_pass_hz": None})
        )


@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        (60, TypeError),
        ("60", TypeError),
        ([True], TypeError),
        (["60"], TypeError),
        ([0], ValueError),
        ([-1], ValueError),
        ([float("nan")], ValueError),
        ([float("inf")], ValueError),
    ],
)
def test_standard_filter_rejects_invalid_notches(value, error):
    with pytest.raises(error, match="notch_freqs"):
        build_preprocessor(
            OmegaConf.create({"name": "time_domain_filter", "notch_freqs": value})
        )


@pytest.mark.parametrize(
    "path", sorted(CONF.glob("*.yaml")), ids=lambda path: path.stem
)
def test_bundled_filters_export_explicit_settings_without_chain_name(path, tmp_path):
    # Supply the normal config parents needed by BrainBERT's interpolations.
    root = OmegaConf.create(
        {
            "preprocessor": OmegaConf.load(path),
            "paths": {"brainbert_checkpoint": "/tmp/brainbert.pt"},
            "model": {"device": "cpu"},
        }
    )
    cfg = root.preprocessor
    assert "name" not in cfg
    stage = next(
        stage for stage in cfg.chain if stage.name.startswith("time_domain_filter")
    )
    build_preprocessor(stage)
    assert isinstance(stage.high_pass_hz, (int, float))
    assert len(stage.notch_freqs) > 0
    internal = build_internal_eval_result(
        provider="neuroprobe",
        task="onset",
        regime="within_session",
        subject_id=1,
        trial_id=1,
        model_name="logistic",
        preprocess_type="chain",
        preprocess_parameters=OmegaConf.to_container(cfg, resolve=True),
        window_slicing_policy={},
        seed=42,
        results_population={},
        subject_load_time=0.0,
        regression_run_time=0.0,
    )
    exported = build_public_export_result(
        internal_result=internal,
        author="Test",
        organization="Test",
        organization_url="",
    )
    output = tmp_path / "population_test.json"
    save_results(exported, output)
    chain = json.loads(output.read_text())["config"]["preprocess"]["chain"]
    exported_filter = next(
        stage for stage in chain if stage["name"].startswith("time_domain_filter")
    )
    assert exported_filter["high_pass_hz"] == stage.high_pass_hz
    assert exported_filter["notch_freqs"] == list(stage.notch_freqs)
