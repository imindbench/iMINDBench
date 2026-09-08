"""Waveform resampling contracts, including the shipped downstream chains."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from imindbench.preprocessors import build_preprocessor

CONF = Path(__file__).resolve().parents[1] / "imindbench/conf/preprocessor"
RESAMPLE_CONFIGS = [
    path for path in sorted(CONF.glob("*.yaml")) if "name: resample" in path.read_text()
]


@pytest.mark.parametrize(
    "source_rate,target_rate,length,expected_length",
    [
        (2048, 500, 2048, 500),
        (1000, 500, 1000, 500),
        (1000, 2048, 1000, 2048),
        (2048, 500, 10, 3),
        (1000, 2048, 10, 21),
        (1000, 1000, 10, 10),
    ],
)
@pytest.mark.parametrize("tensor", [False, True])
def test_resample_length_metadata_and_input_preservation(
    source_rate, target_rate, length, expected_length, tensor
):
    pre = build_preprocessor(
        OmegaConf.create(
            {"name": "resample", "source_rate": source_rate, "target_rate": target_rate}
        )
    )
    original = np.random.default_rng(0).normal(size=(2, length)).astype(np.float32)
    x = torch.tensor(original, requires_grad=True) if tensor else original.copy()
    sample = {"x": x, "sampling_rate": source_rate, "channel_ids": [1, 2], "y": 1}
    out = pre.transform_samples([sample])[0]
    assert out["x"].shape == (2, expected_length)
    assert out["x"].dtype == np.float32
    assert out["sampling_rate"] == target_rate
    assert out["channel_ids"] == [1, 2]
    assert out["y"] == 1
    assert sample["sampling_rate"] == source_rate
    np.testing.assert_array_equal(x.detach().numpy() if tensor else x, original)
    if source_rate == target_rate:
        np.testing.assert_array_equal(out["x"], original)


@pytest.mark.parametrize(
    "source_rate,target_rate", [(2048, 500), (1000, 500), (1000, 2048)]
)
def test_resample_preserves_passband_and_filters_above_target_nyquist(
    source_rate, target_rate
):
    pre = build_preprocessor(
        OmegaConf.create(
            {"name": "resample", "source_rate": source_rate, "target_rate": target_rate}
        )
    )
    time = np.arange(source_rate) / source_rate
    x = np.sin(2 * np.pi * 40 * time)
    if target_rate < source_rate:
        x += np.sin(2 * np.pi * 400 * time)
    out = pre.transform_samples([{"x": x[None, :]}])[0]
    expected = np.sin(2 * np.pi * 40 * np.arange(target_rate) / target_rate)
    # Exclude filter transients at the recording boundaries.
    np.testing.assert_allclose(out["x"][0, 30:-30], expected[30:-30], atol=0.005)
    assert out["sampling_rate"] == target_rate


@pytest.mark.parametrize("key", ["source_rate", "target_rate"])
@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        (True, TypeError),
        (1000.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_resample_rejects_invalid_rates(key, value, error):
    cfg = OmegaConf.create(
        {"name": "resample", "source_rate": 1000, "target_rate": 500}
    )
    cfg[key] = value
    with pytest.raises(error, match=key):
        build_preprocessor(cfg)
    del cfg[key]
    with pytest.raises(ValueError, match=key):
        build_preprocessor(cfg)


@pytest.mark.parametrize("shape", [(10,), (2, 3, 10), (2, 0)])
def test_resample_rejects_non_waveform_inputs(shape):
    pre = build_preprocessor(
        OmegaConf.create({"name": "resample", "source_rate": 1000, "target_rate": 500})
    )
    with pytest.raises(ValueError, match="shape"):
        pre.transform_samples([{"x": np.zeros(shape)}])


@pytest.mark.parametrize("path", RESAMPLE_CONFIGS, ids=lambda path: path.stem)
def test_shipped_resampling_and_downstream_stages(path):
    cfg = OmegaConf.load(path)
    # Context loading/filtering precedes the changed stage; exercise the actual
    # resampling, rereferencing and fitted scaling suffix on one-second samples.
    start = next(i for i, stage in enumerate(cfg.chain) if stage.name == "resample")
    cfg.chain = cfg.chain[start:]
    pre = build_preprocessor(cfg)
    source_rate = cfg.chain[0].source_rate
    target_rate = cfg.chain[0].target_rate
    sample = {
        "x": np.random.default_rng(0).normal(size=(3, source_rate)).astype(np.float32),
        "channel_ids": [1, 2, 3],
        "channel_names": ["A1", "A2", "A3"],
        "sampling_rate": source_rate,
        "y": 1,
    }
    pre.fit_split(iter([sample]))
    out = pre.transform_samples([sample])[0]
    assert out["x"].shape == (3, target_rate)
    assert out["sampling_rate"] == target_rate
    assert out["channel_ids"] == [1, 2, 3]
    assert out["channel_names"] == ["A1", "A2", "A3"]
    assert np.isfinite(out["x"]).all()
