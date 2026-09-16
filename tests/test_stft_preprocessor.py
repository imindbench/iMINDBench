"""Torch STFT feature geometry and context cropping."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from imindbench.preprocessors import build_preprocessor, describe_preprocessor


def test_torch_stft_centered_frames_and_sine_peak():
    cfg = OmegaConf.create(
        {
            "name": "stft",
            "sampling_rate": 1024,
            "nperseg": 256,
            "hop_length": 64,
            "max_frequency": 512,
        }
    )
    wave = np.sin(2 * np.pi * 64 * np.arange(1024) / 1024)
    out = build_preprocessor(cfg).transform_samples([{"x": wave[None, :]}])[0]["x"]
    assert out.shape == (1, 17, 129)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0, 2:-2].argmax(axis=-1), 16)
    np.testing.assert_allclose(out[0, 2:-2, 16], 64, atol=1e-5)


def test_unnamed_chain_runs_resample_then_stft():
    cfg = OmegaConf.create(
        {
            "chain": [
                {"name": "resample", "source_rate": 2048, "target_rate": 1024},
                {
                    "name": "stft",
                    "sampling_rate": 1024,
                    "nperseg": 256,
                    "hop_length": 64,
                    "max_frequency": 512,
                },
            ]
        }
    )
    wave = np.sin(2 * np.pi * 64 * np.arange(2048) / 2048)
    out = build_preprocessor(cfg).transform_samples([{"x": wave[None, :]}])[0]
    assert describe_preprocessor(cfg) == "resample -> stft"
    assert out["x"].shape == (1, 17, 129)
    np.testing.assert_array_equal(out["x"][0, 2:-2].argmax(axis=-1), 16)


def test_laplacian_then_stft_preserves_signal_and_channel_alignment():
    cfg = OmegaConf.create(
        {
            "chain": [
                {"name": "laplacian_rereference", "remove_non_laplacian": True},
                {
                    "name": "stft",
                    "sampling_rate": 1024,
                    "nperseg": 256,
                    "hop_length": 64,
                    "max_frequency": 512,
                },
            ]
        }
    )
    wave = np.sin(2 * np.pi * 64 * np.arange(1024) / 1024)
    # The isolated B1 contact is dropped; A1/A2/A3 reference their neighbors.
    sample = {
        "x": np.asarray([wave, 8 * wave, 2 * wave, 4 * wave]),
        "channel_ids": ["id1", "id2", "id3", "id4"],
        "channel_names": ["sub1/A1", "sub1/B1", "sub1/A2", "sub1/A3"],
        "channel_coords": np.arange(12).reshape(4, 3),
    }
    out = build_preprocessor(cfg).transform_samples([sample])[0]
    assert out["x"].shape == (3, 17, 129)
    np.testing.assert_allclose(out["x"][:, 8, 16], [64, 32, 128], atol=1e-5)
    assert list(out["channel_ids"]) == ["id1", "id3", "id4"]
    assert list(out["channel_names"]) == ["sub1/A1", "sub1/A2", "sub1/A3"]
    np.testing.assert_array_equal(
        out["channel_coords"], sample["channel_coords"][[0, 2, 3]]
    )


@pytest.mark.parametrize(
    "key,value", [("use_scipy", True), ("use_scipy", False), ("boundary", None)]
)
@pytest.mark.parametrize("location", ["stft", "multi_stft", "window"])
def test_removed_scipy_options_fail_at_construction(key, value, location):
    cfg = {"name": "multi_stft", "hop_length": 4, "windows": [{"nperseg": 16}]}
    if location == "stft":
        cfg = {"name": "stft", key: value}
    elif location == "multi_stft":
        cfg[key] = value
    else:
        cfg["windows"][0][key] = value
    with pytest.raises(ValueError, match="STFT uses Torch only"):
        build_preprocessor(OmegaConf.create(cfg))


@pytest.mark.parametrize(
    "padded,target_length,expected_frames",
    [(False, 64, 17), (False, 65, 17), (True, 65, 18)],
)
def test_context_crop_uses_centered_torch_frames(
    padded, target_length, expected_frames
):
    cfg = OmegaConf.create(
        {
            "name": "stft",
            "nperseg": 16,
            "hop_length": 4,
            "sampling_rate": 64,
            "max_frequency": 32,
            "padded": padded,
        }
    )
    sample = {
        "x": np.random.default_rng(1).normal(size=(2, 96)),
        "context_requested_start_sec": 0.0,
        "target_start_sample_in_context": 16,
        "target_num_samples": target_length,
    }
    stft = build_preprocessor(cfg).transform_samples([sample])[0]
    crop = build_preprocessor(OmegaConf.create({"name": "crop_to_target_window"}))
    out = crop.transform_samples([stft])[0]
    assert out["x"].shape == (2, expected_frames, 9)
    np.testing.assert_array_equal(out["x"], stft["x"][:, 4 : 4 + expected_frames, :])
    assert "context_stft_use_scipy" not in stft
    assert "context_stft_boundary" not in stft
