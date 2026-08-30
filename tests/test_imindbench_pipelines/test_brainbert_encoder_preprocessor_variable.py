import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from imindbench.preprocessors.brainbert_encoder_preprocessor import (
    BrainBERTEncoderPreprocessor,
)


class _DummyEncoder(torch.nn.Module):
    def forward(self, x, src_key_mask=None):
        _ = src_key_mask
        # [n_seq, time, freq] -> [n_seq, time, hidden=2]
        summed = x.sum(dim=2, keepdim=True)
        meaned = x.mean(dim=2, keepdim=True)
        return torch.cat([summed, meaned], dim=2)


def _sample(seed: int, n_channels: int, n_time: int = 4, n_freq: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_channels, n_time, n_freq)).astype(np.float32)
    return {
        "x": x,
        "y": int(seed % 2),
        "channel_ids": [f"ch{i}" for i in range(n_channels)],
        "channel_names": [f"ch{i}" for i in range(n_channels)],
        "channel_coords": np.zeros((n_channels, 3), dtype=np.float32),
        "brain_areas": ["region_a"] * n_channels,
        "seq_id": np.zeros((n_channels,), dtype=np.int64),
        "recording_id": f"rec_{seed}",
        "split": "train",
        "sample_idx": seed,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
    }


def _build_preprocessor():
    cfg = DictConfig(
        {
            "name": "brainbert_encoder",
            "upstream_ckpt": "/tmp/fake.ckpt",
            "device": "cpu",
            "pool": "mean",
            "encoder_batch_size": 2,
            "min_batch_size": 1,
        }
    )
    pre = BrainBERTEncoderPreprocessor(cfg)
    pre.model = _DummyEncoder()
    pre.model_cfg = type(
        "DummyCfg",
        (),
        {
            "hidden_dim": 2,
            "nhead": 1,
            "encoder_num_layers": 1,
            "layer_dim_feedforward": 8,
        },
    )()
    pre._load_model = lambda: None
    return pre


def test_transform_samples_requires_fit_state():
    pre = _build_preprocessor()
    with pytest.raises(RuntimeError, match="fit_split"):
        pre.transform_samples([_sample(0, 2)])


def test_fit_split_and_transform_samples_match_encoder_outputs():
    pre = _build_preprocessor()
    samples = [_sample(1, 2), _sample(2, 3), _sample(3, 1)]

    state = pre.fit_split(iter(samples))
    assert state["feature_shape"] == [4, 3]
    assert state["optimal_batch_size"] >= 1

    transformed = pre.transform_samples(samples)
    assert len(transformed) == len(samples)

    for sample_in, sample_out in zip(samples, transformed):
        assert sample_out["x"].shape[0] == sample_in["x"].shape[0]
        expected = pre._encode_sequences(np.asarray(sample_in["x"], dtype=np.float32))
        np.testing.assert_allclose(sample_out["x"], expected, atol=1e-6)

    single = pre.transform_samples([samples[0]])[0]
    np.testing.assert_allclose(single["x"], transformed[0]["x"], atol=1e-6)


def test_chunked_and_non_chunked_transform_samples_match():
    samples = [_sample(11, 2), _sample(12, 3), _sample(13, 2)]

    chunked = _build_preprocessor()
    chunked.fit_split(iter(samples))
    chunked_out = chunked.transform_samples(samples)

    non_chunked = _build_preprocessor()
    non_chunked.manual_batch_size = 10_000
    non_chunked.fit_split(iter(samples))
    non_chunked_out = non_chunked.transform_samples(samples)

    for a, b in zip(chunked_out, non_chunked_out):
        np.testing.assert_allclose(a["x"], b["x"], atol=1e-6)
