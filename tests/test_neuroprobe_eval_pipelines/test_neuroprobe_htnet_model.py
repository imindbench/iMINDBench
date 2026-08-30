import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.models import build_model  # noqa: E402
from neuroprobe_eval.models.htnet_model import HTNetModel  # noqa: E402
from neuroprobe_eval.torch_runner import TorchRunner  # noqa: E402
from neuroprobe_eval.utils.collate import variable_channel_collate  # noqa: E402


def _make_cfg(**overrides):
    cfg = {
        "name": "htnet",
        "device": "cpu",
        "requires_aligned_channels": True,
        "requires_coords": False,
        "f1": 4,
        "d": 2,
        "f2": 8,
        "kernel_length": 16,
        "separable_kernel_length": 8,
        "dropout_rate": 0.0,
        "dropout_type": "Dropout",
        "compute_val": "power",
        "do_log": False,
        "data_srate": 1000,
        "base_split": 4,
        "batch_size": 2,
        "learning_rate": 0.001,
        "training_mode": "epoch_based",
        "max_iter": 1,
        "patience": 1,
        "tol": 1e-4,
        "random_state": 0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_htnet_registry_build_and_forward_shape():
    model = build_model(_make_cfg())
    assert isinstance(model, HTNetModel)

    model.build_model(input_shape=(3, 64), n_classes=2, device="cpu")
    y = model.model(torch.randn(5, 3, 64))

    assert y.shape == (5, 2)


def test_htnet_rejects_stft_like_input_shape():
    model = build_model(_make_cfg())

    with pytest.raises(ValueError, match="time-domain"):
        model.build_model(input_shape=(3, 10, 20), n_classes=2, device="cpu")

    with pytest.raises(ValueError, match="time-domain"):
        model.prepare_batch({"x": torch.randn(2, 3, 10, 20), "y": torch.tensor([0, 1])})


def test_htnet_relative_power_forward_shape():
    model = build_model(_make_cfg(compute_val="relative_power", do_log=False))
    model.build_model(input_shape=(2, 64), n_classes=3, device="cpu")

    y = model.model(torch.randn(4, 2, 64))

    assert y.shape == (4, 3)


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _sample(label: int, split: str, sample_idx: int):
    rng = np.random.default_rng(seed=(label * 100) + sample_idx)
    return {
        "x": rng.normal(size=(3, 64)).astype(np.float32),
        "y": int(label),
        "channel_ids": ["c0", "c1", "c2"],
        "channel_coords": None,
        "seq_id": np.zeros((3,), dtype=np.int64),
        "recording_id": f"{split}_rec",
        "split": split,
        "sample_idx": int(sample_idx),
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
    }


def test_htnet_runs_one_torch_runner_fold():
    cfg = OmegaConf.create(
        {
            "model": _make_cfg(),
            "runner": {"num_workers": 0, "pin_memory": False},
        }
    )
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)

    train_ds = _ListDataset(
        [
            _sample(0, "train", 0),
            _sample(1, "train", 1),
            _sample(0, "train", 2),
            _sample(1, "train", 3),
        ]
    )
    val_ds = _ListDataset([_sample(0, "val", 0), _sample(1, "val", 1)])
    test_ds = _ListDataset([_sample(0, "test", 0), _sample(1, "test", 1)])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )

    result = runner.run_fold(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert result["test_accuracy"] >= 0.0
    assert result["test_roc_auc"] >= 0.0
