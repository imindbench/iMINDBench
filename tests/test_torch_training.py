"""Exercise optimizer updates and validation/stopping timing on real CPU batches."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from imindbench.torch_runner import TorchRunner


def test_determinism_can_be_disabled_after_an_enabled_runner(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        for requested in (True, False):
            cfg = OmegaConf.create(
                {"model": {"device": "cpu"}, "runtime": {"deterministic": requested}}
            )
            TorchRunner(cfg)
            assert torch.are_deterministic_algorithms_enabled() is requested
            assert torch.backends.cudnn.deterministic is requested
    finally:
        torch.use_deterministic_algorithms(enabled)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = benchmark


def run_training(runner_class, mode, early_stop=False):
    torch.manual_seed(7)
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 7},
            "model": {
                "device": "cpu",
                "training_mode": mode,
                "optimizer": "AdamW",
                "learning_rate": 0.01,
                "grad_clip": 0.5,
                "max_iter": 3,
                "patience": 1 if early_stop else 10,
                "tol": 100 if early_stop else 0,
                "total_steps": 7,
                "validation_interval": 3,
                "scheduler": {"name": "cosine_annealing", "total_steps": 9},
            },
        }
    )
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
    )
    model = SimpleNamespace(model=network, accepts_coords=False)
    data = [
        {"x": torch.tensor([float(i), float(i % 2), 1.0]), "y": 2 + 2 * (i % 2)}
        for i in range(10)
    ]
    train = DataLoader(
        data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(8)
    )
    val = DataLoader(data, batch_size=4)
    events = []
    weights = []

    class RecordingRunner(runner_class):
        def _create_optimizer_and_scheduler(self, *args, **kwargs):
            optimizer, scheduler, rate = super()._create_optimizer_and_scheduler(
                *args, **kwargs
            )
            optimizer.register_step_post_hook(lambda *args: events.append("update"))
            return optimizer, scheduler, rate

        def _evaluate_loader(self, *args, **kwargs):
            if kwargs.get("criterion") is not None:
                events.append("validate")
                weights.append(
                    {
                        key: value.detach().clone()
                        for key, value in network.state_dict().items()
                    }
                )
            return super()._evaluate_loader(*args, **kwargs)

    metrics = RecordingRunner(cfg).run_fold(
        model, train_loader=train, val_loader=val, test_loader=val
    )
    return metrics, network.state_dict(), events, weights


@pytest.mark.parametrize(
    "mode,early_stop,expected",
    [
        ("epoch_based", False, (["update"] * 3 + ["validate"]) * 3),
        ("epoch_based", True, (["update"] * 3 + ["validate"]) * 2),
        (
            "steps_based",
            False,
            (["update"] * 3 + ["validate"]) * 2 + ["update", "validate"],
        ),
    ],
)
def test_training_cadence_and_checkpoint_restore(mode, early_stop, expected):
    metrics, state, events, checkpoints = run_training(TorchRunner, mode, early_stop)
    assert events == expected
    assert all(0 <= value <= 1 for value in metrics.values())
    assert any(
        all(torch.equal(state[key], value) for key, value in checkpoint.items())
        for checkpoint in checkpoints
    )
