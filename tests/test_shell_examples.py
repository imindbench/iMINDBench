"""Exercise the user-facing Bash examples without data or training."""

import shlex
import subprocess
from pathlib import Path

import pytest

from imindbench.utils.pipeline_contracts import validate_eval_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "dataset", ["neuroprobev2", "kelesbyd2024", "berezutskayapippi2022"]
)
def test_dataset_default_covers_all_tasks_targets_and_pairings(
    tmp_path, dataset_script, dataset_selections, dataset
):
    script = dataset_script(dataset=dataset)
    result = subprocess.run(
        ["bash", script], cwd=tmp_path, text=True, capture_output=True, check=True
    )
    tasks, targets = dataset_selections(dataset)
    expected_units = {
        (task, *target.removeprefix("sub").split("_sess"))
        for task in tasks
        for target in targets
    }
    rate = 1000 if dataset == "kelesbyd2024" else 2048
    spectral = f"laplacian_multi_stft_{rate}Hz"
    expected_pairings = {("logistic", spectral)}
    actual = set()
    for line in result.stdout.splitlines():
        fields = dict(
            token.split("=", 1) for token in shlex.split(line) if "=" in token
        )
        assert fields["dataset"] == dataset
        assert fields["dataset.regime"] == "within-session"
        actual.add(
            (
                (fields["model"], fields["preprocessor"]),
                (
                    fields["dataset.task"],
                    fields["dataset.test_subject"],
                    fields["dataset.test_session"],
                ),
            )
        )
    assert actual == {
        (pairing, unit) for pairing in expected_pairings for unit in expected_units
    }
    assert len(result.stdout.splitlines()) == len(actual)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("family", ["within_dataset", "multi_dataset"])
@pytest.mark.parametrize(
    "dataset", ["neuroprobev2", "kelesbyd2024", "berezutskayapippi2022"]
)
def test_optional_transfer_uses_main_cohort(tmp_path, dataset_script, family, dataset):
    import json

    script = dataset_script(dataset=dataset, family=family, MODEL="popt")
    result = subprocess.run(
        ["bash", script], text=True, capture_output=True, check=True
    )
    manifest = json.loads(
        (
            ROOT
            / "imindbench/decodable_subject_sessions/stft_or_htnet_500hz_val_mean0p60"
            / f"{dataset}.json"
        ).read_text()
    )
    expected = {
        (task, target)
        for task, values in manifest["tasks"].items()
        for target in values["subject_sessions"]
    }
    actual = set()
    for line in result.stdout.splitlines():
        fields = dict(
            token.split("=", 1) for token in shlex.split(line) if "=" in token
        )
        actual.add(
            (
                fields["dataset.task"],
                f"sub{fields['dataset.test_subject']}_sess{fields['dataset.test_session']}",
            )
        )
        assert fields["dataset"] == (
            dataset if family == "within_dataset" else f"{dataset}_multi_dataset_train"
        )
        assert fields["dataset.regime"] == (
            "hold-in-session" if family == "within_dataset" else "within-session"
        )
    assert actual == expected
    assert len(result.stdout.splitlines()) == len(expected)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("model", ["logistic", "mlp", "cnn", "popt"])
def test_neuroprobe_sample_efficiency_covers_models_units_and_fractions(
    tmp_path, dataset_script, dataset_selections, model
):
    script = dataset_script(
        family="sample_efficiency", MODEL=model, EXPERIMENT=f"multi_stft/{model}"
    )
    result = subprocess.run(
        ["bash", script], text=True, capture_output=True, check=True
    )
    tasks, targets = dataset_selections("neuroprobev2")
    expected = {
        (model, fraction, task, *target.removeprefix("sub").split("_sess"))
        for fraction in ["1.0", "0.5", "0.25", "0.125", "0.0625"]
        for task in tasks
        for target in targets
    }
    actual = set()
    paths = set()
    for line in result.stdout.splitlines():
        fields = dict(
            token.split("=", 1) for token in shlex.split(line) if "=" in token
        )
        actual.add(
            tuple(
                fields[key]
                for key in [
                    "model",
                    "dataset.train_sample_fraction",
                    "dataset.task",
                    "dataset.test_subject",
                    "dataset.test_session",
                ]
            )
        )
        paths.add(fields["hydra.run.dir"])
    assert actual == expected
    assert len(paths) == len(expected) == len(result.stdout.splitlines())
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("model_name", ["logistic", "mlp", "cnn", "htnet_500Hz"])
def test_waveform_baselines_accept_500hz_inputs(model_name):
    import numpy as np
    from hydra import compose, initialize_config_dir

    from imindbench.models import build_model

    with initialize_config_dir(
        config_dir=str(ROOT / "imindbench/conf"), version_base="1.1"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "paths=example",
                "dataset=neuroprobev2",
                f"model={model_name}",
                "preprocessor=laplacian_wav_HPF_global_robust_scalar_long_context_15s_2048Hzto500Hz",
                "experiment=waveform500",
                "model.device=cpu",
            ],
        )
    validate_eval_config(cfg)
    model = build_model(cfg.model)
    # One-second windows at 500 Hz, after waveform preprocessing.
    x = np.random.default_rng(42).normal(size=(8, 3, 500)).astype(np.float32)
    y = np.array([0, 1] * 4)
    if model_name == "logistic":
        batch = model.prepare_batch({"x": x, "y": y})
        model.fit(batch["x"], batch["y"])
        probabilities = model.predict_proba(batch["x"])
    else:
        model.build_model(x.shape[1:], 2, device="cpu")
        probabilities = model.predict_proba(x)
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
