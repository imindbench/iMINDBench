"""Route custom models by their declared interface, independent of names."""

import json

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from imindbench import launch, run_eval
from imindbench.models import MODEL_REGISTRY
from imindbench.models.logistic_model import LogisticModel
from imindbench.utils.pipeline_contracts import validate_eval_config


def _config():
    with initialize_config_module(config_module="imindbench.conf", version_base="1.1"):
        return compose(
            config_name="config",
            overrides=[
                "paths=example",
                "model=logistic",
                "preprocessor=stft_2048Hz",
            ],
        )


@pytest.mark.parametrize("backend", [None, "tensorflow", 1, True])
def test_invalid_backend_fails_at_config_boundary(backend):
    cfg = _config()
    cfg.model.backend = backend
    with pytest.raises((TypeError, ValueError), match="model.backend"):
        validate_eval_config(cfg)
    del cfg.model.backend
    with pytest.raises(TypeError, match="model.backend"):
        validate_eval_config(cfg)


@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_launcher_uses_custom_model_backend_for_device(tmp_path, backend):
    (tmp_path / "model").mkdir()
    model = OmegaConf.load(launch.CONF_DIR / "model/logistic.yaml")
    model.name = "custom_linear"
    model.backend = backend
    if backend == "sklearn":
        del model.device
    OmegaConf.save(model, tmp_path / "model/custom.yaml")
    args = launch.parser().parse_args(
        [
            "--dataset",
            "neuroprobev2",
            "--model",
            "custom",
            "--preprocessor",
            "stft_2048Hz",
            "--task",
            "onset",
            "--target",
            "sub1_sess1",
            "--config-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
        ]
    )
    job = launch.build_commands(args)[0]
    assert ("model.device=cpu" in job["command"]) == (backend == "torch")


@pytest.mark.parametrize("merge_val_into_test", [False, True])
@pytest.mark.parametrize(
    "preprocessor", [{"name": "raw"}, {"chain": [{"name": "raw"}]}]
)
def test_custom_sklearn_model_runs_through_evaluation_entrypoint(
    tmp_path, monkeypatch, merge_val_into_test, preprocessor
):
    cfg = _config()
    cfg.model.name = "custom_linear"
    cfg.preprocessor = preprocessor
    cfg.dataset.merge_val_into_test = merge_val_into_test
    validation_calls = []

    def validate(config):
        validation_calls.append(config)
        validate_eval_config(config)

    monkeypatch.setattr(run_eval, "validate_eval_config", validate)
    monkeypatch.setitem(MODEL_REGISTRY, "custom_linear", LogisticModel)
    fold = {}
    for split in ("train", "val", "test"):
        fold[f"{split}_split"] = [
            {
                "x": np.full((2, 4), i % 2, dtype=np.float32),
                "y": i % 2,
                "split": split,
                "channel_ids": ["a", "b"],
                "channel_coords": None,
                "seq_id": None,
                "recording_id": "synthetic",
                "sample_idx": i,
                "window_start_sec": float(i),
                "window_end_sec": float(i + 1),
            }
            for i in range(8)
        ]
    monkeypatch.setattr(
        run_eval.fold_helpers,
        "iter_variable_channel_folds",
        lambda **kwargs: iter([{"fold_idx": 0, "fold": fold}]),
    )
    output = tmp_path / "population.json"
    monkeypatch.setattr(
        run_eval.logging_utils,
        "resolve_result_output_path",
        lambda **kwargs: str(output),
    )
    run_eval.main.__wrapped__(cfg)
    assert len(validation_calls) == 1
    assert "custom_linear" in output.read_text()
    assert "raw preprocessing" in json.loads(output.read_text())["description"]
