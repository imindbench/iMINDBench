"""Compose external paths with packaged optional-resource defaults."""

import json

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from imindbench.utils.pipeline_contracts import validate_eval_config


def _compose(tmp_path, paths, model="logistic"):
    (tmp_path / "paths").mkdir()
    OmegaConf.save(OmegaConf.create(paths), tmp_path / "paths/local.yaml")
    with initialize_config_module(config_module="imindbench.conf", version_base="1.1"):
        return compose(
            config_name="config",
            overrides=[
                "paths=local",
                f"model={model}",
                "preprocessor=laplacian_stft_2048Hz",
                f"hydra.searchpath={json.dumps(['file://' + str(tmp_path)])}",
            ],
        )


def test_minimal_external_paths_inherit_optional_resources(tmp_path):
    cfg = _compose(
        tmp_path, {"dataset_root": "/processed", "dataset_dirname": "neuroprobe_2025"}
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_eval_config(cfg)
    assert resolved["runtime"]["preprocessed_split_cache_dir"] is None
    assert resolved["runtime"]["train_source_cache_dir"] is None
    assert resolved["paths"]["brainbert_checkpoint"] is None
    assert "clean_laplacian_path" not in resolved["paths"]
    assert "electrode_dir_btb" not in resolved["paths"]
    assert "electrode_dir_byd" not in resolved["paths"]


def test_diver_uses_external_checkpoint_and_cache_paths(tmp_path):
    cfg = _compose(
        tmp_path,
        {
            "dataset_root": "/processed",
            "dataset_dirname": "neuroprobe_2025",
            "diver_checkpoint": "/weights/diver.pt",
            "diver_shape_cache_dir": "/cache/diver",
        },
        model="diver",
    )
    assert cfg.model.upstream_ckpt == "/weights/diver.pt"
    assert cfg.model.model_dir == "/cache/diver"
    validate_eval_config(cfg)


@pytest.mark.parametrize("missing", ["dataset_root", "dataset_dirname"])
def test_required_dataset_paths_still_fail_when_missing(tmp_path, missing):
    paths = {"dataset_root": "/processed", "dataset_dirname": "neuroprobe_2025"}
    del paths[missing]
    cfg = _compose(tmp_path, paths)
    with pytest.raises(InterpolationKeyError, match=missing):
        validate_eval_config(cfg)
