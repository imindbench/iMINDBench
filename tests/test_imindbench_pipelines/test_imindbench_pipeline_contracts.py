import os

import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from imindbench.preprocessors import PREPROCESSOR_REGISTRY
from imindbench.utils.pipeline_contracts import (
    resolve_train_source_configs,
    validate_eval_config,
)


def _cfg(preprocessor):
    return OmegaConf.create(
        {
            "dataset": {
                "root": "/tmp",
                "dirname": "neuroprobe_2025",
                "provider": "neuroprobe2025",
                "subset_tier": "lite",
                "label_mode": "binary",
                "task": "onset",
                "test_subject": 1,
                "test_session": 1,
                "regime": "DS-DM",
                "brain_area_key": "localization_DesikanKilliany",
                "coordinate_profile": "popt_lip",
                "uniquify_channel_ids_with_subject": True,
                "uniquify_channel_ids_with_session": False,
                "merge_val_into_test": False,
            },
            "model": {
                "name": "cnn",
                "requires_aligned_channels": True,
                "requires_coords": False,
            },
            "preprocessor": preprocessor,
            "runtime": {"seed": 0, "overwrite": True},
            "submitter": {
                "author": "tester",
                "organization": "org",
                "organization_url": "https://example.com",
            },
        }
    )


def test_validate_eval_config_accepts_top_level_region_intersection_pool():
    cfg = _cfg({"name": "region_intersection_pool"})
    validate_eval_config(cfg)


def test_validate_eval_config_rejects_missing_region_intersection_pool():
    cfg = _cfg({"name": "raw"})
    with pytest.raises(ValueError, match="requires either a top-level"):
        validate_eval_config(cfg)


def test_validate_eval_config_accepts_default_coordinate_profile_when_missing():
    cfg = _cfg({"name": "region_intersection_pool"})
    del cfg.dataset["coordinate_profile"]

    validate_eval_config(cfg)


def test_validate_eval_config_rejects_unknown_coordinate_profile():
    cfg = _cfg({"name": "region_intersection_pool"})
    cfg.dataset.coordinate_profile = "unknown_profile"

    with pytest.raises(ValueError, match="dataset.coordinate_profile"):
        validate_eval_config(cfg)


def test_validate_eval_config_rejects_removed_seegnificant_profile():
    cfg = _cfg({"name": "region_intersection_pool"})
    cfg.dataset.coordinate_profile = "seegnificant_mni"

    with pytest.raises(ValueError, match="dataset.coordinate_profile"):
        validate_eval_config(cfg)


def test_removed_preprocessors_are_not_registered():
    assert "channel_subselect" not in PREPROCESSOR_REGISTRY
    assert "downsample_pad" not in PREPROCESSOR_REGISTRY
    assert "raw" in PREPROCESSOR_REGISTRY


def test_resolve_train_source_configs_inherit_top_level_coordinate_profile():
    cfg = _cfg({"name": "region_intersection_pool"})
    cfg.dataset.regime = "within-session"
    cfg.dataset.provider = "neuroprobev2"
    cfg.dataset.coordinate_profile = "diver_mni"
    cfg.dataset.train_sources = [
        {
            "provider": "neuroprobev2",
            "root": "/tmp",
            "dirname": "neuroprobe_2025",
            "subset_tier": "lite",
            "test_subject": 1,
            "test_session": 1,
            "regime": "hold-in-session",
            "train_sample_fraction": 1.0,
        },
        {
            "provider": "kelesbyd2024",
            "root": "/tmp",
            "dirname": "keles_byd_2024",
            "subset_tier": "full",
            "test_subject": 41,
            "test_session": 1,
            "regime": "hold-in-session",
            "train_sample_fraction": 1.0,
        },
    ]

    sources = resolve_train_source_configs(cfg.dataset)

    assert [source["coordinate_profile"] for source in sources] == [
        "diver_mni",
        "diver_mni",
    ]


@pytest.mark.parametrize("submitter", [None, {}, {"author": None}, {"author": "A"}])
def test_validate_eval_config_accepts_optional_attribution(submitter):
    cfg = _cfg({"name": "region_intersection_pool"})
    cfg.submitter = submitter
    validate_eval_config(cfg)
    del cfg["submitter"]
    validate_eval_config(cfg)


@pytest.mark.parametrize("submitter", [42, {"author": 42}, {"organization": []}])
def test_validate_eval_config_rejects_invalid_attribution(submitter):
    cfg = _cfg({"name": "region_intersection_pool"})
    cfg.submitter = submitter
    with pytest.raises((TypeError, ValueError), match="submitter"):
        validate_eval_config(cfg)
