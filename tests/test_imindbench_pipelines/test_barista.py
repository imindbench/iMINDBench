"""Smoke tests for the BaRISTA integration.

The port previously shipped with no tests at all. These cover the parts that
fail silently rather than loudly:

* the model registers, and does so through the tolerant registry import, so a
  missing xformers costs only BaRISTA;
* the shipped config carries the keys the model reads with ``cfg.get``, which
  would otherwise fall back to defaults with no warning;
* the two guards that turn a collapsed spatial encoding into an error. Both
  exist because the model will otherwise train happily on an all-UNKNOWN
  region encoding and report plausible but wrong numbers.
"""

import os

import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from imindbench.models import (  # noqa: E402
    MODEL_REGISTRY,
    UNAVAILABLE_MODEL_MODULES,
)

CONF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "imindbench/conf",
)

_HAS_XFORMERS = "barista_model" not in UNAVAILABLE_MODEL_MODULES

requires_barista = pytest.mark.skipif(
    not _HAS_XFORMERS,
    reason=f"barista_model unavailable: {UNAVAILABLE_MODEL_MODULES.get('barista_model')}",
)


def _barista_cfg(**overrides):
    cfg = OmegaConf.load(os.path.join(CONF, "model/barista.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
@requires_barista
def test_barista_is_registered():
    assert "barista" in MODEL_REGISTRY


def test_barista_unavailability_does_not_break_other_models():
    """Whatever happens to BaRISTA, the rest of the registry must still load.

    barista_components/transformer.py imports xformers.ops at module scope, so
    before the registry tolerated import failures this module could take every
    model down with it.
    """
    for name in ("logistic", "mlp", "cnn", "linear_baseline"):
        assert name in MODEL_REGISTRY


# ----------------------------------------------------------------------
# Config contract
# ----------------------------------------------------------------------
def test_barista_config_declares_the_keys_the_model_reads():
    """These are read with cfg.get(), so a missing key silently takes a default."""
    cfg = _barista_cfg()
    assert "min_resolved_region_fraction" in cfg
    assert 0.0 < float(cfg.min_resolved_region_fraction) <= 1.0
    assert "random_init" in cfg
    assert cfg.tokenizer.add_spatial_encoding is True


def test_barista_requires_coords():
    """BaRISTA opts into the coordinate path, so it depends on legacy routing.

    requires_coords=true means _get_channel_meta must resolve a coordinate frame
    for every provider. That is why BaRISTA on KelesBYD2024 and
    BerezutskayaPippi2022 was unreachable until the legacy path started applying
    the profile transform instead of rejecting the 'xyz'/'acpc' tags.
    """
    cfg = _barista_cfg()
    assert bool(cfg.get("requires_coords", False)) is True


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------
@requires_barista
def test_missing_brain_areas_raises_instead_of_collapsing_the_encoding():
    from imindbench.models.barista_model import Barista

    model = Barista.__new__(Barista)
    model.cfg = _barista_cfg()
    with pytest.raises(ValueError, match="carried no brain_areas"):
        model._get_region_enum_ids(["rec0"], None)


@requires_barista
def test_missing_brain_areas_is_allowed_when_spatial_encoding_is_off():
    """Opting out explicitly must stay possible."""
    from imindbench.models.barista_model import Barista

    model = Barista.__new__(Barista)
    model.cfg = _barista_cfg(tokenizer={"add_spatial_encoding": False})
    out = model._get_region_enum_ids(["rec0"], None)
    assert out.shape[0] == 1


@requires_barista
def test_non_destrieux_labels_raise():
    """Desikan-Killiany names resolve to UNKNOWN, which must not pass silently."""
    from imindbench.models.barista_model import Barista
    from imindbench.models.barista_components.atlas import DestrieuxAseg

    model = Barista.__new__(Barista)
    model.cfg = _barista_cfg()
    unknown = DestrieuxAseg.UNKNOWN.value
    with pytest.raises(ValueError, match="Destrieux atlas"):
        model._check_region_labels_are_destrieux(
            recording_id="rec0",
            brain_areas=["ctx-lh-superiortemporal"] * 4,
            enum_ids=[unknown] * 4,
        )


@requires_barista
def test_mostly_resolved_labels_pass():
    from imindbench.models.barista_model import Barista
    from imindbench.models.barista_components.atlas import DestrieuxAseg

    model = Barista.__new__(Barista)
    model.cfg = _barista_cfg()
    unknown = DestrieuxAseg.UNKNOWN.value
    resolved = next(e.value for e in DestrieuxAseg if e.value != unknown)
    # 3 of 4 resolved is above the 0.5 default, so this must not raise.
    model._check_region_labels_are_destrieux(
        recording_id="rec0",
        brain_areas=["ctx_lh_G_temp_sup-Lateral"] * 4,
        enum_ids=[resolved, resolved, resolved, unknown],
    )


@requires_barista
def test_empty_channel_list_is_not_an_error():
    from imindbench.models.barista_model import Barista

    model = Barista.__new__(Barista)
    model.cfg = _barista_cfg()
    model._check_region_labels_are_destrieux(
        recording_id="rec0", brain_areas=[], enum_ids=[]
    )


# ----------------------------------------------------------------------
# Atlas
# ----------------------------------------------------------------------
@requires_barista
def test_destrieux_atlas_resolves_real_region_names():
    """A sanity check that the enum is the Destrieux parcellation it claims."""
    from imindbench.models.barista_components.atlas import DestrieuxAseg

    names = {e.name for e in DestrieuxAseg}
    assert "UNKNOWN" in names
    # Destrieux names are sulcus/gyrus coded; the atlas must be non-trivial.
    assert len(names) > 50
